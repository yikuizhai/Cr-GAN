import argparse
import os
import random
import warnings
import numpy as np
import itertools

import torch
import torch.nn.parallel
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import WeightedRandomSampler
from torchvision.utils import save_image

from crgan import utils
from crgan.models import Generator, Discriminator
from crgan.losses import TVLoss, SSIMLoss, KLLoss, FRLoss, RepelLoss, Virecg
from crgan.data import ltdataset
import logging
import datetime
import sys
from torch.utils.data import sampler

parser = argparse.ArgumentParser(description='CR-GAN Training')

parser.add_argument('--path', metavar='DIR', required=True,
                    help='path to dataset')
parser.add_argument('--log_dir', metavar='DIR',default='runs/crgan',
                    help='path to save state_dict')
parser.add_argument('-w', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--iters', default=15000, type=int, metavar='N',
                    help='number of total iteration to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=10, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--schedule', default=[200], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--factor', default=0, type=int, metavar='N',
                    help='The factor construct the long tailed MSTAR dataset')
parser.add_argument('--seed', default=10, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument("--saved_interval", type=int, default=10, help="interval between saved time")

parser.add_argument('--classes', default=10, type=int, metavar='N',
                    help='number of class (default: 10)')
parser.add_argument('--lr', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate for GAN')
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.9, help="adam: decay of second order momentum of gradient")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--smoothing", type=float, default=0.2, help="label smoothing for real images")
parser.add_argument("--num_samples", type=int, default=2, help="number of data samples fusion")
parser.add_argument("--train_samples", type=int, default=40,
                    help="total balanced samples selected from the input dataset")


def init_logger(log_file=None, log_dir=None, log_level=logging.INFO, mode='w', stdout=True):
    """
    log_dir: 日志文件的文件夹路径
    mode: 'a', append; 'w', 覆盖原文件写入.
    """
    def get_date_str():
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d_%H-%M-%S')

    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    if log_dir is None:
        log_dir = '~/temp/log/'
    if log_file is None:
        log_file = 'log_' + get_date_str() + '.txt'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, log_file)
    # 此处不能使用logging输出
    print('log file path:' + log_file)

    logging.basicConfig(level=logging.DEBUG,
                        format=fmt,
                        filename=log_file,
                        filemode=mode)

    if stdout:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(log_level)
        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    return logging

class ema(object):
    def __init__(self, source, target, decay=0.9999, start_itr=0):
        self.source = source
        self.target = target
        self.decay = decay
        # Optional parameter indicating what iteration to start the decay at
        self.start_itr = start_itr
        # Initialize target's params to be source's
        self.source_dict = self.source.state_dict()
        self.target_dict = self.target.state_dict()
        print('Initializing EMA parameters to be source parameters...')
        with torch.no_grad():
            for key in self.source_dict:
                self.target_dict[key].data.copy_(self.source_dict[key].data)
                # target_dict[key].data = source_dict[key].data # Doesn't work!

    def update(self, itr=None):
        # If an iteration counter is provided and itr is less than the start itr,
        # peg the ema weights to the underlying weights.
        if itr and itr < self.start_itr:
            decay = 0.0
        else:
            decay = self.decay
        with torch.no_grad():
            for key in self.source_dict:
                self.target_dict[key].data.copy_(self.target_dict[key].data * decay
                                                 + self.source_dict[key].data * (1 - decay))

class VAE(nn.Module):
    def __init__(self,E, D):
        super(VAE, self).__init__()
        self.E = E
        self.D = D

    def encode(self, x):
        _,(mu,logvar),_ = self.E(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        gen = self.D(z)
        return gen

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

    def loss_function(self, recon_x, x, mu, logvar, beta=1.0):
        recon = F.l1_loss(recon_x, x)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        print(recon.item(),KLD.item())

        return recon + beta * KLD

def count_trainable_parameters(model):
    """Counts the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    args = parser.parse_args()
    sub_dataset_name = os.path.basename(os.path.normpath(args.path))
    args.log_dir = os.path.join(args.log_dir,sub_dataset_name)
    args.device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    print(args)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # Define Dataset and Dataloader
    if args.channels == 1:
        # Define Dataset and Dataloader
        transformer = transforms.Compose([
                                          transforms.Resize((args.img_size, args.img_size)),
                                          transforms.Grayscale(1),
                                        #   transforms.RandomRotation(10),
                                        #   transforms.RandomHorizontalFlip(),
                                        #   transforms.RandomVerticalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.5], std=[0.5])
                                          ]
                                         )
    else:
        transformer = transforms.Compose([
                                          transforms.Resize((args.img_size, args.img_size)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                        ])

    dataset = ltdataset(
        root=args.path,
        num_samples=args.train_samples,
        transform=transformer,
        seed=args.seed,
    )

    # if args.factor>0:
    #     sampler = get_reverse_sampler(dataset)
    #     loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers)
    # else:
    #     loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=InfiniteSampler(len(dataset)), num_workers=args.workers)

    # create CycleFeatureGAN model
    print("=> creating generator...")
    generator = Generator(args)
    generator_E = Generator(args)
    print(generator)
    print("=> created!")


    print("=> creating discriminator...")
    discriminator = Discriminator(args)
    discriminator_E = Discriminator(args)
    discriminator_E.load_state_dict(discriminator.state_dict())
    print(discriminator)
    print("=> created!")



    generator_E.load_state_dict(generator.state_dict())
    discriminator_E.load_state_dict(discriminator.state_dict())

    # set_requires_grad(generator_E, False)
    # set_requires_grad(discriminator_E, False)

    # Loss function definition
    fr_loss = FRLoss(args)
    kl_loss = KLLoss()
    ssim_loss = SSIMLoss()
    tv_loss = TVLoss()
    repel_loss = RepelLoss()

    generator = generator.to(args.device)
    discriminator = discriminator.to(args.device)
    generator_E = generator_E.to(args.device)
    discriminator_E = discriminator_E.to(args.device)
    fr_loss = fr_loss.to(args.device)
    kl_loss = kl_loss.to(args.device)
    ssim_loss = ssim_loss.to(args.device)
    tv_loss = tv_loss.to(args.device)
    repel_loss = repel_loss.to(args.device)

    # Optimizer definition
    optimizer_g = torch.optim.Adam(generator.parameters(), args.lr*2, betas=(args.b1, args.b2))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), args.lr*2, betas=(args.b1, args.b2))
    optimizer_vae = torch.optim.Adam(itertools.chain(generator.parameters(),discriminator.parameters()), args.lr * 2, betas=(args.b1, args.b2))
    # optimizer_g = torch.optim.RMSprop(generator.parameters(), lr=0.0002)
    # optimizer_d = torch.optim.RMSprop(discriminator.parameters(), lr=0.0002)
    print(discriminator.state_dict().keys())

    train(loader, [generator, discriminator, generator_E, discriminator_E], [fr_loss, kl_loss, ssim_loss, tv_loss,repel_loss], [optimizer_g, optimizer_d, optimizer_vae], args)

def exists(val):
    return val is not None

def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool

def train(loader, models, losses, optimizers, args):
    logging = init_logger(log_dir=args.log_dir)
    generator, discriminator,generator_E, discriminator_E = models[0], models[1], models[2], models[3]
    fr_loss, kl_loss, ssim_loss, tv_loss, repel_loss = losses[0], losses[1], losses[2], losses[3], losses[4]
    optimizer_g, optimizer_d, optimizer_vae = optimizers[0], optimizers[1], optimizers[2]
    vic = Virecg().to(args.device)

    g_ema = ema(generator, generator_E, 0.99, 1000)
    d_ema = ema(discriminator, discriminator_E, 0.99, 0)

    vae = VAE(discriminator,generator).to(args.device)

    global D_history_parameters
    global G_history_parameters

    D_history_parameters = discriminator.parameters()
    G_history_parameters = generator.parameters()
    bank_size = min(10000,int(((len(loader.dataset) // args.batch_size))*args.batch_size))
    print(bank_size)
    mb = utils.memory_bank(args,bank_size).to(args.device)
    fixed_z = torch.randn(64, args.latent_dim, device=args.device)

    a = 0.0

    g_valid_loss, g_img_recon_loss =0,0

    iter_loader = iter(loader)

    for iters in range(1,args.iters+1):

        imgs, labels = next(iter_loader)


        generator.train()
        discriminator.train()

        batch_size = imgs.size(0)

        real_imgs, labels = imgs.to(args.device), labels.to(args.device)


        # Get the general feature, specific mixture feature and smoothing labels
        # mix_mu_var, mu_var, mask = get_feature_probability(discriminator,real_imgs,args)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        set_requires_grad(discriminator,True)
        set_requires_grad(generator,False)
        optimizer_d.zero_grad()
        discriminator.train()

        mix_mu_var = mb.generate_mixture_feature()

        mix_style = reparameterize(mix_mu_var).to(args.device)
        gen_imgs = generator(mix_style)


        # z = torch.randn(batch_size, args.latent_dim).cuda()
        # random_imgs = generator(z)

        # Output for real images
        real_pred, real_feature, real_style = discriminator(real_imgs)

        # style = reparameterize(real_feature)
        # recon_imgs = generator(style)

        # Output for fake images

        fake_pred1, fake_feature,fake_style = discriminator(gen_imgs.detach())

        # fake_pred2, _ ,_= discriminator(random_imgs.detach())

        # fake_pred3, _ = discriminator(recon_imgs.detach())

        # Losses for discriminator
        # d_vality_loss = 0.5 * (
        #         (torch.mean(F.relu(1 + fake_pred1)) + torch.mean(F.relu(1 + fake_pred2)))/2 + torch.mean(
        #     F.relu(1 - real_pred)))
        d_vality_loss = 0.5 * (torch.mean(F.relu(1 + fake_pred1)) + torch.mean(F.relu(1 - real_pred)))

        d_feature_recon_loss = fr_loss(real_feature, fake_feature, mix_mu_var) + vic(fake_style,mix_style,real_style)

        d_loss = d_vality_loss + d_feature_recon_loss

        # Optimize the discriminator
        d_loss.backward()
        optimizer_d.step()

        # print(f"feature loss:{alignment_loss(fake_style,mix_style).item()}")

        # -----------------
        #  Train Generator
        # -----------------
        if iters % 5==0:
            # -----------------
            #  Train VAE
            # -----------------
            set_requires_grad(discriminator, True)
            set_requires_grad(generator, False)
            discriminator.train()
            optimizer_d.zero_grad()
            recon_batch, mu, logvar, style = vae(real_imgs)
            loss = vae.loss_function(recon_batch, real_imgs, mu, logvar)
            loss.backward()
            optimizer_d.step()
            print(f"VAE Loss:{loss.item()}")

            set_requires_grad(discriminator, False)
            set_requires_grad(generator, True)
            generator.train()

            mix_mu_var = mb.generate_mixture_feature()

            optimizer_g.zero_grad()

            recon_imgs = generator(style.detach())

            z = torch.randn(batch_size, args.latent_dim, device=args.device)

            # Generate fake images and reconstruct real images
            mix_style = reparameterize(mix_mu_var).to(args.device)
            gen_imgs = generator(mix_style)

            # style = reparameterize(ema_feature)
            # recon_imgs = generator(style)

            random_imgs = generator(z)

            # Discriminator output for fake images
            validity1, feature, fake_style = discriminator(gen_imgs)

            # validity2, _,_ = discriminator(random_imgs)

            # validity3, _ = discriminator(recon_imgs)

            # Losses for generator
            # g_valid_loss = -(torch.mean(validity1) + torch.mean(validity2))/2
            g_img_recon_loss = F.l1_loss(recon_imgs,real_imgs)
            g_valid_loss = -torch.mean(validity1)
            ms_loss = msloss(torch.cat([mix_style, style.detach(), z], dim=0), torch.cat([gen_imgs, recon_imgs, random_imgs],dim=0))

            g_vic = vic(fake_style,mix_style,real_style.detach())
            g_feature_recon_loss = fr_loss([real_feature[0].detach(),real_feature[1].detach()], feature, mix_mu_var) + g_vic
            g_loss = g_valid_loss + g_img_recon_loss + 0.1*g_feature_recon_loss  + 0.1 * ms_loss

            print(f"g_img_recon_loss:{g_img_recon_loss} g_vic:{g_vic}")

            # Update the parameters of the generator
            g_loss.backward()
            optimizer_g.step()

        discriminator_E.eval()
        with torch.no_grad():
            _, ema_feature,_ = discriminator_E(real_imgs)
            mb.dequeue_and_enqueue(torch.cat(ema_feature, dim=1))

        g_ema.update(iters)
        d_ema.update(iters)

        # Checkpoints saving, visualization and information logging
        g_info = "Iters:{} Generator: valid_loss : {:.6f}  " \
                 "img_recon_loss: {:.6f}".format(iters,
                                                 g_valid_loss,
                                                 g_img_recon_loss)

        d_info = "Iters:{} Discriminator: valid_loss : {:.6f}  " \
                 "feature_recon_loss: {:.6f}".format(
            iters,
            d_vality_loss,
            d_feature_recon_loss
        )
        save_images(iters, generator, args, fixed_z)
        print_info(iters, [g_info, d_info], args, logging)

        save_checkpoint(iters,[generator.state_dict(), discriminator.state_dict()], args)

def alignment_loss(x1,x2):
    x1 = nn.functional.normalize(x1, dim=-1)
    x2 = nn.functional.normalize(x2, dim=-1)
    x2 = x2.to(x1.device)
    return 2 - 2 * (x1 * x2).sum(dim=1).mean()

def msloss(style,img):
    b = style.size(0)
    index = np.arange(b)
    random.shuffle(index)
    style = style[index]
    img = img[index]
    if not img.size(0)%2==0:
        img, style = img[:-1], style[:-1]
    fake_image1, fake_image2 = torch.split(img, int(img.size(0)//2), dim=0)
    style1, style2 = torch.split(style, int(style.size(0) // 2), dim=0)
    lz = torch.mean(torch.abs(fake_image2 - fake_image1)) / torch.mean(
        torch.abs(style2 - style1))
    eps = 1 * 1e-5
    loss_lz = 1 / (lz + eps)
    return loss_lz

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
    def update_average(self, old, new):
        if not exists(old):
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater,ma_model, current_model,type=None,second=False):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

def reset_parameter_averaging(D,G,DE,GE):
    DE.load_state_dict(D.state_dict())
    GE.load_state_dict(G.state_dict())

# def reparameterize(mu_std):
#     mean, logvar = mu_std[0], mu_std[1]
#     eps = torch.randn_like(mean)
#     z = mean + eps * torch.exp(logvar)
#     return z

def reparameterize(mu_std):
    """
    Reparameterization trick to sample from N(mu, var) from
    N(0,1).
    :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
    :return: (Tensor) [B x D]
    """
    mu,std = mu_std[0], mu_std[1]
    std = torch.exp(0.5 * std)
    eps = torch.randn_like(std)
    return eps * std + mu

def get_mask(b,k):
    rm_ls = []
    BK = [np.arange(b) for _ in range(k)]
    mask = np.array(list(itertools.product(*BK)))
    for i, m in enumerate(mask):
        if np.max(np.bincount(m))==k:
            rm_ls.append(i)
    mask = np.delete(mask,rm_ls,axis =0)
    select = np.random.randint(mask.shape[0], size=b)
    return mask[select,...]

def get_feature_probability(model,data,args, T=2):
    model.eval()

    _, f = model(data)

    mu, var = f[0], f[1]

    N = 2
    b, _ = mu.size()
    mask = np.array([np.random.choice(b, N, replace=False) for _ in range(b)])

    p = np.random.randn(b, N).astype(np.float32)

    p = F.softmax(torch.tensor(p / T), dim=1).detach().cpu().numpy()

    mu_ = mu.view(b,100,1,1).detach().cpu().numpy()[mask]
    var_ = var.view(b, 100, 1, 1).detach().cpu().numpy()[mask]

    mu_mix,var_mix = channel_shuffer(p, [mu_, var_])

    return [mu_mix.to(mu.device).view(b,-1), var_mix.to(mu.device).view(b,-1)], [mu, var], mask


def add_noise(features,p=1.0):
    b, c, h, w = features.shape
    noise = torch.randn((c,h,w))
    num_channel = round(c * p)
    for b_index, fk in enumerate(features):
        ls = np.arange(c)
        c_index = np.random.choice(ls, num_channel, replace=False)
        fk[c_index, ...] = noise[c_index, ...]
    return features


def channel_shuffer(probs, u_var, l=1):
    u, var = u_var[0], u_var[1]
    assert u.shape[2] % l == 0
    dim, num = u.shape[2], int(u.shape[2] // l)
    b, k, c, h, w = u.shape
    u_new = np.zeros((b, c, h, w))
    var_new = np.zeros((b, c, h, w))
    for b_index, (uk, vark, pk) in enumerate(zip(u, var, probs)):
        ls = np.arange(num)
        for i, p in enumerate(pk):
            if i == len(pk) - 1:
                patch_index = ls
            else:
                num_patch = round(num * p)
                patch_index = np.random.choice(ls, num_patch, replace=False)
                ls = np.setdiff1d(ls, patch_index)
                if len(ls) == 0:
                    break
            c_index = [i for patch in patch_index for i in range(patch * l, (patch + 1) * l)]
            u_new[b_index, c_index, ...] = uk[i, c_index, ...]
            var_new[b_index, c_index, ...] = vark[i, c_index, ...]
    return torch.tensor(u_new).type(torch.FloatTensor), torch.tensor(var_new).type(torch.FloatTensor)



def save_images(iter, generator, args, fixed_z):
    generator.eval()

    gen_imgs = generator(fixed_z)
    if iter % args.saved_interval == 1:
        save_dir = os.path.join(args.log_dir, "GAN")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_image(gen_imgs, os.path.join(save_dir, f"iter_{iter}.png"), nrow=8,normalize=True)


def save_checkpoint(iter, state, args, best=False):
    if best:
        save_dir = os.path.join(args.log_dir, "Checkpoints")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        torch.save(state[0], os.path.join(save_dir, f"G_best.pkl"))
        torch.save(state[1], os.path.join(save_dir, f"D_best.pkl"))

    if iter % (args.saved_interval*10) == 1:
        save_dir = os.path.join(args.log_dir, "Checkpoints")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        torch.save(state[0], os.path.join(save_dir, f"G_iter_{iter}.pkl"))
        torch.save(state[1], os.path.join(save_dir, f"D_iter_{iter}.pkl"))


def print_info(iter, info, args,logging):
    if iter % args.saved_interval == 1:
        print(info[0])
        print(info[1])
        logging.info(info[0])
        logging.info(info[1])


def update_ema_variables(discriminator,generator, iters, alpha=1):
    # Use the true average until the exponential average is more correct
    global D_history_parameters
    global G_history_parameters

    if iters>0:
        for history_param, param in zip(D_history_parameters, discriminator.parameters()):
            param.data = alpha * history_param.data + (1-alpha) * param.data
        D_history_parameters = discriminator.parameters()

        for history_param, param in zip(G_history_parameters, generator.parameters()):
            param.data = alpha * history_param.data + (1-alpha) * param.data
        G_history_parameters = generator.parameters()

def one_hot(labels, smoothing, classes):
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((labels.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        true_dist.scatter_(1, labels.data.unsqueeze(1), confidence)
    return true_dist


def gram(x):
    (bs, ch,h,w) = x.size()
    f = x.view(ch, bs*h*w)
    f = nn.functional.normalize(f, dim=-1)
    f_T = f.transpose(1, 0)
    G = f.mm(f_T)
    return G


def specific_content_mixture(f):
    (bs, ch, h, w) = f.size()
    g = gram(f)
    f = f.view(bs, ch, -1)
    f_ = torch.matmul(g, f).view(bs, ch, h, w)
    return f_


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_reverse_sampler(dataset):
    weights = list(map(lambda x: 1 / x, dataset.count))
    weights = [weights[target] for (_, target) in dataset]
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(dataset), replacement=True)
    return sampler

class InfiniteSampler(sampler.Sampler):

    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        while True:
            order = np.random.permutation(self.num_samples)
            for i in range(self.num_samples):
                yield order[i]

    def __len__(self):
        return None


def shuffle_tensor(tensor):
    index = [i for i in range(tensor.size(1))]
    for i in (range(tensor.size(0))):
        random.shuffle(index)
        tensor[i] = tensor[i,index,...]
    return tensor

if __name__ == '__main__':
    import time
    start_time = time.time()

    main()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("\n--- Training Finished ---")
    print(f"Total Wall-Clock Time: {elapsed_time:.2f} seconds ({elapsed_time/3600:.2f} hours)")
