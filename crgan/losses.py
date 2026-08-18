import torch
import torch.nn.functional as F
import torch.nn as nn
from math import exp
from torch.autograd import Variable
import numpy as np
import random

# TVLoss for smoothing image generation
class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

# Ssim loss for real images reconstruction
class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


# Loss for label distribution optimize
def KLLoss():
    return nn.KLDivLoss()


def RepelLoss():
    return nn.MSELoss()


class MMD_loss(nn.Module):

    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss

def alignment_loss(x1,x2):
    x1 = nn.functional.normalize(x1, dim=-1)
    x2 = nn.functional.normalize(x2, dim=-1)
    x2 = x2.to(x1.device)
    return 2 - 2 * (x1 * x2).sum(dim=1).mean()

class FRLoss(nn.Module):
    def __init__(self,args):
        super(FRLoss, self).__init__()
        self.loss1 = nn.MSELoss()
        self.repel = uniform(args)
        self.mmd = MMD_loss()

    def forward(self, real, fake, mix):
        real_mu, real_log_var = real[0], real[1]
        fake_mu, fake_log_var = fake[0], fake[1]
        mix_mu, mix_log_var = mix[0], mix[1]

        # loss1 = 0.5*(self.mmd(real_mu,fake_mu) + self.mmd(real_log_var, fake_log_var))

        # kld_loss1 = torch.mean(-0.5 * torch.sum(1 + real_log_var - real_mu ** 2 - real_log_var.exp(), dim=1), dim=0)
        #
        # kld_loss2 = torch.mean(-0.5 * torch.sum(1 + fake_log_var - fake_mu ** 2 - fake_log_var.exp(), dim=1), dim=0)
        #
        # loss2 = 0.5*(kld_loss1+kld_loss2)

        # fake_real_mu_var = torch.cat([fake_mu_var,real_mu_var],dim=0)
        fake_real_mu = torch.cat([fake_mu, real_mu], dim=0)
        fake_real_var = torch.cat([fake_log_var, real_log_var], dim=0)

        # cv_fr = fake_real_mu/(fake_real_var.exp().sqrt()+1e-5)
        # cv_mix = mix_mu/(mix_log_var.exp().sqrt()+1e-5)
        # cv_f = fake_mu/(fake_log_var.exp().sqrt()+1e-5)

        # cv_fr = fake_real_var.exp().sqrt() / (fake_real_mu + 1.0)
        # cv_mix = mix_log_var.exp().sqrt() / (mix_mu + 1.0)
        # cv_f = fake_log_var.exp().sqrt() / (fake_mu + 1.0)
        loss2 = 0.5*(alignment_loss(fake_mu,mix_mu) + alignment_loss(fake_log_var,mix_log_var))
        loss3 =0.5*(self.repel(fake_mu,mix_mu,real_mu) + self.repel(fake_log_var,mix_log_var,real_log_var))

        print(f"Alignment:{loss2} Repel:{loss3}")
        return loss2 + loss3

    def comput_loss(self,f1_list,f2_list):
        loss = 0
        for _,(f1,f2) in enumerate(zip(f1_list,f2_list)):
            loss += self.loss(f1,f2)
        loss /= len(f1_list)
        return loss

    def reparameterize(self, mu, std):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        eps = torch.randn_like(std)
        return eps * std + mu

# class FRLoss(nn.Module):
#     def __init__(self,args,a=0.1):
#         super(FRLoss, self).__init__()
#         self.loss = nn.MSELoss()
#         self.args = args
#
#     def recon_loss(self,I1,I2):
#         return self.loss(I1, I2)
#
#     def forward(self,real_imgs,mask_prob,gen_imgs):
#         gen_imgs_recon_loss = 0.0
#
#         mask, probs = mask_prob[0],mask_prob[1]
#         # real_imgs_recon_loss = self.recon_loss(recon_imgs,real_imgs)
#
#         gen_imgs_target = torch.tensor(real_imgs.detach().cpu().numpy()[mask]).cuda(self.args.gpu)
#         gen_imgs_prob = torch.tensor(probs).cuda(self.args.gpu)
#
#         for _,(prob, real_img, gen_img) in enumerate(zip(gen_imgs_prob, gen_imgs_target, gen_imgs)):
#             for _, (p, real) in enumerate(zip(prob, real_img)):
#                 gen_imgs_recon_loss += p * self.recon_loss(real.unsqueeze(0), gen_img.unsqueeze(0))
#
#         gen_imgs_recon_loss = gen_imgs_recon_loss/real_imgs.size(0)
#         return gen_imgs_recon_loss
#
# def shuffle_tensor(tensor):
#     index = [i for i in range(tensor.size(0))]
#     random.shuffle(index)
#     tensor = tensor[index]
#     return tensor
#
# class Cosclose(nn.Module):
#     def flatten(self,feature):
#         return feature.view(feature.size(0), -1)
#
#     def forward(self, f1, f2):
#         b,c,h,w = f1.size()
#         f1 = f1.view(b,c,h*w)
#         f2 = f2.view(b, c, h * w)
#         f1 = nn.functional.normalize(f1, dim=-1)
#         f2 = nn.functional.normalize(f2, dim=-1)
#         l_pos = torch.mean(torch.einsum('nck,nck->nc', [f1, f2]),dim=-1)
#         return (F.relu(1-l_pos)).mean()

def vae_loss(real,recon_image,real_image):
    real_mu, real_log_var = real[0], real[1]

    loss1 = torch.sum(-0.5 * torch.sum(1 + real_log_var - real_mu ** 2 - real_log_var.exp(), dim=1), dim=0)


    loss2 = F.l1_loss(recon_image,real_image)

    return loss1 + loss2

class Cosclose(nn.Module):
    def flatten(self,feature):
        return feature.view(feature.size(0), -1)

    def forward(self, f1, f2):
        f1 = nn.functional.normalize(f1, dim=-1)
        f2 = nn.functional.normalize(f2, dim=-1)
        l_pos = torch.mean(torch.einsum('ck,ck->c', [f1, f2]),dim=-1)
        return (F.relu(1-l_pos)).mean()


class argrepel(nn.Module):
    def __init__(self,args):
        super(argrepel, self).__init__()
        self.loss = nn.MSELoss()
        self.args = args

    def forward(self, fake, real,second=False):
        b = fake.size(0)
        fake = fake.unsqueeze(0).expand(b,-1,-1,-1,-1)
        real = real.unsqueeze(0).expand(b, -1, -1, -1, -1).permute(1,0,2,3,4)
        index = torch.mean((fake-real)**2,dim=(2,3,4))
        if second:
            diag = torch.diag(index)  # 取 a 对角线元素，输出为 1*3
            a_diag = torch.diag_embed(diag)  # 由 diag 恢复为三维 3*
            index = index-a_diag
            index += 100 * torch.eye(b, device=fake.device)
        index = torch.argmin(index,dim=-1)
        target = real[index]
        return self.loss(fake,target)


class img_recon_loss(nn.Module):
    def __init__(self,args,a=0.1):
        super(img_recon_loss, self).__init__()
        self.loss1 = nn.L1Loss()
        self.loss2 = SSIMLoss()
        self.a = a
        self.args = args

    def recon_loss(self,I1,I2):
        return self.loss1(I1, I2)

    def gen_loss(self,I1,I2):
        return F.relu(0.2-self.loss2(shuffle_tensor(I1),shuffle_tensor(I2)))

    def forward(self,real_imgs,mask_prob,gen_imgs,recon_imgs):
        # gen_imgs_recon_loss = 0.0
        #
        # mask, probs = mask_prob[0],mask_prob[1]
        real_imgs_recon_loss = self.recon_loss(recon_imgs,real_imgs)
        #
        # gen_imgs_target = torch.tensor(real_imgs.detach().cpu().numpy()[mask]).cuda(self.args.gpu)
        # gen_imgs_prob = torch.tensor(probs).cuda(self.args.gpu)
        #
        # for _,(prob, real_img, gen_img) in enumerate(zip(gen_imgs_prob, gen_imgs_target, gen_imgs)):
        #     for _, (p, real) in enumerate(zip(prob, real_img)):
        #         gen_imgs_recon_loss += p * self.recon_loss(real.unsqueeze(0), gen_img.unsqueeze(0))
        #
        # gen_imgs_recon_loss = gen_imgs_recon_loss/real_imgs.size(0)
        # print(real_imgs_recon_loss.item(),gen_imgs_recon_loss.item())
        return real_imgs_recon_loss

def identity_loss(feature,loss):
    b,c = feature.size()
    target = torch.eye(c, device=feature.device)
    g = gram(feature)
    return loss(g,target)

def gram(f):
    # f = nn.functional.normalize(f, dim=-1)
    f_T = f.transpose(1, 0)
    G = f_T.mm(f)
    return G

class uniform(nn.Module):
    def __init__(self,args,T=0.7):
        super(uniform, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        self.T = T
        self.args = args

    def flatten(self,feature):
        # feature = F.adaptive_avg_pool2d(feature,1)
        return feature.view(feature.size(0), -1)

    def forward(self,cv_f,cv_mix,cv_fr):
        cv_f = nn.functional.normalize(cv_f, dim=-1)
        cv_mix = nn.functional.normalize(cv_mix, dim=-1)
        cv_fr = nn.functional.normalize(cv_fr, dim=-1)
        cv_mix = cv_mix.to(cv_f.device)

        l_pos = torch.einsum('ck,ck->c', [cv_f, cv_mix]).unsqueeze(-1)
        l_neg = torch.einsum('ik,jk->ij', [cv_f, cv_fr])

        # diag = torch.diag(l_neg[:,:l_neg.size(0)])  # 取 a 对角线元素，输出为 1*3
        # a_diag = torch.diag_embed(diag+100)  # 由 diag 恢复为三维 3*
        # l_neg[:,:l_neg.size(0)] = l_neg[:,:l_neg.size(0)] - a_diag

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        return self.loss(logits,labels)

# Loss for feature reconstruction
class infonce(nn.Module):
    def __init__(self,  temperature=1.0, use_cosine_similarity=True):
        super(infonce, self).__init__()
        self.temperature = temperature
        self.device = "cuda"
        self.softmax = torch.nn.Softmax(dim=-1)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self,batch_size):
        diag = np.eye(2 * batch_size)
        l1 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
        l2 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        batch_size = zis.size(0)
        self.mask_samples_from_same_repr = self._get_correlated_mask(batch_size).type(torch.bool)

        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, batch_size)
        r_pos = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * batch_size)


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class Virecg(nn.Module):

    def __init__(self):
        super(Virecg, self).__init__()
        self.sim_coeff = 25
        self.std_coeff = 25
        self.cov_coeff = 1

    def forward(self,fake, mix,real):
        fake_gram = gram(fake)
        real_gram = gram(real)

        style_loss = F.mse_loss(fake_gram,real_gram)

        # feature_recon = F.mse_loss(fake,mix)

        feature = torch.cat([fake,real],dim=0)
        std_feature = torch.sqrt(feature.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_feature))


        return style_loss + std_loss
