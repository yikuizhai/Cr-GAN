from torch import nn, einsum
from math import floor, log2
from functools import partial
import torch
from einops import rearrange, repeat
import torch.nn.functional as F
from kornia.filters import filter2d
import math
from crgan.utils import SpectralNorm
import numpy as np

class Generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, args):
        super(Generator, self).__init__()

        # self.label_emb = nn.Embedding(10, 100)
        self.embedding = nn.Linear(args.latent_dim, 1024)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, stride=1, padding=0),  # 1x1 → 4x4
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # 4x4 → 8x8
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 8x8 → 16x16
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 16x16 → 32x32
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 32x32 → 64x64
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2),

            # Adjust the final output channel count.
            nn.Conv2d(32, args.channels, 3, padding=1),
            nn.Tanh()
        )
        self.weights_init_normal()

    def weights_init_normal(self):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find("Conv") != -1:
                torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find("BatchNorm2d") != -1:
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)

    def forward(self, input, label=None):
        # x = torch.cat([input, label], 1)
        # x = torch.mul(self.label_emb(label), input)
        x = self.embedding(input).unsqueeze(-1).unsqueeze(-1)
        x = self.deconv(x)
        return x

# class Generator(nn.Module):
#     def __init__(self,args):
#         super(Generator, self).__init__()
#
#         self.init_size = args.img_size // 4
#         self.l1 = nn.Sequential(nn.Linear(args.latent_dim, 128 * self.init_size ** 2))
#
#         self.conv_blocks = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(128, 128, 3, stride=1, padding=1),
#             nn.BatchNorm2d(128, 0.8),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(128, 64, 3, stride=1, padding=1),
#             nn.BatchNorm2d(64, 0.8),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64, args.channels, 3, stride=1, padding=1),
#             nn.Tanh()
#         )
#
#     def forward(self, z):
#         out = self.l1(z)
#         out = out.view(out.shape[0], 128, self.init_size, self.init_size)
#         img = self.conv_blocks(out)
#         return img

class DiscriminatorBlock(nn.Module):
    def __init__(self, input_dim, output_dim1, output_dim2, first=False,final=False):
        super().__init__()
        self.first = first
        self.final = final

        block1 = [SpectralNorm(nn.Conv2d(input_dim, output_dim1, 3, 2, 1)),
                  nn.LeakyReLU(0.2,inplace=True)]
        self.conv1 = nn.Sequential(*block1)

        block2 = [
            SpectralNorm(nn.Conv2d(input_dim, output_dim2, 3, 2, 1)),
            nn.LeakyReLU(0.2,inplace=True)
            ]
        self.conv2 = nn.Sequential(*block2)

    def forward(self, x, f=None):
        d_in = self.conv1(x)
        # if f is not None:
        #     g_in = self.conv2(x + f)
        # else:
        #     g_in = self.conv2(x)
        return d_in, None

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()

        self.channels = args.channels
        self.img_size = args.img_size
        self.class_num = args.classes
        self.blocks = nn.ModuleList([])
        self.latent_dim = args.latent_dim
        first_channel = 16
        ds_size = 2
        max_channel = 128
        block_num = int(math.log2(args.img_size // ds_size))

        output_dim1 = [min(first_channel * (2 ** i), max_channel) for i in range(block_num)]
        output_dim2 = [min(first_channel * (2 ** i), max_channel) for i in range(block_num)]
        input_dim = [min(first_channel * (2 ** i), max_channel) for i in range(block_num-1)]
        input_dim.insert(0, self.channels)

        for i, (dim, dim1, dim2) in enumerate(zip(input_dim, output_dim1, output_dim2)):
            self.blocks.append(DiscriminatorBlock(input_dim=dim, output_dim1=dim1, output_dim2=dim2))

        # Output layers
        self.adv_layer = nn.Sequential(SpectralNorm(nn.Conv2d(output_dim1[-1], 128, 3, 1, 1)),
                                       nn.Flatten()
                                       )

        self.to_logit = SpectralNorm(nn.Linear(128*ds_size**2, 1))

        self.conv_mu_var =nn.Sequential(
            SpectralNorm(nn.Conv2d(128, 512, 3, 1, 1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Flatten(),
            SpectralNorm(nn.Linear(512 * 2**2, 512)),
            nn.BatchNorm1d(num_features=512,momentum=0.9),
            nn.LeakyReLU(0.2,inplace=True)
        )

        self.mu_layer = nn.Sequential(
            SpectralNorm(nn.Linear(512, self.latent_dim))
        )

        self.var_layer = nn.Sequential(
            SpectralNorm(nn.Linear(512, self.latent_dim))
        )

        self.feature_layer = nn.Sequential(
            SpectralNorm(nn.Linear(512, self.latent_dim))
        )

        self.avg = nn.AdaptiveAvgPool2d(1)

        self.l_y = nn.Embedding(self.class_num, 128 * 4)

        #self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if (isinstance(module,nn.Conv2d) or isinstance(module, nn.Linear)):
                nn.init.normal_(module.weight.data, 0,0.02)

    def forward(self, x, y=None):
        gin = None
        for i, block in enumerate(self.blocks):
            x, gin = block(x,f=gin)

        # adv_x = x.clone().detach()
        x_valid = self.adv_layer(x)
        validity = self.to_logit(x_valid)
        if y is not None:
            label_emb = self.l_y(y)
            validity += torch.sum(label_emb.view(y.size(0), -1) * x, dim=1, keepdim=True)

        mu_var = self.conv_mu_var(x)
        mu = self.mu_layer(mu_var)
        var = self.var_layer(mu_var)
        # feature = self.feature_layer(mu_var)

        return validity, [mu,var], mu_var.view(mu_var.size(0),-1)

class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out

if __name__ == '__main__':
    g = Generator(image_size=128, latent_dim=512)
    print(g)
