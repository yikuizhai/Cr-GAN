import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import init
import numpy as np
from torchvision.models import resnet18
import torch.nn.functional as F

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class ConditionalBatchNorm2d(nn.BatchNorm2d):

    """Conditional Batch Normalization"""

    def __init__(self, num_features, eps=1e-05, momentum=0.1,
                 affine=False, track_running_stats=True):
        super(ConditionalBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )

    def forward(self, input, weight, bias, **kwargs):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        output = F.batch_norm(input, self.running_mean, self.running_var,
                              self.weight, self.bias,
                              self.training or not self.track_running_stats,
                              exponential_average_factor, self.eps)
        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)
        size = output.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * output + bias


class CategoricalConditionalBatchNorm2d(ConditionalBatchNorm2d):

    def __init__(self, num_classes, num_features, eps=1e-5, momentum=0.1,
                 affine=False, track_running_stats=True):
        super(CategoricalConditionalBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.weights = nn.Embedding(num_classes, num_features)
        self.biases = nn.Embedding(num_classes, num_features)

        self._initialize()

    def _initialize(self):
        init.ones_(self.weights.weight.data)
        init.zeros_(self.biases.weight.data)

    def forward(self, input, c, **kwargs):
        weight = self.weights(c)
        bias = self.biases(c)

        return super(CategoricalConditionalBatchNorm2d, self).forward(input, weight, bias)

class memory_bank(nn.Module):

    def __init__(self,args,bank_size=10000):
        super(memory_bank, self).__init__()
        b,  k = args.batch_size, args.latent_dim
        self.bank_size = bank_size
        self.register_buffer("queues", torch.randn((self.bank_size,k*2),dtype=torch.float32))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.b = b
        self.k = k

    def dequeue_and_enqueue(self,feature):
        batch_size = feature.shape[0]

        ptr = int(self.queue_ptr)

        # replace the keys at ptr (dequeue and enqueue)
        self.queues[ptr:ptr + batch_size,:] = feature
        ptr = (ptr + batch_size) % self.bank_size  # move pointer

        self.queue_ptr[0] = ptr

    def generate_mixture_feature(self):
        p1 = np.random.beta(1, 1, size=(self.b, 1))
        p2 = 1 - p1
        p = np.concatenate([p1, p2], axis=-1)
        mask = np.array([np.random.choice(len(self.queues), 2, replace=False) for _ in range(self.b)])
        mu, var = self.queues[:,:self.k],self.queues[:,self.k:]
        mu_ = mu.view(self.bank_size, self.k).detach().cpu().numpy()[mask]
        var_ = var.view(self.bank_size, self.k).detach().cpu().numpy()[mask]
        mu_mix, var_mix = self.feature_interpolate(p, [mu_, var_])

        return [mu_mix, var_mix]

    def channel_shuffer(self,probs, u_var, l=1):
        u, var = u_var[0], u_var[1]
        assert u.shape[2] % l == 0
        dim, num = u.shape[2], int(u.shape[2] // l)
        b, k, c,= u.shape
        u_new = np.zeros((b, c))
        var_new = np.zeros((b, c))
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
                u_new[b_index, c_index] = uk[i, c_index]
                var_new[b_index, c_index] = vark[i, c_index]

        return torch.tensor(u_new).type(torch.FloatTensor), torch.tensor(var_new).type(torch.FloatTensor)

    def feature_interpolate(self, probs, u_var):
        u,var = u_var[0], np.exp(0.5*u_var[1])
        p = probs[:,:,np.newaxis]
        u_new = np.matmul(u.transpose(0,2,1),p)
        var_new = np.matmul(var.transpose(0,2,1),p)
        return torch.tensor(u_new).squeeze(-1).type(torch.FloatTensor), 2*torch.log(torch.tensor(var_new).squeeze(-1).type(torch.FloatTensor))

# class memory_bank(nn.Module):
#
#     def __init__(self,args,bank_size=10000):
#         super(memory_bank, self).__init__()
#         b,  k = args.batch_size, args.latent_dim
#         self.bank_size = bank_size
#         self.register_buffer("queues", torch.randn((self.bank_size,k*2),dtype=torch.float32))
#         self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
#         self.b = b
#         self.k = k
#
#     def dequeue_and_enqueue(self,feature):
#         batch_size = feature.shape[0]
#
#         ptr = int(self.queue_ptr)
#
#         # replace the keys at ptr (dequeue and enqueue)
#         self.queues[ptr:ptr + batch_size,:] = feature
#         ptr = (ptr + batch_size) % self.bank_size  # move pointer
#
#         self.queue_ptr[0] = ptr
#
#     def generate_mixture_feature(self):
#         p1 = np.random.beta(1, 1, size=(self.b, self.k))
#         p1 = p1[:, :, np.newaxis]
#         p2 = 1 - p1
#         p = np.concatenate([p1, p2], axis=-1)
#
#         mask = np.array([np.random.choice(len(self.queues), 2, replace=False) for _ in range(self.b)])
#         mu, var = self.queues[:,:self.k],self.queues[:,self.k:]
#         mu_ = mu.view(self.bank_size, self.k).detach().cpu().numpy()[mask]
#         var_ = var.view(self.bank_size, self.k).detach().cpu().numpy()[mask]
#         mu_mix, var_mix = self.feature_interpolate(p, [mu_, var_])
#
#         return [mu_mix, var_mix]
#
#     def feature_interpolate(self, probs, u_var):
#         u,var = u_var[0], np.exp(0.5*u_var[1])
#         u_new = np.sum(u.transpose(0,2,1) * probs,axis=-1,keepdims=False)
#         var_new = np.sum(var.transpose(0,2,1) * probs,axis=-1,keepdims=False)
#         return torch.tensor(u_new).squeeze(-1).type(torch.FloatTensor), 2*torch.log(torch.tensor(var_new).squeeze(-1).type(torch.FloatTensor))