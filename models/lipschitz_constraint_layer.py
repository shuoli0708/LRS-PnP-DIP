import torch
import torch.nn as nn
import numpy as np
from torch.nn import Parameter

def act(act_fun = 'LeakyReLU'):
    '''
        Either string defining an activation function or module (e.g. nn.ReLU)
    '''
    if isinstance(act_fun, str):
        if act_fun == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
        elif act_fun == 'Swish':
            return Swish()
        elif act_fun == 'ELU':
            return nn.ELU()
        elif act_fun == 'none':
            return nn.Sequential()
        else:
            assert False
    else:
        return act_fun()

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    def __init__(self, module, ln_lambda=1.0, name='weight'):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.ln_lambda = torch.tensor(ln_lambda)
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        
        w = getattr(self.module, self.name + "_bar")
        height = w.data.shape[0]

        _,w_svd,_ = torch.svd(w.view(height,-1).data, some=False, compute_uv=False)
        sigma = w_svd[0]
        sigma = torch.max(torch.ones_like(sigma),sigma/self.ln_lambda)
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)
        w_bar = Parameter(w.data)
        del self.module._parameters[self.name]
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

def conv(in_f, out_f, kernel_size=3, ln_lambda=1, stride=1, bias=True, pad='zero'):
    downsampler = None
    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0

    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)
    nn.init.kaiming_uniform_(convolver.weight, a=0, mode='fan_in')
    if ln_lambda>0:
        convolver = SpectralNorm(convolver, ln_lambda)
    layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
    return nn.Sequential(*layers)


class BatchNormSpectralNorm(object):

    def __init__(self, name='weight', sigma=1.0, eps=1e-12):
        self.name = name
        self.sigma = sigma
        self.eps = eps

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        bias = getattr(module, "bias_orig")
        running_var = getattr(module, "running_var")

        with torch.no_grad():
            cur_sigma = torch.max(torch.abs(weight ))
            #print(cur_sigma)
            cur_sigma = max(float(cur_sigma.cpu().detach().numpy()), self.sigma)
            #print(cur_sigma)
        #cur_sigma = 1
        weight = weight / cur_sigma
        bias = bias / cur_sigma
        return weight, bias

    def remove(self, module):
        weight = getattr(module, self.name)
        bias = getattr(module, "bias")
        delattr(module, self.name)
        delattr(module, self.name + '_orig')
        delattr(module, "bias")
        delattr(module, "bias_orig")
        module.register_parameter(self.name, torch.nn.Parameter(weight.detach()))
        module.register_parameter("bias", torch.nn.Parameter(bias.detach()))

    def __call__(self, module, inputs):
        if module.training:
            weight, bias = self.compute_weight(module)
            setattr(module, self.name, weight)
            setattr(module, "bias", bias)
        else:
            weight_r_g = getattr(module, self.name + '_orig').requires_grad
            bias_r_g = getattr(module, "bias_orig").requires_grad
            getattr(module, self.name).detach_().requires_grad_(weight_r_g)
            getattr(module, "bias").detach_().requires_grad_(bias_r_g)

    @staticmethod
    def apply(module, name, sigma, eps):
        fn = BatchNormSpectralNorm(name, sigma, eps)
        weight = module._parameters[name]
        bias = module._parameters["bias"]

        delattr(module, fn.name)
        delattr(module, "bias")
        module.register_parameter(fn.name + "_orig", weight)
        module.register_parameter("bias_orig", bias)
        # We still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an nn.Parameter and
        # gets added as a parameter. Instead, we register weight.data as a
        # buffer, which will cause weight to be included in the state dict
        # and also supports nn.init due to shared storage.
        module.register_buffer(fn.name, weight.data)
        module.register_buffer("bias", bias.data)

        module.register_forward_pre_hook(fn)
        return fn



def bn_spectral_norm(module, name='weight', sigma=1.0, eps=1e-12):
    BatchNormSpectralNorm.apply(module, name, sigma, eps)
    return module


   
def bn(n_features, lip = 1.0):
    bn = nn.BatchNorm2d(n_features)
    if lip > 0.0:
        return bn_spectral_norm(bn, sigma=lip)
    else:
        return bn





