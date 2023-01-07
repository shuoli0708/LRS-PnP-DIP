import torch
import torch.nn as nn
import numpy as np
from torch.nn import Parameter


def bn(num_features):

    return nn.BatchNorm2d(num_features)

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)



def conv(in_f, out_f, kernel_size=3, ln_lambda=1, stride=1, bias=True, pad='zero'):
    downsampler = None
    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0

    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)
    nn.init.kaiming_uniform_(convolver.weight, a=0, mode='fan_in')
   
    layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
    return nn.Sequential(*layers)

def get_kernel(kernel_width=5, sigma=0.5):

    kernel = np.zeros([kernel_width, kernel_width])
    center = (kernel_width + 1.)/2.
    sigma_sq =  sigma * sigma

    for i in range(1, kernel.shape[0] + 1):
        for j in range(1, kernel.shape[1] + 1):
            di = (i - center)/2.
            dj = (j - center)/2.
            kernel[i - 1][j - 1] = np.exp(-(di * di + dj * dj)/(2 * sigma_sq))
            kernel[i - 1][j - 1] = kernel[i - 1][j - 1]/(2. * np.pi * sigma_sq)

    kernel /= kernel.sum()

    return kernel
