import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common_for_No_Lipschitz_Control import * 

class deep_decoder_No_BN(nn.Module):
    '''
        upsample_mode in ['deconv', 'nearest', 'bilinear', 'gaussian']
        pad in ['zero', 'replication', 'none']
    '''
    def __init__(self, num_input_channels=3, num_output_channels=3, ln_lambda=0, 
                       upsample_mode='gaussian', pad='zero', need_sigmoid=True, need_bias=True):
        super(deep_decoder_No_BN, self).__init__()



        filters = [128, 128]


        layers = []
        layers.append(unetConv2(num_input_channels, filters[0],  need_bias, pad))
        for i in range(len(filters)):
            layers.append(unetUp(filters[i], upsample_mode, need_bias, pad))

        layers.append(conv(filters[-1], num_output_channels, 1, 0, bias=need_bias, pad=pad))
        if need_sigmoid: 
            layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x):

        return self.net(x)


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size,need_bias, pad):
        super(unetConv2, self).__init__()

        self.conv1= nn.Sequential(conv(in_size, out_size, 3, bias=need_bias, pad=pad),
                                   #bn(out_size),
                                   nn.LeakyReLU(),)
        self.conv2= nn.Sequential(conv(out_size, out_size, 3, bias=need_bias, pad=pad),
                                   #bn(out_size),
                                   nn.LeakyReLU(),)
    def forward(self, x):
        x= self.conv1(x)
        x= self.conv2(x)
        return x


class unetUp(nn.Module):
    def __init__(self, out_size, upsample_mode,need_bias, pad):
        super(unetUp, self).__init__()

        num_filt = out_size
        if upsample_mode == 'deconv':
            self.up= nn.ConvTranspose2d(num_filt, out_size, 4, stride=2, padding=1)
            self.conv= conv(out_size, out_size, 3,  bias=need_bias, pad=pad)
        elif upsample_mode=='bilinear' or upsample_mode=='nearest':
            self.up = nn.Upsample(scale_factor=2, mode=upsample_mode)
            self.conv= unetConv2(out_size, out_size,  need_bias, pad)
        elif upsample_mode == 'gaussian':
            self.up = gaussian(out_size, kernel_width=5, sigma=sigma)
            self.conv= unetConv2(out_size, out_size, need_bias, pad)
        else:
            assert False

    def forward(self, x):
        x= self.up(x)
        x = self.conv(x)

        return x

