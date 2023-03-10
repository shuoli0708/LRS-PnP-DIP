import torch
import copy
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch.autograd as autograd
from torch.autograd import Variable
import scipy.io as io
import random
from scipy.io import loadmat
import math 
import matplotlib.pyplot as plt
import time
import torch.jit as jit
from torch.nn import Parameter
from SubLayers import MultiHeadAttention, PositionwiseFeedForward
from .common import *



class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0: 
            idx = len(self) + idx

        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)

class my_self_atten_layer(nn.Module):
    ''' Compose with two layers '''
    def __init__(self, d_model, d_feedforward, n_head, d_k, d_v, dropout=0.1):
        super(my_self_atten_layer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_feedforward, dropout=dropout)
    def forward(self, enc_input):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input)
        enc_output = self.pos_ffn(enc_output)

        return enc_output, enc_slf_attn
        
def get_block(num_channels, norm_layer, act_fun):
    layers = [
        nn.Conv2d(num_channels, num_channels, 3, 1, 1, bias=False),
        norm_layer(num_channels, affine=True),
        act(act_fun),
        nn.Conv2d(num_channels, num_channels, 3, 1, 1, bias=False),
        norm_layer(num_channels, affine=True),
    ]
    return layers



class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, norm_layer, need_bias, pad):
        super(unetConv2, self).__init__()

        print(pad)
        if norm_layer is not None:
            self.conv1= nn.Sequential(conv(in_size, out_size, 3, bias=need_bias, pad=pad),
                                       norm_layer(out_size),
                                       nn.ReLU(),)
            self.conv2= nn.Sequential(conv(out_size, out_size, 3, bias=need_bias, pad=pad),
                                       norm_layer(out_size),
                                       nn.ReLU(),)
        else:
            self.conv1= nn.Sequential(conv(in_size, out_size, 3, bias=need_bias, pad=pad),
                                       nn.ReLU(),)
            self.conv2= nn.Sequential(conv(out_size, out_size, 3, bias=need_bias, pad=pad),
                                       nn.ReLU(),)
    def forward(self, inputs):
        outputs= self.conv1(inputs)
        outputs= self.conv2(outputs)
        return outputs


class unetDown(nn.Module):
    def __init__(self, in_size, out_size, norm_layer, need_bias, pad):
        super(unetDown, self).__init__()
        self.conv= unetConv2(in_size, out_size, norm_layer, need_bias, pad)
        self.down= nn.MaxPool2d(2, 2)

    def forward(self, inputs):
        outputs= self.down(inputs)
        outputs= self.conv(outputs)
        return outputs


class unetUp(nn.Module):
    def __init__(self, out_size, upsample_mode, need_bias, pad, same_num_filt=False):
        super(unetUp, self).__init__()

        num_filt = out_size if same_num_filt else out_size * 2
        if upsample_mode == 'deconv':
            self.up= nn.ConvTranspose2d(num_filt, out_size, 4, stride=2, padding=1)
            self.conv= unetConv2(out_size * 2, out_size, None, need_bias, pad)
        elif upsample_mode=='bilinear' or upsample_mode=='nearest':
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode=upsample_mode),
                                   conv(num_filt, out_size, 3, bias=need_bias, pad=pad))
            self.conv= unetConv2(out_size * 2, out_size, None, need_bias, pad)
        else:
            assert False

    def forward(self, inputs1, inputs2):
        in1_up= self.up(inputs1)
        
        if (inputs2.size(2) != in1_up.size(2)) or (inputs2.size(3) != in1_up.size(3)):
            diff2 = (inputs2.size(2) - in1_up.size(2)) // 2 
            diff3 = (inputs2.size(3) - in1_up.size(3)) // 2 
            inputs2_ = inputs2[:, :, diff2 : diff2 + in1_up.size(2), diff3 : diff3 + in1_up.size(3)]
        else:
            inputs2_ = inputs2

        output= self.conv(torch.cat([in1_up, inputs2_], 1))

        return output











class combined_uNet(nn.Module):
    ''' A encoder model with self attention mechanism. '''
    def __init__(self, num_input_channels, num_output_channels, 
                       feature_scale, more_layers,  d_model, d_feedforward, num_channels,
                       n_layers, n_head, d_k, d_v, 
                       upsample_mode='deconv', pad_CNN='zero', concat_x=False, norm_layer=nn.InstanceNorm2d, need_sigmoid=True, need_bias=True, 
                      dropout=0.2, pad_transformer='reflection', n_position=200,
            ):

        super().__init__()


        ##### Transformer set up #######################################
        self.layer_stack = nn.ModuleList([
            my_self_atten_layer(d_model, d_feedforward, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.num_input_channels = 200
        self.num_output_channels = 200
        self.num_channels = num_channels
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.embedding = nn.Linear(192*192, d_model)
        self.embedding_recovered = nn.Linear(256, 192*192)
        # last layers
        self.last_conv_layer = conv(num_channels, num_output_channels, 3, stride=1, bias=True, pad=pad_transformer)
        self.last_act_after_conv = act('LeakyReLU')
        num_blocks= 4
        num_channels = 32
        act_fun='LeakyReLU'
        s = nn.Sequential
        norm_layer=nn.BatchNorm2d
        self.layers = [
            # nn.ReplicationPad2d(num_blocks * 2 * stride + 3),
            conv(self.num_input_channels, self.num_channels, 3, stride=1, bias=True, pad=pad_transformer),
            act(act_fun)
        ]
        for i in range(num_blocks):
            self.layers += [s(*get_block(num_channels, norm_layer, act_fun))]
       
        self.feature_extract = nn.Sequential(*self.layers)
    
    
    
    
        ##### CNN set up #######################################
        self.feature_scale = feature_scale
        self.more_layers = more_layers
        self.concat_x = concat_x

        filters = [64, 128, 256, 512, 1024]
        filters = [x // self.feature_scale for x in filters]

        self.start = unetConv2(num_input_channels, filters[0] if not concat_x else filters[0] - num_input_channels, norm_layer, need_bias, pad_CNN)

        self.down1 = unetDown(filters[0], filters[1] if not concat_x else filters[1] - num_input_channels, norm_layer, need_bias, pad_CNN)
        self.down2 = unetDown(filters[1], filters[2] if not concat_x else filters[2] - num_input_channels, norm_layer, need_bias, pad_CNN)
        self.down3 = unetDown(filters[2], filters[3] if not concat_x else filters[3] - num_input_channels, norm_layer, need_bias, pad_CNN)
        self.down4 = unetDown(filters[3], filters[4] if not concat_x else filters[4] - num_input_channels, norm_layer, need_bias, pad_CNN)
        # more downsampling layers
        if self.more_layers > 0:
            self.more_downs = [
                unetDown(filters[4], filters[4] if not concat_x else filters[4] - num_input_channels , norm_layer, need_bias, pad_CNN) for i in range(self.more_layers)]
            self.more_ups = [unetUp(filters[4], upsample_mode, need_bias, pad_CNN, same_num_filt =True) for i in range(self.more_layers)]

            self.more_downs = ListModule(*self.more_downs)
            self.more_ups   = ListModule(*self.more_ups)

        self.up4 = unetUp(filters[3], upsample_mode, need_bias, pad_CNN)
        self.up3 = unetUp(filters[2], upsample_mode, need_bias, pad_CNN)
        self.up2 = unetUp(filters[1], upsample_mode, need_bias, pad_CNN)
        self.up1 = unetUp(filters[0], upsample_mode, need_bias, pad_CNN)

        self.final = conv(filters[0], num_output_channels, 1, bias=need_bias, pad=pad_CNN)

        if need_sigmoid: 
            self.final = nn.Sequential(self.final, nn.Sigmoid())
       
    
    
    
    
    
    
    
    
    def forward(self, x, return_attns=False):
		
        ######## Transformer Part ##########################################						
        slf_attn_list = []
       # print('before feature etraxc is:', x.size())
        x = self.feature_extract(x)
        #print('after feature extraction is:', x.size())
        x=x.reshape((self.num_channels,-1))

        x = self.embedding(x)
        #print('after first embedding is:', x.size())
        embed_out  = self.layer_norm(x)

        embed_out =embed_out.reshape((1,self.num_channels,256))
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(embed_out)
            slf_attn_list += [enc_slf_attn] if return_attns else []			
        if return_attns:      
            return enc_output, slf_attn_list        				
        transformer_out  = enc_output    
        #print('transformer output size is:', transformer_out.size())
        transformer_out = self.embedding_recovered(transformer_out)
        transformer_out = transformer_out.reshape((1,self.num_channels,192,192))
        transformer_out = self.last_conv_layer(transformer_out)
        transformer_out = self.last_act_after_conv(transformer_out)



        ######## CNN Part ##########################################				


        #print('size of inputs (i,e: transformer output):', transformer_out.size())
        
        # Downsample 
        
        downs = [transformer_out]
        down = nn.AvgPool2d(2, 2)
        for i in range(4 + self.more_layers):
            downs.append(down(downs[-1]))

        in64 = self.start(transformer_out)
        #print('size after first unetconv2 layer:',in64.size())
        if self.concat_x:
            in64 = torch.cat([in64, downs[0]], 1)
            #print('we are using concatenation!')

        down1 = self.down1(in64)
        #print('size after first donwsample layer:',down1.size())
        if self.concat_x:
            down1 = torch.cat([down1, downs[1]], 1)

        down2 = self.down2(down1)
        #print('size after secd donwsample layer:',down2.size())
        if self.concat_x:
            down2 = torch.cat([down2, downs[2]], 1)

        down3 = self.down3(down2)
        if self.concat_x:
            down3 = torch.cat([down3, downs[3]], 1)

        down4 = self.down4(down3)
        #print('size after last donwsample layer:',down4.size())
        if self.concat_x:
            down4 = torch.cat([down4, downs[4]], 1)

        if self.more_layers > 0:
            #print('we are using more layers!')
            prevs = [down4]
            for kk, d in enumerate(self.more_downs):
                # print(prevs[-1].size())
                out = d(prevs[-1])
                if self.concat_x:
                    out = torch.cat([out,  downs[kk + 5]], 1)

                prevs.append(out)

            up_ = self.more_ups[-1](prevs[-1], prevs[-2])
            for idx in range(self.more_layers - 1):
                l = self.more_ups[self.more - idx - 2]
                up_= l(up_, prevs[self.more - idx - 2])
        else:
            up_= down4
        #print('size before upsample layer:',up_.size())
        up4= self.up4(up_, down3)
        #print('size of zuidixia de upsample layer:',up4.size())
        up3= self.up3(up4, down2)
        #print('size of up3 layer:',up3.size())
        up2= self.up2(up3, down1)
        #print('size of up2 layer:',up2.size())
        up1= self.up1(up2, in64)
        #print('size of zui shang mian layer:',up1.size())
        

        
        return self.final(up1)
