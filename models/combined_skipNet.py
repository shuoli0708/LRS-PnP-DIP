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









class combined_skipNet(nn.Module):
    ''' A encoder model with self attention mechanism. '''
    def __init__(self, num_input_channels, num_output_channels, 
                        d_model, d_feedforward, num_channels,
                       n_layers, n_head, d_k, d_v, 
						num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], num_channels_skip=[4, 4, 4, 4, 4], 
						filter_size_down=3, filter_size_up=3, filter_skip_size=1,
						need_sigmoid=True, need_bias=True, dropout=0.2,
						pad='zero', upsample_mode='nearest',pad_transformer='reflection', downsample_mode='stride', act_fun='LeakyReLU', 
						need1x1_up=True, n_position=200,
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



        ######## skip Net Part ##########################################				
        model = nn.Sequential()
        model_tmp = model
        num_channels_down=[16, 32, 64, 128, 128]
        num_channels_up=[16, 32, 64, 128, 128]
        num_channels_skip=[4, 4, 4, 4, 4]
        input_depth = self.num_input_channels
        upsample_mode='nearest'
        downsample_mode='stride'
        filter_size_down=3
        filter_size_up=3
        filter_skip_size=1
        need_bias=True
        need1x1_up=True
        pad='zero'
        act_fun='LeakyReLU'
        num_input_channels = 200
        num_output_channels = 200
        need_sigmoid=True,

        ##### skip_Net set up #######################################
        assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

        n_scales = len(num_channels_down) 

        if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)):
            upsample_mode   = [upsample_mode]*n_scales

        if not (isinstance(downsample_mode, list)or isinstance(downsample_mode, tuple)):
            downsample_mode   = [downsample_mode]*n_scales
		
        if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)):
            filter_size_down   = [filter_size_down]*n_scales

        if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)):
            filter_size_up   = [filter_size_up]*n_scales

        last_scale = n_scales - 1 

        cur_depth = None




        for i in range(len(num_channels_down)):

            deeper = nn.Sequential()
            skip = nn.Sequential()

            if num_channels_skip[i] != 0:
                model_tmp.add(Concat(1, skip, deeper))
            else:
                model_tmp.add(deeper)
			
            model_tmp.add(bn(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))

            if num_channels_skip[i] != 0:
                skip.add(conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
                skip.add(bn(num_channels_skip[i]))
                skip.add(act(act_fun))
				

            deeper.add(conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[i]))
            deeper.add(bn(num_channels_down[i]))
            deeper.add(act(act_fun))

            deeper.add(conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
            deeper.add(bn(num_channels_down[i]))
            deeper.add(act(act_fun))

            deeper_main = nn.Sequential()

            if i == len(num_channels_down) - 1:
				# The deepest
                k = num_channels_down[i]
            else:
                deeper.add(deeper_main)
                k = num_channels_up[i + 1]

            deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))

            model_tmp.add(conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))


            if need1x1_up:
                model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
                model_tmp.add(bn(num_channels_up[i]))
                model_tmp.add(act(act_fun))

            input_depth = num_channels_down[i]
            model_tmp = deeper_main

        model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
        if need_sigmoid:
            model.add(nn.Sigmoid())
        model = model.cuda()
       
    
    
    
        skip_out = model(transformer_out)

        
        

        
        return skip_out
