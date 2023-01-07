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


class light_transformer(nn.Module):
    ''' A encoder model with self attention mechanism. '''
    def __init__(self, d_model, d_feedforward, num_input_channels, num_output_channels, num_channels,
            n_layers, n_head, d_k, d_v, dropout=0.2, pad='reflection', n_position=200,
            ):

        super().__init__()
        #self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
       
        self.layer_stack = nn.ModuleList([
            my_self_atten_layer(d_model, d_feedforward, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
        self.num_input_channels = 200
        self.num_output_channels = 200
        self.num_channels = num_channels
        
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model


        
        # last layers
        self.last_conv_layer = conv(self.num_channels, num_output_channels, 3, stride=1, bias=True, pad=pad)
        self.last_act_after_conv = act('LeakyReLU')
        
        act_fun='LeakyReLU'


        
        
        
        self.layer = [
            # nn.ReplicationPad2d(num_blocks * 2 * stride + 3),
            conv(200, 64, 3, 2, bias=True, pad=pad, downsample_mode='stride'),
            bn(64),
            act(act_fun),
            conv(64, 64, 3, bias=True, pad=pad),
            bn(64),
            act(act_fun)
        ]
        self.layer += [
            # nn.ReplicationPad2d(num_blocks * 2 * stride + 3),
            conv(64, 64, 3, 2, bias=True, pad=pad, downsample_mode='stride'),
            bn(64),
            act(act_fun),
            conv(64, 64, 3, bias=True, pad=pad),
            bn(64),
            act(act_fun)
        ]
        self.layer += [
            # nn.ReplicationPad2d(num_blocks * 2 * stride + 3),
            conv(64, 64, 3, 2, bias=True, pad=pad, downsample_mode='stride'),
            bn(64),
            act(act_fun),
            conv(64, 64, 3, bias=True, pad=pad),
            bn(64),
            act(act_fun)
        ]
        self.layer += [
            # nn.ReplicationPad2d(num_blocks * 2 * stride + 3),
            conv(64, 64, 3, 2, bias=True, pad=pad, downsample_mode='stride'),
            bn(64),
            act(act_fun),
            conv(64, 64, 3, bias=True, pad=pad),
            bn(64),
            act(act_fun)
        ]
        
        self.feature_extract = nn.Sequential(*self.layer)
    
    
    
        self.up = [
            # nn.ReplicationPad2d(num_blocks * 2 * stride + 3),
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv(64, 64, 3, 1, bias=True, pad=pad),
            bn(64),
            act(act_fun)
        ]
        self.up += [
            # nn.ReplicationPad2d(num_blocks * 2 * stride + 3),
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv(64, 64, 3, 1, bias=True, pad=pad),
            bn(64),
            act(act_fun)
        ]
        self.up += [
            # nn.ReplicationPad2d(num_blocks * 2 * stride + 3),
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv(64, 64, 3, 1, bias=True, pad=pad),
            bn(64),
            act(act_fun)
        ]
        self.up += [
            # nn.ReplicationPad2d(num_blocks * 2 * stride + 3),
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv(64, 64, 3, 1, bias=True, pad=pad),
            bn(64),
            act(act_fun)
        ]
        
        self.up += conv(64, 200, 1, bias=True, pad=pad)     
        
        
        self.recoverd = nn.Sequential(*self.up)    
   
    
    
    def forward(self, x, return_attns=False):
        slf_attn_list = []
        residual = x

        x = self.feature_extract(x)
        #print('after feature extraction is:', x.size())
        x=x.reshape((1,64,-1))

      
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(x)
            slf_attn_list += [enc_slf_attn] if return_attns else []
			
        if return_attns:      
            return enc_output, slf_attn_list        
				
        transformer_out  = enc_output
       
        #print('transformer output size is:', transformer_out.size())
       
        recovered = self.recoverd(transformer_out.reshape((1,64,12,12)))
        #print('size after recovered is:',recovered.size())
        
       

        transformer_out = self.last_act_after_conv(recovered)

        
        return  transformer_out 
