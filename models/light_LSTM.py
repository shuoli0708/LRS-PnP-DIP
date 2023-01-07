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

from .common import *
from models.single_layer_lstm import single_layer_lstm




class light_LSTM(nn.Module):
    ''' A encoder model with self attention mechanism. '''
    def __init__(self, input_size,hidden_size, num_input_channels, num_output_channels, num_channels,
            dropout=0.2, pad='reflection', n_position=200
            ):

        super().__init__()
        #self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
       
       
        self.num_input_channels = 200
        self.num_output_channels = 200
        self.num_channels = num_channels
        
        self.dropout = nn.Dropout(p=dropout)

        
        self.lstm = single_layer_lstm(input_size= 144,hidden_size=144)

        
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

      
        lstm_out,_ = self.lstm(x)
       
        #print('lstm output size is:', lstm_out.size())
       
        recovered = self.recoverd( lstm_out.reshape((1,64,12,12)))
        #print('size after recovered is:',recovered.size())
        
       

        lstm_out = self.last_act_after_conv(recovered)

        
        return   lstm_out
