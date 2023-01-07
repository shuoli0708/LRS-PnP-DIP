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

from models.single_layer_lstm import single_layer_lstm



from .common import *
        
def get_block(num_channels, norm_layer, act_fun):
    layers = [
        nn.Conv2d(num_channels, num_channels, 3, 1, 1, bias=False),
        norm_layer(num_channels, affine=True),
        act(act_fun),
        nn.Conv2d(num_channels, num_channels, 3, 1, 1, bias=False),
        norm_layer(num_channels, affine=True),
    ]
    return layers
    
    
class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.with_conv = True
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)
    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x



class LSTM(nn.Module):
    ''' A encoder model with self attention mechanism. '''
    def __init__(self, input_size,hidden_size, num_input_channels, num_output_channels, num_channels,
            dropout=0.2, pad='reflection', n_position=200,
            ):

        super().__init__()
        #self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
       
       
        self.num_input_channels = 200
        self.num_output_channels = 200
        self.num_channels = num_channels
        
        self.dropout = nn.Dropout(p=dropout)
              
        self.last_conv_layer = conv(num_channels, num_output_channels, 3, stride=1, bias=True, pad=pad)
        self.last_act_after_conv = act('LeakyReLU')

        self.lstm = single_layer_lstm(input_size=input_size,hidden_size=hidden_size)
        
        num_blocks= 3
        num_channels = 32
        act_fun='LeakyReLU'
        s = nn.Sequential
        
        
        norm_layer=nn.BatchNorm2d

        self.act =   act(act_fun)

        self.act_2 =   act(act_fun)
        self.layers = [
            # nn.ReplicationPad2d(num_blocks * 2 * stride + 3),
            conv(self.num_input_channels, self.num_channels, 3, stride=1, bias=True, pad=pad),
            act(act_fun)
        ]
        for i in range(num_blocks):
            self.layers += [s(*get_block(num_channels, norm_layer, act_fun))]
        self.feature_extract = nn.Sequential(*self.layers)
        self.layer_norm = nn.LayerNorm(256, eps=1e-6)

        self.embedding = nn.Linear(192*192, 256)
        self.embedding_recovered = nn.Linear(256, 192*192)   
   
    def forward(self, x, return_attns=False):
        x = self.feature_extract(x)
        #print('after feature extraction is:', x.size())
        x=x.reshape((1,self.num_channels,-1))

        x = self.embedding(x)
        #print('after first embedding is:', x.size())
        norm_out  = self.layer_norm(x)

        norm_out = norm_out.reshape((1,self.num_channels,256))

      

        lstm_in =  x.reshape((1,self.num_channels,-1))

        lstm_out, _  = self.lstm(lstm_in)

        lstm_out = lstm_out.reshape((1,self.num_channels,256))


        lstm_out = self.embedding_recovered(lstm_out)

        lstm_out = lstm_out.reshape((1,self.num_channels,192,192))

        lstm_out = self.last_conv_layer(lstm_out)

        final_out = self.last_act_after_conv(lstm_out)


        
        return  final_out   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
    '''
    def forward(self, x, return_attns=False):
        print('before down sample is:', x.size())
        
        x= self.downsample(x)
        x=self.act(x)
        x= self.downsample_2(x)
        x=self.act_2(x)
        print('after downsample is:', x.size())
       
        x = self.feature_extract(x)
        print('after layer stack is:', x.size())
        x=x.reshape((1,self.num_channels,-1))

        print('lstm input is:', x.size())
          			
        lstm_out, _  = self.lstm(x)
        print('LSTM output is:', lstm_out.size())
        lstm_out = lstm_out.reshape((1,self.num_channels,48,48))
        print('LSTM output after reshape is:', lstm_out.size())
        upsample_out = self.upsample(lstm_out)       
        print('after up sample is:',upsample_out.size())
        
        last_conv_out = self.last_conv_layer(upsample_out)
        print('last conv out is:',last_conv_out.size())
        
        final_out = self.last_act_after_conv(last_conv_out)

        
        return  final_out
    '''
