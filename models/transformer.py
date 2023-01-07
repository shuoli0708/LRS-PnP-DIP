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


class transformer(nn.Module):
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

        self.embedding = nn.Linear(192*192, d_model)
        self.embedding_recovered = nn.Linear(256, 192*192)
        
        # last layers
        self.last_conv_layer = conv(self.num_channels, num_output_channels, 3, stride=1, bias=True, pad=pad)
        self.last_act_after_conv = act('LeakyReLU')
        
        
        
        num_blocks= 4
        act_fun='LeakyReLU'
        s = nn.Sequential
        norm_layer=nn.BatchNorm2d
        
        self.layers = [
            # nn.ReplicationPad2d(num_blocks * 2 * stride + 3),
            conv(self.num_input_channels, self.num_channels, 3, stride=1, bias=True, pad=pad),
            act(act_fun)
        ]
        for i in range(num_blocks):
            self.layers += [s(*get_block(self.num_channels, norm_layer, act_fun))]
       
        self.feature_extract = nn.Sequential(*self.layers)
    '''       
    def forward(self, x, return_attns=False):
        print('before first conv is:', x.size())
        slf_attn_list = []
        residual = x
        x = self.conv_layer(x)
        print('after first conv is:', x.size())
        x = self.act_after_conv(x)
        #print('after first activtion is:', x.size())
        x=x.reshape((self.num_channels,-1))
        print('after first activation is:', x.size())
        x = self.embedding(x)
        print('after first embedding is:', x.size())
        #print('embedding output size is:', x.size())
        embed_out  = self.layer_norm(x)
        print('after layer norm is:', embed_out.size())
        embed_out =embed_out.reshape((1,8,256))
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(embed_out)
            slf_attn_list += [enc_slf_attn] if return_attns else []
        #print('siz of encoder(feed_forward)_ output:,', enc_output.size())					
        if return_attns:      
            return enc_output, slf_attn_list        
				
        transformer_out  = enc_output
       
        print('eocnder output is:',  enc_output.size())
        transformer_out = self.embedding_recovered(transformer_out)
        print('eafter second embedding is is:', transformer_out.size())
        transformer_out = transformer_out.reshape((1,8,192,192))
        print('recovered convc input is:',  transformer_out.size())
        transformer_out = self.last_conv_layer(transformer_out)
        print('after second conv is:', transformer_out.size())
        transformer_out = self.last_act_after_conv(transformer_out)
        print('after second activtion is:', transformer_out.size())
        
        return  transformer_out 
    '''           
    def forward(self, x, return_attns=False):
        slf_attn_list = []
        residual = x
        '''
        
        
        residual = x
        x = self.conv_layer(x)

        x = self.act_after_conv(x)
        
        skip_output = self.testskip_output(x)
        print('skip output size  is:', skip_output.size())
        skip_output= self.testskip_output_3(skip_output)
        print('skip output size  is:', skip_output.size())
        up = self.up(skip_output)
        print('up size  is:', up.size())
        up = torch.cat((up, up), dim=1)
        print('up after concatenation is is:', up.size())
        up_2 = self.conv_up(up)
        print('up convsize is:', up_2.size())
        '''
        
        
        
        x = self.feature_extract(x)
        #print('after feature extraction is:', x.size())
        x=x.reshape((self.num_channels,-1))

        x = self.embedding(x)
        #print('after first embedding is:', x.size())
        #embed_out  = self.layer_norm(x)
        embed_out  = x
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

        
        return  transformer_out 
