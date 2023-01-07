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
class my_own_skip_no_swap(nn.Module):
    ''' A encoder model with self attention mechanism. '''
    def __init__(self, num_input_channels, num_output_channels, num_channels,
            dropout=0.2, pad='reflection', n_position=200,
            ):

        super().__init__()

        self.num_input_channels = 128
        self.num_output_channels = 128
        self.num_channels = num_channels
        
        self.dropout = nn.Dropout(p=dropout)
  
        # last layers
        self.last_conv_layer = conv(self.num_channels, num_output_channels, 3, stride=1, bias=True, pad=pad)
        self.last_act_after_conv = act('LeakyReLU')
        
        act_fun='LeakyReLU'

        self.skip_1 = [
              conv(128, 128, 1, bias=True, pad=pad),
              bn(128),
              act(act_fun)      
        ]
        self.skip_2 = [
              conv(128, 128, 1, bias=True, pad=pad),
              bn(128),
              act(act_fun)         
        ]
        self.skip_3 = [
              conv(128, 128, 1, bias=True, pad=pad),
              bn(128),
              act(act_fun)    
        ]
        self.skip_4 = [
              conv(128, 128, 1, bias=True, pad=pad),
              bn(128),
              act(act_fun)  
        ]
        self.skip_5 = [
              conv(128, 128, 1, bias=True, pad=pad),
              bn(128),
              act(act_fun)  
        ]
        
 
        
        
        self.d_1 = [
            # nn.ReplicationPad2d(num_blocks * 2 * stride + 3),
            conv(128, 128, 3, 2, bias=True, pad=pad, downsample_mode='stride'),
            bn(128),
            act(act_fun),
            conv(128, 128, 3, bias=True, pad=pad),
            bn(128),
            act(act_fun)
        ]
        self.d_2 = [
            # nn.ReplicationPad2d(num_blocks * 2 * stride + 3),
            conv(128, 128, 3, 2, bias=True, pad=pad, downsample_mode='stride'),
            bn(128),
            act(act_fun),
            conv(128, 128, 3, bias=True, pad=pad),
            bn(128),
            act(act_fun)
        ]
        self.d_3 = [
            # nn.ReplicationPad2d(num_blocks * 2 * stride + 3),
            conv(128, 128, 3, 2, bias=True, pad=pad, downsample_mode='stride'),
            bn(128),
            act(act_fun),
            conv(128, 128, 3, bias=True, pad=pad),
            bn(128),
            act(act_fun)
        ]
        self.d_4 = [
            # nn.ReplicationPad2d(num_blocks * 2 * stride + 3),
            conv(128, 128, 3, 2, bias=True, pad=pad, downsample_mode='stride'),
            bn(128),
            act(act_fun),
            conv(128, 128, 3, bias=True, pad=pad),
            bn(128),
            act(act_fun)
        ]
        self.d_5 = [
            # nn.ReplicationPad2d(num_blocks * 2 * stride + 3),
            conv(128, 128, 3, 2, bias=True, pad=pad, downsample_mode='stride'),
            bn(128),
            act(act_fun),
            conv(128, 128, 3, bias=True, pad=pad),
            bn(128),
            act(act_fun)
        ]
        

    
    
        self.up_1 = [
            bn(256),
            conv(256, 128, 3, 1, bias=True, pad=pad),
            bn(128),
            act(act_fun),
            conv(128, 128, 1, bias=True, pad=pad),
            bn(128),
            act(act_fun),
            nn.Upsample(scale_factor=2, mode='nearest')
        ]
        self.up_2 = [
            bn(256),
            conv(256, 128, 3, 1, bias=True, pad=pad),
            bn(128),
            act(act_fun),
            conv(128, 128, 1, bias=True, pad=pad),
            bn(128),
            act(act_fun),
            nn.Upsample(scale_factor=2, mode='nearest')
        ]
        self.up_3 = [
            bn(256),
            conv(256, 128, 3, 1, bias=True, pad=pad),
            bn(128),
            act(act_fun),
            conv(128, 128, 1, bias=True, pad=pad),
            bn(128),
            act(act_fun),
            nn.Upsample(scale_factor=2, mode='nearest')
        ]
        self.up_4 = [
            bn(256),         
            conv(256, 128, 3, 1, bias=True, pad=pad),
            bn(128),
            act(act_fun),
            conv(128, 128, 1, bias=True, pad=pad) ,
            bn(128),
            act(act_fun),  
            nn.Upsample(scale_factor=2, mode='nearest'),
             nn.Sigmoid()
        ]
        
        self.up_5 = [
            bn(256),         
            conv(256, 128, 3, 1, bias=True, pad=pad),
            bn(128),
            act(act_fun),
            conv(128, 128, 1, bias=True, pad=pad) ,
            bn(128),
            act(act_fun),  
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Sigmoid()
        ]
        self.skip_1 = nn.Sequential(*self.skip_1)    
        self.skip_2 = nn.Sequential(*self.skip_2) 
        self.skip_3 = nn.Sequential(*self.skip_3)        
        self.skip_4 = nn.Sequential(*self.skip_4) 
        self.skip_5 = nn.Sequential(*self.skip_5) 
        
        self.d_1 = nn.Sequential(*self.d_1)    
        self.d_2 = nn.Sequential(*self.d_2) 
        self.d_3 = nn.Sequential(*self.d_3) 
        self.d_4 = nn.Sequential(*self.d_4)    
        self.d_5 = nn.Sequential(*self.d_5)  
           
        self.up_1 = nn.Sequential(*self.up_1)    
        self.up_2 = nn.Sequential(*self.up_2) 
        self.up_3 = nn.Sequential(*self.up_3) 
        self.up_4 = nn.Sequential(*self.up_4)    
        self.up_5 = nn.Sequential(*self.up_5)    
        

           
    def forward(self,  x, return_attns=False):

        bs,_,_,_ = x.size()
        x_after_d_1 = self.d_1(x)
        print('after d_1 size:',x_after_d_1.size())

        
    
        res_1 = x_after_d_1
        x_after_d_2 = self.d_2(x_after_d_1)
        print('after d_2 size:',x_after_d_2.size())
        res_2 =  x_after_d_2

        x_after_d_3 = self.d_3(x_after_d_2)

        res_3 = x_after_d_3
        print('after d_3 size:',x_after_d_3.size())
        x_after_d_4 = self.d_4(x_after_d_3)
        print('after d_4 size:',x_after_d_4.size())
        res_4 = x_after_d_4


       
        up_1_out = self.up_1(torch.cat((x_after_d_4, self.skip_1(res_4)), dim=1))  
        print('after up_1 size:',up_1_out.size())
        print('self.skip_2(res_3) size:',self.skip_2(res_3).size())
        up_2_out = self.up_2(torch.cat((up_1_out, self.skip_2(res_3)), dim=1))  
        #print('after up_2 size:',up_2_out.size())
                 
        up_3_out = self.up_3(torch.cat((up_2_out, self.skip_3(res_2)), dim=1)) 
        #print('after up_3 size:',up_3_out.size())
        
      
        up_4_out = self.up_4(torch.cat((up_3_out, self.skip_4(res_1)), dim=1))     
        #print('after up_4 size:',up_4_out.size())


        
        return  up_4_out
