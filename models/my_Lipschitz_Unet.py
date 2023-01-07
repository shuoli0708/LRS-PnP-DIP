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
from .lipschitz_constraint_layer import * 




class my_Lipschitz_Unet(nn.Module):
    ''' A encoder model with self attention mechanism. '''
    def __init__(self, num_input_channels, num_output_channels, 
            ln_lambda, pad='reflection'
            ):

        super().__init__()

        act_fun='LeakyReLU'
        
        self.d_1 = [
            # nn.ReplicationPad2d(num_blocks * 2 * stride + 3),
            conv(128, 128, 3, ln_lambda, 2, bias=True, pad=pad),
            bn(128),
            act(act_fun),
            conv(128, 128, 3,  ln_lambda, bias=True, pad=pad),
            bn(128),
            act(act_fun)
        ]
        self.d_2 = [
            # nn.ReplicationPad2d(num_blocks * 2 * stride + 3),
            conv(128, 128, 3,  ln_lambda,2, bias=True, pad=pad),
            bn(128),
            act(act_fun),
            conv(128, 128, 3,  ln_lambda, bias=True, pad=pad),
            bn(128),
            act(act_fun)
        ]
        self.d_3 = [
            # nn.ReplicationPad2d(num_blocks * 2 * stride + 3),
            conv(128, 128, 3,  ln_lambda, 2, bias=True, pad=pad),
            bn(128),
            act(act_fun),
            conv(128, 128, 3, ln_lambda, bias=True, pad=pad),
            bn(128),
            act(act_fun)
        ]
        self.d_4 = [
            # nn.ReplicationPad2d(num_blocks * 2 * stride + 3),
            conv(128, 128, 3, ln_lambda, 2, bias=True, pad=pad),
            bn(128),
            act(act_fun),
            conv(128, 128, 3, ln_lambda,bias=True, pad=pad),
            bn(128),
            act(act_fun)
        ]
        

    
    
        self.up_1 = [
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv(128, 128, 2, ln_lambda, 1, bias=True, pad=pad),
            bn(128),
            act(act_fun)              
        ]
        self.up_2 = [
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv(128, 128, 2, ln_lambda, 1, bias=True, pad=pad),
            bn(128),
            act(act_fun)
        ]
        self.up_3 = [
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv(128, 128, 3, ln_lambda, 1, bias=True, pad=pad),
            bn(128),
            act(act_fun)
        ]
        self.up_4 = [
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv(128, 128, 3, ln_lambda, 1, bias=True, pad=pad),
            bn(128),
            act(act_fun)
        ]   
        
        self.last = [
            # nn.ReplicationPad2d(num_blocks * 2 * stride + 3),
            conv(128, 128, 1, ln_lambda,  bias=True, pad=pad),
            bn(128),
            act(act_fun),
            conv(128, 128, 1, ln_lambda, bias=True, pad=pad),
            act(act_fun)
        ]
        
        self.d_1 = nn.Sequential(*self.d_1)    
        self.d_2 = nn.Sequential(*self.d_2) 
        self.d_3 = nn.Sequential(*self.d_3) 
        self.d_4 = nn.Sequential(*self.d_4)    
            
        self.up_1 = nn.Sequential(*self.up_1)    
        self.up_2 = nn.Sequential(*self.up_2) 
        self.up_3 = nn.Sequential(*self.up_3) 
        self.up_4 = nn.Sequential(*self.up_4)    
        self.last = nn.Sequential(*self.last)    
    
    def forward(self, x, return_attns=False):

  
        #print('input size:',x.size())
        bs,_,_,_ = x.size()
             
        x_after_d_1 = self.d_1(x)   
        #print('after d_1 size:',x_after_d_1.size())
        x_after_d_2 = self.d_2(x_after_d_1)
        #print('after d_2 size:',x_after_d_2.size())
        x_after_d_3 = self.d_3(x_after_d_2)
        #print('after d_3 size:',x_after_d_3.size())
        x_after_d_4 = self.d_4(x_after_d_3 )
        #print('after d_4size:',x_after_d_4.size())

        
        up_1_out = self.up_1(x_after_d_4)
        #up_1_out = self.up_1(self.skip_1(res_4).reshape((bs,128,9,9)))
        #print('after up_1 size:',up_1_out.size())

        up_2_out = self.up_2(up_1_out)
        #print('after up_2 size:',up_2_out.size())
                 
        up_3_out =  self.up_3(up_2_out)
        #print('after up_3 size:',up_3_out.size())
        
      
        up_4_out =  self.up_4(up_3_out)
        #print('after up_4 size:',up_4_out.size())


        
        return  self.last(up_4_out)

