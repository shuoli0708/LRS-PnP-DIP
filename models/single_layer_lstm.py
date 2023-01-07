#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 17:21:52 2020

@author: shuo
"""
import torch
from torch import nn
import numpy as np
import torch.autograd as autograd
from torch.autograd import Variable

import random
from scipy.io import loadmat
import math 
import matplotlib.pyplot as plt
import time
import torch.jit as jit
from torch.nn import Parameter


'''
class single_layer_lstm(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W = nn.Parameter(torch.Tensor(input_size, hidden_size * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_size * 4))
        self.init_weights()
                
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
         
    def forward(self, x, 
                init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        #print('batch_size ',bs)
       # print('seq lengthe ',seq_sz)
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device), 
                        torch.zeros(bs, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states
         
        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.W + h_t @ self.U + self.bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]), # input
                torch.sigmoid(gates[:, HS:HS*2]), # forget
                torch.tanh(gates[:, HS*2:HS*3]),
                torch.sigmoid(gates[:, HS*3:]), # output
            )
            #print('length of ft:',f_t .size())
            #print('length of ct:',c_t.size())
            #print('length of first part:',(f_t * c_t).size())
            #print('length of second part:',(i_t * g_t).size())
            
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)
    
'''    

 
class single_layer_lstm(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # input gate
        self.W_ii = Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hi = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = Parameter(torch.Tensor(hidden_size))
        # forget gate
        self.W_if = Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hf = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = Parameter(torch.Tensor(hidden_size))
        # canditdate gate
        self.W_ig = Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hg = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_g = Parameter(torch.Tensor(hidden_size))
        # output gate
        self.W_io = Parameter(torch.Tensor(input_size, hidden_size))
        self.W_ho = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = Parameter(torch.Tensor(hidden_size))
        self.init_weights()
        self.f_infinity_norm=0
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
         
          
    def forward(self, x, 
                init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        if init_states is None:
            h_t, c_t = torch.zeros(self.hidden_size).to(x.device), torch.zeros(self.hidden_size).to(x.device)
        else:
            h_t, c_t = init_states
        for t in range(seq_sz): # iterate over the time steps
            x_t = x[:, t, :]
            i_t = torch.sigmoid(x_t @ self.W_ii + h_t @ self.W_hi + self.b_i)
            f_t = torch.sigmoid(x_t @ self.W_if + h_t @ self.W_hf + self.b_f)
            g_t = torch.tanh(x_t @ self.W_ig + h_t @ self.W_hg + self.b_g)
            o_t = torch.sigmoid(x_t @ self.W_io + h_t @ self.W_ho + self.b_o)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        
        #print('saisu is here')
        return hidden_seq, (h_t, c_t)

    
    '''
    def add_weight_constraint(self):
    
        #print('hi i am here')
        with torch.no_grad():
          
          # |Wf|∞ < 0.128  hidden to hidden
          self.W_hf[(self.W_hf)>0.128]=0.128
          self.W_hf[(self.W_hf)<-0.128]=-0.128
          # |bf|∞ < 0.25
          self.b_f[(self.b_f)>0.25]=0.25
          self.b_f[(self.b_f)<-0.25]=-0.25
          # |Uf|∞ < 0.128 input to hidden
          self.W_if[(self.W_if)>0.25]=0.25
          self.W_if[(self.W_if)<-0.25]=-0.25
       
          model.lstm1.W_hf[model.lstm1.W_hf>0]=0
          model.lstm1.b_f[model.lstm1.b_f>0]=0
          model.lstm1.W_if[model.lstm1.W_if>0]=0
      
        return 0
    '''
    
    
    
    
    
    
    
    
    
    
    
    
