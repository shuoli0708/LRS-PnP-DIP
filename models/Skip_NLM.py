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



def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")
        
        
def get_feature_map(inMask, encoder_conv_layers, threshold):
    assert inMask.dim() == 4, "mask must be 4 dimensions"
  
    inMask = inMask.float()
    convs = []
    inMask = Variable(inMask, requires_grad = False)
    pad = 'reflection'
    for id_net in range(encoder_conv_layers):
        pad_layer = nn.ConstantPad2d(1, 1)
        conv = nn.Conv2d(1,1,3,2,0, bias=False)
        conv.weight.data.fill_(1/9)
        convs.append(pad_layer)
        convs.append(conv)
    lnet = nn.Sequential(*convs)
    if inMask.is_cuda:
        lnet = lnet.cuda()
    output = lnet(inMask)

    output = (output > threshold).float().mul_(1)
    output=Variable(output, requires_grad = False)
    
    return output

def extract_index_list(input_mask):   # input: each single mask.

    assert input_mask.dim() == 3, 'mask has to be 3 dimenison!'
    _, H, W = input_mask.shape
    N = H*W

    bkg_index = []
    miss_index = []
    
    tmp_bkg_idx = 0    # number of  background pixels
    tmp_miss_idx = 0   # number of  missing pixels

    mask_flat = input_mask.flatten()
   
    for i in range(N):
        if(mask_flat[i] == 1): # if it is a bkg pixel
            bkg_index.append(i)
            tmp_bkg_idx += 1
        else:
            miss_index.append(i)
            tmp_miss_idx += 1
    
    Num_bkg = tmp_bkg_idx
    Num_miss =  tmp_miss_idx
    bkg_index = torch.Tensor(bkg_index).long()
    miss_index = torch.Tensor(miss_index).long()
    
    return bkg_index, miss_index , Num_bkg, Num_miss  # (1, Num_miss) (1, Num_bkg) ,




def extract_patches(masked_feature_map, bkg_index, miss_index , Num_bkg, Num_miss ):  
    H, W, channel = masked_feature_map.shape
    N =  H*W

    tempt = masked_feature_map.reshape((N,channel))
    patches_miss =   torch.index_select(tempt, 0, miss_index.cuda()).reshape((Num_miss,channel))
    patches_bkg  =   torch.index_select(tempt, 0, bkg_index.cuda()).reshape((Num_bkg,channel))

    return patches_miss,  patches_bkg   # output(Num,128)




def NLM(input_mask, input_feature_map, h_gaussian):     # input_mask: (1, (H, W))       # input_feature_map: (1, (H, W, 128)) 
    H, W, channel = input_feature_map.shape
    h = h_gaussian * h_gaussian
   
   
    bkg_index, miss_index, Num_bkg, Num_miss =  extract_index_list(input_mask)
    '''
    print('Num_bkg is: ',Num_bkg)
    print('Num_miss is: ',Num_miss)
    '''
    patches_miss,  patches_bkg = extract_patches(input_feature_map, bkg_index, miss_index , Num_bkg, Num_miss)

    # patch output size: (Num_,128)
    tempt = input_feature_map.clone().reshape((H*W,channel))

    for i in range(Num_miss):
        x_1 = patches_miss[i,:] 
        weight_sum = 0
        average = 0

        for j in range(Num_bkg):

            x_2 = patches_bkg[j,:]         
            Distance = ((x_1-x_2)*(x_1-x_2)).sum()        
            w = torch.exp(-Distance/h)   
           
            weight_sum += w 
            average += w * x_2
  
        updated_pixel_miss =  average / weight_sum

     # insert pixels back to the input feature map.
        location = miss_index[i]
        tempt[location,:]  =  updated_pixel_miss
       
    out = tempt.reshape((1,H,W,channel))

    return  out

def batch_NLM(batch_mask, batch_feature_map, h_gaussian):  
    bs, H, W, channel = batch_feature_map.shape

    output_feature_map = torch.zeros((bs,H,W,channel))
    
    for i in range(bs):
        each_mask = batch_mask[i,:,:]    # (1, H, W )
        each_feature_map = batch_feature_map[i,:,:,:]
        #tic() 
        output_feature_map[i,:,:,:]  = NLM(each_mask, each_feature_map, h_gaussian)
        #toc()
        #print('each bs NLM ready',i)
    out = output_feature_map.reshape((bs,channel,H,W))
    
    return out



class Skip_NLM (nn.Module):
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
            conv(128, 128, 1, 1, bias=True, pad=pad, downsample_mode='stride'),
            bn(128),
            act(act_fun),
            conv(128, 128, 1, bias=True, pad=pad),
            bn(128),
            act(act_fun)
        ]
        
        

    
    
        self.up_1 = [
            bn(128),
            conv(128, 128, 1, 1, bias=True, pad=pad),
            bn(128),
            act(act_fun),
            conv(128, 128, 1, bias=True, pad=pad),
            bn(128),
            act(act_fun),
        ]
        self.up_2 = [
            bn(256),
            conv(256, 128, 3, 1, bias=True, pad=pad),
            bn(128),
            act(act_fun),
            conv(128, 128, 3, bias=True, pad=pad),
            bn(128),
            act(act_fun),
            nn.Upsample(scale_factor=2, mode='nearest')
        ]
        self.up_3 = [
            bn(256),
            conv(256, 128, 3, 1, bias=True, pad=pad),
            bn(128),
            act(act_fun),
            conv(128, 128, 3, bias=True, pad=pad),
            bn(128),
            act(act_fun),
            nn.Upsample(scale_factor=2, mode='nearest')
        ]
        self.up_4 = [
            bn(256),         
            conv(256, 128, 3, 1, bias=True, pad=pad),
            bn(128),
            act(act_fun),
            conv(128, 128, 3, bias=True, pad=pad) ,
            bn(128),
            act(act_fun),  
            nn.Upsample(scale_factor=2, mode='nearest'),
        ]
        self.up_5 = [
            bn(256),         
            conv(256, 128, 3, 1, bias=True, pad=pad),
            bn(128),
            act(act_fun),
            conv(128, 128, 3, bias=True, pad=pad) ,
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
 
    
    def forward(self, inMask, x, return_attns=False):

  
        #print('inut size:',x.size())
        bs,_,_,_ = x.size()
             
        x_after_d_1 = self.d_1(x)   
        res_1 = x_after_d_1      
        x_after_d_2 = self.d_2(x_after_d_1)
        res_2 =  x_after_d_2 
        x_after_d_3 = self.d_3(x_after_d_2)
        res_3 =  x_after_d_3 
        x_after_d_4 = self.d_4(x_after_d_3)

        x_after_d_4 = x_after_d_4.reshape((bs,9,9,128))  
        

        
        ###################    patch swappping module  ###################
        threshold = 6/9
        encoder_conv_layers = 4
        h_gaussian = 5
        masked_feature_map = get_feature_map(inMask, encoder_conv_layers, threshold)
        updated_feature_map = batch_NLM(masked_feature_map, x_after_d_4, h_gaussian)
        
  
        res_4 = updated_feature_map.cuda()
        
        x_after_d_5 = self.d_5(updated_feature_map.cuda())
        
        res_5 = x_after_d_5

        
        up_1_out = self.up_1(x_after_d_5)  

        up_2_out = self.up_2(torch.cat((up_1_out, self.skip_2(res_4)), dim=1))  
        #print('after up_2 size:',up_2_out.size())
                 
        up_3_out = self.up_3(torch.cat((up_2_out, self.skip_3(res_3)), dim=1)) 
        #print('after up_3 size:',up_3_out.size())
        
      
        up_4_out = self.up_4(torch.cat((up_3_out, self.skip_4(res_2)), dim=1))     
        #print('after up_4 size:',up_4_out.size())

        up_5_out = self.up_5(torch.cat((up_4_out, self.skip_5(res_1)), dim=1))     
        #print('after up_4 size:',up_4_out.size())
        
        return  up_5_out
