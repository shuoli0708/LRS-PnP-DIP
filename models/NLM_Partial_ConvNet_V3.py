import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd


def get_feature_map(inMask, threshold):
    assert inMask.dim() == 4, "mask must be 4 dimensions"
  
    inMask = inMask.float()
    convs = []
    inMask = Variable(inMask, requires_grad = False)

    pad_layer_1 = nn.ConstantPad2d(3, 1)
    pad_layer_2 = nn.ConstantPad2d(2, 1)
    pad_layer_3 = nn.ConstantPad2d(1, 1)
    pad_layer_4 = nn.ConstantPad2d(1, 1)
    conv_1 = nn.Conv2d(1, 1, 7, 2, 0, bias=False)
    conv_1.weight.data.fill_(1/49)
    conv_2 = nn.Conv2d(1, 1, 5, 2, 0, bias=False)
    conv_2.weight.data.fill_(1/25)
    conv_3 = nn.Conv2d(1, 1, 3, 2, 0, bias=False)
    conv_3.weight.data.fill_(1/9)
    conv_4 = nn.Conv2d(1, 1, 3, 2, 0, bias=False)
    conv_4.weight.data.fill_(1/9)


    convs.append(pad_layer_1)   
    convs.append(conv_1)
    convs.append(pad_layer_2)   
    convs.append(conv_2)  
                                     
    convs.append(pad_layer_3)   
    convs.append(conv_3)  
    '''
    convs.append(pad_layer_4)   
    convs.append(conv_4) 
    '''
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
        each_feature_map = batch_feature_map[i,:,:,:] # (1, H, W, channel )
        #tic() 
        output_feature_map[i,:,:,:]  = NLM(each_mask, each_feature_map, h_gaussian)    # NLM output size: 1, H, W,channel
        #toc()
        #print('each bs NLM ready',i)
    #out = output_feature_map.reshape((bs,channel,H,W))
    out = torch.Tensor(output_feature_map.detach().cpu().view((bs,H,W,channel)).numpy().transpose(0,3,1,2))  # final output size: (bs, channel, H, W)
    return out


class PartialConvLayer (nn.Module):

    def __init__(self, in_channels, out_channels, bn=True, bias=False, sample="none-3", activation="relu"):
        super().__init__()
        self.bn = bn

        if sample == "down-7":
            # Kernel Size = 7, Stride = 2, Padding = 3
            self.input_conv = nn.Conv2d(in_channels, out_channels, 7, 2, 3, bias=bias)
            self.mask_conv = nn.Conv2d(1, 1, 7, 2, 3, bias=False)

        elif sample == "down-5":
            self.input_conv = nn.Conv2d(in_channels, out_channels, 5, 2, 2, bias=bias)
            self.mask_conv = nn.Conv2d(1, 1, 5, 2, 2, bias=False)

        elif sample == "down-3":
            self.input_conv = nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=bias)
            self.mask_conv = nn.Conv2d(1, 1, 3, 2, 1, bias=False)

        else:
            self.input_conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=bias)
            self.mask_conv = nn.Conv2d(1, 1, 3, 1, 1, bias=False)

        nn.init.constant_(self.mask_conv.weight, 1.0)

        # "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
        # negative slope of leaky_relu set to 0, same as relu
        # "fan_in" preserved variance from forward pass
        nn.init.kaiming_normal_(self.input_conv.weight, a=0, mode="fan_in")

        for param in self.mask_conv.parameters():
            param.requires_grad = False

        if bn:
            # Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
            # Applying BatchNorm2d layer after Conv will remove the channel mean
            self.batch_normalization = nn.BatchNorm2d(out_channels)

        if activation == "relu":
            # Used between all encoding layers
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            # Used between all decoding layers (Leaky RELU with alpha = 0.2)
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input_x, mask):
       
        #output = W^T dot (X .* M) + b

        output = self.input_conv(input_x * mask)

        # requires_grad = False
        with torch.no_grad():
            # mask = (1 dot M) + 0 = M
            output_mask = self.mask_conv(mask)

        if self.input_conv.bias is not None:
            # spreads existing bias values out along 2nd dimension (channels) and then expands to output size
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(output)
        else:
            output_bias = torch.zeros_like(output)

        # mask_sum is the sum of the binary mask at every partial convolution location
        mask_is_zero = (output_mask == 0)
        # temporarily sets zero values to one to ease output calculation 
        mask_sum = output_mask.masked_fill_(mask_is_zero, 1.0)

        # output at each location as follows:
        # output = (W^T dot (X .* M) + b - b) / M_sum + b ; if M_sum > 0
        # output = 0 ; if M_sum == 0
        output = (output - output_bias) / mask_sum + output_bias
        output = output.masked_fill_(mask_is_zero, 0.0)

        # mask is updated at each location
        new_mask = torch.ones_like(output_mask)
        new_mask = new_mask.masked_fill_(mask_is_zero, 0.0)

        if self.bn:
            output = self.batch_normalization(output)

        if hasattr(self, 'activation'):
            output = self.activation(output)

        return output, new_mask


class NLM_Partial_ConvNet_V3(nn.Module):

	# 256 x 256 image input, 256 = 2^8
    def __init__(self, input_size=144, layers=4):


        super().__init__()
        self.freeze_enc_bn = False
        self.layers = layers
  
  		# ======================= ENCODING LAYERS =======================
  		# 128x144x144 --> 256x72x72
        self.encoder_pre0 = PartialConvLayer(128, 128, bn=False, sample="others")
        self.encoder_pre1 = PartialConvLayer(128, 128, bn=False, sample="others")
        self.encoder_1 = PartialConvLayer(128, 256, bn=False, sample="down-7")
  
  		# 256x72x72 --> 512x36x36
        self.encoder_2 = PartialConvLayer(256, 512, sample="down-5")
        

      
        self.encoder_3 = PartialConvLayer(512, 512, sample="down-3")
        self.encoder_4 = PartialConvLayer(512, 512, sample="down-3")
  
    
  		# ======================= DECODING LAYERS =======================
  		# dec_4: UP(512x9x9) + 512x18x18(enc_3 output) = 1024x18x18 --> 512x18x18
  		# dec_3: UP(512x18x18) + 512x36x36(enc_2 output) = 1024x36x36 --> 512x36x36
          
        for i in range(3, layers + 1):
            name = "decoder_{:d}".format(i)
            setattr(self, name, PartialConvLayer(512 + 512, 512, activation="leaky_relu"))
  
  
  		# UP(512x36x36) + 256x72x72(enc_1 output) = 768x72x72 --> 256x72x72
        self.decoder_2 = PartialConvLayer(512 + 256, 256, activation="leaky_relu")
  
  		# UP(256x72x72) + 128x144x144(original image) = 384x144x144 --> 128x144x144(final output)
        self.decoder_1 = PartialConvLayer(256 + 128, 128, bn=False, activation="leaky_relu", bias=True)
        
        self.decoder_pre1 = PartialConvLayer(128 + 128, 128, activation="leaky_relu")
  
  		# UP(256x72x72) + 128x144x144(original image) = 384x144x144 --> 128x144x144(final output)
        self.decoder_pre0 = PartialConvLayer(128+ 128, 128, bn=False, activation="", bias=True)
	
    def forward(self, input_x, mask):
        mask_tempt = mask.clone()
        encoder_dict = {}
        mask_dict = {}
        buffer_incomplete = (input_x*mask).clone()


        out,mask = self.encoder_pre0(input_x, mask)
        buffer_pre0 = out.clone()
        
        out,mask = self.encoder_pre1(out, mask)
        buffer_pre1 = out.clone()
        
        out,mask = self.encoder_1(out, mask)
        buffer_1 = out.clone()
        
        out,mask = self.encoder_2(out, mask)
        buffer_2 = out.clone()
        
 
        out,mask = self.encoder_3(out, mask)
    
        
        ###################    patch swappping module  ###################
        threshold = 5.5/9
        h_gaussian = 5
        reshaped_encoder_out = torch.Tensor(out.detach().cpu().view((-1,512,18,18)).numpy().transpose(0,2,3,1) ).cuda()
        masked_feature_map = get_feature_map(mask_tempt,  threshold)
        out_feature_map = batch_NLM(masked_feature_map, reshaped_encoder_out, h_gaussian).cuda()  # final output size: (bs, channel, H, W)       
        out = out_feature_map.clone()
        buffer_3 = out.clone() 
                
        
        
        out,mask = self.encoder_4(out, mask)
        buffer_4 = out.clone()        
        
        
        




        out = F.interpolate(out, scale_factor=2)      
        mask = F.interpolate(mask, scale_factor=2)
        out = torch.cat([out, buffer_3], dim=1)
        out,mask = self.decoder_4(out, mask)
        
        
        
        
        out = F.interpolate(out, scale_factor=2)
        mask = F.interpolate(mask, scale_factor=2)
        out = torch.cat([out, buffer_2], dim=1)
        out,mask = self.decoder_3(out, mask)
        
        
        out = F.interpolate(out, scale_factor=2)
        mask = F.interpolate(mask, scale_factor=2)
        out = torch.cat([out, buffer_1], dim=1)
        out,mask = self.decoder_2(out, mask)
        
        out = F.interpolate(out, scale_factor=2)
        mask = F.interpolate(mask, scale_factor=2)


        out = torch.cat([out, buffer_pre1], dim=1)
        out,mask = self.decoder_1(out, mask)
        
       
        out = torch.cat([out, buffer_pre0], dim=1)
        out,mask = self.decoder_pre1(out, mask)
        
        out = torch.cat([out, buffer_incomplete], dim=1)
        out,mask = self.decoder_pre0(out, mask)
        

        



        return  out
