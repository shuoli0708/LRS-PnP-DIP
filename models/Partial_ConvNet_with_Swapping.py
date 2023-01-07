import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd




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


class Partial_ConvNet_with_Swapping(nn.Module):

	# 256 x 256 image input, 256 = 2^8
    def __init__(self, input_size=144, layers=4):


        super().__init__()
        self.freeze_enc_bn = False
        self.layers = layers
  
  		# ======================= ENCODING LAYERS =======================
  		# 128x144x144 --> 256x72x72
        self.encoder_1 = PartialConvLayer(128, 256, bn=False, sample="down-7")
  
  		# 256x72x72 --> 512x36x36
        self.encoder_2 = PartialConvLayer(256, 512, sample="down-5")
  
  		# 512x36x36 --> 512x18x18 --> 512x9x9 
        for i in range(3, layers + 1):
            name = "encoder_{:d}".format(i)
            setattr(self, name, PartialConvLayer(512, 512, sample="down-3"))
  
  		# ======================= DECODING LAYERS =======================
  		# dec_4: UP(512x9x9) + 512x18x18(enc_3 output) = 1024x18x18 --> 512x18x18
  		# dec_3: UP(512x18x18) + 512x36x36(enc_2 output) = 1024x36x36 --> 512x36x36
        for i in range(3, layers + 1):
            name = "decoder_{:d}".format(i)
            setattr(self, name, PartialConvLayer(512 + 512, 512, activation="leaky_relu"))
  
  
  		# UP(512x36x36) + 256x72x72(enc_1 output) = 768x72x72 --> 256x72x72
        self.decoder_2 = PartialConvLayer(512 + 256, 256, activation="leaky_relu")
  
  		# UP(256x72x72) + 128x144x144(original image) = 384x144x144 --> 128x144x144(final output)
        self.decoder_1 = PartialConvLayer(256 + 128, 128, bn=False, activation="", bias=True)
	
    def forward(self, input_x, mask):
        mask_tempt = mask.clone()

        encoder_dict = {}
        mask_dict = {}

        key_prev = "h_0"
        encoder_dict[key_prev], mask_dict[key_prev] = input_x, mask

        for i in range(1, self.layers + 1):
            encoder_key = "encoder_{:d}".format(i)
            key = "h_{:d}".format(i)
			# Passes input and mask through encoding layer
            encoder_dict[key], mask_dict[key] = getattr(self, encoder_key)(encoder_dict[key_prev], mask_dict[key_prev])
            

            #########   NLM Swapping  ###################
            if (i ==2):
                
                ###################    patch swappping module  ###################
                reshaped_encoder_out = torch.Tensor(encoder_dict[key].detach().cpu().view((-1,512,36,36)).numpy().transpose((0,2,3,1))).cuda() # bs x  9 x 9 x 512 reshaped out_data,  output: bs x 72 x 72 x 256 swapped out_data
                h_gaussian = 5      
                new_mask = mask_dict[key].clone()          
                encoder_dict[key] = batch_NLM(new_mask, reshaped_encoder_out, h_gaussian).cuda()
                buffer_swapped_feature_map = encoder_dict[key].clone()
            
            key_prev = key

		# Gets the final output data and mask from the encoding layers
		# 512 x 9 x 9
        out_key = "h_{:d}".format(self.layers)
        out_data, out_mask = encoder_dict[out_key], mask_dict[out_key]






        for i in range(self.layers, 0, -1):
            encoder_key = "h_{:d}".format(i - 1)
            decoder_key = "decoder_{:d}".format(i)

			# Upsample to 2 times scale, matching dimensions of previous encoding layer output
            out_data = F.interpolate(out_data, scale_factor=2)
            out_mask = F.interpolate(out_mask, scale_factor=2)
		
            if (i ==3):
                out_data = torch.cat([out_data, buffer_swapped_feature_map], dim=1)
           
            else:
                out_data = torch.cat([out_data, encoder_dict[encoder_key]], dim=1)
                
			# concatenate upsampled decoder output with encoder output of same H x W dimensions
			# s.t. final decoding layer input will contain the original image
			# also concatenate the masks
            # out_mask = torch.cat([out_mask, mask_dict[encoder_key]], dim=1)
			
			# feed through decoder layers
            out_data, out_mask = getattr(self, decoder_key)(out_data, out_mask)

        return out_data



    def train(self, mode=True):
        super().train(mode)
        if self.freeze_enc_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) and "enc" in name:
					# Sets batch normalization layers to evaluation mode
                    module.eval()






