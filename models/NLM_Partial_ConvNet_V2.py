import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd



class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant(self.W[1].weight, 0)
            nn.init.constant(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant(self.W.weight, 0)
            nn.init.constant(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        # z = W_y + x
        z = W_y 

        return f_div_C, z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer)








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


class NLM_Partial_ConvNet_V2(nn.Module):

	# 256 x 256 image input, 256 = 2^8
    def __init__(self, input_size=144, layers=4):


        super().__init__()
        self.freeze_enc_bn = False
        self.layers = layers
  
  		# ======================= ENCODING LAYERS =======================
  		# 128x144x144 --> 256x72x72
        self.encoder_pre00 = PartialConvLayer(128, 128, bn=False, sample="others")
        self.encoder_pre0 = PartialConvLayer(128, 128, sample="others")
        self.encoder_pre1 = PartialConvLayer(128, 128,  sample="others")
        self.encoder_1 = PartialConvLayer(128, 256,  sample="down-7")
  
  		# 256x72x72 --> 512x36x36
        self.encoder_2 = PartialConvLayer(256, 512, sample="down-5")
        
        
   												
        self.Non_Local_Module_1 = NONLocalBlock2D(256, sub_sample=False, bn_layer=True)
        self.Non_Local_Module_2 = NONLocalBlock2D(512, sub_sample=False, bn_layer=True)
        self.Non_Local_Module_3 = NONLocalBlock2D(512, sub_sample=False, bn_layer=True)
      
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
        self.decoder_1 = PartialConvLayer(256 + 128, 128, activation="leaky_relu")
        
        self.decoder_pre1 = PartialConvLayer(128 + 128, 128, activation="leaky_relu")
  
  		# UP(256x72x72) + 128x144x144(original image) = 384x144x144 --> 128x144x144(final output)
        self.decoder_pre0 = PartialConvLayer(128+ 128, 128, activation="leaky_relu")
        self.decoder_pre00 = PartialConvLayer(128+ 128, 128, bn=False, activation="", bias=True)
	
    def forward(self, input_x, mask):
        mask_tempt = mask.clone()
        encoder_dict = {}
        mask_dict = {}
        buffer_incomplete = (input_x*mask).clone()






        out,mask = self.encoder_pre00(input_x, mask)
        buffer_pre00 = out.clone()




        out,mask = self.encoder_pre0(out, mask)
        buffer_pre0 = out.clone()
        
        out,mask = self.encoder_pre1(out, mask)
        buffer_pre1 = out.clone()
        
        out,mask = self.encoder_1(out, mask)
        a,out = self.Non_Local_Module_1(out)
        buffer_1 = out.clone()
        
        out,mask = self.encoder_2(out, mask)
        a,out = self.Non_Local_Module_2(out)
        buffer_2 = out.clone()
        
        out,mask = self.encoder_3(out, mask)
        a,out = self.Non_Local_Module_3(out)  
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
               
        out = torch.cat([out, buffer_pre00], dim=1)
        out,mask = self.decoder_pre0(out, mask)
        
           
        out = torch.cat([out, buffer_incomplete], dim=1)
        out,mask = self.decoder_pre00(out, mask)



        return  out
