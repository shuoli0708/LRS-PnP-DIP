import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt





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


class Partial_ConvNet(nn.Module):

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
        encoder_dict = {}
        mask_dict = {}





        key_prev = "h_0"
        encoder_dict[key_prev], mask_dict[key_prev] = input_x, mask

        for i in range(1, self.layers + 1):
            encoder_key = "encoder_{:d}".format(i)
            key = "h_{:d}".format(i)
			# Passes input and mask through encoding layer
            encoder_dict[key], mask_dict[key] = getattr(self, encoder_key)(encoder_dict[key_prev], mask_dict[key_prev])
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

			# concatenate upsampled decoder output with encoder output of same H x W dimensions
			# s.t. final decoding layer input will contain the original image
            out_data = torch.cat([out_data, encoder_dict[encoder_key]], dim=1)
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
'''
if __name__ == '__main__':
    size = (10, 128, 144, 144)
    inp = torch.ones(size)
    size_2 = (10, 1, 144, 144)
    input_mask = torch.ones(size_2)
  

    conv = Partial_ConvNet()
    l1 = nn.L1Loss()
    inp.requires_grad = True

    output = conv(inp, input_mask)
    loss = l1(output, torch.randn(10, 128, 144, 144))
    loss.backward()

    assert (torch.sum(inp.grad != inp.grad).item() == 0)
    assert (torch.sum(torch.isnan(conv.decoder_1.input_conv.weight.grad)).item() == 0)
    assert (torch.sum(torch.isnan(conv.decoder_1.input_conv.bias.grad)).item() == 0)
'''
