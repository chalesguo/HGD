"""
*Preliminary* pytorch implementation.

Networks for voxelmorph model

In general, these are fairly specific architectures that were designed for the presented papers.
However, the VoxelMorph concepts are not tied to a very particular architecture, and we
encourage you to explore architectures that fit your needs.
see e.g. more powerful unet function in https://github.com/adalca/neuron/blob/master/neuron/models.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.distributions.normal import Normal
import gc

class unet_core(nn.Module):
    """
    [unet_core] is a class representing the U-Net implementation that takes in
    a fixed image and a moving image and outputs a flow-field
    """

    def __init__(self, dim, enc_nf, dec_nf, full_size=True):
        """
        Instiatiate UNet model
            :param dim: dimension of the image passed into the net
            :param enc_nf: the number of features maps in each layer of encoding stage
            :param dec_nf: the number of features maps in each layer of decoding stage
            :param full_size: boolean value representing whether full amount of decoding
                            layers
        """
        super(unet_core, self).__init__()

        self.full_size = full_size
        self.vm2 = len(dec_nf) == 7

        # Encoder functions
        self.enc = nn.ModuleList()
        for i in range(len(enc_nf)):
            prev_nf = 2 if i == 0 else enc_nf[i - 1]
            self.enc.append(conv_block(dim, prev_nf, enc_nf[i], 2))
            # stride=2,to downsample 2
        self.up_tri = torch.nn.Upsample(scale_factor=2, mode="bilinear")

        self.input_encoder_lvl1 = conv_block1(dim,2, 2, 3, 2, 1)

        self.input_encoder_lvl2 = conv_block1(dim,16, 16, 3, 2, 1)
        self.input_encoder_lvl3 = conv_block1(dim,32, 16, 3, 2, 1)
        self.input_encoder_lvl4 = conv_block1(dim,32, 16, 1, 1, 0)

        self.input_encoder_lvl5 = conv_block1(dim,48, 16, 1, 1, 0)
        self.input_encoder_lvl6 = conv_block1(dim,16, 16, 3, 1, 1)
        self.input_encoder_lvl7 = conv_block2(dim,16, 16, 3, 1, 1)

        self.input_encoder_lvl8 = conv_block1(dim,32, 16, 3, 1, 1)

        self.input_encoder_lvl9 = conv_block1(dim,2, 16, 1, 1, 0)
        self.input_encoder_lvl10 = conv_block1(dim,16, 32, 1, 1, 0)

        # Decoder functions
        self.dec = nn.ModuleList()
        self.dec.append(conv_block(dim, enc_nf[-1], dec_nf[0]))  # 1
        self.dec.append(conv_block(dim, dec_nf[0] * 2, dec_nf[1]))  # 2
        self.dec.append(conv_block(dim, dec_nf[1] * 2, dec_nf[2]))  # 3
        self.dec.append(conv_block(dim, dec_nf[2] + enc_nf[0], dec_nf[3]))  # 4
        self.dec.append(conv_block(dim, dec_nf[3], dec_nf[4]))  # 5

        if self.full_size:
            self.dec.append(conv_block(dim, dec_nf[4] + 2, dec_nf[5], 1))

        if self.vm2:
            self.vm2_conv = conv_block(dim, dec_nf[5], dec_nf[6])

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    # def input_feature_extract(self, in_channels, out_channels, kernel_size, stride, padding, batchnorm=False):
    #     if batchnorm:
    #         layer = nn.Sequential(
    #             nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
    #             nn.BatchNorm2d(out_channels),
    #             nn.ReLU())
    #     else:
    #         layer = nn.Sequential(
    #             nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
    #             nn.LeakyReLU(0.2))
    #         # nn.Conv2d(out_channels, out_channels, kernel_size, stride=stride, padding=padding))
    #     return layer
    #
    # def input_feature_extract2(self, in_channels, out_channels, kernel_size, stride, padding, batchnorm=False):
    #     if batchnorm:
    #         layer = nn.Sequential(
    #             nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
    #             nn.BatchNorm2d(out_channels),
    #             nn.Sigmoid())
    #     else:
    #         layer = nn.Sequential(
    #             nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
    #             nn.Sigmoid())
    #         # nn.Conv2d(out_channels, out_channels, kernel_size, stride=stride, padding=padding))
    #     return layer

    def forward(self, x):
        """
        Pass input x through the UNet forward once
            :param x: concatenated fixed and moving image
        """
        # Get encoder activations
        x_enc = [x]
        for l in self.enc:
            x_enc.append(l(x_enc[-1]))
        A1 = x_enc[-1]
        A2 = x_enc[-2]
        A3 = x_enc[-3]
        A4 = x_enc[-4]
        A5 = x_enc[-5]

        # a1, a3, attn_map, attn_f = self.attention_e_combine(A3, A4, 16, name='e-combine')
        # A55 = self.input_encoder_lvl1(A5)
        # A55 = self.input_encoder_lvl1(A55)
        # A55 = self.input_encoder_lvl1(A55)

        A44 = self.input_encoder_lvl2(A4)
        A44 = self.input_encoder_lvl2(A44)

        A33 = self.input_encoder_lvl3(A3)

        A22 = self.input_encoder_lvl4(A2)

        ecombine = torch.cat([A44, A33, A22], dim=1)

        contextfeature = self.input_encoder_lvl5(ecombine)
        #
        attn_map = self.input_encoder_lvl6(contextfeature)
        attn_map = self.input_encoder_lvl7(attn_map)

        attn_feature = torch.mul(contextfeature, attn_map)

        # A55 = torch.cat([attn_feature, A55], dim=1)
        # A55 = self.input_encoder_lvl8(A55)
        # A55 = self.up_tri(A55)
        # A55 = self.up_tri(A55)
        # A55 = self.up_tri(A55)

        A44 = torch.cat([attn_feature, A44], dim=1)
        A44 = self.input_encoder_lvl8(A44)

        A44 = self.up_tri(A44)
        A44 = self.up_tri(A44)

        A33 = torch.cat([attn_feature, A33], dim=1)
        A33 = self.input_encoder_lvl8(A33)
        A33 = self.input_encoder_lvl10(A33)
        A33 = self.up_tri(A33)

        A22 = torch.cat([attn_feature, A22], dim=1)
        A22 = self.input_encoder_lvl8(A22)
        A22 = self.input_encoder_lvl10(A22)

        for i in range(len(x_enc)):
            x_enc[-4] = A44
            x_enc[-3] = A33
            x_enc[-2] = A22
            # x_enc[-5] = A55
        # Three conv + upsample + concatenate series
        y = x_enc[-1]
        for i in range(3):
            y = self.dec[i](y)
            y = self.upsample(y)
            z = x_enc[-(i + 2)]
            y = torch.cat([y, x_enc[-(i + 2)]], dim=1)

        # Two convs at full_size/2 res
        y = self.dec[3](y)
        y = self.dec[4](y)

        # Upsample to full res, concatenate and conv
        if self.full_size:
            y = self.upsample(y)
            y = torch.cat([y, x_enc[0]], dim=1)
            y = self.dec[5](y)

        # Extra conv for vm2
        if self.vm2:
            y = self.vm2_conv(y)

        return y


class cvpr2018_net1(nn.Module):
    """
    [cvpr2018_net] is a class representing the specific implementation for
    the 2018 implementation of voxelmorph.
    """

    def __init__(self, vol_size, enc_nf, dec_nf, full_size=True):
        """
        Instiatiate 2018 model
            :param vol_size: volume size of the atlas
            :param enc_nf: the number of features maps for encoding stages
            :param dec_nf: the number of features maps for decoding stages
            :param full_size: boolean value full amount of decoding layers
        """
        super(cvpr2018_net1, self).__init__()

        dim = len(vol_size)

        self.unet_model = unet_core(dim, enc_nf, dec_nf, full_size)

        # One conv to get the flow field
        conv_fn = getattr(nn, 'Conv%dd' % dim)  
        self.flow = conv_fn(dec_nf[-1], dim, kernel_size=3,
                            padding=1)  
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(
            self.flow.bias.shape))  
        self.spatial_transform = SpatialTransformer3(vol_size) 
        self.down_avg = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

    def forward(self, src, tgt, iir_iter):
        """
        Pass input x through forward once
            :param src: moving image that we want to shift
            :param tgt: fixed image that we want to shift to
        """

        b_size = src.size()[0]
        h_x1 = src.size()[2]
        # h_x1=h_x1/4
        w_x1 = src.size()[3]
        # w_x1=w_x1/4
        init_dtype = src.dtype
        init_device = src.device

        flow = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        flow1 = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float() 
        src_move = src

        # for j in range(iir_iter):
        cat_input = torch.cat((src, tgt), 1)
        # cat_input = self.down_avg(cat_input)

        down_x = cat_input[:, 0:1, :, :]
        down_y = cat_input[:, 1:2, :, :]
        for j in range(iir_iter):
            cat_input_lvl1 = torch.cat((down_x, down_y), 1)
            x = self.unet_model(cat_input_lvl1)
            flow_fine = self.flow(x)
            flow = flow + flow_fine   
            y = self.spatial_transform(down_x, flow) 
            down_x = y
            # x = torch.cat([src, tgt], dim=1)
            # x = self.unet_model(x)
            # flow_fine = self.flow(x)
            # flow=flow+flow_fine
            # y = self.spatial_transform(src_move, flow)
        flow1 = self.down_avg(flow) 
        fwd = {'Moving': down_x, 'Fixed': down_y, 'Flow': flow, 'Moved': down_x}
        return fwd
      

class cvpr2018_net2(nn.Module):
    """
    [cvpr2018_net] is a class representing the specific implementation for
    the 2018 implementation of voxelmorph.
    """

    def __init__(self, vol_size, enc_nf, dec_nf, full_size=True, model1=None):
        """
        Instiatiate 2018 model
            :param vol_size: volume size of the atlas
            :param enc_nf: the number of features maps for encoding stages
            :param dec_nf: the number of features maps for decoding stages
            :param full_size: boolean value full amount of decoding layers
        """
        super(cvpr2018_net2, self).__init__()

        dim = len(vol_size)

        self.unet_model = unet_core(dim, enc_nf, dec_nf, full_size)

        # One conv to get the flow field
        conv_fn = getattr(nn, 'Conv%dd' % dim)  # conv_fn 为conv.Conv2d
        self.flow = conv_fn(dec_nf[-1], dim, kernel_size=3,
                            padding=1)  
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(
            self.flow.bias.shape)) 
        self.spatial_transform = SpatialTransformer2(vol_size) 
        self.down_avg = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)
        self.model1 = model1
        self.up_tri = torch.nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, src, tgt, iir_iter):
        """
        Pass input x through forward once
            :param src: moving image that we want to shift
            :param tgt: fixed image that we want to shift to
        """

        b_size = src.size()[0]
        h_x1 = src.size()[2] / 2
        w_x1 = src.size()[3] / 2
        init_dtype = src.dtype
        init_device = src.device

        flow100 = torch.zeros(b_size, 2, 96, 96, dtype=init_dtype, device=init_device).float()
        # flow1 = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        src_move = src
        # for j in range(iir_iter):

        src1, flow1, tgt1, flow10 = self.model1(src, tgt, iir_iter)
        lvl1_disp_up = self.up_tri(flow1)
        x_down = self.down_avg(src)
        y_down = self.down_avg(tgt)
        down_x = self.spatial_transform(x_down, lvl1_disp_up)

        # cat_input = torch.cat((down_x, y_down), 1)
        # cat_input = self.down_avg(cat_input)
        # cat_input_lvl1 = self.down_avg(cat_input)
        # down_x = cat_input_lvl1[:, 0:1, :, :]
        # down_y = cat_input_lvl1[:, 1:2, :, :]

        # x = self.unet_model(cat_input)
        # flow_fine = self.flow(x)
        # flow = flow + flow_fine
        # y = self.spatial_transform(down_x, flow)

        for j in range(iir_iter):
            cat_input_lvl1 = torch.cat((down_x, y_down), 1)
            x = self.unet_model(cat_input_lvl1)
            flow_fine = self.flow(x)
            flow = flow100 + flow_fine
            flow100 = flow
            y = self.spatial_transform(down_x, flow100)
            down_x = y
        # x = torch.cat([src, tgt], dim=1)
        # x = self.unet_model(x)
        # flow_fine = self.flow(x)
        # flow=flow+flow_fine
        # y = self.spatial_transform(src_move, flow)

        # src = y
        # tgt = y_down
        return down_x, flow100, y_down,flow



class cvpr2018_net3(nn.Module):
    """
    [cvpr2018_net] is a class representing the specific implementation for
    the 2018 implementation of voxelmorph.
    """

    def __init__(self, vol_size, enc_nf, dec_nf, full_size=True, model1=None, model2=None):
        """
        Instiatiate 2018 model
            :param vol_size: volume size of the atlas
            :param enc_nf: the number of features maps for encoding stages
            :param dec_nf: the number of features maps for decoding stages
            :param full_size: boolean value full amount of decoding layers
        """
        super(cvpr2018_net3, self).__init__()

        dim = len(vol_size)

        self.unet_model = unet_core(dim, enc_nf, dec_nf, full_size)

        # One conv to get the flow field
        conv_fn = getattr(nn, 'Conv%dd' % dim)  # conv_fn 为conv.Conv2d
        self.flow = conv_fn(dec_nf[-1], dim, kernel_size=3,
                            padding=1) 
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(
            self.flow.bias.shape)) 
        self.spatial_transform = SpatialTransformer3(vol_size) 

        self.down_avg = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)
        self.model1 = model1
        self.model2 = model2
        self.up_tri = torch.nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, src, tgt, iir_iter):
        """
        Pass input x through forward once
            :param src: moving image that we want to shift
            :param tgt: fixed image that we want to shift to
        """

        b_size = src.size()[0]
        h_x1 = src.size()[2]
        w_x1 = src.size()[3]
        init_dtype = src.dtype
        init_device = src.device

        flow100 = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        # flow1 = torch.zeros(b_size, 2, h_x1, w_x1, dtype=init_dtype, device=init_device).float()
        src_move = src
        # for j in range(iir_iter):

        src1, flow1, fix1, fix11 = self.model1(src, tgt, iir_iter)
        src2, flow2, fix2, flow3 = self.model2(src, tgt, iir_iter)
        lvl2_disp_up = self.up_tri(flow2)
        lvl1_disp_up = self.up_tri(flow1)
        lvl1_disp_up = self.up_tri(lvl1_disp_up)
        lvl2_disp_up = lvl2_disp_up + lvl1_disp_up
        # x_down = self.down_avg(src)
        # y_down = self.down_avg(tgt)

        down_x = self.spatial_transform(src, lvl2_disp_up)
        for j in range(iir_iter):
            cat_input_lvl1 = torch.cat((down_x, tgt), 1)
            x = self.unet_model(cat_input_lvl1)
            flow_fine = self.flow(x)
            flow = flow100 + flow_fine
            flow100 = flow
            y = self.spatial_transform(down_x, flow100)
            down_x = y
            gc.collect()
        # cat_input = torch.cat((down_x, tgt), 1)
        # # cat_input = self.down_avg(cat_input)
        # # cat_input_lvl1 = self.down_avg(cat_input)
        # # down_x = cat_input_lvl1[:, 0:1, :, :]
        # # down_y = cat_input_lvl1[:, 1:2, :, :]
        # x = self.unet_model(cat_input)
        # flow_fine = self.flow(x)
        # flow = flow + flow_fine
        # y = self.spatial_transform(down_x, flow)

        # x = torch.cat([src, tgt], dim=1)
        # x = self.unet_model(x)
        # flow_fine = self.flow(x)
        # flow=flow+flow_fine
        # y = self.spatial_transform(src_move, flow)

        # src = y
        flow = self.down_avg(flow100)
        flow = self.down_avg(flow)
        flow = self.down_avg(flow)
        flow3 = lvl2_disp_up + flow100
        return down_x, flow100, tgt, flow, flow3


class SpatialTransformer1(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample
    """

    def __init__(self, size, mode='bilinear'):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super(SpatialTransformer1, self).__init__()

        # Create sampling grid
        vectors = [torch.arange(0, s / 4) for s in size]  
        grids = torch.meshgrid(vectors) 
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0) 
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid',
                             grid)                                                

        self.mode = mode

    def forward(self, down_x, flow):
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """
        new_locs = self.grid + flow  

        shape = flow.shape[2:] 
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[
                                                                  i] - 1) - 0.5) 
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1) 
            new_locs = new_locs[..., [1, 0]] 


        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(down_x, new_locs, mode=self.mode)


class SpatialTransformer2(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample
    """

    def __init__(self, size, mode='bilinear'):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super(SpatialTransformer2, self).__init__()

        # Create sampling grid
        vectors = [torch.arange(0, s / 2) for s in size]  
        grids = torch.meshgrid(vectors) 
        grid = torch.stack(grids)  
        grid = torch.unsqueeze(grid, 0) 
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid',
                             grid)                                           

        self.mode = mode

    def forward(self, down_x, flow):
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """
        new_locs = self.grid + flow  

        shape = flow.shape[2:]  
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[
                                                                  i] - 1) - 0.5)  
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)  
            new_locs = new_locs[..., [1, 0]]  


        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(down_x, new_locs, mode=self.mode)


class SpatialTransformer3(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample
    """

    def __init__(self, size, mode='bilinear'):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super(SpatialTransformer3, self).__init__()

        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size] 
        grids = torch.meshgrid(vectors) 
        grid = torch.stack(grids) 
        grid = torch.unsqueeze(grid, 0) 
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid',
                             grid)  

        self.mode = mode

    def forward(self, down_x, flow):
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """
        new_locs = self.grid + flow 

        shape = flow.shape[2:] 
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[
                                                                  i] - 1) - 0.5)  
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)  
            new_locs = new_locs[..., [1, 0]]  


        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(down_x, new_locs, mode=self.mode)





class conv_block(nn.Module):
    """
    [conv_block] represents a single convolution block in the Unet which
    is a convolution based on the size of the input channel and output
    channels and then preforms a Leaky Relu with parameter 0.2.
    """

    def __init__(self, dim, in_channels, out_channels, stride=1):
        """
        Instiatiate the conv block
            :param dim: number of dimensions of the input
            :param in_channels: number of input channels
            :param out_channels: number of output channels
            :param stride: stride of the convolution
        """
        super(conv_block, self).__init__()

        conv_fn = getattr(nn, "Conv{0}d".format(dim))

        if stride == 1:
            ksize = 3
        elif stride == 2:
            ksize = 4
        else:
            raise Exception('stride must be 1 or 2')

        self.main = conv_fn(in_channels, out_channels, ksize, stride, 1)
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode=‘zeros’
        self.activation = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        """
        Pass the input through the conv_block
        """
        out = self.main(x)
        out = self.dropout(out)
        out = self.activation(out)
        return out

class conv_block1(nn.Module):
    """
    [conv_block] represents a single convolution block in the Unet which
    is a convolution based on the size of the input channel and output
    channels and then preforms a Leaky Relu with parameter 0.2.
    """

    def __init__(self, dim, in_channels, out_channels, ksize, stride,padding):
        """
        Instiatiate the conv block
            :param dim: number of dimensions of the input
            :param in_channels: number of input channels
            :param out_channels: number of output channels
            :param stride: stride of the convolution
        """
        super(conv_block1, self).__init__()

        conv_fn = getattr(nn, "Conv{0}d".format(dim))

        self.main = conv_fn(in_channels, out_channels, ksize, stride, padding)
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode=‘zeros’
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        Pass the input through the conv_block
        """
        out = self.main(x)
        out = self.activation(out)
        return out

class conv_block2(nn.Module):
    """
    [conv_block] represents a single convolution block in the Unet which
    is a convolution based on the size of the input channel and output
    channels and then preforms a Leaky Relu with parameter 0.2.
    """

    def __init__(self, dim, in_channels, out_channels, ksize, stride,padding):
        """
        Instiatiate the conv block
            :param dim: number of dimensions of the input
            :param in_channels: number of input channels
            :param out_channels: number of output channels
            :param stride: stride of the convolution
        """
        super(conv_block2, self).__init__()

        conv_fn = getattr(nn, "Conv{0}d".format(dim))

        self.main = conv_fn(in_channels, out_channels, ksize, stride, padding)
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode=‘zeros’
        self.activation = nn.Sigmoid()

    def forward(self, x):
        """
        Pass the input through the conv_block
        """
        out = self.main(x)
        out = self.activation(out)
        return out