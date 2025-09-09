"""The registration model and all it's building blocks."""

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


def spatial_grads_to_sampling_grid_2d(spatial_grads: torch.Tensor):
    

    _, _, h, w = spatial_grads.shape
    y_grad, x_grad = torch.split(spatial_grads, 2, dim=1)
    # y_grad = spatial_grads[:, :1, :, :]
    # x_grad = spatial_grads[:, :1, :, :]

    y_grid = torch.cumsum(y_grad, dim=2)
    x_grid = torch.cumsum(x_grad, dim=3)

    grids = [y_grid, x_grid]

    grids = torch.cat(grids, dim=1)
    # normalized_grids = (grids / dim) * 2 - 1

    # normalized_grids = torch.stack([(grid / dim) * 2 - 1.
    #                                 for dim, grid in zip([h, w], grids)],
    #                                dim=-1).squeeze(1)
    return grids


class Model(nn.Module):
    """Main image registration module."""

    def __init__(self,
                 encoder_filters_per_block: Tuple[int, ...] = (
                         32, 64, 128, 32, 32),
                 encoder_filters_dilation_rates: Tuple[int, ...] = (
                         1, 1, 2, 3, 5),
                 d_decoder_filters_per_block: Tuple[int, ...] = (
                         128, 64, 32, 32, 32),
                 do_affine: bool = True):
        """Inits Model.

        Args:
            encoder_filters_per_block (Tuple[int, ...], optional): number
                of filters for each convolution in the encoder.
                Defaults to ( 32, 64, 128, 32, 32).
            encoder_filters_dilation_rates (Tuple[int, ...], optional):
                dilation rate for each of the convolutions defined by the
                parameter above. Defaults to ( 1, 1, 2, 3, 5).
            d_decoder_filters_per_block (Tuple[int, ...], optional): number
                of convolution filters in the decoder block.
                Defaults to ( 128, 64, 32, 32, 32).
            do_affine (bool, optional): Whether to perform affine registration
                on top of deformable or not. Defaults to True.
        """
        super().__init__()
        self._do_affine = do_affine
        self.encoder = Encoder(in_channels=2,
                               out_channels_per_block=encoder_filters_per_block,
                               dilation_rates=encoder_filters_dilation_rates)
        self.d_decoder = DeformableDecoderBlock2d(
            in_channels=sum(encoder_filters_per_block) + 0,
            out_channels_per_block=d_decoder_filters_per_block)
        if do_affine:
            self.a_decoder = LinearDecoderBlock2d(
                in_channels=sum(encoder_filters_per_block) + 0)
        else:
            self.register_parameter("a_decoder", None)

    def forward(self,
                source: torch.Tensor,
                target: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """forward pass through the model.

        Args:
            source (torch.Tensor): the source image volume.
            target (torch.Tensor): the target image volume.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: returns
                the warped source image, the spatial gradients of the
                deformable component and the affine matrix of the affine
                branch.
        """
        x = self.encoder(source, target)
        spatial_grads = self.d_decoder(x)
        grid = spatial_grads_to_sampling_grid_2d(spatial_grads)
        warped_src = F.grid_sample(source, grid, align_corners=False)
        theta = None
        if self._do_affine:
            theta = self.a_decoder(x)
            grid = F.affine_grid(theta, target.shape, align_corners=False)
            warped_src = F.grid_sample(warped_src, grid, align_corners=False)
        return warped_src, spatial_grads, theta


class Encoder(nn.Module):
    """Encoder of the registration module.

    The module essentially concats the source and target and passes it
    through a bunch of dilated conv blocks.
    """

    def __init__(self,
                 in_channels: int = 2,
                 out_channels_per_block: Tuple[int, ...] = (
                         32, 64, 128, 32, 32),
                 dilation_rates: Tuple[int, ...] = (1, 1, 2, 3, 5),
                 kernel_size: int = 3):
        """Inits Encoder.

        Args:
            in_channels (int, optional): number of input channels to the
                encoder. Defaults to 2.
            out_channels_per_block (Tuple[int, ...], optional): Number of output
                channles per conv block of the encoder. Defaults to
                ( 32, 64, 128, 32, 32).
            dilation_rates (Tuple[int, ...], optional): dilation rate of each
                conv block. Defaults to (1, 1, 2, 3, 5).
            kernel_size (int, optional): size of conv kernel.. Defaults to 3.

        Raises:
            ValueError: If the length of out_channels_per_block tuple does
                not match the length of the dilation rates.
        """
        if not len(out_channels_per_block) == len(dilation_rates):
            raise ValueError(
                "Expected one dilation rate per conv block. Found"
                f" {len(dilation_rates)} dilation rates for"
                f" {len(out_channels_per_block)} blocks.")
        super().__init__()
        self._conv_blocks = nn.ModuleList()
        self.out_channels_per_block = out_channels_per_block
        for num_filters, dilation_rate in zip(out_channels_per_block,
                                              dilation_rates):
            effective_kernel_size = kernel_size + 2 * (dilation_rate - 1)
            self._conv_blocks.append(nn.Sequential(
                nn.Conv2d(in_channels,
                          num_filters,
                          kernel_size=kernel_size,
                          dilation=dilation_rate,
                          bias=False,
                          padding=effective_kernel_size // 2),
                nn.InstanceNorm2d(num_filters),
                nn.LeakyReLU(inplace=True)))
            in_channels = num_filters

    def forward(self,
                source: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([source, target], dim=1)
        outputs = []
        for conv_block in self._conv_blocks:
            out = conv_block(inputs)
            outputs.append(out)
            inputs = out
        outputs = torch.cat([source, target, *outputs], dim=1)
        return outputs


class SqueezeExcitation2d(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, channel // reduction, 1)
        self.conv2 = nn.Conv2d(channel // reduction, channel, 1)

    def forward(self, x):
        pool = F.adaptive_avg_pool2d(x, 1)
        squeeze = F.relu(self.conv1(pool))
        excitation = torch.sigmoid(self.conv2(squeeze))
        return x * excitation


class DeformableDecoderBlock2d(nn.Module):
    """Spatial gradients decoder block.

    This block takes as input the encoded feature maps from the encoder block
    and generates spatial gradients along x and y directions.
    """

    def __init__(self,
                 in_channels: int = 290,
                 out_channels_per_block: Tuple[int, ...] = (128, 64, 32, 32, 32),
                 kernel_size: int = 3,
                 squeeze_factor: int = 16):
        super().__init__()
        self._se = SqueezeExcitation2d(in_channels, squeeze_factor)
        conv_blocks = []
        for num_filters in out_channels_per_block:
            conv_blocks.append(nn.Sequential(
                nn.Conv2d(in_channels,
                          num_filters,
                          bias=False,
                          kernel_size=kernel_size,
                          padding=kernel_size // 2),
                nn.InstanceNorm2d(num_filters),
                nn.LeakyReLU(inplace=True)))
            in_channels = num_filters
        final_conv = nn.Conv2d(in_channels,
                              out_channels=3,
                              kernel_size=kernel_size,
                              padding=kernel_size // 2)
        self._convs = nn.Sequential(*conv_blocks, final_conv)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return 2 * torch.sigmoid(self._convs(self._se(inputs)))


class LinearDecoderBlock2d(nn.Module):
    """Affine matrix decoder block.

    This block takes as input the encoder feature maps and generates the affine
    matrix for the linear registration component of the network.
    """

    def __init__(self, in_channels=290):
        super().__init__()
        self._conv = nn.Conv2d(in_channels, 12, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        pool = F.adaptive_avg_pool2d(inputs, 1)
        return self._conv(pool).view(-1, 3, 4)