"""IO pre- and post-processors for Perceiver."""

import math
import warnings
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import einops

from timm.models.layers import lecun_normal_, trunc_normal_

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from perceiver_io import position_encoding
from perceiver_io.position_encoding import PosEncodingType
from utils.utils import conv_output_shape, init_linear_from_haiku, same_padding, init_conv_from_haiku, \
    init_batchnorm_from_haiku

ModalitySizeT = Mapping[str, int]
PreprocessorOutputT = Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]
PreprocessorT = Callable[..., PreprocessorOutputT]
PostprocessorT = Callable[..., Any]

PreprocessorOutputT = Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]
PreprocessorT = Callable[..., PreprocessorOutputT]
PostprocessorT = Callable[..., Any]

def reverse_space_to_depth(
        frames: torch.Tensor,
        temporal_block_size: int = 1,
        spatial_block_size: int = 1) -> torch.Tensor:
    """Reverse space to depth transform."""
    if len(frames.shape) == 4:
        return einops.rearrange(
            frames, 'b h w (dh dw c) -> b (h dh) (w dw) c',
            dh=spatial_block_size, dw=spatial_block_size)
    elif len(frames.shape) == 5:
        return einops.rearrange(
            frames, 'b t h w (dt dh dw c) -> b (t dt) (h dh) (w dw) c',
            dt=temporal_block_size, dh=spatial_block_size, dw=spatial_block_size)
    else:
        raise ValueError(
            'Frames should be of rank 4 (batch, height, width, channels)'
            ' or rank 5 (batch, time, height, width, channels)')


def space_to_depth(
        frames: torch.Tensor,
        temporal_block_size: int = 1,
        spatial_block_size: int = 1) -> torch.Tensor:
    """Space to depth transform."""
    if len(frames.shape) == 4:
        return einops.rearrange(
            frames, 'b (h dh) (w dw) c -> b h w (dh dw c)',
            dh=spatial_block_size, dw=spatial_block_size)
    elif len(frames.shape) == 5:
        return einops.rearrange(
            frames, 'b (t dt) (h dh) (w dw) c -> b t h w (dt dh dw c)',
            dt=temporal_block_size, dh=spatial_block_size, dw=spatial_block_size)
    else:
        raise ValueError(
            'Frames should be of rank 4 (batch, height, width, channels)'
            ' or rank 5 (batch, time, height, width, channels)')


def extract_patches(images: torch.Tensor,
                    size: Sequence[int],
                    stride: Sequence[int] = 1,
                    dilation: Sequence[int] = 1,
                    padding: str = 'VALID') -> torch.Tensor:
    """Extract patches from images.
  The function extracts patches of shape sizes from the input images in the same
  manner as a convolution with kernel of shape sizes, stride equal to strides,
  and the given padding scheme.
  The patches are stacked in the channel dimension.
  Args:
    images (torch.Tensor): input batch of images of shape [B, C, H, W].
    size (Sequence[int]): size of extracted patches. Must be [patch_height, patch_width].
    stride (Sequence[int]): strides, must be [stride_rows, stride_cols]. Default: 1
    dilation (Sequence[int]): as in dilated convolutions, must be [dilation_rows, dilation_cols]. Default: 1
    padding (str): padding algorithm to use. Default: VALID
  Returns:
    Tensor of shape [B, patch_rows, patch_cols, size_rows * size_cols * C]
  """
    if padding != "VALID":
        raise ValueError(f"Only valid padding is supported. Got {padding}")

    if images.ndim != 4:
        raise ValueError(
            f'Rank of images must be 4 (got tensor of shape {images.shape})')

    n, c, h, w = images.shape
    ph, pw = size

    pad = 0
    out_h, out_w = conv_output_shape((h, w), size, stride, pad, dilation)

    patches = F.unfold(images, size, dilation=dilation, padding=0, stride=stride)

    patches = einops.rearrange(patches, "n (c ph pw) (out_h out_w) -> n out_h out_w (ph pw c)",
                               c=c, ph=ph, pw=pw, out_h=out_h, out_w=out_w)
    return patches


def patches_for_flow(inputs: torch.Tensor) -> torch.Tensor:
    """Extract 3x3x2 image patches for flow inputs.
    Args:
        inputs (torch.Tensor): image inputs (N, 2, C, H, W) """

    batch_size = inputs.shape[0]

    inputs = einops.rearrange(inputs, "N T C H W -> (N T) C H W")
    padded_inputs = F.pad(inputs, [1, 1, 1, 1], mode='constant')
    outputs = extract_patches(
        padded_inputs,
        size=[3, 3],
        stride=1,
        dilation=1,
        padding='VALID')

    outputs = einops.rearrange(outputs, "(N T) H W C-> N T H W C", N=batch_size)

    return outputs


#  ------------------------------------------------------------
#  -------------------  Up/down-sampling  ---------------------
#  ------------------------------------------------------------


class Conv2DDownsample(nn.Module):
    """Downsamples 4x by applying a 2D convolution and doing max pooling."""

    def __init__(
            self,
            num_layers: int = 1,
            in_channels: int = 3,
            num_channels: int = 64,
            use_batchnorm: bool = True
    ):
        """Constructs a Conv2DDownsample model.
    Args:
      num_layers: The number of conv->max_pool layers.
      num_channels: The number of conv output channels.
      use_batchnorm: Whether to use batchnorm.
      name: Name of the module.
    """
        super().__init__()

        self._num_layers = num_layers
        self.norms = None
        if use_batchnorm:
            self.norms = nn.ModuleList()

        self.convs = nn.ModuleList()
        for _ in range(self._num_layers):
            conv = nn.Conv2d(in_channels=in_channels,
                             out_channels=num_channels,
                             kernel_size=7,
                             stride=2,
                             bias=False)
            trunc_normal_(conv.weight, mean=0.0, std=0.01)
            self.convs.append(conv)
            in_channels = num_channels

            if use_batchnorm:
                batchnorm = nn.BatchNorm2d(num_features=num_channels)
                self.norms.append(batchnorm)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = inputs
        for l, conv in enumerate(self.convs):
            pad = same_padding(out.shape[1:], conv.kernel_size, conv.stride, dims=2)
            out = F.pad(out, pad, mode='constant', value=0.0)
            out = conv(out)

            if self.norms is not None:
                out = self.norms[l](out)

            out = F.relu(out)

            pad = same_padding(out.shape[1:], 3, 2, dims=2)
            out = F.pad(out, pad, mode='constant', value=0.0)

            out = F.max_pool2d(out, kernel_size=3, stride=2)

        return out

    def set_haiku_params(self, params, state):
        params = {key[key.find('/') + 1:]: params[key] for key in params.keys()}
        state = {key[key.find('/') + 1:]: state[key] for key in state.keys()}

        for l, conv in enumerate(self.convs):
            suffix = "" if l == 0 else f"_{l}"
            name = "conv" + suffix
            init_conv_from_haiku(conv, params.pop(name))
            if self.norms is not None:
                name = "batchnorm" + suffix

                norm_state = {key[key.find('/') + 1:]: state.pop(key) for key in list(state.keys()) if
                              key.startswith(name)}

                norm_state = {key[key.find('/') + 1:]: norm_state[key] for key in norm_state.keys()}

                init_batchnorm_from_haiku(self.norms[l], params.pop(name), norm_state)

        if len(params) != 0:
            warnings.warn(f"Some parameters couldn't be matched to model: {params.keys()}")

        if len(state) != 0:
            warnings.warn(f"Some state variables couldn't be matched to model: {state.keys()}")


# class Conv2DUpsample(nn.Module):
#     """Upsamples 4x using 2 2D transposed convolutions."""
#
#     def __init__(
#             self,
#             n_outputs: int,
#             in_channels: int = 64,
#     ):
#         """Constructs a Conv2DUpsample model.
#     Args:
#       n_outputs: The number of output channels of the module.
#       name: Name of the module.
#     """
#         super().__init__()
#
#         self.transp_conv1 = nn.ConvTranspose2d(in_channels=in_channels,
#                                                out_channels=n_outputs * 2,
#                                                kernel_size=4,
#                                                stride=2,
#                                                padding=0,
#                                                output_padding=0,
#                                                bias=True)
#
#         self.transp_conv1 = hk.Conv2DTranspose(
#             output_channels=n_outputs * 2,
#             kernel_shape=4,
#             stride=2,
#             with_bias=True,
#             padding='SAME',
#             name='transp_conv_1')
#
#         self.transp_conv2 = nn.ConvTranspose2d(in_channels=n_outputs,
#                                                out_channels=n_outputs,
#                                                kernel_size=4,
#                                                stride=2,
#                                                padding=0,
#                                                output_padding=0,
#                                                bias=True)
#
#         self.transp_conv2 = hk.Conv2DTranspose(
#             output_channels=n_outputs,
#             kernel_shape=4,
#             stride=2,
#             with_bias=True,
#             padding='SAME',
#             name='transp_conv_2')
#
#     def forward(self, inputs: torch.Tensor, *,
#                 test_local_stats: bool = False) -> torch.Tensor:  # TODO what is test_local_stats?
#         out = inputs
#         out = self.transp_conv1(out)
#         out = F.relu(out)
#         out = self.transp_conv2(out)
#
#         return out
#
#
# class Conv3DUpsample(nn.Module):
#     """Simple convolutional auto-encoder."""
#
#     def __init__(self,
#                  n_outputs: int,
#                  n_time_upsamples: int = 2,
#                  n_space_upsamples: int = 4):
#
#         super().__init__()
#
#         self._n_outputs = n_outputs
#         self._n_time_upsamples = n_time_upsamples
#         self._n_space_upsamples = n_space_upsamples
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         n_upsamples = max(self._n_time_upsamples, self._n_space_upsamples)
#
#         time_stride = 2
#         space_stride = 2
#
#         for i in range(n_upsamples):
#             if i >= self._n_time_upsamples:
#                 time_stride = 1
#             if i >= self._n_space_upsamples:
#                 space_stride = 1
#
#             channels = self._n_outputs * pow(2, n_upsamples - 1 - i)
#
#             x = hk.Conv3DTranspose(output_channels=channels,
#                                    stride=[time_stride, space_stride, space_stride],
#                                    kernel_shape=[4, 4, 4],
#                                    name=f'conv3d_transpose_{i}')(x)
#             if i != n_upsamples - 1:
#                 x = F.relu(x)
#
#         return x

