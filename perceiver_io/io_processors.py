# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
from utils.utils import conv_output_shape, init_linear_from_haiku, same_padding, init_conv_from_haiku, \
    init_batchnorm_from_haiku

ModalitySizeT = Mapping[str, int]
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


class Conv2DUpsample(nn.Module):
    """Upsamples 4x using 2 2D transposed convolutions."""

    def __init__(
            self,
            n_outputs: int,
            in_channels: int = 64,
    ):
        """Constructs a Conv2DUpsample model.
    Args:
      n_outputs: The number of output channels of the module.
      name: Name of the module.
    """
        super().__init__()

        self.transp_conv1 = nn.ConvTranspose2d(in_channels=in_channels,
                                               out_channels=n_outputs * 2,
                                               kernel_size=4,
                                               stride=2,
                                               padding=0,
                                               output_padding=0,
                                               bias=True)

        self.transp_conv1 = hk.Conv2DTranspose(
            output_channels=n_outputs * 2,
            kernel_shape=4,
            stride=2,
            with_bias=True,
            padding='SAME',
            name='transp_conv_1')

        self.transp_conv2 = nn.ConvTranspose2d(in_channels=n_outputs,
                                               out_channels=n_outputs,
                                               kernel_size=4,
                                               stride=2,
                                               padding=0,
                                               output_padding=0,
                                               bias=True)

        self.transp_conv2 = hk.Conv2DTranspose(
            output_channels=n_outputs,
            kernel_shape=4,
            stride=2,
            with_bias=True,
            padding='SAME',
            name='transp_conv_2')

    def forward(self, inputs: torch.Tensor, *,
                test_local_stats: bool = False) -> torch.Tensor: #TODO what is test_local_stats?
        out = inputs
        out = self.transp_conv1(out)
        out = F.relu(out)
        out = self.transp_conv2(out)

        return out


class Conv3DUpsample(nn.Module):
    """Simple convolutional auto-encoder."""

    def __init__(self,
                 n_outputs: int,
                 n_time_upsamples: int = 2,
                 n_space_upsamples: int = 4):

        super().__init__()

        self._n_outputs = n_outputs
        self._n_time_upsamples = n_time_upsamples
        self._n_space_upsamples = n_space_upsamples

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_upsamples = max(self._n_time_upsamples, self._n_space_upsamples)

        time_stride = 2
        space_stride = 2

        for i in range(n_upsamples):
            if i >= self._n_time_upsamples:
                time_stride = 1
            if i >= self._n_space_upsamples:
                space_stride = 1

            channels = self._n_outputs * pow(2, n_upsamples - 1 - i)

            x = hk.Conv3DTranspose(output_channels=channels,
                                   stride=[time_stride, space_stride, space_stride],
                                   kernel_shape=[4, 4, 4],
                                   name=f'conv3d_transpose_{i}')(x)
            if i != n_upsamples - 1:
                x = F.relu(x)

        return x


class ImagePreprocessor(nn.Module):
    """Image preprocessing for Perceiver Encoder.
    Args:
        img_size (Sequence[int]): The size of the image to be processed (HxW).
        input_channels (int): The number of channels of the input image. Default: 3.
        prep_type (str): How to process data ('conv' | 'patches' | 'pixels' | 'conv1x1'). Default: 'conv'
        spatial_downsample (int): Factor by which to downsample spatial dimensions. Default: 4
        temporal_downsample (int): Factor by which to downsample temporal dimensiton (e.g. video). Default: 1
        """

    def __init__(
            self,
            img_size: Sequence[int],
            num_frames: int = 1,
            input_channels: int = 3,
            prep_type: str = 'conv',
            spatial_downsample: int = 4,
            temporal_downsample: int = 1,
            position_encoding_type: str = 'fourier',
            n_extra_pos_mlp: int = 0,
            num_channels: int = 64,
            conv_after_patching: bool = False,
            conv2d_use_batchnorm: bool = True,
            concat_or_add_pos: str = 'concat',
            **position_encoding_kwargs):
        super().__init__()

        if prep_type not in ('conv', 'patches', 'pixels', 'conv1x1'):
            raise ValueError('Invalid prep_type!')

        if concat_or_add_pos not in ['concat', 'add']:
            raise ValueError(
                f'Invalid value {concat_or_add_pos} for concat_or_add_pos.')

        self._prep_type = prep_type
        self._spatial_downsample = spatial_downsample
        self._temporal_downsample = temporal_downsample
        self._concat_or_add_pos = concat_or_add_pos
        self._conv_after_patching = conv_after_patching
        self._position_encoding_type = position_encoding_type

        if self._prep_type == 'conv':
            # Downsampling with conv is currently restricted
            convnet_num_layers = math.log(spatial_downsample, 4)
            convnet_num_layers_is_int = (
                    convnet_num_layers == np.round(convnet_num_layers))
            if not convnet_num_layers_is_int or temporal_downsample != 1:
                raise ValueError('Only powers of 4 expected for spatial '
                                 'and 1 expected for temporal '
                                 'downsampling with conv.')

            self.convnet = Conv2DDownsample(
                in_channels=input_channels,
                num_layers=int(convnet_num_layers),
                num_channels=num_channels,
                use_batchnorm=conv2d_use_batchnorm)

        elif self._prep_type == 'conv1x1':
            assert temporal_downsample == 1, 'conv1x1 does not downsample in time.'
            self.convnet_1x1 = nn.Conv2d(
                in_channels=input_channels,
                out_channels=num_channels,
                kernel_size=1,
                # spatial_downsample is unconstrained for 1x1 convolutions.
                stride=[spatial_downsample, spatial_downsample])
            trunc_normal_(self.convnet_1x1.weight, mean=0.0, std=0.01)
            nn.init.constant_(self.convnet_1x1.bias, 0)

        # Dimensions that are indexed by postion encoding
        self.index_dims = [d // spatial_downsample for d in img_size]
        if num_frames > 1:
            self.index_dims = [num_frames // temporal_downsample] + self.index_dims

        self._positional_encoding = position_encoding.build_position_encoding(
            position_encoding_type=position_encoding_type,
            index_dims=self.index_dims,
            **position_encoding_kwargs)

        # Stack MLPs to get a deeper positional embedding.
        self._n_extra_pos_mlp = n_extra_pos_mlp
        if n_extra_pos_mlp != 0:
            raise NotImplementedError

        if self._conv_after_patching:
            self._conv_after_patch_layer = nn.Linear(input_channels * spatial_downsample * temporal_downsample,
                                                     num_channels)
            lecun_normal_(self._conv_after_patch_layer.weight)
            nn.init.constant_(self._conv_after_patch_layer.bias, 0)

        if prep_type == "pixels":
            self.output_channels = input_channels
        elif prep_type == "patches":
            if conv_after_patching:
                self.output_channels = num_channels
            else:
                self.output_channels = input_channels * spatial_downsample ** 2 * temporal_downsample
        else:
            self.output_channels = num_channels

        if concat_or_add_pos == 'concat':
            self.output_channels += self._positional_encoding.n_output_channels()

    def n_output_channels(self):
        return self.output_channels

    def _build_network_inputs(
            self, inputs: torch.Tensor, pos: torch.Tensor,
            network_input_is_1d: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct the final input, including position encoding."""
        batch_size = inputs.shape[0]

        # Reshape input features to a 1D index dimension if necessary.
        if len(inputs.shape) > 3 and network_input_is_1d:
            inputs = torch.reshape(
                inputs, [batch_size, np.prod(self.index_dims), -1])

        # Construct the position encoding.
        pos_enc = self._positional_encoding(batch_size=batch_size, pos=pos)
        pos_enc = pos_enc.to(inputs.device)

        # for i in range(0, self._n_extra_pos_mlp):
        #     pos_enc += hk.Linear(pos_enc.shape[-1])(pos_enc)
        #     if i < (self._n_extra_pos_mlp - 1):
        #         pos_enc = F.relu(pos_enc)

        if not network_input_is_1d:
            # Reshape pos to match the input feature shape
            # if the network takes non-1D inputs
            sh = inputs.shape
            pos_enc = torch.reshape(pos_enc, list(sh)[:-1] + [-1])

        if self._concat_or_add_pos == 'concat':
            inputs_with_pos = torch.cat([inputs, pos_enc], dim=-1)
        elif self._concat_or_add_pos == 'add':
            inputs_with_pos = inputs + pos_enc

        return inputs_with_pos, inputs

    def forward(
            self, inputs: torch.Tensor, *,
            pos=None,
            network_input_is_1d: bool = True) -> PreprocessorOutputT:
        if self._prep_type in ["conv", "conv1x1"]:

            has_temp_dim = len(inputs.shape) == 5

            if has_temp_dim:
                b, t, _, _, _ = inputs.shape
                inputs = inputs.view(b * t, *inputs.shape[2:])

            inputs = inputs.permute(0, 3, 1, 2)
            if self._prep_type == "conv":
                # Convnet image featurization.
                # Downsamples spatially by a factor of 4
                inputs = self.convnet(inputs)
            elif self._prep_type == "conv1x1":
                inputs = self.convnet_1x1(inputs)
            inputs = inputs.permute(0, 2, 3, 1)

            if has_temp_dim:
                inputs = inputs.view(b, t, *inputs.shape[1:])


        elif self._prep_type == 'patches':
            # Space2depth featurization.
            # Video: B x T x H x W x C
            inputs = space_to_depth(
                inputs,
                temporal_block_size=self._temporal_downsample,
                spatial_block_size=self._spatial_downsample)

            if inputs.ndim == 5 and inputs.shape[1] == 1:
                # for flow
                inputs = torch.squeeze(inputs, dim=1)

            if self._conv_after_patching:
                inputs = self._conv_after_patch_layer(inputs)
        elif self._prep_type == 'pixels':
            # if requested, downsamples in the crudest way
            if inputs.ndim == 4:
                inputs = inputs[:,
                         ::self._spatial_downsample, ::self._spatial_downsample]
            elif inputs.ndim == 5:
                inputs = inputs[:, ::self._temporal_downsample,
                         ::self._spatial_downsample, ::self._spatial_downsample]
            else:
                raise ValueError('Unsupported data format for pixels.')

        inputs, inputs_without_pos = self._build_network_inputs(
            inputs, pos, network_input_is_1d)
        modality_sizes = None  # Size for each modality, only needed for multimodal
        return inputs, modality_sizes, inputs_without_pos

    def set_haiku_params(self, params, state=None):
        params = {key[key.find('/') + 1:]: params[key] for key in params.keys()}

        if state is not None:
            state = {key[key.find('/') + 1:]: state[key] for key in state.keys()}

        if self._prep_type == "conv":
            conv2_d_downsample_params = {key[key.find('/') + 1:]: params.pop(key) for key in list(params.keys()) if
                                         key.startswith("conv2_d_downsample")}
            conv2_d_downsample_state = {key[key.find('/') + 1:]: state.pop(key) for key in list(state.keys()) if
                                        key.startswith("conv2_d_downsample")}
            self.convnet.set_haiku_params(conv2_d_downsample_params, conv2_d_downsample_state)
        elif self._prep_type == "conv1x1":
            init_conv_from_haiku(self.convnet_1x1, params.pop("conv2_d"))

        position_encoding_projector = {key[key.find('/') + 1:]: params.pop(key) for key in list(params.keys()) if
                                       key.startswith("position_encoding_projector")}

        if len(position_encoding_projector) > 0:
            if self._position_encoding_type == "trainable":
                self._positional_encoding.set_haiku_params(position_encoding_projector,
                                                           params.pop("trainable_position_encoding"))
            else:
                self._positional_encoding.set_haiku_params(position_encoding_projector)


        elif self._position_encoding_type == "trainable":
            self.pos_enc.set_haiku_params(params.pop("trainable_position_encoding"))

        if self._conv_after_patching:
            init_linear_from_haiku(self._conv_after_patch_layer, params.pop("patches_linear"))

        if len(params) != 0:
            warnings.warn(f"Some parameters couldn't be matched to model: {params.keys()}")


class ImagePostprocessor(nn.Module):
    """Image postprocessing for Perceiver."""

    def __init__(
            self,
            img_size: Sequence[int],
            input_channels: int = 3,
            postproc_type: str = 'pixels',
            spatial_upsample: int = 1,
            temporal_upsample: int = 1,
            n_outputs: int = -1,  # only relevant for 'conv1x1', 'conv', and 'raft'
            input_reshape_size: Optional[Sequence[int]] = None):
        super().__init__()

        if postproc_type not in ('conv', 'patches', 'pixels', 'raft', 'conv1x1'):
            raise ValueError('Invalid postproc_type!')

        # Architecture parameters:
        self._postproc_type = postproc_type

        self._temporal_upsample = temporal_upsample
        self._spatial_upsample = spatial_upsample
        self._input_reshape_size = input_reshape_size

        if self._postproc_type == 'pixels':
            # No postprocessing.
            if self._temporal_upsample != 1 or self._spatial_upsample != 1:
                raise ValueError('Pixels postprocessing should not currently upsample.')
        elif self._postproc_type == 'conv1x1':
            assert self._temporal_upsample == 1, 'conv1x1 does not upsample in time.'
            if n_outputs == -1:
                raise ValueError('Expected value for n_outputs')
            self.conv1x1 = hk.Conv2D(
                n_outputs, kernel_shape=[1, 1],
                # spatial_downsample is unconstrained for 1x1 convolutions.
                stride=[self._spatial_upsample, self._spatial_upsample])
        elif self._postproc_type == 'conv':
            if n_outputs == -1:
                raise ValueError('Expected value for n_outputs')
            if self._temporal_upsample != 1:
                def int_log2(x):
                    return int(np.round(np.log(x) / np.log(2)))

                self.convnet = Conv3DUpsample(
                    n_outputs, int_log2(temporal_upsample), int_log2(spatial_upsample))
            else:
                self.convnet = Conv2DUpsample(n_outputs)

    def forward(
            self, inputs: torch.Tensor, *,
            pos: Optional[torch.Tensor] = None,
            modality_sizes: Optional[ModalitySizeT] = None) -> torch.Tensor:
        if self._input_reshape_size is not None:
            inputs = torch.reshape(
                inputs,
                [inputs.shape[0]] + list(self._input_reshape_size)
                + [inputs.shape[-1]])

        if self._postproc_type == 'conv' or self._postproc_type == 'raft':
            # Convnet image featurization.
            has_temp_dim = len(inputs.shape) == 5

            if len(inputs.shape) == 5 and self._temporal_upsample == 1:
                # Merge batch and time dims.
                b, t, _, _, _ = inputs.shape
                inputs = inputs.view(b * t, *inputs.shape[2:])

            inputs = inputs.permute(0, 3, 1, 2)
            inputs = self.convnet(inputs)
            inputs = inputs.permute(0, 2, 3, 1)

            if len(inputs.shape) == 5 and self._temporal_upsample == 1:
                inputs = inputs.view(b, t, *inputs.shape[1:])

        elif self._postproc_type == 'conv1x1':
            inputs = self.conv1x1(inputs)
        elif self._postproc_type == 'patches':
            inputs = reverse_space_to_depth(
                inputs, self._temporal_upsample, self._spatial_upsample)

        return inputs


class OneHotPreprocessor(nn.Module):
    """One-hot preprocessor for Perceiver Encoder."""

    def __init__(self,
                 input_channels: int, ):
        super().__init__()
        self.input_channels = input_channels

    def n_output_channels(self):
        return self.input_channels

    def forward(self, inputs: torch.Tensor, *,
                pos: Optional[torch.Tensor] = None,
                network_input_is_1d: bool = True) -> PreprocessorOutputT:
        # Add a dummy index dimension.
        inputs = inputs[:, None, :]

        # TODO whats the point of this?

        # No position encodings, so the 1st (input) and 3rd (inputs_without_pos)
        # outputs are identical.
        return inputs, None, inputs


class AudioPreprocessor(nn.Module):
    """Audio preprocessing for Perceiver Encoder."""

    def __init__(
            self,
            samples_per_batch: int,
            prep_type: str = 'patches',
            samples_per_patch: int = 96,
            position_encoding_type: str = 'fourier',
            n_extra_pos_mlp: int = 0,
            concat_or_add_pos: str = 'concat',
            **position_encoding_kwargs):
        super().__init__()

        if prep_type not in ('patches',):
            raise ValueError('Invalid prep_type!')

        if concat_or_add_pos not in ['concat', 'add']:
            raise ValueError(
                f'Invalid value {concat_or_add_pos} for concat_or_add_pos.')

        self._samples_per_patch = samples_per_patch
        self._concat_or_add_pos = concat_or_add_pos

        # TODO check
        self.index_dims = [samples_per_batch // samples_per_patch]

        self._positional_encoding = position_encoding.build_position_encoding(
            index_dims=self.index_dims,
            position_encoding_type=position_encoding_type,
            **position_encoding_kwargs)

        # for deeper positional embeddings
        self._n_extra_pos_mlp = n_extra_pos_mlp

        if self._n_extra_pos_mlp > 0:
            self._extra_pos_mlps = nn.ModuleList()
            for i in range(self._n_extra_pos_mlp):
                linear = nn.Linear(in_features=self._positional_encoding.n_output_channels(),
                                   out_features=self._positional_encoding.n_output_channels())
                lecun_normal_(linear.weight)
                nn.init.constant_(linear.bias, 0)
                self._extra_pos_mlps.append(linear)

        self.output_channels = samples_per_patch

        if concat_or_add_pos == 'concat':
            self.output_channels += self._positional_encoding.n_output_channels()

    def n_output_channels(self):
        return self.output_channels

    def _build_network_inputs(
            self, inputs: torch.Tensor,
            pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct the final input, including position encoding."""
        batch_size = inputs.shape[0]
        index_dims = inputs.shape[1:-1]

        # Construct the position encoding.
        pos_enc = self._positional_encoding(batch_size=batch_size, pos=pos)

        for i in range(0, self._n_extra_pos_mlp):
            # TODO check how this makes it deeper
            pos_enc = pos_enc + self._extra_pos_mlps[i](pos_enc.shape[-1])(pos_enc)
            if i < (self._n_extra_pos_mlp - 1):
                pos_enc = F.relu(pos_enc)

        if self._concat_or_add_pos == 'concat':
            inputs_with_pos = torch.cat([inputs, pos_enc], dim=-1)
        elif self._concat_or_add_pos == 'add':
            inputs_with_pos = inputs + pos_enc

        return inputs_with_pos, inputs

    def forward(self, inputs: torch.Tensor, *,
                pos: Optional[torch.Tensor] = None,
                network_input_is_1d: bool = True) -> PreprocessorOutputT:
        inputs = torch.reshape(inputs, [inputs.shape[0], -1,
                                        self._samples_per_patch])

        inputs, inputs_without_pos = self._build_network_inputs(inputs, pos)
        modality_sizes = None  # Size for each modality, only needed for multimodal
        return inputs, modality_sizes, inputs_without_pos


class AudioPostprocessor(nn.Module):
    """Audio postprocessing for Perceiver."""

    def __init__(
            self,
            postproc_type: str = 'patches',  # 'conv', 'patches', 'pixels',
            in_channels: int = 1024,
            samples_per_patch: int = 96):
        super().__init__()

        if postproc_type not in ('patches',):
            raise ValueError('Invalid postproc_type!')

        # Architecture parameters:
        self._postproc_type = postproc_type

        self.linear = nn.Linear(in_features=in_channels, out_features=samples_per_patch)
        lecun_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, inputs: torch.Tensor, *,
                pos: Optional[torch.Tensor] = None,
                modality_sizes: Optional[ModalitySizeT] = None) -> torch.Tensor:
        out = self.linear(inputs)
        return torch.reshape(out, [inputs.shape[0], -1])

    def set_haiku_params(self, params):
        init_linear_from_haiku(self.linear, params.pop("linear"))
        if len(params) != 0:
            warnings.warn(f"Some parameters couldn't be matched to model: {params.keys()}")


class IdentityPostprocessor(nn.Module):
    """Passes through the inputs unchanged."""

    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor, *,
                pos: Optional[torch.Tensor] = None,
                modality_sizes: Optional[ModalitySizeT] = None) -> torch.Tensor:
        return inputs



class MultimodalPreprocessor(nn.Module):
    """Multimodal preprocessing for Perceiver Encoder.
  Inputs for each modality is preprocessed then padded with trainable position
  embeddings to have the same number of channels.
  """

    def __init__(
            self,
            modalities: Mapping[str, PreprocessorT],
            mask_probs: Optional[Mapping[str, float]] = None,
            min_padding_size: int = 2):
        """Constructor.
    Args:
      modalities: dict mapping modality name to preprocessor
      mask_probs: dict mapping modality name to masking probability of that
        modality
      min_padding_size: the minimum padding size for all modalities.
        The final output will have num_channels equal to the maximum channels
        across all modalities plus min_padding_size.
    """
        super().__init__()
        if type(modalities) is dict:
            modalities = nn.ModuleDict(modalities)

        self._modalities = modalities
        self._min_padding_size = min_padding_size
        self._mask_probs = mask_probs

        self._output_channels = (max(m.n_output_channels() for m in self._modalities.values()) + self._min_padding_size)

        if self._mask_probs is not None:
            self.mask_tokens = nn.ModuleDict()
            for modality_name, modality in self._modalities.items():
                pos = position_encoding.TrainablePositionEncoding(
                    index_dim=1,
                    num_channels=self._output_channels,
                    init_scale=0.02)
                self.mask_tokens[modality_name] = pos

        self.padding_embeddings = nn.ModuleDict()
        for modality_name, modality in self._modalities.items():
            pos = position_encoding.TrainablePositionEncoding(
                index_dim=1,
                num_channels=self._output_channels - modality.n_output_channels(),
                init_scale=0.02)
            self.padding_embeddings[modality_name] = pos

    def n_output_channels(self):
        return self._output_channels

    def forward(self, inputs: torch.Tensor, *,
                pos: Optional[torch.Tensor] = None,
                network_input_is_1d: bool = True) -> PreprocessorOutputT:
        outputs = {}
        inputs_without_pos = {}
        for modality, preprocessor in self._modalities.items():
            outputs[modality], _, inputs_without_pos[modality] = preprocessor(
                inputs[modality], pos=pos,
                network_input_is_1d=network_input_is_1d)

        padded = {}
        modality_sizes = {}
        for modality, output in outputs.items():
            pos_enc = self.padding_embeddings[modality](output.shape[0])
            padding = torch.broadcast_to(
                pos_enc,
                [output.shape[0], output.shape[1],
                 self._output_channels - output.shape[2]])
            output_padded = torch.cat([output, padding], dim=2)

            if self._mask_probs is not None:
                # Randomly mask out each token corresponding to this modality
                mask_token = self.mask_tokens[modality](output.shape[0])
                mask_prob = self._mask_probs[modality]

                mask = torch.bernoulli(torch.full([output.shape[0], output.shape[1]], fill_value=mask_prob))
                mask = torch.unsqueeze(mask, dim=2)
                output_padded = (1 - mask) * output_padded + mask * mask_token

            padded[modality] = output_padded
            modality_sizes[modality] = output_padded.shape[1]

        # Apply a predictable ordering to the modalities
        padded_ls = [padded[k] for k in sorted(padded.keys())]
        return (torch.cat(padded_ls, dim=1),
                modality_sizes,
                inputs_without_pos)

    def set_haiku_params(self, params):
        for modality in self._modalities.keys():
            self.padding_embeddings[modality].set_haiku_params(params.pop(f"{modality}_padding"))
            if self._mask_probs is not None:
                self.mask_tokens[modality].set_haiku_params(params.pop(f"{modality}_mask_token"))

        if len(params) != 0:
            warnings.warn(f"Some parameters couldn't be matched to model: {params.keys()}")

        # for modality, preprocessor in self._modalities.items():
        #     preprocessor.set_haiku_params(params)



class ClassificationPostprocessor(nn.Module):
    """Classification postprocessing for Perceiver."""

    def __init__(
            self,
            num_input_channels: int,
            num_classes: int):
        super().__init__()
        self._num_classes = num_classes
        self.linear = nn.Linear(num_input_channels, num_classes)
        lecun_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, inputs: torch.Tensor, *,
                pos: Optional[torch.Tensor] = None,
                modality_sizes: Optional[ModalitySizeT] = None) -> torch.Tensor:
        logits = self.linear(inputs)
        return logits[:, 0, :]

    def set_haiku_params(self, params):
        init_linear_from_haiku(self.linear, params.pop("linear"))
        if len(params) != 0:
            warnings.warn(f"Some parameters couldn't be matched to model: {params.keys()}")


class ProjectionPostprocessor(nn.Module):
    """Projection postprocessing for Perceiver."""

    def __init__(
            self,
            num_inputs: int,
            num_outputs: int):
        super().__init__()
        self._num_outputs = num_outputs

        self.projection = nn.Linear(num_inputs, num_outputs)
        lecun_normal_(self.projection.weight)
        nn.init.constant_(self.projection.bias, 0)

    def forward(self, inputs: torch.Tensor, *,
                pos: Optional[torch.Tensor] = None,
                modality_sizes: Optional[ModalitySizeT] = None) -> torch.Tensor:
        logits = self.projection(inputs)
        return logits

    def set_haiku_params(self, params):
        init_linear_from_haiku(self.projection, params.pop("linear"))
        if len(params) != 0:
            warnings.warn(f"Some parameters couldn't be matched to model: {params.keys()}")


class EmbeddingDecoder(nn.Module):
    """Pytorech module to decode embeddings."""

    def __init__(self, embedding: nn.Embedding):
        """Constructs the module.
    Args:
      embedding: Embedding module to use.
    """
        super().__init__()
        self._embedding = embedding
        self._vocab_size, self._d_model = embedding.weight.shape
        self.bias = nn.Parameter(torch.zeros(self._vocab_size))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = embeddings.shape
        output = torch.matmul(
            embeddings.reshape([-1, self._d_model]),  # Flatten batch dim
            self._embedding.weight.T)
        output = output + self.bias
        return output.reshape([batch_size, seq_len, self._vocab_size])

    def set_haiku_params(self, params):
        with torch.no_grad():
            self.bias.copy_(torch.from_numpy(params["bias"]).float())
