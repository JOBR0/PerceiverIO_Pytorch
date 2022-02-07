import math
import warnings
from typing import Optional, Sequence, Tuple

from timm.models.layers import lecun_normal_, trunc_normal_

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from perceiver_io import position_encoding
from perceiver_io.io_processors.processor_utils import Conv2DDownsample, space_to_depth
from perceiver_io.position_encoding import PosEncodingType, TrainablePositionEncoding
from utils.utils import init_linear_from_haiku, init_conv_from_haiku, init_embedding_from_haiku

PreprocessorOutputT = Tuple[torch.Tensor, torch.Tensor]


class EmbeddingPreprocessor(nn.Module):
    """Preprocessor for Language Embedding.
        Args:
            vocab_size (int): Size of the vocabulary.
            max_seq_len (int): Maximum sequence length.
            embedding_dims (int): Embedding dimension.
    """

    def __init__(self,
                 vocab_size: int,
                 max_seq_len: int,
                 embedding_dims: int):
        super().__init__()

        self.input_pos_encoding = TrainablePositionEncoding(
            index_dim=max_seq_len,
            num_channels=embedding_dims)

        self.embed = nn.Embedding(num_embeddings=vocab_size,
                                  embedding_dim=embedding_dims)

        self._output_channels = embedding_dims

    def n_output_channels(self):
        return self._output_channels

    def forward(self, inputs: torch.Tensor, *,
                pos: Optional[torch.Tensor] = None) -> PreprocessorOutputT:
        batch_size = inputs.shape[0]

        embedding_inputs = self.embed(inputs)

        input_pos_encoding = self.input_pos_encoding(batch_size)

        embeddings_with_pos_encoding = embedding_inputs + input_pos_encoding

        return embeddings_with_pos_encoding, embedding_inputs

    def set_haiku_params(self, params):
        self.input_pos_encoding.set_haiku_params(params.pop("trainable_position_encoding"))

        init_embedding_from_haiku(self.embed, params.pop("embed"))

        if len(params) != 0:
            warnings.warn(f"Some parameters couldn't be matched to model: {params.keys()}")


class ImagePreprocessor(nn.Module):
    """Image preprocessing for Perceiver Encoder.
    Args:
        img_size (Sequence[int]): The size of the image to be processed (HxW).
        num_frames (int): The number of frames to be processed.
        input_channels (int): The number of channels of the input image. Default: 3.
        prep_type (str): How to process data ("conv" | "patches" | "pixels" | "conv1x1"). Default: "conv"
        spatial_downsample (int): Factor by which to downsample spatial dimensions. Default: 4
        temporal_downsample (int): Factor by which to downsample temporal dimensiton (e.g. video). Default: 1
        position_encoding_type (PosEncodingType): The type of position encoding to use. Default: PosEncodingType.FOURIER
        n_extra_pos_mlp (int): Number of linear layers that are applied to the positional encoding. Default: 0
        num_channels (int): Number of output_channels. Default: 64
        conv_after_patching: (bool) Whether to apply 1x1 conv after patching. Default: False
        conv_2d_use_batchnorm (bool): Whether to use batchnorm for "conv" preprocessing. Default: True
        concat_or_add_pos (str): Whether to concatenate or add the positional encoding. Default: "concat"
        **position_encoding_kwargs: Keyword arguments for position encoding.
        """

    def __init__(
            self,
            img_size: Sequence[int],
            num_frames: int = 1,
            input_channels: int = 3,
            prep_type: str = "conv",
            spatial_downsample: int = 4,
            temporal_downsample: int = 1,
            position_encoding_type: PosEncodingType = PosEncodingType.FOURIER,
            n_extra_pos_mlp: int = 0,
            num_channels: int = 64,
            conv_after_patching: bool = False,
            conv2d_use_batchnorm: bool = True,
            concat_or_add_pos: str = "concat",
            **position_encoding_kwargs):
        super().__init__()

        if prep_type not in ("conv", "patches", "pixels", "conv1x1"):
            raise ValueError("Invalid prep_type!")

        if concat_or_add_pos not in ["concat", "add"]:
            raise ValueError(
                f"Invalid value {concat_or_add_pos} for concat_or_add_pos.")

        self._prep_type = prep_type
        self._spatial_downsample = spatial_downsample
        self._temporal_downsample = temporal_downsample
        self._concat_or_add_pos = concat_or_add_pos
        self._conv_after_patching = conv_after_patching
        self._position_encoding_type = position_encoding_type

        if self._prep_type == "conv":
            # Downsampling with conv is currently restricted
            convnet_num_layers = math.log(spatial_downsample, 4)
            convnet_num_layers_is_int = (
                    convnet_num_layers == np.round(convnet_num_layers))
            if not convnet_num_layers_is_int or temporal_downsample != 1:
                raise ValueError("Only powers of 4 expected for spatial "
                                 "and 1 expected for temporal "
                                 "downsampling with conv.")

            self.convnet = Conv2DDownsample(
                in_channels=input_channels,
                num_layers=int(convnet_num_layers),
                num_channels=num_channels,
                use_batchnorm=conv2d_use_batchnorm)

        elif self._prep_type == "conv1x1":
            assert temporal_downsample == 1, "conv1x1 does not downsample in time."
            self.convnet_1x1 = nn.Conv2d(
                in_channels=input_channels,
                out_channels=num_channels,
                kernel_size=1,
                # spatial_downsample is unconstrained for 1x1 convolutions.
                stride=(spatial_downsample, spatial_downsample))
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
        if self._n_extra_pos_mlp > 0:
            self._extra_pos_mlps = nn.ModuleList()
            for i in range(self._n_extra_pos_mlp):
                linear = nn.Linear(in_features=self._positional_encoding.n_output_channels(),
                                   out_features=self._positional_encoding.n_output_channels())
                lecun_normal_(linear.weight)
                nn.init.constant_(linear.bias, 0)
                self._extra_pos_mlps.append(linear)

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

        if concat_or_add_pos == "concat":
            self.output_channels += self._positional_encoding.n_output_channels()

    def n_output_channels(self):
        return self.output_channels

    def _build_network_inputs(
            self, inputs: torch.Tensor, pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct the final input, including position encoding."""
        batch_size = inputs.shape[0]

        # Reshape input features to a 1D index dimension if necessary.
        if len(inputs.shape) > 3:
            inputs = torch.reshape(
                inputs, [batch_size, np.prod(self.index_dims), -1])

        # Construct the position encoding.
        pos_enc = self._positional_encoding(batch_size=batch_size, pos=pos)
        pos_enc = pos_enc.to(inputs.device)

        for i in range(0, self._n_extra_pos_mlp):
            pos_enc = pos_enc + self._extra_pos_mlps[i](pos_enc.shape[-1])(pos_enc)
            if i < (self._n_extra_pos_mlp - 1):
                pos_enc = F.relu(pos_enc)

        if self._concat_or_add_pos == "concat":
            inputs_with_pos = torch.cat([inputs, pos_enc], dim=-1)
        elif self._concat_or_add_pos == "add":
            inputs_with_pos = inputs + pos_enc

        return inputs_with_pos, inputs

    def forward(
            self, inputs: torch.Tensor, *,
            pos=None) -> PreprocessorOutputT:
        """inputs should have pytorch image shape [.., channel, height, width]"""
        if self._prep_type in ["conv", "conv1x1"]:

            has_temp_dim = len(inputs.shape) == 5

            if has_temp_dim:
                b, t, _, _, _ = inputs.shape
                inputs = inputs.view(b * t, *inputs.shape[2:])

            if self._prep_type == "conv":
                # Convnet image featurization.
                # Downsamples spatially by a factor of 4
                inputs = self.convnet(inputs)
            elif self._prep_type == "conv1x1":
                inputs = self.convnet_1x1(inputs)

            # Move channel dimension to the end
            inputs = inputs.movedim(-3, -1)

            if has_temp_dim:
                inputs = inputs.view(b, t, *inputs.shape[1:])

        elif self._prep_type == "patches":
            # Move channel dimension to the end
            inputs = inputs.movedim(-3, -1)
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
        elif self._prep_type == "pixels":
            # Move channel dimension to the end
            inputs = inputs.movedim(-3, -1)
            # if requested, downsamples in the crudest way
            if inputs.ndim == 4:
                inputs = inputs[:,
                         ::self._spatial_downsample, ::self._spatial_downsample]
            elif inputs.ndim == 5:
                inputs = inputs[:, ::self._temporal_downsample,
                         ::self._spatial_downsample, ::self._spatial_downsample]
            else:
                raise ValueError("Unsupported data format for pixels.")

        inputs, inputs_without_pos = self._build_network_inputs(
            inputs, pos)
        return inputs, inputs_without_pos

    def set_haiku_params(self, params, state=None):
        params = {key[key.find("/") + 1:]: params[key] for key in params.keys()}

        if state is not None:
            state = {key[key.find("/") + 1:]: state[key] for key in state.keys()}

        if self._prep_type == "conv":
            conv2_d_downsample_params = {key[key.find("/") + 1:]: params.pop(key) for key in list(params.keys()) if
                                         key.startswith("conv2_d_downsample")}
            conv2_d_downsample_state = {key[key.find("/") + 1:]: state.pop(key) for key in list(state.keys()) if
                                        key.startswith("conv2_d_downsample")}
            self.convnet.set_haiku_params(conv2_d_downsample_params, conv2_d_downsample_state)
        elif self._prep_type == "conv1x1":
            init_conv_from_haiku(self.convnet_1x1, params.pop("conv2_d"))

        position_encoding_projector = {key[key.find("/") + 1:]: params.pop(key) for key in list(params.keys()) if
                                       key.startswith("position_encoding_projector")}

        if len(position_encoding_projector) > 0:
            if self._position_encoding_type == PosEncodingType.TRAINABLE:
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


class OneHotPreprocessor(nn.Module):
    """One-hot preprocessor for Perceiver Encoder.
        Args:
            input_channels (int): Number of input channels.
    """

    def __init__(self,
                 input_channels: int, ):
        super().__init__()
        self.input_channels = input_channels

    def n_output_channels(self):
        return self.input_channels

    def forward(self, inputs: torch.Tensor, *,
                pos: Optional[torch.Tensor] = None) -> PreprocessorOutputT:
        # Add a dummy index dimension.
        inputs = inputs[:, None, :]

        # No position encodings, so the 1st (input) and 2nd (inputs_without_pos)
        # outputs are identical.
        return inputs, inputs


class AudioPreprocessor(nn.Module):
    """Audio preprocessing for Perceiver Encoder."""

    def __init__(
            self,
            samples_per_batch: int,
            prep_type: str = "patches",
            samples_per_patch: int = 96,
            position_encoding_type: PosEncodingType = PosEncodingType.FOURIER,
            n_extra_pos_mlp: int = 0,
            concat_or_add_pos: str = "concat",
            **position_encoding_kwargs):
        super().__init__()

        if prep_type not in ("patches",):
            raise ValueError("Invalid prep_type!")

        if concat_or_add_pos not in ["concat", "add"]:
            raise ValueError(
                f"Invalid value {concat_or_add_pos} for concat_or_add_pos.")

        self._samples_per_patch = samples_per_patch
        self._concat_or_add_pos = concat_or_add_pos

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

        if concat_or_add_pos == "concat":
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
        pos_enc = self._positional_encoding(batch_size=batch_size, pos=pos).to(inputs.device)

        for i in range(0, self._n_extra_pos_mlp):
            pos_enc = pos_enc + self._extra_pos_mlps[i](pos_enc.shape[-1])(pos_enc)
            if i < (self._n_extra_pos_mlp - 1):
                pos_enc = F.relu(pos_enc)

        if self._concat_or_add_pos == "concat":
            inputs_with_pos = torch.cat([inputs, pos_enc], dim=-1)
        elif self._concat_or_add_pos == "add":
            inputs_with_pos = inputs + pos_enc

        return inputs_with_pos, inputs

    def forward(self, inputs: torch.Tensor, *,
                pos: Optional[torch.Tensor] = None) -> PreprocessorOutputT:
        inputs = torch.reshape(inputs, [inputs.shape[0], -1,
                                        self._samples_per_patch])

        inputs, inputs_without_pos = self._build_network_inputs(inputs, pos)
        return inputs, inputs_without_pos
