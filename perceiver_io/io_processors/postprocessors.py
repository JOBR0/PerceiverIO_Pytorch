import warnings
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import einops

from timm.models.layers import lecun_normal_, trunc_normal_

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from perceiver_io import position_encoding
from perceiver_io.io_processors.processor_utils import ModalitySizeT, reverse_space_to_depth
from perceiver_io.position_encoding import PosEncodingType
from utils.utils import conv_output_shape, init_linear_from_haiku, same_padding, init_conv_from_haiku, \
    init_batchnorm_from_haiku, init_embedding_from_haiku


class EmbeddingPostprocessor(nn.Module):
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

    def forward(
            self, inputs: torch.Tensor, *,
            pos: Optional[torch.Tensor] = None,
            modality_sizes: Optional[ModalitySizeT] = None) -> torch.Tensor:
        batch_size, seq_len, _ = inputs.shape
        output = torch.matmul(
            inputs.reshape([-1, self._d_model]),  # Flatten batch dim
            self._embedding.weight.T)
        output = output + self.bias
        return output.reshape([batch_size, seq_len, self._vocab_size])

    def set_haiku_params(self, params):
        with torch.no_grad():
            self.bias.copy_(torch.from_numpy(params["bias"]).float())


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
            raise NotImplementedError
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
                raise NotImplementedError

                def int_log2(x):
                    return int(np.round(np.log(x) / np.log(2)))

                self.convnet = Conv3DUpsample(
                    n_outputs, int_log2(temporal_upsample), int_log2(spatial_upsample))
            else:
                raise NotImplementedError
                # TODO
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


class FlowPostprocessor(nn.Module):
    """Flow postprocessing for Perceiver."""

    def __init__(
            self,
            img_size: Sequence[int],
            flow_scale_factor: float = 1.0,
    ):
        super().__init__()
        self.flow_scale_factor = flow_scale_factor
        self.img_size = img_size

    def forward(
            self, inputs: torch.Tensor, *,
            pos: Optional[torch.Tensor] = None,
            modality_sizes: Optional[Mapping[str, int]] = None) -> torch.Tensor:
        batch_size = inputs.shape[0]
        inputs = inputs * self.flow_scale_factor

        return inputs.reshape([batch_size, *self.img_size, 2]).permute(0, 3, 1, 2)
