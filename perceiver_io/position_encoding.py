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
"""Position encodings and utilities."""

import abc
import functools
import math
import warnings

import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, lecun_normal_

from utils.utils import init_linear_from_haiku


def generate_fourier_features(
        pos, num_bands, max_resolution=(224, 224),
        concat_pos=True, sine_only=False):
    """Generate a Fourier frequency position encoding with linear spacing.
  Args:
    pos: The position of n points in d dimensional space.
      A torch tensor of shape [n, d].
    num_bands: The number of bands (K) to use.
    max_resolution: The maximum resolution (i.e. the number of pixels per dim).
      A tuple representing resolution for each dimension
    concat_pos: Concatenate the input position encoding to the Fourier features?
    sine_only: Whether to use a single phase (sin) or two (sin/cos) for each
      frequency band.
  Returns:
    embedding: A 1D tensor of shape [n, n_channels]. If concat_pos is True
      and sine_only is False, output dimensions are ordered as:
        [dim_1, dim_2, ..., dim_d,
         sin(pi*f_1*dim_1), ..., sin(pi*f_K*dim_1), ...,
         sin(pi*f_1*dim_d), ..., sin(pi*f_K*dim_d),
         cos(pi*f_1*dim_1), ..., cos(pi*f_K*dim_1), ...,
         cos(pi*f_1*dim_d), ..., cos(pi*f_K*dim_d)],
       where dim_i is pos[:, i] and f_k is the kth frequency band.
  """
    min_freq = 1.0
    # Nyquist frequency at the target resolution:

    freq_bands = torch.stack([
        torch.linspace(min_freq, res / 2, steps=num_bands)
        for res in max_resolution], axis=0)

    # Get frequency bands for each spatial dimension.
    # Output is size [n, d * num_bands]
    per_pos_features = pos[:, :, None] * freq_bands[None, :, :]
    per_pos_features = torch.reshape(per_pos_features,
                                     [-1, np.prod(per_pos_features.shape[1:])])

    if sine_only:
        # Output is size [n, d * num_bands]
        per_pos_features = torch.sin(math.pi * (per_pos_features))
    else:
        # Output is size [n, 2 * d * num_bands]
        per_pos_features = torch.cat(
            [torch.sin(math.pi * per_pos_features),
             torch.cos(math.pi * per_pos_features)], dim=-1)
    # Concatenate the raw input positions.
    if concat_pos:
        # Adds d bands to the encoding.
        per_pos_features = torch.cat([pos, per_pos_features], dim=-1)
    return per_pos_features


def build_linear_positions(index_dims, output_range=(-1.0, 1.0)):
    """Generate an array of position indices for an N-D input array.
  Args:
    index_dims: The shape of the index dimensions of the input array.
    output_range: The min and max values taken by each input index dimension.
  Returns:
    A torch tensor of shape [index_dims[0], index_dims[1], .., index_dims[-1], N].
  """

    def _linspace(n_xels_per_dim):
        return torch.linspace(
            output_range[0], output_range[1],
            steps=n_xels_per_dim,
            dtype=torch.float32)

    dim_ranges = [
        _linspace(n_xels_per_dim) for n_xels_per_dim in index_dims]
    array_index_grid = torch.meshgrid(*dim_ranges, indexing="ij")

    return torch.stack(array_index_grid, axis=-1)


class AbstractPositionEncoding(nn.Module, metaclass=abc.ABCMeta):
    """Abstract Position Encoding."""

    @abc.abstractmethod
    def forward(self, batch_size, pos):
        raise NotImplementedError

    @abc.abstractmethod
    def n_output_channels(self):
        raise NotImplementedError


class TrainablePositionEncoding(AbstractPositionEncoding):
    """Trainable position encoding."""

    def __init__(self, index_dim, num_channels: int = 128, init_scale: float = 0.02):
        super().__init__()
        # size = list(index_dim) + [num_channels]
        self.pos_embs = nn.Parameter(torch.zeros((index_dim, num_channels)))
        trunc_normal_(self.pos_embs, std=init_scale)

        self._output_channels = num_channels

    def forward(self, batch_size, pos=None):
        del pos  # Unused but required from super class
        if batch_size is not None:
            pos_embs = torch.broadcast_to(self.pos_embs[None, :, :], (batch_size,) + self.pos_embs.shape)
        return pos_embs

    def n_output_channels(self):
        return self._output_channels

    def set_haiku_params(self, params):
        with torch.no_grad():
            self.pos_embs.copy_(torch.from_numpy(params['pos_embs']).float())


def _check_or_build_spatial_positions(pos, index_dims, batch_size):
    """Checks or builds spatial position features (x, y, ...).
  Args:
    pos: None, or an array of position features. If None, position features
      are built. Otherwise, their size is checked.
    index_dims: An iterable giving the spatial/index size of the data to be
      featurized.
    batch_size: The batch size of the data to be featurized.
  Returns:
    An array of position features, of shape [batch_size, prod(index_dims)].
  """
    if pos is None:
        pos = build_linear_positions(index_dims)
        pos = torch.broadcast_to(pos[None], (batch_size,) + pos.shape)
        pos = torch.reshape(pos, [batch_size, np.prod(index_dims), -1])
    else:
        # Just a warning label: you probably don't want your spatial features to
        # have a different spatial layout than your pos coordinate system.
        # But feel free to override if you think it'll work!
        assert pos.shape[-1] == len(index_dims)

    return pos


class FourierPositionEncoding(AbstractPositionEncoding):
    """Fourier (Sinusoidal) position encoding."""

    def __init__(self, index_dims, num_bands, concat_pos=True,
                 max_resolution=None, sine_only=False):
        super().__init__()
        self._num_bands = num_bands
        self._concat_pos = concat_pos
        self._sine_only = sine_only
        self._index_dims = index_dims
        # Use the index dims as the maximum resolution if it's not provided.
        self._max_resolution = max_resolution or index_dims

        self._output_channels = num_bands if sine_only else num_bands * 2
        self._output_channels *= len(self._max_resolution)

        if concat_pos:
            self._output_channels += len(self._max_resolution)

    def forward(self, batch_size, pos=None):
        pos = _check_or_build_spatial_positions(pos, self._index_dims, batch_size)
        pos = generate_fourier_features(
            pos[0],
            num_bands=self._num_bands,
            max_resolution=self._max_resolution,
            concat_pos=self._concat_pos,
            sine_only=self._sine_only)
        if batch_size is not None:
            pos = torch.broadcast_to(pos[None, :, :], (batch_size,) + pos.shape)
        return pos

    def n_output_channels(self):
        return self._output_channels


class PositionEncodingProjector(AbstractPositionEncoding):
    """Projects a position encoding to a target size."""

    def __init__(self, input_size, output_size, base_position_encoding):
        super().__init__()
        self._base_position_encoding = base_position_encoding
        self._projector = nn.Linear(input_size, output_size)
        self._output_channels = output_size
        lecun_normal_(self._projector.weight)
        nn.init.constant_(self._projector.bias, 0)

    def forward(self, batch_size, pos=None):
        base_pos = self._base_position_encoding(batch_size, pos)
        projected_pos = self._projector(base_pos)
        return projected_pos

    def n_output_channels(self):
        return self._output_channels

    def set_haiku_params(self, params, base_params = None):
        init_linear_from_haiku(self._projector, params.pop("linear"))
        if base_params is not None:
            self._base_position_encoding.set_haiku_params(base_params)

        if len(params) != 0:
            warnings.warn(f"Some parameters couldn't be matched to model: {params.keys()}")




def build_position_encoding(
        position_encoding_type,
        index_dims,
        project_pos_dim=-1,
        trainable_position_encoding_kwargs=None,
        fourier_position_encoding_kwargs=None):
    """Builds the position encoding."""

    if position_encoding_type == 'trainable':
        assert trainable_position_encoding_kwargs is not None
        output_pos_enc = TrainablePositionEncoding(
            # Construct 1D features:
            index_dim=np.prod(index_dims),
            **trainable_position_encoding_kwargs)
    elif position_encoding_type == 'fourier':
        assert fourier_position_encoding_kwargs is not None
        output_pos_enc = FourierPositionEncoding(
            index_dims=index_dims,
            **fourier_position_encoding_kwargs)
    else:
        raise ValueError(f'Unknown position encoding: {position_encoding_type}.')

    if project_pos_dim > 0:
        # Project the position encoding to a target dimension:
        output_pos_enc = PositionEncodingProjector(
            input_size=output_pos_enc.n_output_channels(),
            output_size=project_pos_dim,
            base_position_encoding=output_pos_enc)

    return output_pos_enc
