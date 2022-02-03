import warnings
from enum import Enum
from typing import Sequence

import torch
import torch.nn as nn

from perceiver_io import position_encoding
from perceiver_io.position_encoding import AbstractPositionEncoding, build_position_encoding, PosEncodingType
from utils.utils import unravel_index, init_linear_from_haiku


class BasicQuery(nn.Module):
    """Cross-attention-based decoder.
    Args:
        output_num_channels (int): Number of channels to which output is projected if final_project is True.
        position_encoding_type (str) Default: 'trainable'.
        # Ignored if position_encoding_type == 'none':
        output_index_dims (int):  Default: None.
        subsampled_index_dims (int): None,
        num_latent_channels (int):  Number of channels in latent variables. Default: 1024,
        qk_channels (int):  Default: None,
        v_channels (int): Default: None,
        use_query_residual (bool):  Default: False,
        output_w_init: str = "lecun_normal",
        concat_preprocessed_input: bool = False,
        num_heads (int): Number of heads for attention. Default: 1,
        final_project (bool): Whether to apply final linear layer. Default: True,
        query_channels (int): Number of channels of the query features that are used for the cross-attention.
            If set to None, the channels are set according to teh position encoding. Default: None.
        **position_encoding_kwargs
    """

    def __init__(self,
                 output_index_dims: int = None,
                 concat_preprocessed_input: bool = False,
                 preprocessed_input_channels: int = None,
                 position_encoding_type: PosEncodingType = PosEncodingType.TRAINABLE,
                 **position_encoding_kwargs):
        super().__init__()

        self._output_index_dim = output_index_dims
        self._concat_preprocessed_input = concat_preprocessed_input
        self._position_encoding_type = position_encoding_type

        if position_encoding_type != PosEncodingType.NONE and position_encoding_type is not None:
            self._position_encoding = position_encoding.build_position_encoding(
                position_encoding_type,
                index_dims=output_index_dims,
                **position_encoding_kwargs)
            self._n_query_channels = self._position_encoding.n_output_channels()

        else:
            self._position_encoding = None
            assert concat_preprocessed_input is True, "concat_preprocessed_input must be True if position_encoding_type is None"
            self._n_query_channels = 0

        if concat_preprocessed_input:
            assert preprocessed_input_channels is not None, "preprocessed_input_channels must be set if concat_preprocessed_input is True"
            self._n_query_channels += preprocessed_input_channels

    def n_query_channels(self):
        return self._n_query_channels

    def forward(self, inputs, inputs_without_pos=None, subsampled_points=None):
        N = inputs.shape[0]



        if self._position_encoding is not None:
            if subsampled_points is not None:
                # unravel_index returns a tuple (x_idx, y_idx, ...)
                # stack to get the [n, d] tensor of coordinates
                # pos = torch.stack(torch.unravel_index(subsampled_points, self._output_index_dim), dim=1)

                pos = unravel_index(subsampled_points, self._output_index_dim)
                # Map these coordinates to [-1, 1]
                pos = -1 + 2 * pos / torch.tensor(self._output_index_dim)[None, :]
                pos = torch.broadcast_to(pos[None],
                                         [N, pos.shape[0], pos.shape[1]])
                pos_emb = self._position_encoding(
                    batch_size=N,
                    pos=pos)
                pos_emb = torch.reshape(pos_emb, [N, -1, pos_emb.shape[-1]])
            else:
                pos_emb = self._position_encoding(batch_size=N)
        else:
            pos_emb = None



        if self._concat_preprocessed_input:
            if inputs_without_pos is None:
                raise ValueError('Value is required for inputs_without_pos if'
                                 ' concat_preprocessed_input is True')
            if pos_emb is None:
                pos_emb = inputs
            else:
                pos_emb = torch.cat([inputs_without_pos, pos_emb], dim=-1)

        return pos_emb

    def set_haiku_params(self, params):
        # params = {key[key.find("/~/") + 1:]: params[key] for key in
        #           params.keys()}
        if self._position_encoding_type == PosEncodingType.TRAINABLE:
            params = {key[key.find('/') + 1:]: params[key] for key in params.keys()}

            self._position_encoding.set_haiku_params(params.pop("trainable_position_encoding"))

        if len(params) != 0:
            warnings.warn(f"Some parameters couldn't be matched to model: {params.keys()}")


class TrainableQuery(BasicQuery):
    def __init__(self,
                 output_index_dims: int = None,
                 concat_preprocessed_input: bool = False,
                 preprocessed_input_channels: int = None,
                 num_channels: int = 128,
                 init_scale: float = 0.02

                 ):
        trainable_position_encoding_kwargs = dict(
            num_channels=num_channels,
            init_scale=init_scale
        )
        super().__init__(output_index_dims=output_index_dims,
                         concat_preprocessed_input=concat_preprocessed_input,
                         preprocessed_input_channels=preprocessed_input_channels,
                         position_encoding_type=PosEncodingType.TRAINABLE,
                         trainable_position_encoding_kwargs=trainable_position_encoding_kwargs)


class FourierQuery(BasicQuery):
    def __init__(self,
                 output_index_dims: int = None,
                 concat_preprocessed_input: bool = False,
                 preprocessed_input_channels: int = None,
                 num_bands=64,
                 concat_pos=True,
                 max_resolution=None,
                 sine_only=False
                 ):
        fourier_position_encoding_kwargs = dict(
            num_bands=num_bands,
            max_resolution=max_resolution,
            sine_only=sine_only,
            concat_pos=concat_pos,
        )
        super().__init__(output_index_dims=output_index_dims,
                         concat_preprocessed_input=concat_preprocessed_input,
                         preprocessed_input_channels=preprocessed_input_channels,
                         position_encoding_type=PosEncodingType.FOURIER,
                         fourier_position_encoding_kwargs=fourier_position_encoding_kwargs)


class FlowQuery(BasicQuery):
    """This query just passes the inputs as query."""
    def __init__(self,
                 preprocessed_input_channels: int,
                 output_img_size: Sequence[int],
                 output_num_channels: int = 2,):
        super().__init__(output_index_dims=tuple(output_img_size) + (output_num_channels,),
                         concat_preprocessed_input=True,
                         preprocessed_input_channels=preprocessed_input_channels,
                         position_encoding_type=PosEncodingType.NONE)
