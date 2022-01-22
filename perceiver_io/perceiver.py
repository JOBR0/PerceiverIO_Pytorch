import abc
import warnings

import torch
import torch.nn as nn
from timm.models.layers import lecun_normal_

from perceiver_io import position_encoding, io_processors
from perceiver_io.transformer_primitives import CrossAttention, SelfAttention, make_cross_attention_mask
from utils.utils import init_linear_from_haiku


class AbstractPerceiverDecoder(nn.Module, metaclass=abc.ABCMeta):
    """Abstract Perceiver decoder."""

    @abc.abstractmethod
    def decoder_query(self, inputs, modality_sizes=None, inputs_without_pos=None,
                      subsampled_points=None):
        raise NotImplementedError

    @abc.abstractmethod
    def output_shape(self, inputs):
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, query, z, *, is_training, query_mask=None):
        raise NotImplementedError


class PerceiverEncoder(nn.Module):
    """The Perceiver Encoder: a scalable, fully attentional encoder.
        The encoder has a total of num_self_attends_per_block * num_blocks self-attend layers. We share weights between blocks.
        Args:
            num_input_channels (int): Number of channels in inputs.
            num_self_attends_per_block (int): Number of self attends per block. Default: 6
            num_blocks (int): Number of blocks. All blocks share weights. Default: 1
            z_index_dim (int): Number of latent variables. Default: 512
            num_z_channels (int): Number of channels for latent variables. Default: 1024
            qk_channels (int): Number of channels used for queries and keys. Default:
            v_channels (int): Number of channels for values. Default:
            num_cross_attend_heads (int): Number of attention heads for cross attention. Default: 1
            num_self_attend_heads (int): Number of attention heads for self attention. Default: 8
            cross_attend_widening_factor (int): Widening factor for MLP in cross attention. Default:1
            self_attend_widening_factor (int): Widening factor for MLP in self attention. Default:1
            dropout_prob (float): Dropout probability in self and cross attention. Default: 0.0
            z_pos_enc_init_scale (float): Default: 0.02
            cross_attention_shape_for_attn (str) Default: 'kv'
            use_query_residual (bool): Default: True
    """

    def __init__(
            self,
            num_input_channels: int,
            num_self_attends_per_block: int = 6,
            num_blocks: int = 8,
            z_index_dim: int = 512,
            num_z_channels: int = 1024,
            qk_channels: int = None,
            v_channels: int = None,
            num_cross_attend_heads: int = 1,
            num_self_attend_heads: int = 8,
            cross_attend_widening_factor: int = 1,
            self_attend_widening_factor: int = 1,
            dropout_prob: float = 0.0,
            z_pos_enc_init_scale: float = 0.02,
            cross_attention_shape_for_attn: str = 'kv',
            use_query_residual: bool = True):
        super().__init__()

        # Check that we can use multihead-attention with these shapes.
        if num_z_channels % num_self_attend_heads != 0:
            raise ValueError(f'num_z_channels ({num_z_channels}) must be divisible by'
                             f' num_self_attend_heads ({num_self_attend_heads}).')
        if num_z_channels % num_cross_attend_heads != 0:
            raise ValueError(f'num_z_channels ({num_z_channels}) must be divisible by'
                             f' num_cross_attend_heads ({num_cross_attend_heads}).')

        self._input_is_1d = True

        self._num_blocks = num_blocks

        # Construct the latent array initial state.
        self.z_pos_enc = position_encoding.TrainablePositionEncoding(
            index_dim=z_index_dim,
            num_channels=num_z_channels,
            init_scale=z_pos_enc_init_scale)

        # Construct the cross attend:
        self.cross_attend = CrossAttention(
            q_in_channels=num_z_channels,
            kv_in_channels=num_input_channels,
            dropout_prob=dropout_prob,
            num_heads=num_cross_attend_heads,
            widening_factor=cross_attend_widening_factor,
            shape_for_attn=cross_attention_shape_for_attn,
            qk_channels=qk_channels,
            v_channels=v_channels,
            use_query_residual=use_query_residual)

        # Construct the block of self-attend layers.
        # We get deeper architectures by applying this block more than once.
        self.self_attends = nn.ModuleList()
        for _ in range(num_self_attends_per_block):
            self_attend = SelfAttention(
                in_channels=num_z_channels,
                num_heads=num_self_attend_heads,
                dropout_prob=dropout_prob,
                qk_channels=qk_channels,
                v_channels=v_channels,
                widening_factor=self_attend_widening_factor)
            self.self_attends.append(self_attend)

    def latents(self, inputs):
        # Initialize the latent array for the initial cross-attend.
        return self.z_pos_enc(batch_size=inputs.shape[0])

    def forward(self, inputs, z, *, input_mask=None):
        attention_mask = None
        if input_mask is not None:
            attention_mask = make_cross_attention_mask(
                query_mask=torch.ones(z.shape[:2], dtype=torch.bool),
                kv_mask=input_mask)
        z = self.cross_attend(z, inputs, attention_mask=attention_mask)
        for _ in range(self._num_blocks):
            for self_attend in self.self_attends:
                z = self_attend(z)
        return z

    def set_haiku_params(self, params):
        params = {key[key.find('/') + 1:]: params[key] for key in params.keys()}

        cross_attention_params = {key[key.find('/') + 1:]: params.pop(key) for key in list(params.keys()) if
                                  key.startswith("cross_attention")}
        self.cross_attend.set_haiku_params(cross_attention_params)

        for i, self_attend in enumerate(self.self_attends):
            suffix = "" if i == 0 else f"_{i}"
            name = "self_attention" + suffix + "/"
            self_attention_params = {key[key.find('/') + 1:]: params.pop(key) for key in list(params.keys()) if
                                     key.startswith(name)}
            self_attend.set_haiku_params(self_attention_params)

        pos_encoding_params = params.pop("trainable_position_encoding")
        self.z_pos_enc.set_haiku_params(pos_encoding_params)

        if len(params) != 0:
            warnings.warn(f"Some parameters couldn't be matched to model: {params.keys()}")


class BasicDecoder(AbstractPerceiverDecoder):
    """Cross-attention-based decoder.
    Args:
        q_in_channels (int): Number of channels in queries.
        output_num_channels (int): Number of channels to which output is projected if final_project is True.
        position_encoding_type (str) Default: 'trainable'.
        # Ignored if position_encoding_type == 'none':
        output_index_dims (int):  Default: None.
        subsampled_index_dims (int): None,
        num_z_channels (int):  Number of channels in latent variables. Default: 1024,
        qk_channels (int):  Default: None,
        v_channels (int): Default: None,
        use_query_residual (bool):  Default: False,
        output_w_init: str = "lecun_normal",
        concat_preprocessed_input: bool = False,
        num_heads (int): Number of heads for attention. Default: 1,
        final_project (bool): Whether to apply final linear layer. Default: True,
        **position_encoding_kwargs
    """

    def __init__(self,
                 q_in_channels: int,
                 output_num_channels: int,
                 position_encoding_type: str = 'trainable',
                 output_index_dims: int = None,
                 subsampled_index_dims: int = None,
                 num_z_channels: int = 1024,
                 qk_channels: int = None,
                 v_channels: int = None,
                 use_query_residual: bool = False,
                 output_w_init: str = "lecun_normal",
                 concat_preprocessed_input: bool = False,
                 num_heads: int = 1,
                 final_project: bool = True,
                 **position_encoding_kwargs):
        super().__init__()
        self._position_encoding_type = position_encoding_type

        # If `none`, the decoder will not construct any position encodings.
        # You should construct your own when quering the decoder.
        self.output_pos_enc = None
        if self._position_encoding_type != 'none':
            self.output_pos_enc = position_encoding.build_position_encoding(
                position_encoding_type,
                index_dims=output_index_dims,
                **position_encoding_kwargs)

        self._output_index_dim = output_index_dims
        if subsampled_index_dims is None:
            subsampled_index_dims = output_index_dims
        self._subsampled_index_dims = subsampled_index_dims
        self._output_num_channels = output_num_channels
        self._output_w_init = output_w_init
        self._use_query_residual = use_query_residual
        self._qk_channels = qk_channels
        self._v_channels = v_channels
        self._final_project = final_project
        self._num_heads = num_heads

        self._concat_preprocessed_input = concat_preprocessed_input

        self.decoding_cross_attn = CrossAttention(
            q_in_channels=q_in_channels,
            kv_in_channels=num_z_channels,
            dropout_prob=0.0,
            num_heads=self._num_heads,
            widening_factor=1,
            shape_for_attn='kv',
            qk_channels=self._qk_channels,
            v_channels=self._v_channels,
            use_query_residual=self._use_query_residual)

        if self._final_project:
            self.final_layer = nn.Linear(q_in_channels, self._output_num_channels)
            if self._output_w_init == "lecun_normal":
                lecun_normal_(self.final_layer.weight)
            elif self._output_w_init == "zeros":
                nn.init.constant_(self.final_layer.weight, 0)
            else:
                raise ValueError(f"{self._output_w_init} not supported as output_w_init")
            nn.init.constant_(self.final_layer.bias, 0)

    def output_shape(self, inputs):
        return ((inputs[0], self._subsampled_index_dims, self._output_num_channels),
                None)

    def decoder_query(self, inputs, modality_sizes=None, inputs_without_pos=None, subsampled_points=None):
        assert self._position_encoding_type != 'none'  # Queries come from elsewhere

        N = inputs.shape[0]

        if subsampled_points is not None:
            raise NotImplementedError("Subsampled points not implemented")
            # # unravel_index returns a tuple (x_idx, y_idx, ...)
            # # stack to get the [n, d] tensor of coordinates
            # pos = torch.stack(jnp.unravel_index(subsampled_points, self._output_index_dim), axis=1)
            # # Map these coordinates to [-1, 1]
            # pos = -1 + 2 * pos / torch.tensor(self._output_index_dim)[None, :]
            # pos = torch.broadcast_to(pos[None],
            #                        [N, pos.shape[0], pos.shape[1]])
            # pos_emb = self.output_pos_enc(
            #     batch_size=inputs.shape[0],
            #     pos=pos)
            # pos_emb = torch.reshape(pos_emb, [N, -1, pos_emb.shape[-1]])
        else:
            pos_emb = self.output_pos_enc(batch_size=N)
        if self._concat_preprocessed_input:
            if inputs_without_pos is None:
                raise ValueError('Value is required for inputs_without_pos if'
                                 ' concat_preprocessed_input is True')
            pos_emb = torch.concatenate([inputs_without_pos, pos_emb], axis=-1)

        return pos_emb

    def forward(self, query, z, *, query_mask=None):
        # Cross-attention decoding.
        # key, value: B x N x K; query: B x M x K
        # Attention maps -> B x N x M
        # Output -> B x M x K
        attention_mask = None
        if query_mask is not None:
            attention_mask = make_cross_attention_mask(
                query_mask=query_mask,
                kv_mask=torch.ones(z.shape[:2], dtype=torch.bool))

        output = self.decoding_cross_attn(query, z, attention_mask=attention_mask)
        if self._final_project:
            output = self.final_layer(output)
        return output

    def set_haiku_params(self, params):
        cross_attention_params = {key[key.find('/') + 1:]: params.pop(key) for key in list(params.keys()) if
                                  key.startswith("cross_attention")}

        self.decoding_cross_attn.set_haiku_params(cross_attention_params)

        if self._final_project:
            init_linear_from_haiku(self.final_layer, params.pop("output"))


        if self._position_encoding_type == 'trainable':
            params = {key[key.find('/') + 1:]: params[key] for key in params.keys()}

            self.output_pos_enc.set_haiku_params(params.pop("trainable_position_encoding"))

        if len(params) != 0:
            warnings.warn(f"Some parameters couldn't be matched to model: {params.keys()}")


class Perceiver(nn.Module):
    """The Perceiver: a scalable, fully attentional architecture.
    Args:
        encoder (PerceiverEncoder): The encoder to use.
        decoder (AbstractPerceiverDecoder): The decoder to use.
        input_preprocessor: The input preprocessor to use. Default: None.
        output_postrocessor: The output preprocessor to use. Default: None.
    """

    def __init__(
            self,
            encoder: PerceiverEncoder,
            decoder: AbstractPerceiverDecoder,
            input_preprocessor=None,
            output_postprocessor=None):
        super().__init__()

        # Feature and task parameters:
        self._input_preprocessor = input_preprocessor
        self._output_postprocessor = output_postprocessor
        self._decoder = decoder
        self._encoder = encoder

    def forward(self, inputs, *, subsampled_output_points=None,
                pos=None, input_mask=None, query_mask=None):
        if self._input_preprocessor:
            network_input_is_1d = self._encoder._input_is_1d
            inputs, modality_sizes, inputs_without_pos = self._input_preprocessor(
                inputs, pos=pos,
                network_input_is_1d=network_input_is_1d)
        else:
            modality_sizes = None
            inputs_without_pos = None

        # Get the queries for encoder and decoder cross-attends.
        encoder_query = self._encoder.latents(inputs)
        decoder_query = self._decoder.decoder_query(
            inputs, modality_sizes, inputs_without_pos,
            subsampled_points=subsampled_output_points)

        # Run the network forward:
        z = self._encoder(inputs, encoder_query, input_mask=input_mask)
        _, output_modality_sizes = self._decoder.output_shape(
            inputs)
        output_modality_sizes = output_modality_sizes or modality_sizes

        outputs = self._decoder(decoder_query, z, query_mask=query_mask)

        if self._output_postprocessor:
            outputs = self._output_postprocessor(outputs,
                                                 modality_sizes=output_modality_sizes)

        return outputs


class BasicVideoAutoencodingDecoder(AbstractPerceiverDecoder):
  """Cross-attention based video-autoencoding decoder.

  Light-weight wrapper of `BasicDecoder` with video reshaping logic.
  """

  def __init__(self,
               output_shape,
               position_encoding_type,
               **decoder_kwargs):
    super().__init__()
    if len(output_shape) != 4:  # B, T, H, W
      raise ValueError(f'Expected rank 4 output_shape, got {output_shape}.')
    # Build the decoder components:
    self._output_shape = output_shape
    self._output_num_channels = decoder_kwargs['output_num_channels']

    self.decoder = BasicDecoder(
        output_index_dims=self._output_shape[1:4],  # T*H*W
        position_encoding_type=position_encoding_type,
        **decoder_kwargs)

  def decoder_query(self, inputs, modality_sizes=None,
                    inputs_without_pos=None, subsampled_points=None):
    return self.decoder.decoder_query(inputs,
                                      modality_sizes=modality_sizes,
                                      inputs_without_pos=inputs_without_pos,
                                      subsampled_points=subsampled_points)

  def output_shape(self, inputs):
    return ([inputs.shape[0]] + self._output_shape[1:] +
            [self._output_num_channels], None)

  def forward(self, query, z, *, is_training, query_mask=None):
    output = self.decoder(query, z, is_training=is_training)

    output = torch.reshape(output, self._output_shape + [output.shape[-1]])
    return output


class MultimodalDecoder(AbstractPerceiverDecoder):
  """Multimodal decoding by composing uni-modal decoders.

  The modalities argument of the constructor is a dictionary mapping modality
  name to the decoder of that modality. That decoder will be used to construct
  queries for that modality. However, there is a shared cross attention across
  all modalities, using the concatenated per-modality query vectors.
  """

  def __init__(self, modalities, num_outputs, output_num_channels,
               min_padding_size=2,
               subsampled_index_dims=None,
               **decoder_kwargs):
    super().__init__()
    self._modalities = modalities
    self._subsampled_index_dims = subsampled_index_dims
    self._min_padding_size = min_padding_size
    self._output_num_channels = output_num_channels
    self._num_outputs = num_outputs
    self._decoder = BasicDecoder(
        output_index_dims=(num_outputs,),
        output_num_channels=output_num_channels,
        position_encoding_type='none',
        **decoder_kwargs)

  def decoder_query(self, inputs, modality_sizes, inputs_without_pos=None,
                    subsampled_points=None):
    # Partition the flat inputs among the different modalities
    inputs = io_processors.restructure(modality_sizes, inputs)
    # Obtain modality-specific decoders' queries
    subsampled_points = subsampled_points or dict()
    decoder_queries = dict()
    for modality, decoder in self._modalities.items():
      # Get input_without_pos for this modality if it exists.
      input_without_pos = None
      if inputs_without_pos is not None:
        input_without_pos = inputs_without_pos.get(modality, None)
      decoder_queries[modality] = decoder.decoder_query(
          inputs=inputs[modality],
          modality_sizes=None,
          inputs_without_pos=input_without_pos,
          subsampled_points=subsampled_points.get(modality, None)
      )

    # Pad all queries with trainable position encodings to make them
    # have the same channels
    num_channels = (max(query.shape[2] for query in decoder_queries.values())
                    + self._min_padding_size)

    def embed(modality, x):
      x = jnp.reshape(x, [x.shape[0], np.prod(x.shape[1:-1]), x.shape[-1]])
      pos = position_encoding.TrainablePositionEncoding(
          1, num_channels=num_channels - x.shape[2],
          init_scale=0.02, name=f'{modality}_padding')(x.shape[0])
      pos = jnp.broadcast_to(
          pos, [x.shape[0], x.shape[1], num_channels - x.shape[2]])
      return jnp.concatenate([x, pos], axis=2)

    # Apply a predictable ordering to the modalities
    return jnp.concatenate([
        embed(modality, decoder_queries[modality])
        for modality in sorted(self._modalities.keys())
    ], axis=1)

  def output_shape(self, inputs):
    if self._subsampled_index_dims is not None:
      subsampled_index_dims = sum(self._subsampled_index_dims.values())
    else:
      subsampled_index_dims = self._num_outputs
    return ((inputs.shape[0], subsampled_index_dims, self._output_num_channels),
            self._subsampled_index_dims)

  def forward(self, query, z, *, is_training, query_mask=None):
    # B x 1 x num_classes -> B x num_classes
    return self._decoder(query, z)
