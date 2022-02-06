import warnings
from typing import Mapping, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import lecun_normal_

from perceiver_io import position_encoding
from perceiver_io.io_processors.processor_utils import PreprocessorT, PreprocessorOutputT
from perceiver_io.transformer_primitives import CrossAttention, SelfAttention, make_cross_attention_mask
from utils.utils import init_linear_from_haiku

ModalitySizeT = Mapping[str, int]


class PerceiverEncoder(nn.Module):
    """The Perceiver Encoder: a scalable, fully attentional encoder.
        The encoder has a total of num_self_attends_per_block * num_blocks self-attend layers. We share weights between blocks.
        Args:
            num_input_channels (int): Number of channels in inputs.
            num_self_attends_per_block (int): Number of self attends per block. Default: 6
            num_blocks (int): Number of blocks. All blocks share weights. Default: 1
            num_latents (int): Number of latent variables. Default: 512
            num_latent_channels (int): Number of channels for latent variables. Default: 1024
            qk_channels (int): Number of channels used for queries and keys. Default:
            v_channels (int): Number of channels for values. Default:
            num_cross_attend_heads (int): Number of attention heads for cross attention. Default: 1
            num_self_attend_heads (int): Number of attention heads for self attention. Default: 8
            cross_attend_widening_factor (int): Widening factor for MLP in cross attention. Default:1
            self_attend_widening_factor (int): Widening factor for MLP in self attention. Default:1
            dropout_prob (float): Dropout probability in self and cross attention. Default: 0.0
            latent_pos_enc_init_scale (float): Default: 0.02
            cross_attention_shape_for_attn (str) Default: "kv"
            use_query_residual (bool): Default: True
    """

    def __init__(
            self,
            num_input_channels: int,
            num_self_attends_per_block: int = 6,
            num_blocks: int = 8,
            num_latents: int = 512,
            num_latent_channels: int = 1024,
            qk_channels: int = None,
            v_channels: int = None,
            num_cross_attend_heads: int = 1,
            num_self_attend_heads: int = 8,
            cross_attend_widening_factor: int = 1,
            self_attend_widening_factor: int = 1,
            dropout_prob: float = 0.0,
            latent_pos_enc_init_scale: float = 0.02,
            cross_attention_shape_for_attn: str = "kv",
            use_query_residual: bool = True):
        super().__init__()

        # Check that we can use multihead-attention with these shapes.
        if num_latent_channels % num_self_attend_heads != 0:
            raise ValueError(f"num_z_channels ({num_latent_channels}) must be divisible by"
                             f" num_self_attend_heads ({num_self_attend_heads}).")
        if num_latent_channels % num_cross_attend_heads != 0:
            raise ValueError(f"num_z_channels ({num_latent_channels}) must be divisible by"
                             f" num_cross_attend_heads ({num_cross_attend_heads}).")

        self._input_is_1d = True

        self._num_blocks = num_blocks

        # Construct the latent array initial state.
        self.latent_pos_enc = position_encoding.TrainablePositionEncoding(
            index_dim=num_latents,
            num_channels=num_latent_channels,
            init_scale=latent_pos_enc_init_scale)

        # Construct the cross attend:
        self.cross_attend = CrossAttention(
            q_in_channels=num_latent_channels,
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
                in_channels=num_latent_channels,
                num_heads=num_self_attend_heads,
                dropout_prob=dropout_prob,
                qk_channels=qk_channels,
                v_channels=v_channels,
                widening_factor=self_attend_widening_factor)
            self.self_attends.append(self_attend)

    def latents(self, inputs):
        # Initialize the latent array for the initial cross-attend.
        return self.latent_pos_enc(batch_size=inputs.shape[0])

    def forward(self, inputs, latents, *, input_mask=None):
        attention_mask = None
        if input_mask is not None:
            attention_mask = make_cross_attention_mask(
                query_mask=torch.ones(latents.shape[:2], dtype=torch.bool, device=inputs.device), kv_mask=input_mask)
        latents = self.cross_attend(latents, inputs, attention_mask=attention_mask)
        for _ in range(self._num_blocks):
            for self_attend in self.self_attends:
                latents = self_attend(latents)
        return latents

    def set_haiku_params(self, params):
        params = {key[key.find("/") + 1:]: params[key] for key in params.keys()}

        cross_attention_params = {key[key.find("/") + 1:]: params.pop(key) for key in list(params.keys()) if
                                  key.startswith("cross_attention")}
        self.cross_attend.set_haiku_params(cross_attention_params)

        for i, self_attend in enumerate(self.self_attends):
            suffix = "" if i == 0 else f"_{i}"
            name = "self_attention" + suffix + "/"
            self_attention_params = {key[key.find("/") + 1:]: params.pop(key) for key in list(params.keys()) if
                                     key.startswith(name)}
            self_attend.set_haiku_params(self_attention_params)

        pos_encoding_params = params.pop("trainable_position_encoding")
        self.latent_pos_enc.set_haiku_params(pos_encoding_params)

        if len(params) != 0:
            warnings.warn(f"Some parameters couldn't be matched to model: {params.keys()}")


class PerceiverDecoder(nn.Module):
    """Cross-attention-based decoder.
    Args:
        final_project_out_channels (int): Number of channels to which output is projected if final_project is True.
        num_latent_channels (int):  Number of channels in latent variables. Default: 1024,
        qk_channels (int):  Default: None,
        v_channels (int): Default: None,
        use_query_residual (bool):  Default: False,
        output_w_init: str = "lecun_normal",
        num_heads (int): Number of heads for attention. Default: 1,
        final_project (bool): Whether to apply final linear layer. Default: True,
    """

    def __init__(self,
                 query_channels: int,
                 final_project_out_channels: int,
                 num_latent_channels: int = 1024,
                 qk_channels: int = None,
                 v_channels: int = None,
                 use_query_residual: bool = False,
                 output_w_init: str = "lecun_normal",
                 num_heads: int = 1,
                 final_project: bool = True,):
        super().__init__()

        self._output_num_channels = final_project_out_channels
        self._output_w_init = output_w_init
        self._use_query_residual = use_query_residual
        self._qk_channels = qk_channels
        self._v_channels = v_channels
        self._final_project = final_project
        self._num_heads = num_heads

        self.query_channels = query_channels

        self.decoding_cross_attn = CrossAttention(
            q_in_channels=query_channels,
            kv_in_channels=num_latent_channels,
            dropout_prob=0.0,
            num_heads=self._num_heads,
            widening_factor=1,
            shape_for_attn="kv",
            qk_channels=self._qk_channels,
            v_channels=self._v_channels,
            use_query_residual=self._use_query_residual)

        if self._final_project:
            self.final_layer = nn.Linear(query_channels, self._output_num_channels)
            if self._output_w_init == "lecun_normal":
                lecun_normal_(self.final_layer.weight)
            elif self._output_w_init == "zeros":
                nn.init.constant_(self.final_layer.weight, 0)
            else:
                raise ValueError(f"{self._output_w_init} not supported as output_w_init")
            nn.init.constant_(self.final_layer.bias, 0)

    # def output_shape(self, inputs):
    #     return ((inputs[0], self._subsampled_index_dims, self._output_num_channels),
    #             None)


    def forward(self, query, latents, *, query_mask=None):
        # Cross-attention decoding.
        # key, value: B x N x K; query: B x M x K
        # Attention maps -> B x N x M
        # Output -> B x M x K
        attention_mask = None
        if query_mask is not None:
            attention_mask = make_cross_attention_mask(
                query_mask=query_mask,
                kv_mask=torch.ones(latents.shape[:2], dtype=torch.bool, device=query.device))

        output = self.decoding_cross_attn(query, latents, attention_mask=attention_mask)
        if self._final_project:
            output = self.final_layer(output)
        return output

    def set_haiku_params(self, params):
        cross_attention_params = {key[key.find("/") + 1:]: params.pop(key) for key in list(params.keys()) if
                                  key.startswith("cross_attention")}

        self.decoding_cross_attn.set_haiku_params(cross_attention_params)

        if self._final_project:
            init_linear_from_haiku(self.final_layer, params.pop("output"))

        if len(params) != 0:
            warnings.warn(f"Some parameters couldn't be matched to model: {params.keys()}")


class Perceiver(nn.Module):
    """The Perceiver: a scalable, fully attentional architecture.
    Args:
        num_blocks (int): Number of times the block is applied with shared weights. Default: 8
        num_self_attends_per_block (int): Number of self-attentions in the block. Default: 6,
        num_latents: (int): Number of latent vectors. Default 512,
        num_latent_channels (int): Number of channels for the latent vectors. Default: 1024,
        final_project (bool): Whether to apply a linear layer to the outputs before the post-processors. Default: True,
        final_project_out_channels (int): Number of output channels for the final projection layer. Default: None,
        perceiver_encoder_kwargs (Dict): Additional arguments for the perceiver encoder {},
        perceiver_decoder_kwargs (Dict): Additional arguments for the perceiver decoder {},
        input_preprocessors (dict / nn.Module): Optional input preprocessors. 1 or none for each modality. Default: None,
        output_postprocessors (dict / nn.Module): Optional output postprocessors. 1 or none for each modality. Default: None,
        output_queries (dict / nn.Module): Modules that create the output queries. 1 for each modality. Default: None,
        output_query_padding_channels (int): Number of learnable features channels that are added to the output queries. Default: 0,
        input_padding_channels (int): Number of learnable features channels that are added to the preprocessed inputs. Default: 0,
        input_channels (dict, int): = The number of input channels need to be specified if NO preprocessor is used. Otherwise,
                                    the number will be inferred from the preprocessor. Default: None,
        input_mask_probs (dict): Probability with which each input modality will be masked out. Default None,
    """

    def __init__(
            self,
            num_blocks: int = 8,
            num_self_attends_per_block: int = 6,
            num_latents: int = 512,
            num_latent_channels: int = 1024,
            final_project: bool = True,
            final_project_out_channels: int = None,
            perceiver_encoder_kwargs: Dict = {},
            perceiver_decoder_kwargs: Dict = {},
            input_preprocessors: Union[dict, nn.Module] = None,
            output_postprocessors: Union[dict, nn.Module] = None,
            output_queries: Union[dict, nn.Module] = None,
            output_query_padding_channels: int = 0,
            input_padding_channels: int = 0,
            input_channels: Union[dict, int] = None,
            input_mask_probs: dict = None,
    ):
        super().__init__()

        if final_project_out_channels is None:
            final_project_out_channels = num_latent_channels

        if type(input_channels) is int:
            input_channels = {"default": input_channels}

        # convert to ModuleDict to register all modules
        if type(input_preprocessors) is dict:
            input_preprocessors = nn.ModuleDict(input_preprocessors)
        elif issubclass(type(input_preprocessors), nn.Module):
            # Single preprocessor
            input_preprocessors = nn.ModuleDict({"default": input_preprocessors})

        self._multi_preprocessor = MultimodalPreprocessor(input_preprocessors=input_preprocessors,
                                                          mask_probs=input_mask_probs,
                                                          min_padding_size=input_padding_channels,
                                                          input_channels=input_channels)

        # convert to ModuleDict to register all modules
        if type(output_postprocessors) is dict:
            output_postprocessors = nn.ModuleDict(output_postprocessors)
        elif issubclass(type(output_postprocessors), nn.Module):
            # Single preprocessor
            output_postprocessors = nn.ModuleDict({"default": output_postprocessors})
        self._output_postprocessors = output_postprocessors

        # convert to ModuleDict to register all modules
        if type(output_queries) is dict:
            output_queries = nn.ModuleDict(output_queries)
        elif issubclass(type(output_queries), nn.Module):
            # Single preprocessor
            output_queries = nn.ModuleDict({"default": output_queries})
        self._output_queries = output_queries

        query_channels = (
                max(m.n_query_channels() for m in self._output_queries.values()) + output_query_padding_channels)
        self.query_channels = query_channels

        self.padding_embeddings = nn.ModuleDict()

        # Use trainable encodings to pad all queries to the same number of channels.
        for modality_name, query in self._output_queries.items():
            pos = position_encoding.TrainablePositionEncoding(
                index_dim=1,
                num_channels=query_channels - query.n_query_channels(),
                init_scale=0.02)
            self.padding_embeddings[modality_name] = pos

        self._encoder = PerceiverEncoder(
            num_input_channels=self._multi_preprocessor.n_output_channels(),
            num_blocks=num_blocks,
            num_self_attends_per_block=num_self_attends_per_block,
            num_latents=num_latents,
            num_latent_channels=num_latent_channels,
            **perceiver_encoder_kwargs)

        self._decoder = PerceiverDecoder(
            query_channels=query_channels,
            final_project=final_project,
            final_project_out_channels=final_project_out_channels,
            num_latent_channels=num_latent_channels,
            **perceiver_decoder_kwargs)

    def forward(self, inputs, *, subsampled_output_points=None,
                pos=None, input_mask=None, query_mask=None):

        if type(inputs) is torch.Tensor:
            # Single input
            inputs = {"default": inputs}

        if self._multi_preprocessor is not None:
            network_input_is_1d = self._encoder._input_is_1d
            inputs, modality_sizes, inputs_without_pos = self._multi_preprocessor(
                inputs, pos=pos,
                network_input_is_1d=network_input_is_1d)
        else:
            modality_sizes = None
            inputs_without_pos = None

        # Get the queries for encoder and decoder cross-attends.
        encoder_query = self._encoder.latents(inputs)
        decoder_query = self.decoder_query(
            inputs, modality_sizes, inputs_without_pos,
            subsampled_points=subsampled_output_points)

        # Run the network forward:
        latents = self._encoder(inputs, encoder_query, input_mask=input_mask)
        # _, output_modality_sizes = self._decoder.output_shape(inputs)
        # output_modality_sizes = output_modality_sizes or modality_sizes

        output_modality_sizes = modality_sizes
        if subsampled_output_points is not None:
            for modality in subsampled_output_points.keys():
                if subsampled_output_points[modality] is not None:
                    output_modality_sizes[modality] = subsampled_output_points[modality].shape[0]

        outputs = self._decoder(decoder_query, latents, query_mask=query_mask)

        if self._output_postprocessors:
            if type(outputs) == torch.Tensor:
                # Slice up modalities by their sizes.
                assert modality_sizes is not None
                outputs = restructure(modality_sizes=output_modality_sizes, inputs=outputs)
            outputs = {modality: postprocessor(
                outputs[modality], pos=None, modality_sizes=None)
                for modality, postprocessor in self._output_postprocessors.items()}

        if type(outputs) is not torch.Tensor and list(outputs.keys()) == ["default"]:
            # return tensor directly if input was given as tensor
            outputs = outputs["default"]

        return outputs

    def decoder_query(self, inputs, modality_sizes, inputs_without_pos=None,
                      subsampled_points=None):
        # Partition the flat inputs among the different modalities
        inputs = restructure(modality_sizes, inputs)
        # Obtain modality-specific decoders' queries
        subsampled_points = subsampled_points or dict()
        decoder_queries = dict()
        for modality, output_query in self._output_queries.items():
            # Get input_without_pos for this modality if it exists.
            input_without_pos = None
            if inputs_without_pos is not None:
                input_without_pos = inputs_without_pos.get(modality, None)
            query = output_query(inputs[modality],
                                 inputs_without_pos=input_without_pos,
                                 subsampled_points=subsampled_points.get(modality, None))

            query = query.reshape([query.shape[0], np.prod(query.shape[1:-1]), query.shape[-1]])

            pos = self.padding_embeddings[modality](query.shape[0])

            pos = torch.broadcast_to(pos, [query.shape[0], query.shape[1], self.query_channels - query.shape[2]])

            query = torch.cat([query, pos], dim=2)

            decoder_queries[modality] = query

        # Apply a predictable ordering to the modalities
        return torch.cat([
            decoder_queries[modality]
            for modality in sorted(decoder_queries.keys())
        ], dim=1)

    def set_haiku_params(self, params):
        for modality in self.padding_embeddings.keys():
            self.padding_embeddings[modality].set_haiku_params(params.pop(f"{modality}_padding"))

        if len(params) != 0:
            warnings.warn(f"Some parameters couldn't be matched to model: {params.keys()}")


def restructure(modality_sizes: ModalitySizeT,
                inputs: torch.Tensor) -> Mapping[str, torch.Tensor]:
    """Partitions a [B, N, C] tensor into tensors for each modality.
  Args:
    modality_sizes: dict specifying the size of the modality
    inputs: input tensor
  Returns:
    dict mapping name of modality to its associated tensor.
  """
    outputs = {}
    index = 0
    # Apply a predictable ordering to the modalities
    for modality in sorted(modality_sizes.keys()):
        size = modality_sizes[modality]
        inp = inputs[:, index:index + size]
        index += size
        outputs[modality] = inp
    return outputs


class MultimodalPreprocessor(nn.Module):
    """Multimodal preprocessing for Perceiver Encoder.
  Inputs for each modality is preprocessed then padded with trainable position
  embeddings to have the same number of channels.
  """

    def __init__(
            self,
            input_preprocessors: Mapping[str, PreprocessorT] = None,
            mask_probs: Optional[Mapping[str, float]] = None,
            min_padding_size: int = 2,
            input_channels: Mapping[str, float] = None):
        """Constructor.
    Args:
      input_preprocessors: dict mapping modality name to preprocessor
      mask_probs: dict mapping modality name to masking probability of that
        modality
      min_padding_size: the minimum padding size for all modalities.
        The final output will have num_channels equal to the maximum channels
        across all modalities plus min_padding_size.
    """
        super().__init__()

        self._preprocessors = input_preprocessors
        self._min_padding_size = min_padding_size
        self._mask_probs = mask_probs

        self._common_channels = None

        if input_preprocessors is not None:
            assert input_channels is None, "input_channels and modalities are mutually exclusive"
            input_channels = {modality: p.n_output_channels() for modality, p in self._preprocessors.items()}
            self._common_channels = (max(input_channels.values()) + self._min_padding_size)
        else:
            assert input_channels is not None, "if no preprocessors, input_channels must be specified"
            self._common_channels = (max(input_channels.values()) + self._min_padding_size)

        if self._mask_probs is not None:
            self.mask_tokens = nn.ModuleDict()
            for modality_name, modality in self._preprocessors.items():
                pos = position_encoding.TrainablePositionEncoding(
                    index_dim=1,
                    num_channels=self._common_channels,
                    init_scale=0.02)
                self.mask_tokens[modality_name] = pos

        self.padding_embeddings = None
        # No padding if all modalities have the same number of channels and no extra padding is required.
        if input_channels is not None:
            if max(input_channels.values()) != min(input_channels.values()) or min_padding_size != 0:
                self.padding_embeddings = nn.ModuleDict()
                for modality_name, modality in self._preprocessors.items():
                    pos = position_encoding.TrainablePositionEncoding(
                        index_dim=1,
                        num_channels=self._common_channels - modality.n_output_channels(),
                        init_scale=0.02)
                    self.padding_embeddings[modality_name] = pos

    def n_output_channels(self):
        return self._common_channels

    def forward(self, inputs: torch.Tensor, *,
                pos: Optional[torch.Tensor] = None,
                network_input_is_1d: bool = True) -> PreprocessorOutputT:
        if self._preprocessors is None:
            outputs = inputs
            inputs_without_pos = inputs
        else:
            outputs = {}
            inputs_without_pos = {}
            for modality, preprocessor in self._preprocessors.items():
                outputs[modality], inputs_without_pos[modality] = preprocessor(
                    inputs[modality], pos=pos,
                    network_input_is_1d=network_input_is_1d)

        if self.padding_embeddings is not None:
            modality_sizes = {}

            padded = {}

            for modality, output in outputs.items():
                pos_enc = self.padding_embeddings[modality](output.shape[0])
                padding = torch.broadcast_to(
                    pos_enc,
                    [output.shape[0], output.shape[1],
                     self._common_channels - output.shape[2]])
                output_padded = torch.cat([output, padding], dim=2)
                padded[modality] = output_padded
                modality_sizes[modality] = output_padded.shape[1]
            outputs = padded
        else:
            modality_sizes = {modality: outputs[modality].shape[1] for modality in outputs.keys()}

        if self._mask_probs is not None:
            masked = {}
            for modality, output in outputs.items():
                # Randomly mask out each token corresponding to this modality
                mask_token = self.mask_tokens[modality](output.shape[0])
                mask_prob = self._mask_probs[modality]

                mask = torch.bernoulli(torch.full([output.shape[0], output.shape[1]], fill_value=mask_prob)).to(
                    mask_token.device)
                mask = torch.unsqueeze(mask, dim=2)
                output_masked = (1 - mask) * outputs[modality] + mask * mask_token
                masked[modality] = output_masked
            outputs = masked

        # Apply a predictable ordering to the modalities
        outputs = [outputs[k] for k in sorted(outputs.keys())]
        return (torch.cat(outputs, dim=1),
                modality_sizes,
                inputs_without_pos)

    def set_haiku_params(self, params):
        for modality in self._preprocessors.keys():
            self.padding_embeddings[modality].set_haiku_params(params.pop(f"{modality}_padding"))
            if self._mask_probs is not None:
                self.mask_tokens[modality].set_haiku_params(params.pop(f"{modality}_mask_token"))

        if len(params) != 0:
            warnings.warn(f"Some parameters couldn't be matched to model: {params.keys()}")
