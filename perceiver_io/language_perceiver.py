import itertools
import pickle
import warnings
from typing import Sequence

import torch.nn as nn
import torch

from perceiver_io.bytes_tokenizer import BytesTokenizer
from perceiver_io.io_processors import EmbeddingDecoder
from perceiver_io.perceiver import PerceiverEncoder, Perceiver
from perceiver_io import io_processors
from timm.models.layers import to_2tuple

import torch.nn.functional as F
from torch.cuda.amp import autocast

from perceiver_io.perceiver import AbstractPerceiverDecoder, BasicDecoder
from perceiver_io.position_encoding import TrainablePositionEncoding
from perceiver_io.utils import init_embedding_from_haiku


class LanguagePerceiver(nn.Module):
    """
    LanguagePerceiver: Perceiver for masked language modeling.
    """

    def __init__(self,
                 vocab_size: int = 262):
        super().__init__()

        D_MODEL = 768
        D_LATENTS = 1280
        MAX_SEQ_LEN = 2048

        encoder = PerceiverEncoder(
            num_input_channels=D_MODEL,  # TODO change
            num_self_attends_per_block=26,
            num_blocks=1,
            z_index_dim=256,
            num_z_channels=D_LATENTS,
            num_self_attend_heads=8,
            num_cross_attend_heads=8,
            qk_channels=8 * 32,
            v_channels=D_LATENTS,
            use_query_residual=True,
            cross_attend_widening_factor=1,
            self_attend_widening_factor=1)

        decoder = BasicDecoder(
            q_in_channels=D_MODEL,  # TODO change
            output_num_channels=D_LATENTS,
            position_encoding_type='trainable',
            output_index_dims=MAX_SEQ_LEN,
            num_z_channels=D_LATENTS,
            qk_channels=8 * 32,
            v_channels=D_MODEL,
            num_heads=8,
            final_project=False,
            use_query_residual=False,
            trainable_position_encoding_kwargs=dict(num_channels=D_MODEL)
        )

        self.perceiver = Perceiver(
            input_preprocessor=None,
            encoder=encoder,
            decoder=decoder,
            output_postprocessor=None)

        self.input_pos_encoding = TrainablePositionEncoding(
            index_dim=MAX_SEQ_LEN, num_channels=D_MODEL)

        self.embed = nn.Embedding(num_embeddings=vocab_size,
                                  embedding_dim=D_MODEL)

        # TODO check this
        self.emebdding_decoder = EmbeddingDecoder(self.embed)

        # TODO init embedding

    def load_haiku_params(self, file):
        with open(file, "rb") as f:
            params = pickle.loads(f.read())

            # convert to dict
            params = {key: params[key] for key in params}
            encoder_params = {key[key.find('/') + 1:]: params.pop(key) for key in list(params.keys()) if
                              key.startswith("perceiver_encoder")}
            self.perceiver._encoder.set_haiku_params(encoder_params)
            decoder_params = {key[key.find('/') + 1:]: params.pop(key) for key in list(params.keys()) if
                              key.startswith("basic_decoder")}
            self.perceiver._decoder.set_haiku_params(decoder_params)

            self.input_pos_encoding.set_haiku_params(params.pop("trainable_position_encoding"))

            self.emebdding_decoder.set_haiku_params(params.pop("embedding_decoder"))

            init_embedding_from_haiku(self.embed, params.pop("embed"))

            if len(params) != 0:
                warnings.warn(f"Some parameters couldn't be matched to model: {params.keys()}")

    def forward(self, inputs: torch.Tensor, input_masks: torch.Tensor):
        """"""
        embedding_inputs = self.embed(inputs)

        batch_size = embedding_inputs.shape[0]
        input_pos_encoding = self.input_pos_encoding(batch_size)

        embedding_inputs = embedding_inputs + input_pos_encoding

        output_embeddings = self.perceiver(embedding_inputs, input_mask=input_masks, query_mask=input_masks)

        logits = self.emebdding_decoder(output_embeddings)

        return logits