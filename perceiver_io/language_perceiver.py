import pickle
import warnings

import torch.nn as nn
import torch

from perceiver_io.io_processors.postprocessors import EmbeddingPostprocessor
from perceiver_io.io_processors.preprocessors import EmbeddingPreprocessor
from perceiver_io.output_queries import TrainableQuery
from perceiver_io.perceiver import Perceiver


class LanguagePerceiver(nn.Module):
    """
    LanguagePerceiver: Perceiver for masked language modeling.
    """

    def __init__(self,
                 vocab_size: int = 262,
                 max_seq_len: int = 2048,
                 latent_channels=1280,
                 embed_dim: int = 768):
        super().__init__()

        perceiver_encoder_kwargs = dict(
            num_self_attend_heads=8,
            num_cross_attend_heads=8,
            qk_channels=8 * 32,
            v_channels=latent_channels,
            use_query_residual=True, )

        perceiver_decoder_kwargs = dict(
            qk_channels=8 * 32,
            v_channels=embed_dim,
            num_heads=8,
            use_query_residual=False,
        )

        output_query = TrainableQuery(
            output_index_dims=max_seq_len,
            num_channels=embed_dim,
        )

        input_preprocessor = EmbeddingPreprocessor(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            embedding_dims=embed_dim,
        )
        output_postprocessor = EmbeddingPostprocessor(input_preprocessor.embed)

        self.perceiver = Perceiver(
            final_project=False,
            num_self_attends_per_block=26,
            num_blocks=1,
            num_latents=256,
            num_latent_channels=latent_channels,
            input_preprocessors=input_preprocessor,
            output_postprocessors=output_postprocessor,
            perceiver_encoder_kwargs=perceiver_encoder_kwargs,
            perceiver_decoder_kwargs=perceiver_decoder_kwargs,
            output_queries=output_query)

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
            query_params = {key: decoder_params.pop(key) for key in list(decoder_params.keys()) if
                            key.startswith("~")}

            self.perceiver._output_queries["default"].set_haiku_params(query_params)

            self.perceiver._decoder.set_haiku_params(decoder_params)

            self.perceiver._output_postprocessors["default"].set_haiku_params(params.pop("embedding_decoder"))

            self.perceiver._multi_preprocessor._preprocessors["default"].set_haiku_params(params)

            # init_embedding_from_haiku(self.embed, params.pop("embed"))

            if len(params) != 0:
                warnings.warn(f"Some parameters couldn't be matched to model: {params.keys()}")

    def forward(self, inputs: torch.Tensor, input_masks: torch.Tensor):
        logits = self.perceiver(inputs, input_mask=input_masks, query_mask=input_masks)
        return logits
