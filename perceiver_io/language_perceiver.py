import pickle
import warnings


import torch.nn as nn
import torch


from perceiver_io.io_processors import EmbeddingDecoder
from perceiver_io.perceiver import PerceiverEncoder, Perceiver


from perceiver_io.perceiver import AbstractPerceiverDecoder, BasicDecoder
from perceiver_io.position_encoding import TrainablePositionEncoding
from utils.utils import init_embedding_from_haiku


class LanguagePerceiver(nn.Module):
    """
    LanguagePerceiver: Perceiver for masked language modeling.
    """

    def __init__(self,
                 vocab_size: int = 262,
                 max_seq_len: int = 2048,
                 latent_channels = 1280,
                 embed_dim: int = 768):
        super().__init__()

        encoder = PerceiverEncoder(
            num_input_channels=embed_dim,  # TODO change
            num_self_attends_per_block=26,
            num_blocks=1,
            num_latents=256,
            num_latent_channels=latent_channels,
            num_self_attend_heads=8,
            num_cross_attend_heads=8,
            qk_channels=8 * 32,
            v_channels=latent_channels,
            use_query_residual=True,
            cross_attend_widening_factor=1,
            self_attend_widening_factor=1)

        decoder = BasicDecoder(
            query_channels=embed_dim,  # TODO change
            output_num_channels=latent_channels,
            position_encoding_type='trainable',
            output_index_dims=max_seq_len,
            num_latent_channels=latent_channels,
            qk_channels=8 * 32,
            v_channels=embed_dim,
            num_heads=8,
            final_project=False,
            use_query_residual=False,
            trainable_position_encoding_kwargs=dict(num_channels=embed_dim)
        )

        self.perceiver = Perceiver(
            input_preprocessors=None,
            encoder=encoder,
            decoder=decoder,
            output_postprocessor=None)

        self.input_pos_encoding = TrainablePositionEncoding(
            index_dim=max_seq_len, num_channels=embed_dim)

        self.embed = nn.Embedding(num_embeddings=vocab_size,
                                  embedding_dim=embed_dim)

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
