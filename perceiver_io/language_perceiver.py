import torch.nn as nn
import torch

from perceiver_io.io_processors.postprocessors import EmbeddingPostprocessor
from perceiver_io.io_processors.preprocessors import EmbeddingPreprocessor
from perceiver_io.output_queries import TrainableQuery
from perceiver_io.perceiver import PerceiverIO


class LanguagePerceiver(nn.Module):
    """
    LanguagePerceiver: Perceiver for masked language modeling.
    Args:
        vocab_size (int): size of the vocabulary. Default: 262
        max_seq_len (int): maximum length of the sequence. Default: 2048
        embed_dim (int): output dimensionality of the input embedding. Default: 768
        num_self_attends_per_block (int): Number of self attends per block. Default: 26
        num_blocks (int): Number of blocks. All blocks share weights. Default: 1
        num_latents (int): Number of latent variables. Default: 256
        num_latent_channels (int): Number of channels for latent variables. Default: 1280

    """

    def __init__(self,
                 vocab_size: int = 262,
                 max_seq_len: int = 2048,
                 embed_dim: int = 768,
                 num_self_attends_per_block: int = 26,
                 num_blocks: int = 1,
                 num_latents: int = 256,
                 num_latent_channels: int = 1280,):
        super().__init__()

        perceiver_encoder_kwargs = dict(
            num_self_attend_heads=8,
            num_cross_attend_heads=8,
            qk_channels=8 * 32,
            v_channels=num_latent_channels,
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

        self.perceiver = PerceiverIO(
            final_project=False,
            num_self_attends_per_block=num_self_attends_per_block,
            num_blocks=num_blocks,
            num_latents=num_latents,
            num_latent_channels=num_latent_channels,
            input_preprocessors=input_preprocessor,
            output_postprocessors=output_postprocessor,
            perceiver_encoder_kwargs=perceiver_encoder_kwargs,
            perceiver_decoder_kwargs=perceiver_decoder_kwargs,
            output_queries=output_query)

    def forward(self, inputs: torch.Tensor, input_masks: torch.Tensor):
        logits = self.perceiver(inputs, input_mask=input_masks, query_mask=input_masks)
        return logits
