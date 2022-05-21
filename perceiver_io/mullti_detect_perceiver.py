from enum import Enum
from typing import Sequence, List

import torch.nn as nn
import torch

from perceiver_io.io_processors.postprocessors import ClassificationPostprocessor
from perceiver_io.io_processors.preprocessors import ImagePreprocessor
from perceiver_io.output_queries import TrainableQuery
from perceiver_io.perceiver import PerceiverIO

from perceiver_io.position_encoding import PosEncodingType


class PrepType(Enum):
    FOURIER_POS_CONVNET = 1
    LEARNED_POS_1X1CONV = 2
    FOURIER_POS_PIXEL = 3


class MultiDetectPerceiver(nn.Module):
    """
    ClassificationPerceiver: Perceiver for image classification
    Args:
        num_classes (int): Number of classes. Default: 1000.
        img_size (Sequence[int]): Size of images [H, W]. Default: (224, 224).
        img_channels (int): Number of image channels. Default: 3.
        prep_type (str): Preprocessing type. Default: PrepType.FOURIER_POS_CONVNET.
        num_self_attends_per_block (int): Number of self attends per block. Default: 6
        num_blocks (int): Number of blocks. All blocks share weights. Default: 8
        num_latents (int): Number of latent variables. Default: 512
        num_latent_channels (int): Number of channels for latent variables. Default: 1024
    """

    def __init__(self,
                 num_classes: int = 1000,
                 img_size: Sequence[int] = (224, 224),
                 img_channels: int = 3,
                 num_self_attends_per_block: int = 6,
                 num_blocks: int = 8,
                 num_latents: int = 512,
                 num_latent_channels: int = 1024,
                 ):
        super().__init__()

        # TODO Change pos_embed dims
        # TODO maybe make them equal to 244x224
        # TODO pad inputs

        img_preprocessor_big_h = ImagePreprocessor(
            img_size=(164, 224),
            input_channels=img_channels,
            position_encoding_type=PosEncodingType.FOURIER,
            fourier_position_encoding_kwargs=dict(
                concat_pos=True,
                max_resolution=(41, 56),
                num_bands=64,
                sine_only=False
            ),
            prep_type="conv")

        img_preprocessor_small_h = ImagePreprocessor(
            img_size=(104, 224),
            input_channels=img_channels,
            shared_conv_net=img_preprocessor_big_h.convnet,
            position_encoding_type=PosEncodingType.FOURIER,
            fourier_position_encoding_kwargs=dict(
                concat_pos=True,
                max_resolution=(56, 56),
                num_bands=64,
                sine_only=False
            ),
            prep_type="conv")

        input_preprocessors = {
            "img_0": img_preprocessor_big_h,
            "img_1": img_preprocessor_big_h,
            "img_2": img_preprocessor_small_h,
            "img_3": img_preprocessor_big_h,
            "img_4": img_preprocessor_small_h,
        }

        perceiver_encoder_kwargs = dict(
            num_self_attend_heads=8,
            use_query_residual=True, )

        decoder_query_residual = True

        perceiver_decoder_kwargs = dict(
            use_query_residual=decoder_query_residual,
        )

        output_query = TrainableQuery(
            output_index_dims=num_classes,
            num_channels=1024,
            init_scale=0.02,
        )

        output_postprocessor = ClassificationPostprocessor(
            num_classes=num_classes,
            num_input_channels=num_classes,
            project=False
        )

        self.perceiver = PerceiverIO(
            num_blocks=num_blocks,
            num_self_attends_per_block=num_self_attends_per_block,
            num_latents=num_latents,
            num_latent_channels=num_latent_channels,
            input_preprocessors=input_preprocessors,
            perceiver_encoder_kwargs=perceiver_encoder_kwargs,
            output_queries=output_query,
            perceiver_decoder_kwargs=perceiver_decoder_kwargs,
            final_project_out_channels=num_classes,
            output_postprocessors=output_postprocessor)

    def forward(self, img_list: List[torch.Tensor]):
        """
        :param img: (batch_size, 3, H, W)
        """
        input_dict = {f"img_{i}": img for i, img in enumerate(img_list)}
        return self.perceiver(input_dict)
