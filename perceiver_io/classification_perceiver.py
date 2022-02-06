import pickle
import warnings
from enum import Enum
from typing import Sequence

import torch.nn as nn
import torch

from perceiver_io.io_processors.postprocessors import ClassificationPostprocessor
from perceiver_io.io_processors.preprocessors import ImagePreprocessor
from perceiver_io.output_queries import TrainableQuery
from perceiver_io.perceiver import Perceiver

from perceiver_io.position_encoding import PosEncodingType


class PrepType(Enum):
    FOURIER_POS_CONVNET = 1
    LEARNED_POS_1X1CONV = 2
    FOURIER_POS_PIXEL = 3


class ClassificationPerceiver(nn.Module):
    """
    ClassificationPerceiver: Perceiver for image classification
    Args:
        num_classes (int): Number of classes. Default: 1000.
        img_size (Sequence[int]): Size of images [H, W]. Default: (224, 224).
        prep_type (str): Preprocessing type. Default: PrepType.FOURIER_POS_CONVNET.
    """

    def __init__(self,
                 num_classes: int = 1000,
                 img_size: Sequence[int] = (224, 224),
                 img_channels: int = 3,
                 prep_type: PrepType = PrepType.FOURIER_POS_CONVNET):
        super().__init__()

        if prep_type == PrepType.FOURIER_POS_CONVNET:
            input_preprocessor = ImagePreprocessor(
                img_size=img_size,
                input_channels=img_channels,
                position_encoding_type=PosEncodingType.FOURIER,
                fourier_position_encoding_kwargs=dict(
                    concat_pos=True,
                    max_resolution=(56, 56),
                    num_bands=64,
                    sine_only=False
                ),
                prep_type="conv")

        elif prep_type == PrepType.LEARNED_POS_1X1CONV:
            input_preprocessor = ImagePreprocessor(
                img_size=img_size,
                input_channels=img_channels,
                position_encoding_type=PosEncodingType.TRAINABLE,
                trainable_position_encoding_kwargs=dict(
                    init_scale=0.02,
                    num_channels=256,
                ),
                prep_type="conv1x1",
                project_pos_dim=256,
                num_channels=256,
                spatial_downsample=1,
                concat_or_add_pos="concat",
            )

        elif prep_type == PrepType.FOURIER_POS_PIXEL:
            input_preprocessor = ImagePreprocessor(
                img_size=img_size,
                input_channels=img_channels,
                position_encoding_type=PosEncodingType.FOURIER,
                fourier_position_encoding_kwargs=dict(
                    concat_pos=True,
                    max_resolution=(224, 224),
                    num_bands=64,
                    sine_only=False
                ),
                prep_type="pixels",
                spatial_downsample=1,
            )
        else:
            raise ValueError(f"Unknown prep_type type: {prep_type}")

        perceiver_encoder_kwargs = dict(
            num_self_attend_heads=8,
            use_query_residual=True, )

        decoder_query_residual = False if prep_type == PrepType.LEARNED_POS_1X1CONV else True

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

        self.perceiver = Perceiver(
            num_blocks=8,
            num_self_attends_per_block=6,
            num_latents=512,
            num_latent_channels=1024,
            input_preprocessors=input_preprocessor,
            perceiver_encoder_kwargs=perceiver_encoder_kwargs,
            output_queries=output_query,
            perceiver_decoder_kwargs=perceiver_decoder_kwargs,
            final_project_out_channels=num_classes,
            output_postprocessors=output_postprocessor)

    def load_haiku_params(self, file):
        with open(file, "rb") as f:
            dict = pickle.loads(f.read())
            params = dict["params"]
            state = dict["state"]
            # convert to dict
            state = {key: state[key] for key in state}
            encoder_params = {key[key.find("/") + 1:]: params.pop(key) for key in list(params.keys()) if
                              key.startswith("perceiver_encoder")}
            self.perceiver._encoder.set_haiku_params(encoder_params)
            # decoder_params = {key[key.find("/") + 1:]: params.pop(key) for key in list(params.keys()) if
            #                   key.startswith("classification_decoder")}

            decoder_params = {key[key.find("basic_decoder/") + len("basic_decoder/"):]: params.pop(key) for key in
                              list(params.keys()) if
                              key.startswith("classification_decoder")}

            query_params = {key: decoder_params.pop(key) for key in list(decoder_params.keys()) if
                            key.startswith("~")}

            self.perceiver._output_queries["default"].set_haiku_params(query_params)
            self.perceiver._decoder.set_haiku_params(decoder_params)

            preprocessor_params = {key[key.find("/") + 1:]: params.pop(key) for key in list(params.keys()) if
                                   key.startswith("image_preprocessor")}
            preprocessor_state = {key[key.find("/") + 1:]: state.pop(key) for key in list(state.keys()) if
                                  key.startswith("image_preprocessor")}
            self.perceiver._multi_preprocessor._preprocessors["default"].set_haiku_params(preprocessor_params,
                                                                                          preprocessor_state)

            if len(params) != 0:
                warnings.warn(f"Some parameters couldn't be matched to model: {params.keys()}")

    def forward(self, img: torch.Tensor):
        """
        :param img: (batch_size, 3, H, W)
        """
        return self.perceiver(img)
