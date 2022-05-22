from enum import Enum
from typing import Sequence, List

import torch.nn as nn
import torch

from perceiver_io.io_processors.postprocessors import ClassificationPostprocessor, ProjectionPostprocessor
from perceiver_io.io_processors.preprocessors import ImagePreprocessor
from perceiver_io.output_queries import TrainableQuery
from perceiver_io.perceiver import PerceiverIO

from perceiver_io.position_encoding import PosEncodingType



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
                 num_classes: int = 4,
                 max_predictions: int = 200,
                 img_size: Sequence[int] = (224, 224),
                 img_channels: int = 3,
                 num_self_attends_per_block: int = 6,
                 num_blocks: int = 8,
                 num_latents: int = 512,
                 num_latent_channels: int = 1024,
                 ):
        super().__init__()

        # TYPE_VEHICLE = 1; (index 0)
        # TYPE_PEDESTRIAN = 2; (index 1)
        # TYPE_SIGN = 3; (index 2)
        # TYPE_CYCLIST = 4 (index 3)

        # TODO (pad inputs)

        self.num_classes = num_classes

        small_resolution = (104, 224)
        big_resolution = (164, 224)

        img_preprocessor_img0 = ImagePreprocessor(
            img_size=big_resolution,
            input_channels=img_channels,
            position_encoding_type=PosEncodingType.TRAINABLE,
            trainable_position_encoding_kwargs=dict(
                init_scale=0.02,
                num_channels=258,
            ),
            prep_type="conv")

        img_preprocessor_img1 = ImagePreprocessor(
            img_size=big_resolution,
            input_channels=img_channels,
            shared_conv_net=img_preprocessor_img0.convnet,
            position_encoding_type=PosEncodingType.TRAINABLE,
            trainable_position_encoding_kwargs=dict(
                init_scale=0.02,
                num_channels=258,
            ),
            prep_type="conv")

        img_preprocessor_img2 = ImagePreprocessor(
            img_size=small_resolution,
            input_channels=img_channels,
            shared_conv_net=img_preprocessor_img0.convnet,
            position_encoding_type=PosEncodingType.TRAINABLE,
            trainable_position_encoding_kwargs=dict(
                init_scale=0.02,
                num_channels=258,
            ),
            prep_type="conv")

        img_preprocessor_img3 = ImagePreprocessor(
            img_size=big_resolution,
            input_channels=img_channels,
            shared_conv_net=img_preprocessor_img0.convnet,
            position_encoding_type=PosEncodingType.TRAINABLE,
            trainable_position_encoding_kwargs=dict(
                init_scale=0.02,
                num_channels=258,
            ),
            prep_type="conv")

        img_preprocessor_img4 = ImagePreprocessor(
            img_size=small_resolution,
            input_channels=img_channels,
            shared_conv_net=img_preprocessor_img0.convnet,
            position_encoding_type=PosEncodingType.TRAINABLE,
            trainable_position_encoding_kwargs=dict(
                init_scale=0.02,
                num_channels=258,
            ),
            prep_type="conv")

        input_preprocessors = {
            "img_0": img_preprocessor_img0,
            "img_1": img_preprocessor_img1,
            "img_2": img_preprocessor_img2,
            "img_3": img_preprocessor_img3,
            "img_4": img_preprocessor_img4,
        }

        perceiver_encoder_kwargs = dict(
            num_self_attend_heads=8,
            use_query_residual=True, )

        decoder_query_residual = True

        perceiver_decoder_kwargs = dict(
            use_query_residual=decoder_query_residual,
        )

        output_query = TrainableQuery(
            output_index_dims=max_predictions,
            num_channels=1024,
            init_scale=0.02,
        )

        n_box_coordinates = 7 # [center_x, center_y, center_z, length, width, height, heading]
        final_project_out_channels = 1000
        n_predicted_values = 1 + num_classes + n_box_coordinates

        output_postprocessor = ProjectionPostprocessor(
            num_inputs=final_project_out_channels,
            num_outputs=n_predicted_values
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
            final_project_out_channels=final_project_out_channels,
            output_postprocessors=output_postprocessor)

    def forward(self, input_dict):
        """
        :param img: (batch_size, 3, H, W)
        """
        #input_dict = {f"img_{i}": img for i, img in enumerate(img_list)}
        outputs = self.perceiver(input_dict)
        class_predictions = outputs[:, :, :self.num_classes + 1]
        box_predictions = outputs[:, :, self.num_classes + 1:]

        output = {"pred_logits": class_predictions, "pred_boxes": box_predictions}

        return class_predictions, box_predictions
