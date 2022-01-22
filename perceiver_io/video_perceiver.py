import itertools
import pickle
import warnings
from typing import Sequence

import torch.nn as nn
import torch

from perceiver_io.perceiver import PerceiverEncoder, Perceiver, AbstractPerceiverDecoder, BasicDecoder, \
    BasicVideoAutoencodingDecoder
from perceiver_io import io_processors
from timm.models.layers import to_2tuple

import torch.nn.functional as F
from torch.cuda.amp import autocast


class VideoPerceiver(nn.Module):
    """
    VideoPerceiver: Perceiver for video data.
    """

    def __init__(
            self):
        super().__init__()


        NUM_FRAMES = 16
        AUDIO_SAMPLES_PER_FRAME = 48000 // 25
        SAMPLES_PER_PATCH = 16
        NUM_CLASSES = 700
        IMG_SZ = 56
        n_audio_samples = NUM_FRAMES * AUDIO_SAMPLES_PER_FRAME

        # subsampled_index_dims = {
        #     'audio': subsampling['audio'].shape[0],
        #     'image': subsampling['image'].shape[0],
        #     'label': 1,
        # }

        input_preprocessor = io_processors.MultimodalPreprocessor(
            min_padding_size=4,
            modalities={
                'audio': io_processors.AudioPreprocessor(
                    position_encoding_type='fourier',
                    fourier_position_encoding_kwargs=dict(
                        num_bands=192,
                        max_resolution=(n_audio_samples,),
                        sine_only=False,
                        concat_pos=True,
                    ),
                    n_extra_pos_mlp=0,
                    prep_type='patches',
                    samples_per_patch=16),
                'image': io_processors.ImagePreprocessor(
                    input_shape=(IMG_SZ, IMG_SZ, 3), # TODO check
                    position_encoding_type='fourier',
                    fourier_position_encoding_kwargs=dict(
                        num_bands=32,
                        max_resolution=(NUM_FRAMES, IMG_SZ, IMG_SZ),
                        sine_only=False,
                        concat_pos=True,
                    ),
                    n_extra_pos_mlp=0,
                    prep_type='patches',
                    spatial_downsample=4,
                    temporal_downsample=1),
                'label': io_processors.OneHotPreprocessor(),
            },
            mask_probs={'image': 0.0, 'audio': 0.0, 'label': 1.0}
        )

        output_postprocessor = io_processors.MultimodalPostprocessor(
            modalities={
                'audio': io_processors.AudioPostprocessor(
                    samples_per_patch=SAMPLES_PER_PATCH),
                'image': io_processors.ProjectionPostprocessor(
                    num_inputs=3,#todo check
                    num_outputs=3),
                'label': io_processors.ClassificationPostprocessor(
                    num_classes=NUM_CLASSES),
            })


        #encoder_input_channels = input_preprocessor.n_output_channels()
        num_z_channels = 512

        encoder = PerceiverEncoder(
            num_input_channels=3, #TODO change
            num_self_attends_per_block=8,
            # Weights won't be shared if num_blocks is set to 1.
            num_blocks=1,
            z_index_dim=28 * 28 * 1,
            num_z_channels=512,
            num_cross_attend_heads=1,
            num_self_attend_heads=8,
            cross_attend_widening_factor=1,
            self_attend_widening_factor=1,
            dropout_prob=0.0,
            z_pos_enc_init_scale=0.02,
            cross_attention_shape_for_attn='kv')

        image_decoder = BasicVideoAutoencodingDecoder(
            # Autoencoding, don't pass inputs to the queries.
            concat_preprocessed_input=False,
            subsampled_index_dims=subsampling['image'],
            output_shape=images.shape[:4],
            num_z_channels=1024,
            output_num_channels=512,
            use_query_residual=False,
            position_encoding_type='fourier',
            fourier_position_encoding_kwargs=dict(
                num_bands=32,
                max_resolution=(NUM_FRAMES, IMG_SZ, IMG_SZ),
                sine_only=False,
                concat_pos=True,
            ),
        )

        decoder = MultimodalDecoder(
            # Autoencoding, don't pass inputs to the queries.
            concat_preprocessed_input=False,
            subsampled_index_dims=subsampled_index_dims,
            # Modality specific decoders are used ONLY to generate queries.
            # All modalties are decoded together using a unified decoder.
            modalities={
                'audio': perceiver.BasicDecoder(
                    # Autoencoding, don't pass inputs to the queries.
                    concat_preprocessed_input=False,
                    subsampled_index_dims=subsampling['audio'],
                    output_index_dims=(n_audio_samples // SAMPLES_PER_PATCH,),
                    num_z_channels=1024,
                    output_num_channels=512,
                    use_query_residual=False,
                    position_encoding_type='fourier',
                    fourier_position_encoding_kwargs=dict(
                        num_bands=192,
                        max_resolution=(n_audio_samples,),
                        sine_only=False,
                        concat_pos=True,
                    ),
                ),
                'image': image_decoder,
                'label': perceiver.ClassificationDecoder(
                    # Autoencoding, don't pass inputs to the queries.
                    concat_preprocessed_input=False,
                    num_classes=NUM_CLASSES,
                    num_z_channels=1024,
                    use_query_residual=False,
                    position_encoding_type='trainable',
                    trainable_position_encoding_kwargs=dict(
                        num_channels=1024,
                        init_scale=0.02,
                    ),
                ),
            },
            num_outputs=None,
            output_num_channels=512,
            use_query_residual=False, )

        self.perceiver = Perceiver(
            input_preprocessor=input_preprocessor,
            encoder=encoder,
            decoder=decoder,
            output_postprocessor=output_postprocessor)



    def load_haiku_params(self, file):
        with open(file, "rb") as f:
            params = pickle.loads(f.read())
            encoder_params = {key[key.find('/') + 1:]: params.pop(key) for key in list(params.keys()) if
                              key.startswith("perceiver_encoder")}
            self.perceiver._encoder.set_haiku_params(encoder_params)
            decoder_params = {key[key.find('/') + 1:]: params.pop(key) for key in list(params.keys()) if
                              key.startswith("flow_decoder")}
            self.perceiver._decoder.set_haiku_params(decoder_params)

            preprocessor_params = {key[key.find('/') + 1:]: params.pop(key) for key in list(params.keys()) if
                                   key.startswith("image_preprocessor")}
            self.perceiver._input_preprocessor.set_haiku_params(preprocessor_params)

            if len(params) != 0:
                warnings.warn(f"Some parameters couldn't be matched to model: {params.keys()}")



    def forward(self, image1: torch.Tensor, image2: torch.Tensor, test_mode: bool = False, min_overlap: int = 20):
        """"""
        pass


