import itertools
import pickle
import warnings
from typing import Sequence

import torch.nn as nn
import torch

from perceiver_io.classification_perceiver import ClassificationDecoder
from perceiver_io.perceiver import PerceiverEncoder, Perceiver, AbstractPerceiverDecoder, BasicDecoder, \
    BasicVideoAutoencodingDecoder, MultimodalDecoder
from perceiver_io import io_processors
from timm.models.layers import to_2tuple

import torch.nn.functional as F
from torch.cuda.amp import autocast


class MultiModalPerceiver(nn.Module):
    """
    MultiModalPerceiver: Perceiver for video data.
    """

    def __init__(
            self,
            img_size: Sequence[int] = (56, 56),
            img_channels: int = 3,
            num_frames: int = 16,
            num_classes: int = 700,
            audio_samples_per_frame: int = 48000 // 25,
            num_latent_channels: int = 512):

        super().__init__()

        self.H, self.W = img_size


        SAMPLES_PER_PATCH = 16
        n_audio_samples = num_frames * audio_samples_per_frame

        # subsampled_index_dims = {
        #     'audio': subsampling['audio'].shape[0],
        #     'image': subsampling['image'].shape[0],
        #     'label': 1,
        # }

        # TODO fix this
        image_chunk_size =  1000 #np.prod(images.shape[1:-1]) // nchunks
        audio_chunk_size = 1000 #audio.shape[1] // SAMPLES_PER_PATCH // nchunks
        chunk_idx = 0

        subsampling = {
            'image': torch.arange(
                image_chunk_size * chunk_idx, image_chunk_size * (chunk_idx + 1)),
            'audio': torch.arange(
                audio_chunk_size * chunk_idx, audio_chunk_size * (chunk_idx + 1)),
            'label': None,
        }

        input_preprocessor = io_processors.MultimodalPreprocessor(
            min_padding_size=4,
            modalities={
                'audio': io_processors.AudioPreprocessor(
                    input_channels=1,#TODO
                    samples_per_batch=n_audio_samples,#TODO check
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
                    img_size=(self.H, self.W),
                    input_channels=3,
                    position_encoding_type='fourier',
                    fourier_position_encoding_kwargs=dict(
                        num_bands=32,
                        max_resolution=(num_frames, self.H, self.W),
                        sine_only=False,
                        concat_pos=True,
                    ),
                    n_extra_pos_mlp=0,
                    prep_type='patches',
                    spatial_downsample=4,
                    temporal_downsample=1),
                'label': io_processors.OneHotPreprocessor(
                    input_channels=num_classes,#TODO
                ),
            },
            mask_probs={'image': 0.0, 'audio': 0.0, 'label': 1.0}
        )

        output_postprocessor = io_processors.MultimodalPostprocessor(
            modalities={
                'audio': io_processors.AudioPostprocessor(
                    in_channels=512,
                    samples_per_patch=SAMPLES_PER_PATCH),
                'image': io_processors.ProjectionPostprocessor(
                    num_inputs=512,#todo check
                    num_outputs=3),
                'label': io_processors.ClassificationPostprocessor(
                    num_input_channels=512,#todo check what's the point of this postprocessor combined with classification decoder
                    num_classes=num_classes),
            })


        #encoder_input_channels = input_preprocessor.n_output_channels()

        encoder = PerceiverEncoder(
            num_input_channels=input_preprocessor.n_output_channels(),
            num_self_attends_per_block=8,
            # Weights won't be shared if num_blocks is set to 1.
            num_blocks=1,
            num_latents=28 * 28 * 1,
            num_latent_channels=num_latent_channels,
            num_cross_attend_heads=1,
            num_self_attend_heads=8,
            cross_attend_widening_factor=1,
            self_attend_widening_factor=1,
            dropout_prob=0.0,
            latent_pos_enc_init_scale=0.02,
            cross_attention_shape_for_attn='kv')

        image_decoder = BasicVideoAutoencodingDecoder(
            # Autoencoding, don't pass inputs to the queries.
            concat_preprocessed_input=False,
            subsampled_index_dims=subsampling['image'],
            output_shape=(5, 3, 224, 224), #TODO change#images.shape[:4],
            num_latent_channels=1024,# TODO check why does it differ from encoder
            output_num_channels=512,
            use_query_residual=False,
            position_encoding_type='fourier',
            fourier_position_encoding_kwargs=dict(
                num_bands=32,
                max_resolution=(num_frames, self.H, self.W),
                sine_only=False,
                concat_pos=True,
            ),
        )

        #TODO change
        subsampled_index_dims = 1

        decoder = MultimodalDecoder(
            # Autoencoding, don't pass inputs to the queries.
            concat_preprocessed_input=False,
            subsampled_index_dims=subsampled_index_dims,
            # Modality specific decoders are used ONLY to generate queries.
            # All modalties are decoded together using a unified decoder.
            modalities={
                'audio': BasicDecoder(
                    # Autoencoding, don't pass inputs to the queries.
                    concat_preprocessed_input=False,
                    subsampled_index_dims=subsampling['audio'],
                    output_index_dims=(n_audio_samples // SAMPLES_PER_PATCH,),
                    num_latent_channels=1024,# TODO check why does it differ from encoder
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
                'label': ClassificationDecoder(
                    # Autoencoding, don't pass inputs to the queries.
                    concat_preprocessed_input=False,
                    num_classes=num_classes,
                    num_latent_channels=1024,# TODO check why does it differ from encoder
                    use_query_residual=False,
                    final_project = False,
                    position_encoding_type='trainable',
                    trainable_position_encoding_kwargs=dict(
                        num_channels=1024,
                        init_scale=0.02,
                    ),
                ),
            },
            num_outputs=None,
            output_num_channels=512,
            num_latent_channels=num_latent_channels,
            use_query_residual=False)

        self.perceiver = Perceiver(
            input_preprocessor=input_preprocessor,
            encoder=encoder,
            decoder=decoder,
            output_postprocessor=output_postprocessor)



    def load_haiku_params(self, file):
        with open(file, "rb") as f:
            params = pickle.loads(f.read())
            encoder_params = {key[key.find('/') + 1:]: params.pop(key) for key in list(params.keys()) if
                              key.startswith("encoder")}

            self.perceiver._encoder.set_haiku_params(encoder_params)

            multimodal_decoder_params = {key[key.find('/') + 1:]: params.pop(key) for key in list(params.keys()) if
                              key.startswith("multimodal_decoder")}

            self.perceiver._decoder.set_haiku_params(multimodal_decoder_params)

            preprocessor_params = {key[key.find('/') + 1:]: params.pop(key) for key in list(params.keys()) if
                                   key.startswith("multimodal_preprocessor")}

            self.perceiver._input_preprocessor.set_haiku_params(preprocessor_params)

            classification_decoder_params = {key[key.find('/') + 1:]: params.pop(key) for key in list(params.keys()) if
                                   key.startswith("classification_decoder")}

            self.perceiver._decoder._modalities["label"].set_haiku_params(classification_decoder_params)

            projection_postprocessor_params = {key[key.find('/') + 1:]: params.pop(key) for key in list(params.keys()) if
                                             key.startswith("projection_postprocessor")}

            audio_postprocessor_params = {key[key.find('/') + 1:]: params.pop(key) for key in list(params.keys()) if
                                             key.startswith("audio_postprocessor")}

            classification_postprocessor_params = {key[key.find('/') + 1:]: params.pop(key) for key in list(params.keys()) if
                                             key.startswith("classification_postprocessor")}

            self.perceiver._output_postprocessor._modalities["label"].set_haiku_params(classification_postprocessor_params)
            self.perceiver._output_postprocessor._modalities["audio"].set_haiku_params(audio_postprocessor_params)
            self.perceiver._output_postprocessor._modalities["image"].set_haiku_params(projection_postprocessor_params)

            if len(params) != 0:
                warnings.warn(f"Some parameters couldn't be matched to model: {params.keys()}")



    def forward(self, image1: torch.Tensor, image2: torch.Tensor, test_mode: bool = False, min_overlap: int = 20):
        """"""
        pass


