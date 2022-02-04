import pickle
import warnings
from typing import Sequence

import torch.nn as nn
import torch

import numpy as np

from perceiver_io.io_processors.postprocessors import AudioPostprocessor, ProjectionPostprocessor, \
    ClassificationPostprocessor
from perceiver_io.io_processors.preprocessors import AudioPreprocessor, ImagePreprocessor, OneHotPreprocessor
from perceiver_io.output_queries import FourierQuery, TrainableQuery
from perceiver_io.perceiver import Perceiver
from perceiver_io.position_encoding import PosEncodingType


class MultiModalPerceiver(nn.Module):
    """
    MultiModalPerceiver: Perceiver for autoencoding video data.
    """

    def __init__(
            self,
            img_size: Sequence[int] = (224, 224),
            img_channels: int = 3,
            num_frames: int = 16,
            num_classes: int = 700,
            audio_samples_per_frame: int = 48000 // 25,
            audio_samples_per_patch: int = 16,
            num_latent_channels: int = 512):

        super().__init__()

        self.H, self.W = img_size
        self.num_classes = num_classes
        self.audio_samples_per_frame = audio_samples_per_frame
        self.audio_samples_per_patch = audio_samples_per_patch

        n_audio_samples = num_frames * audio_samples_per_frame

        input_preprocessors = {
            "audio": AudioPreprocessor(
                samples_per_batch=n_audio_samples,
                position_encoding_type=PosEncodingType.FOURIER,
                fourier_position_encoding_kwargs=dict(
                    num_bands=192,
                    max_resolution=(n_audio_samples,),
                    sine_only=False,
                    concat_pos=True,
                ),
                n_extra_pos_mlp=0,
                prep_type="patches",
                samples_per_patch=audio_samples_per_patch),
            "image": ImagePreprocessor(
                img_size=(self.H, self.W),
                input_channels=img_channels,
                num_frames=num_frames,
                position_encoding_type=PosEncodingType.FOURIER,
                fourier_position_encoding_kwargs=dict(
                    num_bands=32,
                    max_resolution=(num_frames, self.H // 4, self.W // 4),
                    sine_only=False,
                    concat_pos=True,
                ),
                n_extra_pos_mlp=0,
                prep_type="patches",
                spatial_downsample=4,
                temporal_downsample=1),
            "label": OneHotPreprocessor(
                input_channels=num_classes,
            ),
        }

        output_postprocessors = {
            "audio": AudioPostprocessor(
                in_channels=512,
                samples_per_patch=audio_samples_per_patch),
            "image": ProjectionPostprocessor(
                num_inputs=512,
                num_outputs=3),
            "label": ClassificationPostprocessor(
                num_input_channels=512,
                # todo check what"s the point of this postprocessor combined with classification decoder
                num_classes=num_classes),
        }

        image_out_query = FourierQuery(
            concat_preprocessed_input=False,
            output_index_dims=(5, self.H, self.W),  # TODO change#images.shape[:4],
            num_bands=32,
            max_resolution=(num_frames, self.H // 4, self.W // 4),
            sine_only=False,
            concat_pos=True,
        )

        audio_out_query = FourierQuery(
            concat_preprocessed_input=False,
            output_index_dims = (n_audio_samples // self.audio_samples_per_patch,),
            num_bands=192,
            max_resolution=(n_audio_samples,),
            sine_only=False,
            concat_pos=True, )

        label_out_query = TrainableQuery(
            output_index_dims=(1,),
            concat_preprocessed_input=False,
            num_channels=1024,#TODO check
            init_scale=0.02)

        output_queries = {
            "audio": audio_out_query,
            "image": image_out_query,
            "label": label_out_query, }



        self.perceiver = Perceiver(
            num_self_attends_per_block=8,
            # Weights won"t be shared if num_blocks is set to 1.
            num_blocks=1,
            num_latents=28 * 28 * 1,
            num_latent_channels=num_latent_channels,
            input_preprocessors=input_preprocessors,
            output_postprocessors=output_postprocessors,
            output_queries=output_queries,
            input_padding_channels=4,
            output_query_padding_channels=2,
            input_mask_probs={"image": 0.0, "audio": 0.0, "label": 1.0}, )

    def load_haiku_params(self, file):
        with open(file, "rb") as f:
            params = pickle.loads(f.read())
            encoder_params = {key[key.find("/") + 1:]: params.pop(key) for key in list(params.keys()) if
                              key.startswith("encoder")}

            self.perceiver._encoder.set_haiku_params(encoder_params)

            multimodal_decoder_params = {key[key.find("/") + 1:]: params.pop(key) for key in list(params.keys()) if
                                         key.startswith("multimodal_decoder")}
            multimodal_decoder_params = {key[key.find('/') + 1:]: multimodal_decoder_params[key] for key in multimodal_decoder_params.keys()}

            basic_decoder_params = {key[key.find("/") + 1:]: multimodal_decoder_params.pop(key) for key in list(multimodal_decoder_params.keys()) if
                                         key.startswith("basic_decoder")}
            self.perceiver._decoder.set_haiku_params(basic_decoder_params)
            self.perceiver.set_haiku_params(multimodal_decoder_params)


            preprocessor_params = {key[key.find("/") + 1:]: params.pop(key) for key in list(params.keys()) if
                                   key.startswith("multimodal_preprocessor")}

            self.perceiver._multi_preprocessor.set_haiku_params(preprocessor_params)

            classification_decoder_params = {key[key.find("/") + 1 + len("~/basic_decoder/"):]: params.pop(key) for key in list(params.keys()) if
                                             key.startswith("classification_decoder")}

            self.perceiver._output_queries["label"].set_haiku_params(classification_decoder_params)


            projection_postprocessor_params = {key[key.find("/") + 1:]: params.pop(key) for key in list(params.keys())
                                               if
                                               key.startswith("projection_postprocessor")}

            audio_postprocessor_params = {key[key.find("/") + 1:]: params.pop(key) for key in list(params.keys()) if
                                          key.startswith("audio_postprocessor")}

            classification_postprocessor_params = {key[key.find("/") + 1:]: params.pop(key) for key in
                                                   list(params.keys()) if
                                                   key.startswith("classification_postprocessor")}

            self.perceiver._output_postprocessors["label"].set_haiku_params(classification_postprocessor_params)
            self.perceiver._output_postprocessors["audio"].set_haiku_params(audio_postprocessor_params)
            self.perceiver._output_postprocessors["image"].set_haiku_params(projection_postprocessor_params)

            if len(params) != 0:
                warnings.warn(f"Some parameters couldn't be matched to model: {params.keys()}")

    def forward(self, images: torch.Tensor, audio: torch.Tensor, n_chunks: int = 128):
        """"""
        # TODO check image channel position
        batch_size = images.shape[0]

        image_chunk_size = np.prod(images.shape[1:-1]).item() // n_chunks
        audio_chunk_size = audio.shape[1] // self.audio_samples_per_patch // n_chunks

        reconstruction = {"image": [], "audio": [], "label": None}

        for chunk_idx in range(n_chunks):
            subsampling = {
                "image": torch.arange(
                    image_chunk_size * chunk_idx, image_chunk_size * (chunk_idx + 1)),
                "audio": torch.arange(
                    audio_chunk_size * chunk_idx, audio_chunk_size * (chunk_idx + 1)),
                "label": None,
            }
            output = self.perceiver(
                {"image": images, "audio": audio, "label": torch.zeros((batch_size, self.num_classes), device=images.device)},
                subsampled_output_points=subsampling)

            reconstruction["image"].append(output["image"])
            reconstruction["audio"].append(output["audio"])
            reconstruction["label"] = output["label"]

        reconstruction["image"] = torch.cat(reconstruction["image"], dim=1).reshape(images.shape)
        reconstruction["audio"] = torch.cat(reconstruction["audio"], dim=1).reshape(audio.shape)

        return reconstruction
