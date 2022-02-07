from typing import Sequence

import torch.nn as nn
import torch

from perceiver_io.io_processors.postprocessors import AudioPostprocessor, ProjectionPostprocessor, \
    ClassificationPostprocessor
from perceiver_io.io_processors.preprocessors import AudioPreprocessor, ImagePreprocessor, OneHotPreprocessor
from perceiver_io.output_queries import FourierQuery, TrainableQuery
from perceiver_io.perceiver import PerceiverIO
from perceiver_io.position_encoding import PosEncodingType


class MultiModalPerceiver(nn.Module):
    """
    MultiModalPerceiver: Perceiver for auto-encoding video data.
    Args:
        img_size (Sequence[int]): Size of the image. Default: (224, 224)
        img_channels (int): Number of channels of the image. Default: 3
        num_frames (int): Number of frames to use for the video. Default: 16
        num_classes (int): Number of possible classes. Default: 700
        audio_samples_per_frame (int): Number of audio samples per video frame. Default: 128
        audio_samples_per_patch (int): Number of audio samples that are combined as a patch. Default: 16
        num_self_attends_per_block (int): Number of self attends per block. Default: 8
        num_blocks (int): Number of blocks. All blocks share weights. Default: 1
        num_latents (int): Number of latent variables. Default: 784
        num_latent_channels (int): Number of channels for latent variables. Default: 512
    """

    def __init__(
            self,
            img_size: Sequence[int] = (224, 224),
            img_channels: int = 3,
            num_frames: int = 16,
            num_classes: int = 700,
            audio_samples_per_frame: int = 48000 // 25,
            audio_samples_per_patch: int = 16,
            num_self_attends_per_block: int = 8,
            num_blocks: int = 1,
            num_latents: int = 28 * 28 * 1,
            num_latent_channels: int = 512,):

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
                num_classes=num_classes),
        }

        image_out_query = FourierQuery(
            concat_preprocessed_input=False,
            output_index_dims=(num_frames, self.H, self.W),
            num_bands=32,
            max_resolution=(num_frames, self.H // 4, self.W // 4),
            sine_only=False,
            concat_pos=True,
        )

        audio_out_query = FourierQuery(
            concat_preprocessed_input=False,
            output_index_dims=(n_audio_samples // self.audio_samples_per_patch,),
            num_bands=192,
            max_resolution=(n_audio_samples,),
            sine_only=False,
            concat_pos=True, )

        label_out_query = TrainableQuery(
            output_index_dims=(1,),
            concat_preprocessed_input=False,
            num_channels=1024,
            init_scale=0.02)

        output_queries = {
            "audio": audio_out_query,
            "image": image_out_query,
            "label": label_out_query, }

        self.perceiver = PerceiverIO(
            num_self_attends_per_block=num_self_attends_per_block,
            num_blocks=num_blocks,
            num_latents=num_latents,
            num_latent_channels=num_latent_channels,
            input_preprocessors=input_preprocessors,
            output_postprocessors=output_postprocessors,
            output_queries=output_queries,
            input_padding_channels=4,
            output_query_padding_channels=2,
            input_mask_probs={"image": 0.0, "audio": 0.0, "label": 1.0}, )

    def forward(self, images: torch.Tensor, audio: torch.Tensor, n_chunks: int = 128):
        """"""
        batch_size, t, c, h, w = images.shape

        image_chunk_size = t*h*w // n_chunks
        audio_chunk_size = audio.shape[1] // self.audio_samples_per_patch // n_chunks

        reconstruction = {"image": [], "audio": [], "label": []}

        for chunk_idx in range(n_chunks):
            subsampling = {
                "image": torch.arange(
                    image_chunk_size * chunk_idx, image_chunk_size * (chunk_idx + 1)),
                "audio": torch.arange(
                    audio_chunk_size * chunk_idx, audio_chunk_size * (chunk_idx + 1)),
                "label": None,
            }
            output = self.perceiver(
                {"image": images, "audio": audio,
                 "label": torch.zeros((batch_size, self.num_classes), device=images.device)},
                subsampled_output_points=subsampling)

            reconstruction["image"].append(output["image"])
            reconstruction["audio"].append(output["audio"])
            reconstruction["label"].append(output["label"][:,None])

        reconstruction["image"] = torch.cat(reconstruction["image"], dim=1).reshape([batch_size, t, h, w, c]).moveaxis(-1, -3)
        reconstruction["audio"] = torch.cat(reconstruction["audio"], dim=1).reshape(audio.shape)
        reconstruction["label"] = torch.cat(reconstruction["label"], dim=1).mean(dim=1)

        return reconstruction
