import itertools
import pickle
import warnings
from typing import Sequence

import torch.nn as nn
import torch

from core.Transformers.perceiver import PerceiverEncoder, FlowDecoder, Perceiver
from core.Transformers import io_processors
from timm.models.layers import to_2tuple

import torch.nn.functional as F
from torch.cuda.amp import autocast


class FlowPerceiver(nn.Module):
    """
    FlowPerceiver: Perceiver for optical flow
    Args:
        img_size (Sequence[int]): Size of training images (height x width). Default: (368, 496)
        flow_scale_factor (int): https://github.com/deepmind/deepmind-research/issues/266. Default: 20
        n_latents (int): Number of latent variables. Default: 2048
        n_self_attends (int): Number of self attention layers. Default: 24
        mixed_precision (bool): Whether to run the perceiver in mixed precision. Default: False
    """

    def __init__(
            self,
            img_size: Sequence[int] = (368, 496),
            flow_scale_factor: int = 20,
            n_latents: int = 2048,
            n_self_attends: int = 24,
            mixed_precision: bool = False):
        super().__init__()
        self._flow_scale_factor = flow_scale_factor
        self.mixed_precision = mixed_precision

        channels = 3
        patch_size = 3

        preprocessor_channels = 64

        input_preprocessor = io_processors.ImagePreprocessor(
            input_shape=list(img_size) + [channels * patch_size ** 2],
            position_encoding_type='fourier',
            fourier_position_encoding_kwargs=dict(
                num_bands=64,
                max_resolution=img_size,
                sine_only=False,
                concat_pos=True,
            ),
            n_extra_pos_mlp=0,
            prep_type='patches',
            spatial_downsample=1,
            conv_after_patching=True,
            temporal_downsample=2,
            num_channels=preprocessor_channels)

        # TODO generalize
        fourier_dimensions = 258
        encoder_input_channels = preprocessor_channels + fourier_dimensions
        num_z_channels = 512

        encoder = PerceiverEncoder(
            num_input_channels=encoder_input_channels,
            num_self_attends_per_block=n_self_attends,
            # Weights won't be shared if num_blocks is set to 1.
            num_blocks=1,
            z_index_dim=n_latents,
            num_cross_attend_heads=1,
            num_z_channels=num_z_channels,
            num_self_attend_heads=16,
            cross_attend_widening_factor=1,
            self_attend_widening_factor=1,
            dropout_prob=0.0,
            z_pos_enc_init_scale=0.02,
            cross_attention_shape_for_attn='kv')

        decoder = FlowDecoder(
            q_in_channels=encoder_input_channels,
            num_z_channels=num_z_channels,
            output_image_shape=img_size,
            rescale_factor=100.0,
            use_query_residual=False,
            output_num_channels=2,
            output_w_init="zeros",
            # We query the decoder using the first frame features
            # rather than a standard decoder position encoding.
            position_encoding_type='fourier',
            fourier_position_encoding_kwargs=dict(
                concat_pos=True,
                max_resolution=img_size,
                num_bands=64,
                sine_only=False
            )
        )

        self.perceiver = Perceiver(
            input_preprocessor=input_preprocessor,
            encoder=encoder,
            decoder=decoder,
            output_postprocessor=None)

        self.H, self.W = to_2tuple(img_size)

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

    def compute_grid_indices(self, image_shape: tuple, min_overlap: int):
        """
        Compute top-left corner coordinates for patches
        Args:
            image_shape (tuple): Height and width of the input image
            min_overlap (int): Minimum number of pixels that two patches overlap
        """
        if min_overlap >= self.H or min_overlap >= self.W:
            raise ValueError(
                f"Overlap should be less than size of patch (got {min_overlap}"
                f"for patch size {(self.H, self.W)}).")

        ys = list(range(0, image_shape[0], self.H - min_overlap))
        xs = list(range(0, image_shape[1], self.W - min_overlap))
        # Make sure the final patch is flush with the image boundary
        ys[-1] = image_shape[0] - self.H
        xs[-1] = image_shape[1] - self.W

        # Avoid predicting same patch multiple times
        if image_shape[0] == self.H:
            ys = [0]
        if image_shape[1] == self.W:
            xs = [0]

        return itertools.product(ys, xs)

    def _predict_patch(self, patch):
        """Predict flow for one image patch as big as training images"""
        with autocast(enabled=self.mixed_precision):
            # Extract overlapping 3x3 patches
            patch = io_processors.patches_for_flow(patch)
            output = self.perceiver(patch)
            output = output.permute(0, 3, 1, 2)
        return self._flow_scale_factor * output

    def forward(self, image1: torch.Tensor, image2: torch.Tensor, test_mode: bool = False, min_overlap: int = 20):
        """
        Computes forward pass for flow perceiver
        Args:
            image1 (torch.Tensor): source images (N, C, H, W).
            image2 (torch.Tensor): target images (N, C, H, W).
            test_mode (bool): If in test mode. Default: False
            min_overlap (int): Minimum overlap of patches if images are bigger than training size. Default: 20
        """
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        height = image1.shape[2]
        width = image1.shape[3]

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        # Stack in time dimension
        inputs = torch.stack([image1, image2], axis=1)

        if height < self.H:
            raise ValueError(
                f"Height of image (shape: {image1.shape}) must be at least {self.H:}."
                "Please pad or resize your image to the minimum dimension."
            )
        if width < self.W:
            raise ValueError(
                f"Width of image (shape: {image1.shape}) must be at least {self.W}."
                "Please pad or resize your image to the minimum dimension."
            )

        if test_mode:
            # in test_mode, image size can be arbitrary
            # the flow is predicted for patches of training size and than stitched together
            flows = 0
            flow_count = 0

            grid_indices = self.compute_grid_indices((height, width), min_overlap)

            for y, x in grid_indices:
                inp_piece = inputs[..., y: y + self.H, x: x + self.W]
                flow_piece = self._predict_patch(inp_piece)

                # weights should give more weight to flow from center of patches
                weights_y, weights_x = torch.meshgrid(torch.arange(self.H), torch.arange(self.W))
                weights_x = torch.minimum(weights_x + 1, self.W - weights_x)
                weights_y = torch.minimum(weights_y + 1, self.H - weights_y)
                weights = torch.minimum(weights_x, weights_y)[None, None, :, :]
                weights = weights.to(flow_piece.device)

                padding = (x, width - x - self.W, y, height - y - self.H)
                flows = flows + F.pad(flow_piece * weights, padding)
                flow_count = flow_count + F.pad(weights, padding)

            flows = flows / flow_count
            output = flows

        else:
            assert height == self.H and width == self.W, \
                f"In training mode images must have size equal to specified img_size {(self.H, self.W)}"
            output = self._predict_patch(inputs)

        return output
