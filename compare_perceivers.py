import itertools
import random
import unittest
import numpy as np
import jax.numpy as jnp
import torch

import haiku as hk
import jax
from perceiver_io.flow_perceiver import FlowPerceiver

from perceiver.position_encoding import generate_fourier_features

import matplotlib.pyplot as plt

import sys
sys.path.insert(0, 'perceiver')

from perceiver.position_encoding import \
    generate_fourier_features as generate_fourier_features_jax, \
    FourierPositionEncoding as FourierPositionEncoding_jax, TrainablePositionEncoding as TrainablePositionEncoding_jax, \
    _check_or_build_spatial_positions as _check_or_build_spatial_positions_jax, build_linear_positions as build_linear_positions_jax
from perceiver.io_processors import extract_patches as extract_patches_jax, \
    patches_for_flow as patches_for_flow_jax, ImagePreprocessor as ImagePreprocessor_jax
from perceiver_io.io_processors import extract_patches as extract_patches_torch, \
    patches_for_flow as patches_for_flow_torch, ImagePreprocessor as ImagePreprocessor_torch
from perceiver_io.position_encoding import generate_fourier_features as generate_fourier_features_torch, \
    FourierPositionEncoding as FourierPositionEncoding_torch, \
    TrainablePositionEncoding as TrainablePositionEncoding_torch, \
    _check_or_build_spatial_positions as _check_or_build_spatial_positions_torch, build_linear_positions as build_linear_positions_torch


def create_jax_torch_data(sizes):
    data = np.random.rand(*sizes).astype(np.float32)
    torch_d = torch.from_numpy(data)
    jax_d = jnp.array(data)
    return jax_d, torch_d


def jax_allclose_torch(jax_data, torch_data, rtol=1e-05, atol=1e-08):
    np1 = np.array(jax_data)
    np2 = torch_data.detach().cpu().numpy()

    # np.max(np.abs(np1-np2))
    return np.allclose(np1, np2, rtol=rtol, atol=atol)


class ComparePerceivers(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Create some arbitrary values that can be used for tests
        self.img_size = (368, 496)
        self.batch_size = 3

    def test_patching(self):
        height = 128
        width = 224
        batch_size = 2
        img_sizes = (batch_size, height, width, 3)
        jax_images, torch_images = create_jax_torch_data(img_sizes)

        # torch version expects channels as second dimension
        torch_images = torch_images.permute(0, 3, 1, 2)

        sizes_cases = [(3, 3), (1, 4)]
        strides_cases = [(1, 1), (3, 2)]
        dilation_cases = [(1, 1), (2, 1)]

        padding = "VALID"

        for sizes, strides, dilation in itertools.product(sizes_cases, strides_cases, dilation_cases):
            rates = (1,) + dilation + (1,)
            jax_sizes = (1,) + sizes + (1,)
            jax_strides = (1,) + strides + (1,)

            jax_result = extract_patches_jax(jax_images, jax_sizes, jax_strides, rates, padding)
            torch_result = extract_patches_torch(torch_images, sizes, strides, dilation, padding)

            self.assertTrue(jax_allclose_torch(jax_result, torch_result))

    def test_patches_for_flow(self):
        height = 128
        width = 224
        batch_size = 2
        img_sizes = (batch_size, 2, height, width, 3)
        jax_images, torch_images = create_jax_torch_data(img_sizes)

        # torch version expects channels as third dimension
        torch_images = torch_images.permute(0, 1, 4, 2, 3)

        jax_result = patches_for_flow_jax(jax_images)
        torch_result = patches_for_flow_torch(torch_images)

        self.assertTrue(jax_allclose_torch(jax_result, torch_result))

    def test_trainable_encoding(self):
        index_dims = [6] + list(self.img_size)

        enc_torch = TrainablePositionEncoding_torch(np.prod(index_dims))

        def enc_func_jax(batch_size):
            enc_jax = TrainablePositionEncoding_jax(np.prod(index_dims))
            return enc_jax(batch_size)

        rng = jax.random.PRNGKey(42)
        enc_func_jax = hk.transform(enc_func_jax)
        params = enc_func_jax.init(rng, self.batch_size)
        jax_result = enc_func_jax.apply(params, rng, self.batch_size)

        torch_result = enc_torch(self.batch_size)

        # Tests only for same shape as encoding is initialized randomly
        self.assertTrue(jax_result.shape == torch_result.shape)


    def test_build_linear_positions(self):
        torch_result = build_linear_positions_torch(self.img_size)
        jax_result = build_linear_positions_jax(self.img_size)

        self.assertTrue(jax_allclose_torch(jax_result, torch_result, atol=1e-7))

    def test_spatial_positions(self):
        torch_result = _check_or_build_spatial_positions_torch(None, self.img_size, self.batch_size)
        jax_result = _check_or_build_spatial_positions_jax(None, self.img_size, self.batch_size)

        self.assertTrue(jax_allclose_torch(jax_result, torch_result, atol=1e-7))

    def test_fourier_encoding(self):
        num_bands = 64
        max_resolution = self.img_size
        sine_only = False
        concat_pos = True
        index_dims = list(self.img_size)

        enc_torch = FourierPositionEncoding_torch(index_dims, num_bands, concat_pos, max_resolution, sine_only)

        def enc_func_jax(batch_size):
            enc_jax = FourierPositionEncoding_jax(index_dims, num_bands, concat_pos, max_resolution, sine_only)
            return enc_jax(batch_size)

        rng = jax.random.PRNGKey(42)
        enc_func_jax = hk.transform(enc_func_jax)
        params = enc_func_jax.init(rng, self.batch_size)
        jax_result = enc_func_jax.apply(params, rng, self.batch_size)

        torch_result = enc_torch(self.batch_size)
        self.assertTrue(jax_allclose_torch(jax_result, torch_result, atol=1e-3))

    def test_image_preprocessor(self):
        kwargs = dict(
            position_encoding_type='fourier',
            fourier_position_encoding_kwargs=dict(
                num_bands=64,
                max_resolution=self.img_size,
                sine_only=False,
                concat_pos=True,
            ),
            n_extra_pos_mlp=0,
            prep_type='patches',
            spatial_downsample=1,
            conv_after_patching=True,
            temporal_downsample=2)

        input_is_1d = True
        pos = None
        input_size = [self.batch_size, 2] + list(self.img_size) + [3 * 3 * 3]
        jax_inputs, torch_inputs = create_jax_torch_data(input_size)

        # torch_inputs = torch_inputs.permute(0, 1, 4, 2, 3)

        input_shape = list(self.img_size) + [3 * 3 * 3]

        prep_torch = ImagePreprocessor_torch(input_shape=input_shape, **kwargs)
        torch_inputs, _, torch_inputs_without_pos = prep_torch(torch_inputs, pos=pos, network_input_is_1d=input_is_1d)

        def prep_func_jax(inputs, pos, network_input_is_1d):
            prep_jax = ImagePreprocessor_jax(**kwargs)
            return prep_jax(inputs, is_training=True, pos=pos, network_input_is_1d=network_input_is_1d)

        rng = jax.random.PRNGKey(42)
        prep_func_jax = hk.transform(prep_func_jax)
        params = prep_func_jax.init(rng, jax_inputs, pos=pos, network_input_is_1d=input_is_1d)
        jax_inputs, _, jax_inputs_without_pos = prep_func_jax.apply(params, rng, jax_inputs, pos=pos,
                                                                    network_input_is_1d=input_is_1d)

        self.assertTrue(jax_allclose_torch(jax_inputs, torch_inputs))
        self.assertTrue(jax_allclose_torch(jax_inputs_without_pos, torch_inputs_without_pos))

    # def test_flow_perceiver(self):
    #     img_size = (10, 10)
    #     batch_size = 2
    #     img1 = torch.rand((batch_size, 3) + img_size)
    #     img2 = torch.rand((batch_size, 3) + img_size)
    #
    #     flow_perceiver = FlowPerceiver(img_size=img_size)
    #     out = flow_perceiver(img1, img2)


if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    # flow_perceiver = FlowPerceiver()

    unittest.main()
