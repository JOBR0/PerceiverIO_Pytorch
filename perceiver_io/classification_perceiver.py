import itertools
import pickle
import warnings
from typing import Sequence

import torch.nn as nn
import torch

from perceiver_io.perceiver import PerceiverEncoder, Perceiver
from perceiver_io import io_processors
from timm.models.layers import to_2tuple

import torch.nn.functional as F
from torch.cuda.amp import autocast

from perceiver_io.perceiver import AbstractPerceiverDecoder, BasicDecoder


class ClassificationPerceiver(nn.Module):

    def __init__(self, num_classes: int = 1000, img_size: Sequence[int] = (224, 224)):
        super().__init__()
        channels = 3

        input_preprocessor = io_processors.ImagePreprocessor(
            # TODO check input_shape
            input_shape=list(img_size) + [channels],
            position_encoding_type='fourier',
            fourier_position_encoding_kwargs=dict(
                concat_pos=True,
                max_resolution=(56, 56),
                num_bands=64,
                sine_only=False
            ),
            prep_type='conv')

        encoder = PerceiverEncoder(
            num_input_channels=input_preprocessor.output_channels,
            cross_attend_widening_factor=1,
            cross_attention_shape_for_attn='kv',
            dropout_prob=0,
            num_blocks=8,
            num_cross_attend_heads=1,
            num_self_attend_heads=8,
            num_self_attends_per_block=6,
            num_z_channels=1024,
            self_attend_widening_factor=1,
            use_query_residual=True,
            z_index_dim=512,
            z_pos_enc_init_scale=0.02)

        decoder = ClassificationDecoder(
            q_in_channels=1024, # corresponds to num_channels of position encoding
            num_classes=num_classes,
            num_z_channels=1024,
            position_encoding_type='trainable',
            trainable_position_encoding_kwargs=dict(
                init_scale=0.02,
                num_channels=1024,
            ),
            use_query_residual=True,
        )

        self.perceiver = Perceiver(
            input_preprocessor=input_preprocessor,
            encoder=encoder,
            decoder=decoder,
            output_postprocessor=None)

    def load_haiku_params(self, file):
        with open(file, "rb") as f:
            dict = pickle.loads(f.read())
            params = dict["params"]
            #TODO handle state
            state = dict["state"]
            # convert to dict
            state = {key: state[key] for key in state}
            encoder_params = {key[key.find('/') + 1:]: params.pop(key) for key in list(params.keys()) if
                              key.startswith("perceiver_encoder")}
            self.perceiver._encoder.set_haiku_params(encoder_params)
            decoder_params = {key[key.find('/') + 1:]: params.pop(key) for key in list(params.keys()) if
                              key.startswith("classification_decoder")}
            self.perceiver._decoder.set_haiku_params(decoder_params)

            preprocessor_params = {key[key.find('/') + 1:]: params.pop(key) for key in list(params.keys()) if
                                   key.startswith("image_preprocessor")}
            preprocessor_state = {key[key.find('/') + 1:]: state.pop(key) for key in list(state.keys()) if
                                  key.startswith("image_preprocessor")}
            self.perceiver._input_preprocessor.set_haiku_params(preprocessor_params, preprocessor_state)

            if len(params) != 0:
                warnings.warn(f"Some parameters couldn't be matched to model: {params.keys()}")

    def forward(self, img: torch.Tensor):
        return self.perceiver(img)


class ClassificationDecoder(AbstractPerceiverDecoder):
    """Cross-attention based classification decoder.
  Light-weight wrapper of `BasicDecoder` for logit output.
  """

    def __init__(self,
                 num_classes: int,
                 **decoder_kwargs):
        super().__init__()

        self._num_classes = num_classes
        self.decoder = BasicDecoder(
            output_index_dims=(1,),  # Predict a single logit array.
            output_num_channels=num_classes,
            **decoder_kwargs)

    def decoder_query(self, inputs, modality_sizes=None,
                      inputs_without_pos=None, subsampled_points=None):
        return self.decoder.decoder_query(inputs, modality_sizes,
                                          inputs_without_pos,
                                          subsampled_points=subsampled_points)

    def output_shape(self, inputs):
        return (inputs.shape[0], self._num_classes), None

    def forward(self, query, z, *, query_mask=None):
        # B x 1 x num_classes -> B x num_classes
        logits = self.decoder(query, z)
        return logits[:, 0, :]

    def set_haiku_params(self, params):
        # TODO where does "~" come from?
        params = {key[key.find('/')+1:]: params[key] for key in params.keys()}

        basic_decoder_params = {key[key.find('/')+1:]: params.pop(key) for key in list(params.keys()) if
                            key.startswith("basic_decoder")}

        self.decoder.set_haiku_params(basic_decoder_params)

        if len(params) != 0:
            warnings.warn(f"Some parameters couldn't be matched to model: {params.keys()}")
