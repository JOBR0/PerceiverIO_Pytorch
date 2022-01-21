from typing import Union

import torch
import numpy as np
import pickle

from perceiver_io.language_perceiver import LanguagePerceiver
from perceiver_io.bytes_tokenizer import BytesTokenizer

tokenizer = BytesTokenizer()

model = LanguagePerceiver(vocab_size=tokenizer.vocab_size)

model.load_haiku_params("haiku_models/language_perceiver_io_bytes.pickle")

D_MODEL = 768
D_LATENTS = 1280
MAX_SEQ_LEN = 2048

input_str = "This is an incomplete sentence where some words are missing."

input_tokens = tokenizer.to_int(input_str)

# Mask " missing.". Note that the model performs much better if the masked chunk
# starts with a space.
input_tokens[51:60] = tokenizer.mask_token
print("Tokenized string without masked bytes:")
print(tokenizer.to_string(input_tokens))

# @title Pad and reshape inputs
inputs = input_tokens[None]
input_mask = np.ones_like(inputs)


def pad(max_sequence_length: int, inputs, input_mask):
    input_len = inputs.shape[1]
    assert input_len <= max_sequence_length
    pad_len = max_sequence_length - input_len
    padded_inputs = np.pad(
        inputs,
        pad_width=((0, 0), (0, pad_len)),
        constant_values=tokenizer.pad_token)
    padded_mask = np.pad(
        input_mask,
        pad_width=((0, 0), (0, pad_len)),
        constant_values=0)
    return padded_inputs, padded_mask


inputs, input_mask = pad(MAX_SEQ_LEN, inputs, input_mask)

inputs = torch.from_numpy(inputs)
input_mask = torch.from_numpy(input_mask)

out = model(inputs, input_masks=input_mask)

masked_tokens_predictions = out[0, 51:60].argmax(axis=-1)
print("Greedy predictions:")
print(masked_tokens_predictions)
print()
print("Predicted string:")
print(tokenizer.to_string(masked_tokens_predictions))
