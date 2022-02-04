import os

import torch
import numpy as np

from perceiver_io.language_perceiver import LanguagePerceiver
from utils.bytes_tokenizer import BytesTokenizer
from utils.utils import dump_pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = BytesTokenizer()

perceiver = LanguagePerceiver(vocab_size=tokenizer.vocab_size)
perceiver.to(device)
perceiver.eval()

ckpt_file = "./pytorch_checkpoints/language_perceiver_io_bytes.pth"

# check if file exists
if not os.path.isfile(ckpt_file):
    raise ValueError("Please download the model checkpoint and place it in /pytorch_checkpoints")

checkpoint = torch.load(ckpt_file, map_location=device)

perceiver.load_state_dict(checkpoint['model_state_dict'])

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

inputs = torch.from_numpy(inputs).to(device)
input_mask = torch.from_numpy(input_mask).bool().to(device)

# Predict
with torch.inference_mode():
    out = perceiver(inputs, input_masks=input_mask)

dump_pickle(out.cpu().numpy(), "temp/output_language_torch.pickle")

masked_tokens_predictions = out[0, 51:60].argmax(axis=-1)
print("Greedy predictions:")
print(masked_tokens_predictions)
print()
print("Predicted string:")
print(tokenizer.to_string(masked_tokens_predictions.cpu().numpy()))
