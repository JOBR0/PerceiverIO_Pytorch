import numpy as np
from utils.utils import load_pickle, dump_pickle

print("Multmodal")
output_multi_haiku = load_pickle("temp/output_multi_haiku.pickle")
output_multi_torch = load_pickle("temp/output_multi_torch.pickle")
output_diff = {k: (output_multi_torch[k] - output_multi_haiku[k]) ** 2 for k in output_multi_haiku.keys()}
max_diffs = {k: np.max(output_diff[k]) for k in output_diff.keys()}
print(max_diffs)

print("Language")

output_language_haiku = load_pickle("temp/output_language_haiku.pickle")
output_language_torch = load_pickle("temp/output_language_torch.pickle")
output_diff = output_language_haiku - output_language_torch
max_diffs = np.max(output_diff)
print(max_diffs)


pass