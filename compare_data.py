import numpy as np
from utils.utils import load_pickle, dump_pickle

print("Multmodal")
output_multi_haiku = load_pickle("temp/output_multi_haiku.pickle")
output_multi_torch = load_pickle("temp/output_multi_torch.pickle")
output_diff = {k: (output_multi_torch[k] - output_multi_haiku[k]) ** 2 for k in output_multi_haiku.keys()}
max_diffs = {k: np.max(output_diff[k]) for k in output_diff.keys()}
std_diffs = {k: np.std(output_diff[k]) for k in output_diff.keys()}
print("max", max_diffs)
print("std", std_diffs)

print("Language")

output_language_haiku = load_pickle("temp/output_language_haiku.pickle")
output_language_torch = load_pickle("temp/output_language_torch.pickle")
output_diff = output_language_haiku - output_language_torch
max_diffs = np.max(output_diff)
std_diffs = np.std(output_diff)
print("max", max_diffs)
print("std", std_diffs)

print("Classification")

print("FOURIER_POS_CONVNET")
output_class_haiku_1 = load_pickle("temp/output_conv_preprocessing_haiku.pickle")
output_class_torch_1 = load_pickle("temp/output_PrepType.FOURIER_POS_CONVNET_torch.pickle")
output_diff = output_class_haiku_1 - output_class_torch_1
max_diffs = np.max(output_diff)
std_diffs = np.std(output_diff)
print("max", max_diffs)
print("std", std_diffs)

print("LEARNED_POS_1X1CONV")
output_class_haiku_2 = load_pickle("temp/output_learned_position_encoding_haiku.pickle")
output_class_torch_2 = load_pickle("temp/output_PrepType.LEARNED_POS_1X1CONV_torch.pickle")
output_diff = output_class_haiku_2 - output_class_torch_2
max_diffs = np.max(output_diff)
std_diffs = np.std(output_diff)
print("max", max_diffs)
print("std", std_diffs)

print("FOURIER_POS_PIXEL")
output_class_haiku_3 = load_pickle("temp/output_fourier_position_encoding_haiku.pickle")
output_class_torch_3 = load_pickle("temp/output_PrepType.FOURIER_POS_PIXEL_torch.pickle")
output_diff = output_class_haiku_3 - output_class_torch_3
max_diffs = np.max(output_diff)
std_diffs = np.std(output_diff)
print("max", max_diffs)
print("std", std_diffs)

print("Flow")

output_flow_haiku = load_pickle("temp/output_flow_haiku.pickle")
output_flow_torch = load_pickle("temp/output_flow_torch.pickle")
output_flow_torch = np.moveaxis(output_flow_torch,1,-1)
output_diff = output_flow_haiku - output_flow_torch
max_diffs = np.max(output_diff)
std_diffs = np.std(output_diff)
print("max", max_diffs)
print("std", std_diffs)

