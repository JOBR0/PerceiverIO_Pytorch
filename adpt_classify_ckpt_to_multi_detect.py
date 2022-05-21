import os

import torch


ckpt_file = "./pytorch_checkpoints/imagenet_conv_preprocessing.pth"

# check if file exists
if not os.path.isfile(ckpt_file):
    raise ValueError("Please download the model checkpoint and place it in /pytorch_checkpoints")

checkpoint = torch.load(ckpt_file, map_location="cpu")

models_state_dict = checkpoint["model_state_dict"]

new_state_dict = {}

for key in models_state_dict:
    if "_preprocessors.__default" in key:
        new_key = key.replace("_preprocessors.__default", "_preprocessors.img_0")
        new_state_dict[new_key] = models_state_dict[key]
    else:
        new_state_dict[key] = models_state_dict[key]


checkpoint["model_state_dict"] = new_state_dict


torch.save(checkpoint, "./pytorch_checkpoints/imagenet_conv_preprocessing_multi_detect.pth")