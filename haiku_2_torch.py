import os

import torch

from perceiver_io.language_perceiver import LanguagePerceiver
from perceiver_io.flow_perceiver import FlowPerceiver
from perceiver_io.classification_perceiver import ClassificationPerceiver


model = LanguagePerceiver()
haiku_file = "haiku_checkpoints/language_perceiver_io_bytes.pickle"
model.load_haiku_params(haiku_file)

model_name = "language_perceiver_io_bytes.pth"

state_dicts = {
    "model_state_dict": model.state_dict()
}
torch.save(state_dicts, os.path.join("pytorch_checkpoints", model_name))

model = FlowPerceiver()
haiku_file = "haiku_checkpoints/optical_flow_checkpoint.pystate"
model.load_haiku_params(haiku_file)

model_name = "flow_perceiver_io.pth"

state_dicts = {
    "model_state_dict": model.state_dict()
}
torch.save(state_dicts, os.path.join("pytorch_checkpoints", model_name))

model = ClassificationPerceiver()


model = ClassificationPerceiver(prep_type="conv_preprocessing")
haiku_file = "haiku_checkpoints/imagenet_conv_preprocessing.pystate"
model.load_haiku_params(haiku_file)

model_name = "classify_perceiver_io_conv_preprocessing.pth"

state_dicts = {
    "model_state_dict": model.state_dict()
}
torch.save(state_dicts, os.path.join("pytorch_checkpoints", model_name))

model = ClassificationPerceiver(prep_type="learned_pos_encoding")
haiku_file = "haiku_checkpoints/imagenet_learned_position_encoding.pystate"
model.load_haiku_params(haiku_file)

model_name = "classify_perceiver_io_learned_preprocessing.pth"

state_dicts = {
    "model_state_dict": model.state_dict()
}
torch.save(state_dicts, os.path.join("pytorch_checkpoints", model_name))

model = ClassificationPerceiver(prep_type="fourier_pos_encoding")
haiku_file = "haiku_checkpoints/imagenet_fourier_position_encoding.pystate"
model.load_haiku_params(haiku_file)

model_name = "classify_perceiver_io_fourier_preprocessing.pth"

state_dicts = {
    "model_state_dict": model.state_dict()
}
torch.save(state_dicts, os.path.join("pytorch_checkpoints", model_name))


