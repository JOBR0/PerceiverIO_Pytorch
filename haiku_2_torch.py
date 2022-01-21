import os

import torch

from perceiver_io.language_perceiver import LanguagePerceiver
from perceiver_io.flow_perceiver import FlowPerceiver
from perceiver_io.classification_perceiver import ClassificationPerceiver


model = LanguagePerceiver()
haiku_file = "haiku_models/language_perceiver_io_bytes.pickle"
model.load_haiku_params(haiku_file)

model_name = "language_perceiver_io_bytes.pth"

state_dicts = {
    "model_state_dict": model.state_dict()
}
torch.save(state_dicts, os.path.join("pytorch_checkpoints", model_name))

