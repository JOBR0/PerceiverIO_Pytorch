import torch

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from flow_utils import flow_to_image
from perceiver_io.classification_perceiver import ClassificationPerceiver
from perceiver_io.utils import load_image


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

perceiver = ClassificationPerceiver()

perceiver.load_haiku_params("./haiku_models/optical_flow_checkpoint.pystate")

img = load_image("./sample_data/dalmation.jpg", device)


MEAN_RGB = (0.485 * 255, 0.456 * 255, 0.406 * 255)
STDDEV_RGB = (0.229 * 255, 0.224 * 255, 0.225 * 255)

img = img - np.array(MEAN_RGB) / np.array(STDDEV_RGB)

with torch.inference_mode():
    pred = perceiver(img)

# Show prediciton
plt.imshow(flow_to_image(flow))




