import os
import torch

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from utils.flow_utils import flow_to_image
from perceiver_io.flow_perceiver import FlowPerceiver
from utils.utils import load_image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

perceiver = FlowPerceiver(img_size=(368, 496))

# check if file exists
if not os.path.isfile("./pytorch_checkpoints/flow_perceiver_io.pth"):
    raise ValueError("Please download the model checkpoint and place it in /pytorch_checkpoints")

checkpoint = torch.load("./pytorch_checkpoints/flow_perceiver_io.pth", map_location=device)

perceiver.load_state_dict(checkpoint['model_state_dict'])

img1 = load_image("./sample_data/frame_0016.png", device)
img2 = load_image("./sample_data/frame_0017.png", device)

# Normalize images
img1 = 2 * (img1 / 255.0) - 1.0
img2 = 2 * (img2 / 255.0) - 1.0

# Predict Flow
with torch.inference_mode():
    flow = perceiver(img1, img2, test_mode=True)

# Show prediction
figure = plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(img1[0].permute(1, 2, 0).cpu().numpy()/255)
plt.subplot(2, 2, 2)
plt.imshow(img2[0].permute(1, 2, 0).cpu().numpy()/255)
plt.subplot(2, 2, 3)
plt.imshow(flow_to_image(flow[0].permute(1, 2, 0).cpu().numpy()))
plt.show()




