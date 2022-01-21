import torch

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from utils.flow_utils import flow_to_image
from perceiver_io.flow_perceiver import FlowPerceiver



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

perceiver = FlowPerceiver()

perceiver.load_haiku_params("./haiku_checkpoints/optical_flow_checkpoint.pystate")

img1 = load_image("./sample_data/frame_0016.png", device)
img2 = load_image("./sample_data/frame_0017.png", device)

with torch.inference_mode():
    flow = perceiver(img1, img2, test_mode=True)

# Show prediciton
figure = plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(img1[0].permute(1, 2, 0).cpu().numpy()/255)
plt.subplot(2, 2, 2)
plt.imshow(img2[0].permute(1, 2, 0).cpu().numpy()/255)
plt.subplot(2, 2, 3)
plt.imshow(flow_to_image(flow[0].permute(1, 2, 0).cpu().numpy()))
plt.show()




