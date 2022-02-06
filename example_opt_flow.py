import os

import torch
import matplotlib.pyplot as plt

from utils.flow_utils import flow_to_image
from utils.utils import load_image, dump_pickle
from perceiver_io.flow_perceiver import FlowPerceiver


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

perceiver = FlowPerceiver(img_size=(368, 496))

perceiver.eval()
perceiver.to(device)

ckpt_file = "./pytorch_checkpoints/optical_flow_checkpoint.pth"

# check if file exists
if not os.path.isfile(ckpt_file):
    raise ValueError("Please download the model checkpoint and place it in /pytorch_checkpoints")

checkpoint = torch.load(ckpt_file, map_location=device)

perceiver.load_state_dict(checkpoint["model_state_dict"])

img1 = load_image("./sample_data/frame_0016.png", device)
img2 = load_image("./sample_data/frame_0017.png", device)

# Normalize images
img1_norm = 2 * (img1 / 255.0) - 1.0
img2_norm = 2 * (img2 / 255.0) - 1.0

img1_norm = img1_norm.to(device)
img2_norm = img2_norm.to(device)

# Predict Flow
with torch.inference_mode():
    flow = perceiver(img1_norm, img2_norm, test_mode=True)

dump_pickle(flow.cpu().numpy(), "temp/output_flow_torch.pickle") # TODO: remove

# Show prediction
figure = plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(img1[0].permute(1, 2, 0).cpu().numpy()/255)
plt.subplot(2, 2, 2)
plt.imshow(img2[0].permute(1, 2, 0).cpu().numpy()/255)
plt.subplot(2, 2, 3)
plt.imshow(flow_to_image(flow[0].permute(1, 2, 0).cpu().numpy()))
plt.show()




