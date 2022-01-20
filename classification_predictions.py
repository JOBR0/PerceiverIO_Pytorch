import torch

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from flow_utils import flow_to_image
from perceiver_io.classification_perceiver import ClassificationPerceiver
from perceiver_io.utils import load_image

from torchvision import transforms

from imagenet_labels import IMAGENET_LABELS



MEAN_RGB = (0.485 * 255, 0.456 * 255, 0.406 * 255)
STDDEV_RGB = (0.229 * 255, 0.224 * 255, 0.225 * 255)

img_size = (224, 224)

normalize = transforms.Normalize(MEAN_RGB, STDDEV_RGB)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

perceiver = ClassificationPerceiver()

perceiver.load_haiku_params("./haiku_models/imagenet_conv_preprocessing.pystate")
perceiver.eval()

img = load_image("./sample_data/dalmation.jpg", device)



h, w = img.shape[2:]
# crop to square and then resize to 224x224
min_size = min(h, w)
img_norm = transforms.functional.resized_crop(img, top=int(h/2-min_size/2), left=int(w/2-min_size/2), height=min_size, width=min_size, size=img_size)

img_norm = normalize(img_norm)


with torch.inference_mode():
    logits = perceiver(img_norm)
    # get top 5 predictions
    top_preds = torch.topk(logits, 5)[1]

    # get top 5 class labels
    top_labels = np.array(IMAGENET_LABELS)[top_preds.cpu().numpy()]


    print(top_preds)
    print(top_labels)


#print(logits)

# Show prediciton
plt.imshow(img[0].permute(1, 2, 0).cpu().numpy()/255)
plt.show()





