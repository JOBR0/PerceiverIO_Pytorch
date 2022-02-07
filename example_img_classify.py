import os

import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import transforms

from perceiver_io.classification_perceiver import ClassificationPerceiver, PrepType
from utils.utils import load_image, dump_pickle
from utils.imagenet_labels import IMAGENET_LABELS

import torch.nn.functional as F


def img_classify_example():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    MEAN_RGB = (0.485 * 255, 0.456 * 255, 0.406 * 255)
    STDDEV_RGB = (0.229 * 255, 0.224 * 255, 0.225 * 255)

    img_size = (224, 224)

    normalize = transforms.Normalize(MEAN_RGB, STDDEV_RGB)

    # There are three different pretrained models that use different input preprocessing:
    prep_type = PrepType.FOURIER_POS_CONVNET
    if prep_type == PrepType.FOURIER_POS_CONVNET:
        # Uses a 2D fourier position encoding and a 2D conv-net as preprocessing
        ckpt_file = "./pytorch_checkpoints/imagenet_conv_preprocessing.pth"
    elif prep_type == PrepType.LEARNED_POS_1X1CONV:
        #  Uses a learned position encoding over the image and 1x1 convolution over the pixels
        ckpt_file = "./pytorch_checkpoints/imagenet_learned_position_encoding.pth"
    elif prep_type == PrepType.FOURIER_POS_PIXEL:
        # Uses a 2D fourier position encoding and the raw pixels
        ckpt_file = "./pytorch_checkpoints/imagenet_fourier_position_encoding.pth"

    perceiver = ClassificationPerceiver(num_classes=1000,
                                        img_size=img_size,
                                        prep_type=prep_type)
    perceiver.eval()
    perceiver.to(device)

    # check if file exists
    if not os.path.isfile(ckpt_file):
        raise ValueError("Please download the model checkpoint and place it in /pytorch_checkpoints")

    checkpoint = torch.load(ckpt_file, map_location=device)

    perceiver.load_state_dict(checkpoint["model_state_dict"])

    img = load_image("./sample_data/dalmation.jpg", device)

    h, w = img.shape[2:]
    # crop to square and then resize to 224x224
    min_size = min(h, w)
    img_norm = transforms.functional.resized_crop(img, top=int(h / 2 - min_size / 2), left=int(w / 2 - min_size / 2),
                                                  height=min_size, width=min_size, size=img_size)
    img_norm = normalize(img_norm)
    img_norm = img_norm.to(device)

    with torch.inference_mode():
        logits = perceiver(img_norm)
        # get top 5 predictions
        top_preds = torch.topk(logits, 5)[1]

        probs = F.softmax(logits, dim=-1).squeeze()

        # get top 5 class labels
        top_labels = np.array(IMAGENET_LABELS)[top_preds.cpu().numpy()]
        top_probs = probs[top_preds]

    print("Top 5 labels:")
    for i in range(top_probs.shape[1]):
        # print as percentage
        print(f"{top_labels[0, i]}: {top_probs[0, i] * 100:.1f}%")

    dump_pickle(logits.cpu().numpy(), f"temp/output_{str(prep_type)}_torch.pickle") # TODO remove

    # Show prediction
    plt.imshow((img[0].permute(1, 2, 0).cpu().numpy() / 255))
    plt.title(f"Label: {top_labels[0]}")
    plt.show()


if __name__ == "__main__":
    img_classify_example()
