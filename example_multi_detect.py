import os
import warnings

import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import transforms

from perceiver_io.mullti_detect_perceiver import MultiDetectPerceiver, PrepType
from utils.utils import load_image
from utils.imagenet_labels import IMAGENET_LABELS

import torch.nn.functional as F


def img_classify_example():
    device = "cpu" #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    MEAN_RGB = (0.485 * 255, 0.456 * 255, 0.406 * 255)
    STDDEV_RGB = (0.229 * 255, 0.224 * 255, 0.225 * 255)

    img_size = (224, 224)

    normalize = transforms.Normalize(MEAN_RGB, STDDEV_RGB)


    perceiver = MultiDetectPerceiver(num_classes=1000,
                                        img_size=img_size)

    perceiver.eval()
    perceiver.to(device)

    ckpt_file = "./pytorch_checkpoints/imagenet_conv_preprocessing_multi_detect.pth"

    # check if file exists
    if not os.path.isfile(ckpt_file):
        raise ValueError("Please download the model checkpoint and place it in /pytorch_checkpoints")

    checkpoint = torch.load(ckpt_file, map_location=device)
    result = perceiver.load_state_dict(checkpoint["model_state_dict"], strict=False)

    if result.missing_keys:
        warnings.warn(f"Missing keys while loading model weights: {result.missing_keys}")
    if result.unexpected_keys:
        warnings.warn(f"Unexpected keys while loading model weights: {result.unexpected_keys}")



    img_names = ["training_processed_segment-10017090168044687777_6380_000_6400_000_with_camera_labels_frame_0_cam_0.jpg",
            "training_processed_segment-10017090168044687777_6380_000_6400_000_with_camera_labels_frame_0_cam_1.jpg",
            "training_processed_segment-10017090168044687777_6380_000_6400_000_with_camera_labels_frame_0_cam_2.jpg",
            "training_processed_segment-10017090168044687777_6380_000_6400_000_with_camera_labels_frame_0_cam_3.jpg",
            "training_processed_segment-10017090168044687777_6380_000_6400_000_with_camera_labels_frame_0_cam_4.jpg"]

    img_names = ["dalmation_big.jpg",
                 "dalmation_big.jpg",
                 "dalmation_small.jpg",
                 "dalmation_big.jpg",
                 "dalmation_small.jpg"]

    imgs = []
    for img_name in img_names:
        imgs.append(load_image(os.path.join("C:/Users/Jonas/Code/temp", img_name), device))

    normed_imgs = []


    for img in imgs:
        h, w = img.shape[2:]

        if h == 1280:
            img_size = (163, 224)
        elif h == 886:
            img_size = (103, 224)
        else:
            raise Exception

        img_norm = transforms.functional.resize(img, size=img_size)

        img_norm = normalize(img_norm)
        img_norm = img_norm.to(device)

        normed_imgs.append(img_norm)


    with torch.inference_mode():
        logits = perceiver(normed_imgs)
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

    # Show prediction
    plt.imshow((img[0].permute(1, 2, 0).cpu().numpy() / 255))
    plt.title(f"Label: {top_labels[0]}")
    plt.show()


if __name__ == "__main__":
    img_classify_example()
