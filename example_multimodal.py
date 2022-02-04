import base64
import os

import cv2
import imageio
import numpy as np
import scipy.io.wavfile

import torch
# import functional torch
import torch.nn.functional as F

from perceiver_io.multimodal_perceiver import MultiModalPerceiver

# Utilities to fetch videos from UCF101 dataset
from utils.kinetics_700_classes import KINETICS_CLASSES

from utils.utils import show_animation


def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]


def load_video(path, max_frames=0, resize=(224, 224)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames) / 255.0


sample_rate, audio = scipy.io.wavfile.read("output.wav")
if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 2 ** 15
elif audio.dtype != np.float32:
    raise ValueError('Unexpected datatype. Model expects sound samples to lie in [-1, 1]')

video_path = "./sample_data/video.avi"
video = load_video(video_path)

# Visualize inputs
#show_animation(video)

# @title Model construction
NUM_FRAMES = 16
AUDIO_SAMPLES_PER_FRAME = 48000 // 25
SAMPLES_PER_PATCH = 16
NUM_CLASSES = 700
IMG_SZ = 224

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

perceiver = MultiModalPerceiver(
    num_frames=NUM_FRAMES,
    audio_samples_per_frame=AUDIO_SAMPLES_PER_FRAME,
    audio_samples_per_patch=SAMPLES_PER_PATCH,
    num_classes=NUM_CLASSES,
    img_size=(IMG_SZ, IMG_SZ),
)
perceiver.eval()
perceiver.to(device)

ckpt_file = "./pytorch_checkpoints/video_autoencoding_checkpoint.pth"

# check if file exists
if not os.path.isfile(ckpt_file):
    raise ValueError("Please download the model checkpoint and place it in /pytorch_checkpoints")

checkpoint = torch.load(ckpt_file, map_location=device)

perceiver.load_state_dict(checkpoint['model_state_dict'])

video_input = torch.from_numpy(video[None, :16]).float().to(device)
audio_input = torch.from_numpy(audio[None, :16 * AUDIO_SAMPLES_PER_FRAME, 0:1]).float().to(device)

# Auto-encode the first 16 frames of the video and one of the audio channels
with torch.inference_mode():
    reconstruction = perceiver(video_input, audio_input)

from utils.utils import dump_pickle

output_torch = {k: reconstruction[k].numpy() for k in reconstruction.keys()}
dump_pickle(output_torch, "temp/output_multi_torch.pickle")


# Visualize reconstruction of first 16 frames
show_animation(reconstruction["image"][0])

# Kinetics 700 Labels
scores, indices = torch.top_k(F.softmax(reconstruction["label"]), 5)

for score, index in zip(scores[0], indices[0]):
    print("%s: %s" % (KINETICS_CLASSES[index], score))

# Auto-encode the entire video, one chunk at a time

# Partial video and audio into 16-frame chunks
nframes = video.shape[0]
# Truncate to be divisible by 16
nframes = nframes - (nframes % 16)

video_chunks = np.reshape(video[:nframes], [nframes // 16, 16, 224, 224, 3])
audio_chunks = np.reshape(audio[:nframes * AUDIO_SAMPLES_PER_FRAME],
                          [nframes // 16, 16 * AUDIO_SAMPLES_PER_FRAME, 2])

with torch.inference_mode():
    reconstruction = {"image": [], "audio": [], "label": None}
    for i in range(nframes // 16):
        output = perceiver(video_chunks[None, i], audio_chunks[None, i, :, 0:1])

        reconstruction["image"].append(output["image"])
        reconstruction["audio"].append(output["audio"])
        # TODO check what other implementations does here
        reconstruction["label"] = output["label"]

    reconstruction["image"] = torch.cat(reconstruction["image"], dim=1)
    reconstruction["audio"] = torch.cat(reconstruction["audio"], dim=1)



# Visualize reconstruction of entire video
show_animation(reconstruction["image"][0])
