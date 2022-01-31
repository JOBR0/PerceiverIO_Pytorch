import base64
import functools
import os
import pickle
import ssl
import re
import tempfile

from urllib import request

import cv2
import haiku as hk
import imageio
import jax
import jax.numpy as jnp
import numpy as np
import scipy.io.wavfile

import torch

from perceiver_io.multimodal_perceiver import MultiModalPerceiver

from IPython.display import HTML

from perceiver import perceiver, io_processors

# Utilities to fetch videos from UCF101 dataset
from utils.kinetics_700_classes import KINETICS_CLASSES


# Utilities to open video files using CV2
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


def to_gif(images):
    converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
    imageio.mimsave('./animation.gif', converted_images, fps=25)
    with open('./animation.gif', 'rb') as f:
        gif_64 = base64.b64encode(f.read()).decode('utf-8')
    return HTML('<img src="data:image/gif;base64,%s"/>' % gif_64)


def play_audio(data, sample_rate=48000):
    scipy.io.wavfile.write('tmp_audio.wav', sample_rate, data)

    with open('./tmp_audio.wav', 'rb') as f:
        audio_64 = base64.b64encode(f.read()).decode('utf-8')
    return HTML('<audio controls src="data:audio/wav;base64,%s"/>' % audio_64)





sample_rate, audio = scipy.io.wavfile.read("output.wav")
if audio.dtype == np.int16:
  audio = audio.astype(np.float32) / 2**15
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
reconstruction = perceiver(video_input, audio_input)

# Visualize reconstruction of first 16 frames
show_animation(reconstruction["image"][0])

# Kinetics 700 Labels
scores, indices = jax.lax.top_k(jax.nn.softmax(reconstruction["label"]), 5)

for score, index in zip(scores[0], indices[0]):
    print("%s: %s" % (KINETICS_CLASSES[index], score))

# Auto-encode the entire video, one chunk at a time

# Partial video and audio into 16-frame chunks
nframes = video.shape[0]
# Truncate to be divisible by 16
nframes = nframes - (nframes % 16)
video_chunks = jnp.reshape(video[:nframes], [nframes // 16, 16, 224, 224, 3])
audio_chunks = jnp.reshape(audio[:nframes * AUDIO_SAMPLES_PER_FRAME],
                           [nframes // 16, 16 * AUDIO_SAMPLES_PER_FRAME, 2])

encode = jax.jit(functools.partial(autoencode_video, params, state, rng))

# Logically, what we do is the following code. We write out the loop to allocate
# GPU memory for only one chunk
#
# reconstruction = jax.vmap(encode, in_axes=1, out_axes=1)(
#     video_chunks[None, :], audio_chunks[None, :, :, 0:1])

chunks = []
for i in range(nframes // 16):
    reconstruction = encode(video_chunks[None, i], audio_chunks[None, i, :, 0:1])
    chunks.append(jax.tree_map(lambda x: np.array(x), reconstruction))

reconstruction = jax.tree_multimap(lambda *args: np.stack(args, axis=1),
                                   *chunks)

reconstruction = jax.tree_map(lambda x: np.reshape(x, [-1] + list(x.shape[2:])), reconstruction)

# Visualize reconstruction of entire video
table([to_gif(reconstruction['image'][0]), play_audio(np.array(reconstruction['audio'][0]))])
