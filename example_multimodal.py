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

from perceiver_io.video_perceiver import MultiModalPerceiver

from IPython.display import HTML

from perceiver import perceiver, io_processors

# Utilities to fetch videos from UCF101 dataset
from utils.kinetics_700_classes import KINETICS_CLASSES


# Utilities to open video files using CV2
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


def table(elements):
    row = ['<td>%s</td>' % el.data for el in elements]
    return HTML('<table><tr>%s</tr></table>' % ''.join(row))


sample_rate, audio = scipy.io.wavfile.read("output.wav")
if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 2 ** 15
elif audio.dtype != np.float32:
    raise ValueError('Unexpected datatype. Model expects sound samples to lie in [-1, 1]')

video_path = "./sample_data/video.avi"
video = load_video(video_path)

# Visualize inputs
table([to_gif(video), play_audio(audio)])

# @title Model construction
NUM_FRAMES = 16
AUDIO_SAMPLES_PER_FRAME = 48000 // 25
SAMPLES_PER_PATCH = 16
NUM_CLASSES = 700
IMG_SZ = 56

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

perceiver = MultiModalPerceiver()

perceiver.eval()


def autoencode_video(images, audio):
    nchunks = 128
    reconstruction = {}
    for chunk_idx in range(nchunks):
        image_chunk_size = np.prod(images.shape[1:-1]) // nchunks
        audio_chunk_size = audio.shape[1] // SAMPLES_PER_PATCH // nchunks
        subsampling = {
            'image': jnp.arange(
                image_chunk_size * chunk_idx, image_chunk_size * (chunk_idx + 1)),
            'audio': jnp.arange(
                audio_chunk_size * chunk_idx, audio_chunk_size * (chunk_idx + 1)),
            'label': None,
        }
        output = perceiver(images, audio, subsampling)

        reconstruction['label'] = output['label']
        if 'image' not in reconstruction:
            reconstruction['image'] = output['image']
            reconstruction['audio'] = output['audio']
        else:
            reconstruction['image'] = jnp.concatenate(
                [reconstruction['image'], output['image']], axis=1)
            reconstruction['audio'] = jnp.concatenate(
                [reconstruction['audio'], output['audio']], axis=1)

    reconstruction['image'] = jnp.reshape(reconstruction['image'], images.shape)
    reconstruction['audio'] = jnp.reshape(reconstruction['audio'], audio.shape)
    return reconstruction


# Auto-encode the first 16 frames of the video and one of the audio channels
reconstruction = autoencode_video(video[None, :16], audio[None, :16 * AUDIO_SAMPLES_PER_FRAME, 0:1])

# Visualize reconstruction of first 16 frames
table([to_gif(reconstruction["image"][0]), play_audio(np.array(reconstruction["audio"][0]))])

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
