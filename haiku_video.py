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

from IPython.display import HTML

from perceiver import perceiver, io_processors

# Utilities to fetch videos from UCF101 dataset
from utils.kinetics_700_classes import KINETICS_CLASSES

UCF_ROOT = 'https://www.crcv.ucf.edu/THUMOS14/UCF101/UCF101/'
_VIDEO_LIST = None
_CACHE_DIR = tempfile.mkdtemp()
# As of July 2020, crcv.ucf.edu doesn't use a certificate accepted by the
# default Colab environment anymore.
unverified_context = ssl._create_unverified_context()


def list_ucf_videos():
    """Lists videos available in UCF101 dataset."""
    global _VIDEO_LIST
    if not _VIDEO_LIST:
        index = request.urlopen(UCF_ROOT, context=unverified_context).read().decode('utf-8')
        videos = re.findall('(v_[\w_]+\.avi)', index)
        _VIDEO_LIST = sorted(set(videos))
    return list(_VIDEO_LIST)


def fetch_ucf_video(video):
    """Fetchs a video and cache into local filesystem."""
    cache_path = os.path.join("./", video)
    if not os.path.exists(cache_path):
        urlpath = request.urljoin(UCF_ROOT, video)
        print('Fetching %s => %s' % (urlpath, cache_path))
        data = request.urlopen(urlpath, context=unverified_context).read()
        open(cache_path, "wb").write(data)
    return cache_path


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


#@title Load video and audio from UCF

video_names = list_ucf_videos()
video_path = fetch_ucf_video(video_names[0])

# Extract audio using FFMPEG and encode as pcm float wavfile (only format readable by scipy.io.wavfile).
#!yes | ffmpeg -i "$video_path"  -c copy  -f wav -map 0:a pcm_f32le -ar 48000 output.wav

sample_rate, audio = scipy.io.wavfile.read("output.wav")
if audio.dtype == np.int16:
  audio = audio.astype(np.float32) / 2**15
elif audio.dtype != np.float32:
  raise ValueError('Unexpected datatype. Model expects sound samples to lie in [-1, 1]')

video = load_video(video_path)

# Visualize inputs
table([to_gif(video), play_audio(audio)])

# @title Model construction
NUM_FRAMES = 16
AUDIO_SAMPLES_PER_FRAME = 48000 // 25
SAMPLES_PER_PATCH = 16
NUM_CLASSES = 700
IMG_SZ = 56


def video_autoencoder(images, audio, subsampling):
    n_audio_samples = NUM_FRAMES * AUDIO_SAMPLES_PER_FRAME
    input_preprocessor = io_processors.MultimodalPreprocessor(
        min_padding_size=4,
        modalities={
            'audio': io_processors.AudioPreprocessor(
                position_encoding_type='fourier',
                fourier_position_encoding_kwargs=dict(
                    num_bands=192,
                    max_resolution=(n_audio_samples,),
                    sine_only=False,
                    concat_pos=True,
                ),
                n_extra_pos_mlp=0,
                prep_type='patches',
                samples_per_patch=16),
            'image': io_processors.ImagePreprocessor(
                position_encoding_type='fourier',
                fourier_position_encoding_kwargs=dict(
                    num_bands=32,
                    max_resolution=(NUM_FRAMES, IMG_SZ, IMG_SZ),
                    sine_only=False,
                    concat_pos=True,
                ),
                n_extra_pos_mlp=0,
                prep_type='patches',
                spatial_downsample=4,
                temporal_downsample=1),
            'label': io_processors.OneHotPreprocessor(),
        },
        mask_probs={'image': 0.0, 'audio': 0.0, 'label': 1.0},
    )

    output_postprocessor = io_processors.MultimodalPostprocessor(
        modalities={
            'audio': io_processors.AudioPostprocessor(
                samples_per_patch=SAMPLES_PER_PATCH),
            'image': io_processors.ProjectionPostprocessor(
                num_outputs=3),
            'label': io_processors.ClassificationPostprocessor(
                num_classes=NUM_CLASSES),
        })

    encoder = perceiver.PerceiverEncoder(
        num_self_attends_per_block=8,
        # Weights won't be shared if num_blocks is set to 1.
        num_blocks=1,
        z_index_dim=28 * 28 * 1,
        num_z_channels=512,
        num_cross_attend_heads=1,
        num_self_attend_heads=8,
        cross_attend_widening_factor=1,
        self_attend_widening_factor=1,
        dropout_prob=0.0,
        z_pos_enc_init_scale=0.02,
        cross_attention_shape_for_attn='kv',
        name='encoder')

    subsampled_index_dims = {
        'audio': subsampling['audio'].shape[0],
        'image': subsampling['image'].shape[0],
        'label': 1,
    }
    image_decoder = perceiver.BasicVideoAutoencodingDecoder(
        # Autoencoding, don't pass inputs to the queries.
        concat_preprocessed_input=False,
        subsampled_index_dims=subsampling['image'],
        output_shape=images.shape[:4],
        num_z_channels=1024,
        output_num_channels=512,
        use_query_residual=False,
        position_encoding_type='fourier',
        fourier_position_encoding_kwargs=dict(
            num_bands=32,
            max_resolution=(NUM_FRAMES, IMG_SZ, IMG_SZ),
            sine_only=False,
            concat_pos=True,
        ),
    )

    decoder = perceiver.MultimodalDecoder(
        # Autoencoding, don't pass inputs to the queries.
        concat_preprocessed_input=False,
        subsampled_index_dims=subsampled_index_dims,
        # Modality specific decoders are used ONLY to generate queries.
        # All modalties are decoded together using a unified decoder.
        modalities={
            'audio': perceiver.BasicDecoder(
                # Autoencoding, don't pass inputs to the queries.
                concat_preprocessed_input=False,
                subsampled_index_dims=subsampling['audio'],
                output_index_dims=(n_audio_samples // SAMPLES_PER_PATCH,),
                num_z_channels=1024,
                output_num_channels=512,
                use_query_residual=False,
                position_encoding_type='fourier',
                fourier_position_encoding_kwargs=dict(
                    num_bands=192,
                    max_resolution=(n_audio_samples,),
                    sine_only=False,
                    concat_pos=True,
                ),
            ),
            'image': image_decoder,
            'label': perceiver.ClassificationDecoder(
                # Autoencoding, don't pass inputs to the queries.
                concat_preprocessed_input=False,
                num_classes=NUM_CLASSES,
                num_z_channels=1024,
                use_query_residual=False,
                position_encoding_type='trainable',
                trainable_position_encoding_kwargs=dict(
                    num_channels=1024,
                    init_scale=0.02,
                ),
            ),
        },
        num_outputs=None,
        output_num_channels=512,
        use_query_residual=False, )

    model = perceiver.Perceiver(
        input_preprocessor=input_preprocessor,
        encoder=encoder,
        decoder=decoder,
        output_postprocessor=output_postprocessor)

    return model({'image': images,
                  'audio': audio,
                  'label': np.zeros((images.shape[0], 700))},
                 is_training=False, subsampled_output_points=subsampling)


video_autoencoder = hk.transform_with_state(video_autoencoder)


# @title Model application


def autoencode_video(params, state, rng, images, audio):
    #image shape  = (batch_size, num_frames, img_sz, img_sz, 3)
    #audio shape  = (batch_size, num_sample, 1)

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
        output, state = video_autoencoder.apply(
            params, state, rng, images, audio, subsampling)
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

#@title Load parameters from checkpoint


rng = jax.random.PRNGKey(42)
with open("haiku_checkpoints/video_autoencoding_checkpoint.pystate", "rb") as f:
  params = pickle.loads(f.read())

state = {}

# Auto-encode the first 16 frames of the video and one of the audio channels
reconstruction = autoencode_video(params, state, rng, video[None, :16], audio[None, :16*AUDIO_SAMPLES_PER_FRAME, 0:1])

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