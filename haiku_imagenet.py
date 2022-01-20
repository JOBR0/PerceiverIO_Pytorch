import functools
import itertools
import pickle

import cv2
import haiku as hk
import imageio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from perceiver import perceiver, io_processors

from imagenet_labels import IMAGENET_LABELS

# One of learned_position_encoding, fourier_position_encoding, or conv_preprocessing'
# learned_position_encoding: Uses a learned position encoding over the image
#     and 1x1 convolution over the pixels
# fourier_position_encoding: Uses a 2D fourier position encoding
#     and the raw pixels
# conv_preprocessing: Uses a 2D fourier position encoding
#     and a 2D conv-net as preprocessing
model_type = 'conv_preprocessing' #@param ['learned_position_encoding', 'fourier_position_encoding', 'conv_preprocessing']

IMAGE_SIZE = (224, 224)

learned_pos_configs = dict(
    input_preprocessor=dict(
        position_encoding_type='trainable',
        trainable_position_encoding_kwargs=dict(
            init_scale=0.02,
            num_channels=256,
        ),
        prep_type='conv1x1',
        project_pos_dim=256,
        num_channels=256,
        spatial_downsample=1,
        concat_or_add_pos='concat',
    ),
    encoder=dict(
        cross_attend_widening_factor=1,
        cross_attention_shape_for_attn='kv',
        dropout_prob=0,
        num_blocks=8,
        num_cross_attend_heads=1,
        num_self_attend_heads=8,
        num_self_attends_per_block=6,
        num_z_channels=1024,
        self_attend_widening_factor=1,
        use_query_residual=True,
        z_index_dim=512,
        z_pos_enc_init_scale=0.02
    ),
    decoder=dict(
        num_z_channels=1024,
        position_encoding_type='trainable',
        trainable_position_encoding_kwargs=dict(
            init_scale=0.02,
            num_channels=1024,
        ),
        use_query_residual=False,
    )
)

fourier_pos_configs = dict(
    input_preprocessor=dict(
        position_encoding_type='fourier',
        fourier_position_encoding_kwargs=dict(
            concat_pos=True,
            max_resolution=(224, 224),
            num_bands=64,
            sine_only=False
        ),
        prep_type='pixels',
        spatial_downsample=1,
    ),
    encoder=dict(
        cross_attend_widening_factor=1,
        cross_attention_shape_for_attn='kv',
        dropout_prob=0,
        num_blocks=8,
        num_cross_attend_heads=1,
        num_self_attend_heads=8,
        num_self_attends_per_block=6,
        num_z_channels=1024,
        self_attend_widening_factor=1,
        use_query_residual=True,
        z_index_dim=512,
        z_pos_enc_init_scale=0.02
    ),
    decoder=dict(
        num_z_channels=1024,
        position_encoding_type='trainable',
        trainable_position_encoding_kwargs=dict(
            init_scale=0.02,
            num_channels=1024,
        ),
        use_query_residual=True,
    )
)

conv_maxpool_configs = dict(
    input_preprocessor=dict(
        position_encoding_type='fourier',
        fourier_position_encoding_kwargs=dict(
            concat_pos=True,
            max_resolution=(56, 56),
            num_bands=64,
            sine_only=False
        ),
        prep_type='conv',
    ),
    encoder=dict(
        cross_attend_widening_factor=1,
        cross_attention_shape_for_attn='kv',
        dropout_prob=0,
        num_blocks=8,
        num_cross_attend_heads=1,
        num_self_attend_heads=8,
        num_self_attends_per_block=6,
        num_z_channels=1024,
        self_attend_widening_factor=1,
        use_query_residual=True,
        z_index_dim=512,
        z_pos_enc_init_scale=0.02
    ),
    decoder=dict(
        num_z_channels=1024,
        position_encoding_type='trainable',
        trainable_position_encoding_kwargs=dict(
            init_scale=0.02,
            num_channels=1024,
        ),
        use_query_residual=True,
    )
)

CONFIGS = {
    'learned_position_encoding': learned_pos_configs,
    'fourier_position_encoding': fourier_pos_configs,
    'conv_preprocessing': conv_maxpool_configs,
}

def imagenet_classifier(config, images):
  input_preprocessor = io_processors.ImagePreprocessor(
      **config['input_preprocessor'])
  encoder = perceiver.PerceiverEncoder(**config['encoder'])
  decoder = perceiver.ClassificationDecoder(
      1000,
      **config['decoder'])
  model = perceiver.Perceiver(
      encoder=encoder,
      decoder=decoder,
      input_preprocessor=input_preprocessor)
  logits = model(images, is_training=False)
  return logits


imagenet_classifier = hk.transform_with_state(imagenet_classifier)

rng = jax.random.PRNGKey(42)





rng = jax.random.PRNGKey(42)
with open("haiku_models/imagenet_conv_preprocessing.pystate", 'rb') as f:
  ckpt = pickle.loads(f.read())

params = ckpt['params']
state = ckpt['state']


with open('sample_data/dalmation.jpg', 'rb') as f:
  img = imageio.imread(f)


MEAN_RGB = (0.485 * 255, 0.456 * 255, 0.406 * 255)
STDDEV_RGB = (0.229 * 255, 0.224 * 255, 0.225 * 255)

def normalize(im):
  return (im - np.array(MEAN_RGB)) / np.array(STDDEV_RGB)

def resize_and_center_crop(image):
  """Crops to center of image with padding then scales."""
  shape = image.shape

  image_height = shape[0]
  image_width = shape[1]

  padded_center_crop_size = ((224 / (224 + 32)) *
       np.minimum(image_height, image_width).astype(np.float32)).astype(np.int32)

  offset_height = ((image_height - padded_center_crop_size) + 1) // 2
  offset_width = ((image_width - padded_center_crop_size) + 1) // 2
  crop_window = [offset_height, offset_width,
                 padded_center_crop_size, padded_center_crop_size]

  # image = tf.image.crop_to_bounding_box(image_bytes, *crop_window)
  image = image[crop_window[0]:crop_window[0] + crop_window[2], crop_window[1]:crop_window[1]+crop_window[3]]
  return cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)

# Imagenet classification

# Obtain a [224, 224] crop of the image while preserving aspect ratio.
# With Fourier position encoding, no resize is needed -- the model can
# generalize to image sizes it never saw in training
centered_img = resize_and_center_crop(img)  # img
logits, _ = imagenet_classifier.apply(params, state, rng, CONFIGS[model_type], normalize(centered_img)[None])


_, indices = jax.lax.top_k(logits[0], 5)
probs = jax.nn.softmax(logits[0])

plt.imshow(img)
plt.axis('off')
print('Top 5 labels:')
for i in list(indices):
  print(f'{IMAGENET_LABELS[i]}: {probs[i]}')