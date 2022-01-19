import math
from typing import Union, Sequence

import torch
from PIL import Image
import numpy as np

def load_image(imfile, device):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(device)


def same_padding(input_size: Sequence[int], kernel_size: Union[int, Sequence[int]],
                      stride: Union[int, Sequence[int]] = 1, dims: int = 2):
    """
    Calculates the padding for a convolution with same padding and stride
    If the padding isn't divisible by two, right and bottom will be 1 pixel larger
    Args:
        kernel_size (int): Size of kernel
        stride (int): stride of convolution
        dims (int): Number of dimension over which to perform convolution. Default: 2 for conv2D
    """
    if type(kernel_size) is int:
        kernel_size = [kernel_size] * dims
    if type(stride) is int:
        stride = [stride] * dims

    padding = []
    for d in range(dims-1, 0, -1):
        if input_size[d] % stride[d] == 0:
            total_padding = kernel_size[d] - stride[d]
        else:
            total_padding = kernel_size[d] - (input_size[d] % stride[d])

        left_padding = math.floor(total_padding / 2)
        right_padding = math.ceil(total_padding / 2)
        padding.append(left_padding)
        padding.append(right_padding)
    return padding

def conv_output_shape(input_size: Sequence[int],
                      kernel_size: Union[int, Sequence[int]],
                      stride: Union[int, Sequence[int]] = 1,
                      padding: Union[int, Sequence[int]] = 0,
                      dilation: Union[int, Sequence[int]] = 1,
                      dims: int = 2):
    """
    Calculates the output shape of a tensor for a convolution
    Args:
        input_size (Sequence[int]): Shape of input tensor
        kernel_size (int or Sequence[int]): Size of kernel
        stride (int or Sequence[int]): stride of convolution. Default: 1
        padding (int or Sequence[int]): padding of convolution. Default: 0
        dilation (int or Sequence[int]): dilation of convolution. Default: 1
        dims (int): Number of dimension over which to perform convolution. Default: 2 for conv2D
    """

    # Dimensions which stay the same
    skip_dims = len(input_size) - dims

    # Create lists from integer arguments
    if type(kernel_size) is int:
        kernel_size = [kernel_size] * dims
    if type(stride) is int:
        stride = [stride] * dims
    if type(padding) is int:
        padding = [padding] * dims
    if type(dilation) is int:
        dilation = [dilation] * dims

    output_size = list(input_size[:skip_dims])
    for i in range(len(input_size)):
        out = math.floor(
            (input_size[skip_dims + i] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) / stride[i] + 1)
        output_size.append(out)

    return output_size

def init_linear_from_haiku(linear_layer: torch.nn.Linear, haiku_params):
    with torch.no_grad():
        linear_layer.weight.copy_(torch.from_numpy(haiku_params['w'].T).float())
        linear_layer.bias.copy_(torch.from_numpy(haiku_params['b'].T).float())


def init_layer_norm_from_haiku(layer_norm: torch.nn.LayerNorm, haiku_params):
    with torch.no_grad():
        layer_norm.weight.copy_(torch.from_numpy(haiku_params['scale']).float())
        layer_norm.bias.copy_(torch.from_numpy(haiku_params['offset']).float())