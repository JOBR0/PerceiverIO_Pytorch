import math
from typing import Union, Sequence

import torch

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