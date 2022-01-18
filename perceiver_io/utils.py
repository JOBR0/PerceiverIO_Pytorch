import torch

def init_linear_from_haiku(linear_layer: torch.nn.Linear, haiku_params):
    with torch.no_grad():
        linear_layer.weight.copy_(torch.from_numpy(haiku_params['w'].T).float())
        linear_layer.bias.copy_(torch.from_numpy(haiku_params['b'].T).float())


def init_layer_norm_from_haiku(layer_norm: torch.nn.LayerNorm, haiku_params):
    with torch.no_grad():
        layer_norm.weight.copy_(torch.from_numpy(haiku_params['scale']).float())
        layer_norm.bias.copy_(torch.from_numpy(haiku_params['offset']).float())