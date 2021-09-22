import math
import warnings

import torch.nn as nn
import torch.nn.functional as F
import torch

from timm.models.layers import variance_scaling_

from core.utils.utils import init_linear_from_haiku, init_layer_norm_from_haiku


def make_cross_attention_mask(query_mask, kv_mask):
    batch_size, query_len = query_mask.shape
    _, key_len = kv_mask.shape
    mask = torch.vmap(torch.outer)(query_mask, kv_mask)
    assert mask.shape == (batch_size, query_len, key_len)
    return mask


class Attention(nn.Module):
    """Multi-headed {cross, self}-attention.
    Args:
        q_in_channels (int): Number of channels in queries.
        k_in_channels (int): Number of channels in keys. Default: q_in_channels if None
        v_in_channels (int): Number of channels int values. Default: k_in_channel if None
        num_heads (int): Number of attention heads. Default: 8
        init_scale (float): Scale for variance_scaling_ of q, k and v projections
        with_final_bias (bool): Whether to have a bias in the final linear layer
        final_init_scale_multiplier (float): Multiplier for scale for variance_scaling_ for final layer. Default: 1.0
        dropout_prob (float): Dropout probability. Default: 0.0
        qk_out_channels (int): Number of channels to which queries and keys are projected. Default: q_in_channels if None
        v_out_channels (int): Number of channels to which values are projected. Default: qk_out_channels if None
        output_channels (int): Number of output channels. Default: v_out_channels if None
    """

    def __init__(self,
                 q_in_channels: int,
                 k_in_channels: int = None,
                 v_in_channels: int = None,
                 num_heads: int = 8,
                 init_scale: float = 1.0,
                 with_final_bias: bool = True,
                 final_init_scale_multiplier: float = 1.,
                 dropout_prob: float = 0.0,
                 qk_out_channels: int = None,
                 v_out_channels: int = None,
                 output_channels: int = None):
        super().__init__()
        self._num_heads = num_heads
        final_init_scale = final_init_scale_multiplier * init_scale

        # Q and K must have the same number of channels.
        # Default to preserving Q's input's shape.
        if qk_out_channels is None:
            qk_out_channels = q_in_channels
        # V's num_channels determines the shape of the output of QKV-attention.
        # Default to the same number of channels used in the key-query operation.
        if v_out_channels is None:
            v_out_channels = qk_out_channels
        # Project the output of QKV attention to a desired number of channels.
        # Default to the same number as the output of the QKV attention operation.
        if output_channels is None:
            output_channels = v_out_channels

        self._qk_channels_per_head = qk_out_channels // num_heads
        self._v_channels_per_head = v_out_channels // num_heads

        if qk_out_channels % num_heads != 0:
            raise ValueError(f'qk_out_channels ({qk_out_channels}) must be divisible by'
                             f' num_heads ({num_heads}).')
        if v_out_channels % num_heads != 0:
            raise ValueError(f'v_channels ({v_out_channels}) must be divisible by'
                             f' num_heads ({num_heads}).')

        self.proj_q = nn.Linear(q_in_channels, qk_out_channels, bias=True)
        self.proj_k = nn.Linear(k_in_channels, qk_out_channels, bias=True)
        self.proj_v = nn.Linear(v_in_channels, v_out_channels, bias=True)

        variance_scaling_(self.proj_q.weight, scale=init_scale, mode='fan_in', distribution='truncated_normal')
        nn.init.constant_(self.proj_q.bias, 0)
        variance_scaling_(self.proj_k.weight, scale=init_scale, mode='fan_in', distribution='truncated_normal')
        nn.init.constant_(self.proj_k.bias, 0)
        variance_scaling_(self.proj_v.weight, scale=init_scale, mode='fan_in', distribution='truncated_normal')
        nn.init.constant_(self.proj_v.bias, 0)

        self.dropout = nn.Dropout(dropout_prob)

        self.final = nn.Linear(v_out_channels, output_channels, bias=with_final_bias)
        variance_scaling_(self.final.weight, scale=final_init_scale, mode='fan_in', distribution='truncated_normal')
        nn.init.constant_(self.final.bias, 0)

    def forward(self, inputs_q, inputs_k, inputs_v, attention_mask=None, attention_bias = None, return_matrix=False):

        # Project QKV to a common feature dimension.
        q = self.proj_q(inputs_q)
        k = self.proj_k(inputs_k)
        v = self.proj_v(inputs_v)

        # Reshape channels for multi-head attention.
        batch, q_time, _ = q.shape
        _, kv_time, _ = k.shape
        q = torch.reshape(q, [batch, q_time, self._num_heads, self._qk_channels_per_head])
        k = torch.reshape(k, [batch, kv_time, self._num_heads, self._qk_channels_per_head])
        v = torch.reshape(v, [batch, kv_time, self._num_heads, self._v_channels_per_head])

        result = self.attend(q, k, v, attention_mask=attention_mask, attention_bias=attention_bias, return_matrix = return_matrix)

        if return_matrix:
            attention_matrix, result = result

        result = self.final(result)

        if return_matrix:
            return attention_matrix, result

        return result

    def attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attention_mask: bool = None, attention_bias = None, return_matrix: bool=False):
        """Computes multi-head attention using a query, key and value.
      Args:
        q: Query with shape [batch, q_indices, num_heads, head_dim].
        k: Key with shape [batch, kv_indices, num_heads, head_dim].
        v: Value with shape [batch, kv_indices, num_heads, head_dim].
        attention_mask: Array of shape [batch, q_indices, kv_indices] indicating
          which attentions are valid
      Returns:
        Output of the attention with shape [batch, q_indices, hiddens]
      """
        batch, q_indices, num_heads, q_head_dim = q.shape
        _, _, _, v_head_dim = v.shape
        hiddens = num_heads * v_head_dim

        #attention = torch.einsum('bthd,bThd->bhtT', q, k)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attention = (q @ k.transpose(-2, -1))


        if attention_bias is not None:
            attention += attention_bias

        scale = 1. / math.sqrt(q_head_dim)
        attention *= scale

        if attention_mask is not None:
            # Use large_k instead of np.NINF because np.NINF breaks for causal-masked
            # left-padded sampling.
            large_k = torch.tensor(1e4 if attention.dtype == torch.float16 else 1e30,
                                   dtype=attention.dtype)

            attention = torch.where(attention_mask[:, None, :, :], attention,
                                    -large_k)

        normalized = F.softmax(attention, dim=-1)

        self.dropout(normalized)

        #summed = torch.einsum('bhtT,bThd->bthd', normalized, v)
        summed = normalized @ v
        summed = summed.permute(0, 2, 1, 3)


        summed = torch.reshape(summed, [batch, q_indices, hiddens])

        if attention_mask is not None:
            # If all attended tokens are masked, or for masked tokens
            # some rows of logits gets completely masked, in which case the softmax
            # gives a uniform row and we obtain non-zero outputs where it should be
            # zero. We force zeros.
            wipe_attn = torch.all(
                attention_mask == 0, axis=2, keepdims=True)  # shape (B, T, 1)
            summed = torch.where(wipe_attn, torch.zeros_like(summed), summed)

        if return_matrix:
            return normalized, summed

        return summed

    def set_haiku_params(self, params):
        init_linear_from_haiku(self.proj_q, params.pop("linear"))
        init_linear_from_haiku(self.proj_k, params.pop("linear_1"))
        init_linear_from_haiku(self.proj_v, params.pop("linear_2"))
        init_linear_from_haiku(self.final, params.pop("linear_3"))

        if len(params) != 0:
            warnings.warn(f"Some parameters couldn't be matched to model: {params.keys()}")


class MLP(nn.Module):
    """A Transformer-style dense module to follow attention.
    Args:
        in_channels (int): Number of channels in input.
        out_channels (int): Number of channels in output. Default: set to in_channels if None
        widening_factor (int): Number of channels between layers are widening_factor*in_channels. Default: 4
        dropout_prob (float): Dropout probability. Default: 0.0
        init_scale (float): Scale for variance scaling initialization. Default: 1.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int = None,
                 widening_factor: int = 4,
                 dropout_prob: float = 0.0,
                 init_scale: float = 1.):
        super().__init__()
        # Make out channels equal to in_channels if not specified
        out_channels = out_channels or in_channels

        self.fc1 = nn.Linear(in_channels, widening_factor * in_channels)
        variance_scaling_(self.fc1.weight, scale=init_scale, mode='fan_in', distribution='truncated_normal')
        nn.init.constant_(self.fc1.bias, 0)
        self.fc2 = nn.Linear(widening_factor * in_channels, out_channels)
        variance_scaling_(self.fc2.weight, scale=init_scale, mode='fan_in', distribution='truncated_normal')
        nn.init.constant_(self.fc2.bias, 0)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return self.dropout(x)

    def set_haiku_params(self, params):
        init_linear_from_haiku(self.fc1, params.pop("linear"))
        init_linear_from_haiku(self.fc2, params.pop("linear_1"))

        if len(params) != 0:
            warnings.warn(f"Some parameters couldn't be matched to model: {params.keys()}")


class SelfAttention(nn.Module):
    """A self-attention module, including a dense block.
    Args:
        in_channels (int): number of channels of inputs
        widening_factor (int): Widening factor for MLP. Default: 4
        dropout_prob (float): Dropout probability. Default: 0.0
        dropout_attn_prob (float): Dropout probability for attention. Default: 0.0
        num_heads (int): Number of attention heads. Default: 8
        att_init_scale (float):
        dense_init_scale (float):
        qk_channels (int): Number of channels for queries and keys. Default: If None set to in_channels
        v_channels (int): Number of channels for values. Default: If None set to qk_channels
    """

    def __init__(self,
                 in_channels: int,
                 widening_factor: int = 4,
                 dropout_prob: float = 0.0,
                 dropout_attn_prob: float = 0.0,
                 num_heads: int = 8,
                 att_init_scale: float = 1.0,
                 dense_init_scale: float = 1.0,
                 qk_channels: int = None,
                 v_channels: int = None):
        super().__init__()

        # Q and K must have the same number of channels.
        # Default to preserving Q's input's shape.
        if qk_channels is None:
            qk_channels = in_channels
        # V's num_channels determines the shape of the output of QKV-attention.
        # Default to the same number of channels used in the key-query operation.
        if v_channels is None:
            v_channels = qk_channels

        self.mlp = MLP(
            in_channels=v_channels,
            widening_factor=widening_factor,
            dropout_prob=dropout_prob,
            init_scale=dense_init_scale)

        self.attention = Attention(
            q_in_channels=in_channels,
            k_in_channels=in_channels,
            v_in_channels=in_channels,
            num_heads=num_heads,
            init_scale=att_init_scale,
            qk_out_channels=qk_channels,
            v_out_channels=v_channels,
            dropout_prob=dropout_attn_prob)

        self.layer_norm1 = nn.LayerNorm(in_channels)
        self.layer_norm2 = nn.LayerNorm(v_channels)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self,
                inputs,
                *,
                attention_mask=None,
                attention_bias=None,
                return_matrix: bool=False):
        x = inputs
        qkv_inputs = self.layer_norm1(inputs)
        attention = self.attention(qkv_inputs, qkv_inputs, qkv_inputs,
                                   attention_mask=attention_mask, attention_bias=attention_bias,
                                   return_matrix=return_matrix)
        if return_matrix:
            attention_matrix, attention = attention

        attention = self.dropout(attention)
        x = x + attention

        x = x + self.mlp(self.layer_norm2(x))

        if return_matrix:
            return attention_matrix, x

        return x

    def set_haiku_params(self, params):
        mlp_params = {key[key.find('/') + 1:]: params.pop(key) for key in list(params.keys()) if
                      key.startswith("mlp")}
        self.mlp.set_haiku_params(mlp_params)

        attention_params = {key[key.find('/') + 1:]: params.pop(key) for key in list(params.keys()) if
                            key.startswith("attention")}
        self.attention.set_haiku_params(attention_params)

        init_layer_norm_from_haiku(self.layer_norm1, params.pop("layer_norm"))
        init_layer_norm_from_haiku(self.layer_norm2, params.pop("layer_norm_1"))

        if len(params) != 0:
            warnings.warn(f"Some parameters couldn't be matched to model: {params.keys()}")


class CrossAttention(nn.Module):
    """A cross-attention module, including a dense block.
    Args:
        q_in_channels (int): Number of channels for queries.
        kv_in_channels (int): Number of channels for keys and values.
        widening_factor (int): Widening factor for MLP. Default: 1
        dropout_prob (float): Dropout probability. Default: 0.0
        dropout_attn_prob (float): Dropout probability for attention. Default: 0.0
        num_heads (int): Number of attention heads. Default: 8
        attn_init_scale (float): Scale for Variance scaling initialization of final linear layer in Attention. Default: 1.0
        mlp_init_scale (float): Scale for Variance scaling initialization of MLP. Default: 1.0
        shape_for_attn (str): Number of channels used for key and query scalar product. "kv" for same number as keys and values. "q" for same number as queries.
            Ignored if qk_channels is not None. Default: "kv"
        use_query_residual (bool):  Add query to attention output
        qk_channels (int): Number of channels for queries and keys. Default: If None set to q_in_channels
        v_channels (int) Number of channels for values
        """

    def __init__(self,
                 q_in_channels: int,
                 kv_in_channels: int,
                 widening_factor: int = 1,
                 dropout_prob: float = 0.0,
                 dropout_attn_prob: float = 0.0,
                 num_heads: int = 8,
                 attn_init_scale: float = 1.0,
                 mlp_init_scale: float = 1.0,
                 shape_for_attn: str = 'kv',
                 use_query_residual: bool = True,
                 qk_channels: int = None,
                 v_channels: int = None):
        super().__init__()

        self._use_query_residual = use_query_residual

        output_channels = q_in_channels
        if qk_channels is None:
            if shape_for_attn == 'q':
                qk_channels = q_in_channels
            elif shape_for_attn == 'kv':
                qk_channels = kv_in_channels
            else:
                raise ValueError(f'Unknown value {shape_for_attn} for '
                                 'shape_for_attention.')

        if v_channels is None:
            v_channels = qk_channels

        self.attention = Attention(
            q_in_channels=q_in_channels,
            k_in_channels=kv_in_channels,
            v_in_channels=kv_in_channels,
            num_heads=num_heads,
            init_scale=attn_init_scale,
            dropout_prob=dropout_attn_prob,
            qk_out_channels=qk_channels,
            v_out_channels=v_channels,
            output_channels=output_channels)

        self.mlp = MLP(
            in_channels=output_channels,
            widening_factor=widening_factor,
            dropout_prob=dropout_prob,
            init_scale=mlp_init_scale)

        self.layer_norm_q = nn.LayerNorm(q_in_channels)
        self.layer_norm_kv = nn.LayerNorm(kv_in_channels)
        self.layer_norm2 = nn.LayerNorm(output_channels)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self,
                inputs_q,
                inputs_kv,
                *,
                attention_mask=None,
                attention_bias=None,
                return_matrix: bool=False):

        inputs_kv = self.layer_norm_kv(inputs_kv)

        attention = self.attention(inputs_q=self.layer_norm_q(inputs_q),
                                   inputs_k=inputs_kv,
                                   inputs_v=inputs_kv,
                                   attention_mask=attention_mask,
                                   attention_bias=attention_bias,
                                   return_matrix = return_matrix)

        if return_matrix:
            attention_matrix, attention = attention
        attention = self.dropout(attention)

        # Optionally include a residual to the query.
        # Consider omitting the residual if the semantics of query and output
        # are different, e.g. if queries are positions and outputs are pixels.
        if self._use_query_residual:
            x = inputs_q + attention
        else:
            x = attention

        x = x + self.mlp(self.layer_norm2(x))

        if return_matrix:
            return attention_matrix, x

        return x

    def set_haiku_params(self, params):
        mlp_params = {key[key.find('/') + 1:]: params.pop(key) for key in list(params.keys()) if
                      key.startswith("mlp")}
        self.mlp.set_haiku_params(mlp_params)

        attention_params = {key[key.find('/') + 1:]: params.pop(key) for key in list(params.keys()) if
                            key.startswith("attention")}
        self.attention.set_haiku_params(attention_params)

        init_layer_norm_from_haiku(self.layer_norm_q, params.pop("layer_norm"))
        init_layer_norm_from_haiku(self.layer_norm_kv, params.pop("layer_norm_1"))
        init_layer_norm_from_haiku(self.layer_norm2, params.pop("layer_norm_2"))

        if len(params) != 0:
            warnings.warn(f"Some parameters couldn't be matched to model: {params.keys()}")
