# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------



import math
import mlx.core as mx
import mlx.nn as nn
import mlx.nn.losses as losses
import math

class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(
        self,
        num_features: int,
    ):
        super().__init__()

        self.num_features = num_features
        self.weight = mx.ones((num_features,))
        self.bias = mx.zeros((num_features,))
        self.running_mean = mx.zeros((num_features,))
        self.running_var = mx.ones((num_features,))
        self.freeze(keys=["running_mean", "running_var"], recurse=False)

    def __call__(self, x):
        w = self.weight
        b = self.bias
        rv = self.running_var
        rm = self.running_mean
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


def gen_encoder_output_proposals(memory, memory_padding_mask, spatial_shapes, learnedwh=None):
    """
    Input:
        - memory: bs, \sum{hw}, d_model
        - memory_padding_mask: bs, \sum{hw}
        - spatial_shapes: nlevel, 2
        - learnedwh: 2
    Output:
        - output_memory: bs, \sum{hw}, d_model
        - output_proposals: bs, \sum{hw}, 4
    """
    N_, S_, C_ = memory.shape
    base_scale = 4.0
    proposals = []
    _cur = 0
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        H_, W_ = int(H_), int(W_)
        mask_flatten_ = mx.reshape(memory_padding_mask[:, _cur:(_cur + H_ * W_)], (N_, H_, W_, 1))
        valid_H = mx.sum(~mask_flatten_[:, :, 0, 0], axis=1)
        valid_W = mx.sum(~mask_flatten_[:, 0, :, 0], axis=1)

        grid_y, grid_x = mx.meshgrid(mx.linspace(0, H_ - 1, H_, dtype=mx.float32),
                                     mx.linspace(0, W_ - 1, W_, dtype=mx.float32), indexing='ij')
        grid = mx.concatenate([mx.expand_dims(grid_x, axis=-1), mx.expand_dims(grid_y, axis=-1)], axis=-1)  # H_, W_, 2

        scale = mx.reshape(mx.concatenate([mx.expand_dims(valid_W, axis=-1), mx.expand_dims(valid_H, axis=-1)], axis=1), (N_, 1, 1, 2))
        grid = (mx.expand_dims(grid, axis=0) + 0.5) / scale
        if learnedwh is not None:
            wh = mx.ones_like(grid) * nn.sigmoid(learnedwh) * (2.0 ** lvl)
        else:
            wh = mx.ones_like(grid) * 0.05 * (2.0 ** lvl)

        proposal = mx.reshape(mx.concatenate([grid, wh], axis=-1), (N_, -1, 4))
        proposals.append(proposal)
        _cur += (H_ * W_)

    output_proposals = mx.concatenate(proposals, axis=1)
    output_proposals_valid = mx.all((output_proposals > 0.01) & (output_proposals < 0.99), axis=-1, keepdims=True)
    output_proposals = mx.log(output_proposals / (1 - output_proposals))  # unsigmoid
    expanded_mask = mx.expand_dims(memory_padding_mask, axis=-1)

    output_proposals = mx.where(expanded_mask, mx.ones_like(output_proposals) * float('inf'), output_proposals)

    # output_proposals = mx.where(memory_padding_mask, mx.inf, output_proposals)
    # output_proposals = mx.where(~output_proposals_valid, mx.ones_like(output_proposals) * float('inf'), output_proposals)
    output_proposals = mx.where(~output_proposals_valid, mx.ones_like(output_proposals) * float('inf'), output_proposals)

    output_memory = memory
    output_memory = mx.where(expanded_mask, mx.zeros_like(output_memory), output_memory)
    output_memory = mx.where(~output_proposals_valid, mx.zeros_like(output_memory), output_memory)

    return output_memory, output_proposals


class RandomBoxPerturber:
    def __init__(self, x_noise_scale=0.2, y_noise_scale=0.2, w_noise_scale=0.2, h_noise_scale=0.2):
        self.noise_scale = mx.array([x_noise_scale, y_noise_scale, w_noise_scale, h_noise_scale])

    def __call__(self, refanchors):
        nq, bs, query_dim = refanchors.shape
        device = refanchors.device

        noise_raw = mx.random.uniform(shape=refanchors.shape)
        noise_scale = mx.array(self.noise_scale, device=device)[:query_dim]

        new_refanchors = refanchors * (1 + (noise_raw - 0.5) * noise_scale)
        return mx.clip(new_refanchors, 0, 1)


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha=0.25, gamma=2):
    """
    Focal loss function used in RetinaNet for dense detection.
    """
    prob = nn.sigmoid(inputs)
    ce_loss = losses.binary_cross_entropy(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return mx.sum(mx.mean(loss, axis=1)) / num_boxes


class MLP(nn.Module):
    """ Simple multi-layer perceptron (also called FFN) """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        h = [hidden_dim] * (num_layers - 1)
        self.layers = [nn.Linear(idim, odim) for idim, odim in zip([input_dim] + h, h + [output_dim])]

    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = nn.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


def _get_activation_fn(activation):
    """Returns an activation function"""
    if activation == "relu":
        return nn.relu
    elif activation == "gelu":
        return nn.gelu
    elif activation == "glu":
        return nn.glu
    elif activation == "prelu":
        return nn.PReLU
    elif activation == "selu":
        return nn.selu
    else:
        raise RuntimeError(f"Unknown activation function: {activation}")


def gen_sineembed_for_position(pos_tensor):
    scale = 2 * math.pi
    dim_t = mx.arange(128, dtype=mx.float32)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = mx.stack((mx.sin(pos_x[:, :, 0::2]), mx.cos(pos_x[:, :, 1::2])), axis=3).flatten(2)
    pos_y = mx.stack((mx.sin(pos_y[:, :, 0::2]), mx.cos(pos_y[:, :, 1::2])), axis=3).flatten(2)

    if pos_tensor.shape[-1] == 2:
        pos = mx.concatenate((pos_y, pos_x), axis=2)
    elif pos_tensor.shape[-1] == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        h_embed = pos_tensor[:, :, 3] * scale
        w_embed = w_embed[:, :, None] / dim_t
        h_embed = h_embed[:, :, None] / dim_t
        pos_w = mx.stack((mx.sin(w_embed[:, :, 0::2]), mx.cos(w_embed[:, :, 1::2])), axis=3).flatten(2)
        pos_h = mx.stack((mx.sin(h_embed[:, :, 0::2]), mx.cos(h_embed[:, :, 1::2])), axis=3).flatten(2)
        pos = mx.concatenate((pos_y, pos_x, pos_w, pos_h), axis=2)
    else:
        raise ValueError(f"Unknown pos_tensor shape: {pos_tensor.shape[-1]}")
    
    return pos

