import numpy as np
from typing import Optional
import math
import copy
import warnings
import numpy as np
import random
import mlx.core as mx
import mlx.nn as nn
from .deformable_attn import MSDeformAttn
from .utils import gen_encoder_output_proposals, MLP, _get_activation_fn, gen_sineembed_for_position
    
class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 add_channel_attention=False,
                 use_deformable_box_attn=False,
                 box_attn_type='roi_align'):
        super().__init__()
        self.params = locals()
        del self.params['self']
        del self.params['__class__']
        # Self-attention mechanism
        if use_deformable_box_attn:
            self.self_attn = nn.MSDeformableBoxAttention(d_model, n_levels, n_heads, n_boxes=n_points, used_func=box_attn_type)
        else:
            self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)  # Use MLX activation functions
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # Channel attention mechanism
        self.add_channel_attention = add_channel_attention
        if add_channel_attention:
            self.activ_channel = _get_activation_fn('relu', d_model=d_model)
            self.norm_channel = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def __call__(self, src, pos, reference_points, spatial_shapes, level_start_index, key_padding_mask=None):
        # Self-attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # Feedforward network
        src = self.forward_ffn(src)

        # Channel attention if applicable
        if self.add_channel_attention:
            src = self.norm_channel(src + self.activ_channel(src))

        return src
