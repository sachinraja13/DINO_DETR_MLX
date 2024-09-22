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

    
    
class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 use_deformable_box_attn=False,
                 box_attn_type='roi_align',
                 key_aware_type=None,
                 decoder_sa_type='sa',
                 module_seq=['sa', 'ca', 'ffn']):
        super().__init__()
        self.params = locals()
        del self.params['self']
        del self.params['__class__']
        self.module_seq = module_seq
        assert sorted(module_seq) == ['ca', 'ffn', 'sa']

        # Cross attention
        if use_deformable_box_attn:
            self.cross_attn = nmn.MSDeformableBoxAttention(d_model, n_levels, n_heads, n_boxes=n_points, used_func=box_attn_type)
        else:
            self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Self attention
        self.self_attn = nn.MultiHeadAttention(d_model, n_heads, bias=True)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # Feedforward network (FFN)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.key_aware_type = key_aware_type
        self.key_aware_proj = None
        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ['sa', 'ca_label', 'ca_content']

        if decoder_sa_type == 'ca_content':
            self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)

    def rm_self_attn_modules(self):
        self.self_attn = None
        self.dropout2 = None
        self.norm2 = None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_sa(self, tgt, tgt_query_pos=None, tgt_query_sine_embed=None, tgt_key_padding_mask=None, 
                   tgt_reference_points=None, memory=None, memory_key_padding_mask=None, 
                   memory_level_start_index=None, memory_spatial_shapes=None, memory_pos=None, 
                   self_attn_mask=None, cross_attn_mask=None):
        
        # Self-attention
        if self.self_attn is not None:
            if self.decoder_sa_type == 'sa':
                q = k = self.with_pos_embed(tgt, tgt_query_pos)
                tgt2 = self.self_attn(q.transpose(1,0,2), k.transpose(1,0,2), tgt.transpose(1,0,2), mask=self_attn_mask).transpose(1,0,2)
                tgt = tgt + self.dropout2(tgt2)
                tgt = self.norm2(tgt)
            elif self.decoder_sa_type == 'ca_label':
                bs = tgt.shape[1]
                k = v = self.label_embedding.weight[:, None, :].repeat(1, bs, 1)
                tgt2 = self.self_attn(tgt.transpose(1,0,2), k.transpose(1,0,2), v.transpose(1,0,2), mask=self_attn_mask).transpose(1,0,2)
                tgt = tgt + self.dropout2(tgt2)
                tgt = self.norm2(tgt)
            elif self.decoder_sa_type == 'ca_content':
                tgt2 = self.self_attn(self.with_pos_embed(tgt, tgt_query_pos).transpose(1, 0, 2),
                                      tgt_reference_points.transpose(1, 0, 2, 3),
                                      memory.transpose(1, 0, 2), memory_spatial_shapes, memory_level_start_index, memory_key_padding_mask).transpose(1, 0, 2)
                tgt = tgt + self.dropout2(tgt2)
                tgt = self.norm2(tgt)
            else:
                raise NotImplementedError(f"Unknown decoder_sa_type {self.decoder_sa_type}")

        return tgt

    def forward_ca(self, tgt, tgt_query_pos=None, tgt_query_sine_embed=None, tgt_key_padding_mask=None, 
                   tgt_reference_points=None, memory=None, memory_key_padding_mask=None, 
                   memory_level_start_index=None, memory_spatial_shapes=None, memory_pos=None, 
                   self_attn_mask=None, cross_attn_mask=None):

        # Cross-attention
        if self.key_aware_type is not None:
            if self.key_aware_type == 'mean':
                tgt = tgt + memory.mean(0, keepdim=True)
            elif self.key_aware_type == 'proj_mean':
                tgt = tgt + self.key_aware_proj(memory).mean(0, keepdim=True)
            else:
                raise NotImplementedError(f"Unknown key_aware_type: {self.key_aware_type}")
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, tgt_query_pos).transpose(1, 0, 2),
                               tgt_reference_points.transpose(1, 0, 2, 3),
                               memory.transpose(1, 0, 2), memory_spatial_shapes, memory_level_start_index, memory_key_padding_mask).transpose(1, 0, 2)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        return tgt

    def __call__(self, tgt, tgt_query_pos=None, tgt_query_sine_embed=None, tgt_key_padding_mask=None, 
                tgt_reference_points=None, memory=None, memory_key_padding_mask=None, 
                memory_level_start_index=None, memory_spatial_shapes=None, memory_pos=None, 
                self_attn_mask=None, cross_attn_mask=None):

        for funcname in self.module_seq:
            if funcname == 'ffn':
                tgt = self.forward_ffn(tgt)
            elif funcname == 'ca':
                tgt = self.forward_ca(tgt, tgt_query_pos, tgt_query_sine_embed, 
                                      tgt_key_padding_mask, tgt_reference_points, 
                                      memory, memory_key_padding_mask, memory_level_start_index, 
                                      memory_spatial_shapes, memory_pos, self_attn_mask, cross_attn_mask)
            elif funcname == 'sa':
                tgt = self.forward_sa(tgt, tgt_query_pos, tgt_query_sine_embed, 
                                      tgt_key_padding_mask, tgt_reference_points, 
                                      memory, memory_key_padding_mask, memory_level_start_index, 
                                      memory_spatial_shapes, memory_pos, self_attn_mask, cross_attn_mask)
            else:
                raise ValueError(f"Unknown funcname {funcname}")

        return tgt
    
