import numpy as np
from typing import Optional
import math
import copy
import warnings
import numpy as np
import random
import mlx.core as mx
import mlx.nn as nn
from .deformable_transformer_encoder_layer import DeformableTransformerEncoderLayer
from .utils import gen_encoder_output_proposals, MLP, _get_activation_fn, gen_sineembed_for_position


def _get_encoder_clones(module, N, layer_share=False):
    if layer_share:
        return [module for i in range(N)]
    else:
        modules = []
        for i in range(N):
            modules.append(DeformableTransformerEncoderLayer(**module.params))
        return modules


class TransformerEncoder(nn.Module):
    def __init__(self,
                 encoder_layer, num_layers, norm=None, d_model=256,
                 num_queries=300,
                 deformable_encoder=False,
                 enc_layer_share=False, enc_layer_dropout_prob=None,
                 # ['no', 'standard', 'early', 'combine', 'enceachlayer', 'enclayer1']
                 two_stage_type='no'
                 ):
        super().__init__()
        self.params = locals()
        del self.params['self']
        del self.params['__class__']
        # Prepare layers
        if num_layers > 0:
            self.layers = _get_encoder_clones(
                encoder_layer, num_layers, layer_share=enc_layer_share)
        else:
            self.layers = []
            del encoder_layer

        self.query_scale = None
        self.num_queries = num_queries
        self.deformable_encoder = deformable_encoder
        self.num_layers = num_layers
        self.norm = norm
        self.d_model = d_model

        # Dropout probabilities for each encoder layer
        self.enc_layer_dropout_prob = enc_layer_dropout_prob
        if enc_layer_dropout_prob is not None:
            assert isinstance(enc_layer_dropout_prob, list)
            assert len(enc_layer_dropout_prob) == num_layers
            for prob in enc_layer_dropout_prob:
                assert 0.0 <= prob <= 1.0

        self.two_stage_type = two_stage_type
        if two_stage_type in ['enceachlayer', 'enclayer1']:
            _proj_layer = nn.Linear(d_model, d_model)
            _norm_layer = nn.LayerNorm(d_model)
            if two_stage_type == 'enclayer1':
                self.enc_norm = [_norm_layer]
                self.enc_proj = [_proj_layer]
            else:
                self.enc_norm = [nn.LayerNorm(d_model)
                                 for _ in range(num_layers - 1)]
                self.enc_proj = [nn.Linear(d_model, d_model)
                                 for _ in range(num_layers - 1)]
            self.class_embed = [nn.Linear(d_model, 91)
                                for _ in range(num_layers - 1)]

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            H_ = int(H_)
            W_ = int(W_)
            ref_y, ref_x = mx.meshgrid(mx.linspace(0.5, H_ - 0.5, H_, dtype=mx.float32),
                                       mx.linspace(0.5, W_ - 0.5, W_, dtype=mx.float32), indexing='ij')
            ref_y = ref_y.reshape(-1)[None] / \
                (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / \
                (valid_ratios[:, None, lvl, 0] * W_)
            ref = mx.stack([ref_x, ref_y], axis=-1)
            reference_points_list.append(ref)
        reference_points = mx.concatenate(reference_points_list, axis=1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def __call__(self,
                 src: mx.array,
                 pos: mx.array,
                 spatial_shapes: mx.array,
                 level_start_index: mx.array,
                 valid_ratios: mx.array,
                 key_padding_mask: mx.array,
                 ref_token_index: Optional[mx.array] = None,
                 ref_token_coord: Optional[mx.array] = None
                 ):
        """
        Inputs:
            - src: [bs, sum(hi*wi), 256]
            - pos: pos embed for src. [bs, sum(hi*wi), 256]
            - spatial_shapes: h,w of each level [num_level, 2]
            - level_start_index: [num_level] start point of level in sum(hi*wi).
            - valid_ratios: [bs, num_level, 2]
            - key_padding_mask: [bs, sum(hi*wi)]
            - ref_token_index: bs, nq
            - ref_token_coord: bs, nq, 4
        Outputs: 
            - output: [bs, sum(hi*wi), 256]
        """
        if self.two_stage_type in ['no', 'standard', 'enceachlayer', 'enclayer1']:
            assert ref_token_index is None

        output = src

        if self.num_layers > 0 and self.deformable_encoder:
            reference_points = self.get_reference_points(
                spatial_shapes, valid_ratios)
        intermediate_output = []
        intermediate_ref = []
        if ref_token_index is not None:
            out_i = output[mx.arange(batch_size)[
                :, None, None], ref_token_index[:, :, None], mx.arange(self.d_model)[None, None, :]]
            intermediate_output.append(out_i)
            intermediate_ref.append(ref_token_coord)

        # Main process across layers
        for layer_id, layer in enumerate(self.layers):
            # Apply dropout for the layer if specified
            dropflag = False
            if self.enc_layer_dropout_prob is not None and random.random() < self.enc_layer_dropout_prob[layer_id]:
                dropflag = True

            if not dropflag:
                if self.deformable_encoder:
                    output = layer(src=output, pos=pos, reference_points=reference_points,
                                   spatial_shapes=spatial_shapes, level_start_index=level_start_index,
                                   key_padding_mask=key_padding_mask)
                else:
                    output = layer(src=output.transpose(0, 1), pos=pos.transpose(
                        0, 1), key_padding_mask=key_padding_mask).transpose(0, 1)

            if ((layer_id == 0 and self.two_stage_type in ['enceachlayer', 'enclayer1'])
                or (self.two_stage_type == 'enceachlayer')) \
                    and (layer_id != self.num_layers - 1):
                output_memory, output_proposals = gen_encoder_output_proposals(
                    output, key_padding_mask, spatial_shapes)
                output_memory = self.enc_norm[layer_id](
                    self.enc_proj[layer_id](output_memory))

                # Gather reference tokens
                topk = self.num_queries
                enc_outputs_class = self.class_embed[layer_id](output_memory)
                # ref_token_index = mx.repeat(mx.arange(enc_outputs_class.shape[1])[None, :], batch_size, axis=0)
                ref_token_index = mx.argpartition(
                    enc_outputs_class.max(axis=-1) * -1, topk, axis=1)[:, :topk]
                # Final gathered output
                ref_token_coord = output_proposals[mx.arange(batch_size)[
                    :, None, None], ref_token_index[..., None], mx.arange(4)[None, None, :]]

                output = output_memory

            # Auxiliary loss
            if (layer_id != self.num_layers - 1) and ref_token_index is not None:
                out_i = output[mx.arange(batch_size)[
                    :, None, None], ref_token_index[:, :, None], mx.arange(self.d_model)[None, None, :]]
                intermediate_output.append(out_i)
                intermediate_ref.append(ref_token_coord)
        # Final normalization
        if self.norm is not None:
            output = self.norm(output)
        # print(intermediate_output)
        if ref_token_index is not None:
            intermediate_output = mx.stack(intermediate_output)
            intermediate_ref = mx.stack(intermediate_ref)
        else:
            intermediate_output = intermediate_ref = None

        return output, intermediate_output, intermediate_ref
