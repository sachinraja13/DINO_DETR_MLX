import numpy as np
from typing import Optional
import math
import copy
import warnings
import numpy as np
import random
import mlx.core as mx
import mlx.nn as nn
from .deformable_transformer_decoder_layer import DeformableTransformerDecoderLayer
from .utils import gen_encoder_output_proposals, MLP, _get_activation_fn, gen_sineembed_for_position


def _get_decoder_clones(module, N, layer_share=False):
    if layer_share:
        return [module for i in range(N)]
    else:
        modules = []
        for i in range(N):
            modules.append(DeformableTransformerDecoderLayer(**module.params))
        return modules


def inverse_sigmoid(x, eps=1e-3):
    x = mx.clip(x, 0, 1)
    x1 = mx.clip(x, eps, None)
    x2 = mx.clip(1 - x, eps, None)
    return mx.log(x1/x2)


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None,
                 return_intermediate=False,
                 d_model=256, query_dim=4,
                 modulate_hw_attn=False,
                 num_feature_levels=1,
                 deformable_decoder=False,
                 decoder_query_perturber=None,
                 dec_layer_number=None,  # Number of queries for each layer in the decoder
                 rm_dec_query_scale=False,
                 dec_layer_share=False,
                 dec_layer_dropout_prob=None,
                 use_detached_boxes_dec_out=False):
        super().__init__()
        self.params = locals()
        del self.params['self']
        del self.params['__class__']
        # Initialize decoder layers
        if num_layers > 0:
            self.layers = _get_decoder_clones(
                decoder_layer, num_layers, layer_share=dec_layer_share)
        else:
            self.layers = []

        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate, "return_intermediate is required"

        self.query_dim = query_dim
        assert query_dim in [
            2, 4], f"query_dim should be 2 or 4, but got {query_dim}"

        self.num_feature_levels = num_feature_levels
        self.use_detached_boxes_dec_out = use_detached_boxes_dec_out
        self.d_model = d_model
        self.modulate_hw_attn = modulate_hw_attn
        self.deformable_decoder = deformable_decoder

        # Initialize query position head
        self.ref_point_head = MLP(
            query_dim // 2 * d_model, d_model, d_model, 2)

        if not deformable_decoder:
            self.query_pos_sine_scale = MLP(d_model, d_model, d_model, 2)
        else:
            self.query_pos_sine_scale = None

        if rm_dec_query_scale:
            self.query_scale = None
        else:
            raise NotImplementedError(
                "Query scaling has not been implemented.")

        self.bbox_embed = None
        self.class_embed = None
        if not deformable_decoder and modulate_hw_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 2, 2)
        else:
            self.ref_anchor_head = None

        self.decoder_query_perturber = decoder_query_perturber
        self.box_pred_damping = None

        self.dec_layer_number = dec_layer_number
        if dec_layer_number is not None:
            assert isinstance(dec_layer_number, list)
            assert len(dec_layer_number) == num_layers

        self.dec_layer_dropout_prob = dec_layer_dropout_prob
        if dec_layer_dropout_prob is not None:
            assert isinstance(dec_layer_dropout_prob, list)
            assert len(dec_layer_dropout_prob) == num_layers
            for prob in dec_layer_dropout_prob:
                assert 0.0 <= prob <= 1.0

        self.rm_detach = None

    def __call__(self, tgt, memory,
                 tgt_mask: Optional[mx.array] = None,
                 memory_mask: Optional[mx.array] = None,
                 tgt_key_padding_mask: Optional[mx.array] = None,
                 memory_key_padding_mask: Optional[mx.array] = None,
                 pos: Optional[mx.array] = None,
                 # num_queries, bs, 2
                 refpoints_unsigmoid: Optional[mx.array] = None,
                 level_start_index: Optional[mx.array] = None,  # num_levels
                 # bs, num_levels, 2
                 spatial_shapes: Optional[mx.array] = None,
                 valid_ratios: Optional[mx.array] = None):
        """
        Input:
            - tgt: Target sequence, shape [nq, bs, d_model]
            - memory: Memory from encoder, shape [hw, bs, d_model]
            - pos: Positional encoding for memory, shape [hw, bs, d_model]
            - refpoints_unsigmoid: Unsigmoided reference points, shape [nq, bs, 2/4]
            - valid_ratios/spatial_shapes: Shape [bs, nlevel, 2]
        """
        output = tgt

        intermediate = []
        # Apply sigmoid to reference points
        reference_points = mx.sigmoid(refpoints_unsigmoid)
        ref_points = [reference_points]  # Store reference points

        # Process through decoder layers
        for layer_id, layer in enumerate(self.layers):
            # Preprocess reference points
            if self.training and self.decoder_query_perturber is not None and layer_id != 0:
                reference_points = self.decoder_query_perturber(
                    reference_points)

            if self.deformable_decoder:
                if reference_points.shape[-1] == 4:
                    reference_points_input = reference_points[:, :, None] * mx.concatenate(
                        [valid_ratios, valid_ratios], axis=-1)[None, :]
                else:
                    assert reference_points.shape[-1] == 2
                    reference_points_input = reference_points[:,
                                                              :, None] * valid_ratios[None, :]
                query_sine_embed = gen_sineembed_for_position(
                    reference_points_input[:, :, 0, :])
            else:
                query_sine_embed = gen_sineembed_for_position(reference_points)
                reference_points_input = None
            # Conditional query
            raw_query_pos = self.ref_point_head(query_sine_embed)
            pos_scale = self.query_scale(
                output) if self.query_scale is not None else 1
            query_pos = pos_scale * raw_query_pos
            if not self.deformable_decoder:
                query_sine_embed = query_sine_embed[...,
                                                    :self.d_model] * self.query_pos_sine_scale(output)

            # Modulated HW attentions
            if not self.deformable_decoder and self.modulate_hw_attn:
                refHW_cond = mx.sigmoid(self.ref_anchor_head(output))
                query_sine_embed[..., self.d_model // 2:] *= (
                    refHW_cond[..., 0] / reference_points[..., 2])[..., None]
                query_sine_embed[..., :self.d_model // 2] *= (
                    refHW_cond[..., 1] / reference_points[..., 3])[..., None]

            # Dropout mechanism for some layers if required
            dropflag = False
            if self.dec_layer_dropout_prob is not None:
                prob = random.random()
                if prob < self.dec_layer_dropout_prob[layer_id]:
                    dropflag = True

            if not dropflag:
                output = layer(
                    tgt=output,
                    tgt_query_pos=query_pos,
                    tgt_query_sine_embed=query_sine_embed,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    tgt_reference_points=reference_points_input,
                    memory=memory,
                    memory_key_padding_mask=memory_key_padding_mask,
                    memory_level_start_index=level_start_index,
                    memory_spatial_shapes=spatial_shapes,
                    memory_pos=pos,
                    self_attn_mask=tgt_mask,
                    cross_attn_mask=memory_mask
                )

            # Iterative update
            if self.bbox_embed is not None:
                reference_before_sigmoid = inverse_sigmoid(reference_points)
                delta_unsig = self.bbox_embed[layer_id](output)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = mx.sigmoid(outputs_unsig)

                # Select reference points
                if self.dec_layer_number is not None and layer_id != self.num_layers - 1:
                    nq_now = new_reference_points.shape[0]
                    select_number = self.dec_layer_number[layer_id + 1]
                    if nq_now != select_number:
                        class_unselected = self.class_embed[layer_id](output)
                        topk_proposals = mx.argpartition(class_unselected.max(
                            axis=-1) * -1, select_number, axis=0)[:topk, :]
                        new_reference_points = new_reference_points[topk_proposals[..., None], mx.arange(
                            batch_size)[None, :, None], mx.arange(4)[None, None, :]]

                if self.rm_detach and 'dec' in self.rm_detach:
                    reference_points = new_reference_points
                else:
                    reference_points = mx.stop_gradient(new_reference_points)
                if self.use_detached_boxes_dec_out:
                    ref_points.append(reference_points)
                else:
                    ref_points.append(new_reference_points)

            intermediate.append(self.norm(output))

            if self.dec_layer_number is not None and layer_id != self.num_layers - 1:
                if nq_now != select_number:
                    output = output[topk_proposals[..., None], mx.arange(
                        batch_size)[None, :, None], mx.arange(d_model)[None, None, :]]

        return [
            [itm_out.transpose(1, 0, 2) for itm_out in intermediate],
            [itm_refpoint.transpose(1, 0, 2) for itm_refpoint in ref_points]
        ]
