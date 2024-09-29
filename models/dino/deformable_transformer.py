import numpy as np
from typing import Optional
import math
import copy
import warnings
import numpy as np
import random
import mlx.core as mx
import mlx.nn as nn
from .deformable_attn import *
from .deformable_transformer_decoder import *
from .deformable_transformer_encoder import *
from util.misc import inverse_sigmoid
from .utils import gen_encoder_output_proposals, MLP, _get_activation_fn, gen_sineembed_for_position

# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR Transformer class.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------


class DeformableTransformer(nn.Module):

    def __init__(self, d_model=256, nhead=8,
                 num_queries=300,
                 num_encoder_layers=6,
                 num_unicoder_layers=0,
                 num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=True, query_dim=4,
                 num_patterns=0,
                 modulate_hw_attn=False,
                 deformable_encoder=True,
                 deformable_decoder=True,
                 num_feature_levels=4,
                 enc_n_points=4,
                 dec_n_points=4,
                 use_deformable_box_attn=False,
                 box_attn_type='roi_align',
                 learnable_tgt_init=True,
                 decoder_query_perturber=None,
                 add_channel_attention=False,
                 add_pos_value=False,
                 random_refpoints_xy=False,
                 two_stage_type='no',
                 two_stage_pat_embed=0,
                 two_stage_add_query_num=0,
                 two_stage_learn_wh=False,
                 two_stage_keep_all_tokens=False,
                 dec_layer_number=None,
                 rm_enc_query_scale=True,
                 rm_dec_query_scale=True,
                 rm_self_attn_layers=None,
                 key_aware_type=None,
                 layer_share_type=None,
                 rm_detach=None,
                 decoder_sa_type='sa',
                 module_seq=['sa', 'ca', 'ffn'],
                 embed_init_tgt=False,
                 use_detached_boxes_dec_out=False):
        super().__init__()
        self.params = locals()
        del self.params['self']
        del self.params['__class__']
        self.num_feature_levels = num_feature_levels
        self.num_encoder_layers = num_encoder_layers
        self.num_unicoder_layers = num_unicoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.deformable_encoder = deformable_encoder
        self.deformable_decoder = deformable_decoder
        self.two_stage_keep_all_tokens = two_stage_keep_all_tokens
        self.num_queries = num_queries
        self.random_refpoints_xy = random_refpoints_xy
        self.use_detached_boxes_dec_out = use_detached_boxes_dec_out
        assert query_dim == 4

        if num_feature_levels > 1:
            assert deformable_encoder, "only support deformable_encoder for num_feature_levels > 1"
        if use_deformable_box_attn:
            assert deformable_encoder or deformable_encoder

        if layer_share_type in ['encoder', 'both']:
            enc_layer_share = True
        else:
            enc_layer_share = False
        if layer_share_type in ['decoder', 'both']:
            dec_layer_share = True
        else:
            dec_layer_share = False
        assert layer_share_type is None

        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ['sa', 'ca_label', 'ca_content']

        # Encoder layer selection
        if deformable_encoder:
            encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward, dropout, activation,
                                                              num_feature_levels, nhead, enc_n_points,
                                                              add_channel_attention=add_channel_attention,
                                                              use_deformable_box_attn=use_deformable_box_attn,
                                                              box_attn_type=box_attn_type)
        else:
            raise NotImplementedError

        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers,
            encoder_norm, d_model=d_model,
            num_queries=num_queries,
            deformable_encoder=deformable_encoder,
            enc_layer_share=enc_layer_share,
            two_stage_type=two_stage_type
        )

        # Decoder layer selection
        if deformable_decoder:
            decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward, dropout, activation,
                                                              num_feature_levels, nhead, dec_n_points,
                                                              use_deformable_box_attn=use_deformable_box_attn,
                                                              box_attn_type=box_attn_type,
                                                              key_aware_type=key_aware_type,
                                                              decoder_sa_type=decoder_sa_type,
                                                              module_seq=module_seq)
        else:
            raise NotImplementedError

        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm,
            return_intermediate=return_intermediate_dec,
            d_model=d_model, query_dim=query_dim,
            modulate_hw_attn=modulate_hw_attn,
            num_feature_levels=num_feature_levels,
            deformable_decoder=deformable_decoder,
            decoder_query_perturber=decoder_query_perturber,
            dec_layer_number=dec_layer_number,
            rm_dec_query_scale=rm_dec_query_scale,
            dec_layer_share=dec_layer_share,
            use_detached_boxes_dec_out=use_detached_boxes_dec_out
        )

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries  # Useful for single-stage models
        self.num_patterns = num_patterns
        if not isinstance(num_patterns, int):
            Warning(f"num_patterns should be int but got {type(num_patterns)}")
            self.num_patterns = 0

        if self.num_patterns > 0:
            self.patterns = nn.Embedding(self.num_patterns, d_model)
            nn.init.normal()(self.patterns.weight)

        # Multi-level embedding setup
        if num_feature_levels > 1:
            if self.num_encoder_layers > 0:
                self.level_embed = nn.init.normal()(mx.zeros((num_feature_levels, d_model)))
            else:
                self.level_embed = None

        self.learnable_tgt_init = learnable_tgt_init
        assert learnable_tgt_init, "Why not learnable_tgt_init?"
        self.embed_init_tgt = embed_init_tgt

        if (two_stage_type != 'no' and embed_init_tgt) or (two_stage_type == 'no'):
            self.tgt_embed = nn.Embedding(self.num_queries, d_model)
            nn.init.normal()(self.tgt_embed.weight)
        else:
            self.tgt_embed = None

        # Two-stage initialization
        self.two_stage_type = two_stage_type
        self.two_stage_pat_embed = two_stage_pat_embed
        self.two_stage_add_query_num = two_stage_add_query_num
        self.two_stage_learn_wh = two_stage_learn_wh
        assert two_stage_type in [
            'no', 'standard'], f"Unknown param {two_stage_type} for two_stage_type"
        if two_stage_type == 'standard':
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            if self.two_stage_pat_embed > 0:
                self.pat_embed_for_2stage = nn.Embedding(
                    self.two_stage_pat_embed, d_model)
                nn.init.normal()(self.pat_embed_for_2stage.weight)
            if two_stage_add_query_num > 0:
                self.tgt_embed = nn.Embedding(
                    self.two_stage_add_query_num, d_model)
            if two_stage_learn_wh:
                self.two_stage_wh_embedding = nn.Embedding(1, 2)
                nn.init.constant(math.log(0.05 / (1 - 0.05))
                                 )(self.two_stage_wh_embedding.weight)
            else:
                self.two_stage_wh_embedding = None

        if two_stage_type == 'no':
            self.init_ref_points(num_queries)

        self.enc_out_class_embed = None
        self.enc_out_bbox_embed = None

        self.dec_layer_number = dec_layer_number
        if dec_layer_number is not None:
            if self.two_stage_type != 'no' or num_patterns == 0:
                assert dec_layer_number[
                    0] == num_queries, f"dec_layer_number[0]({dec_layer_number[0]}) != num_queries({num_queries})"
            else:
                assert dec_layer_number[0] == num_queries * \
                    num_patterns, f"dec_layer_number[0]({dec_layer_number[0]}) != num_queries({num_queries}) * num_patterns({num_patterns})"

        self.rm_self_attn_layers = rm_self_attn_layers
        if rm_self_attn_layers is not None:
            print(
                f"Removing the self-attn in {rm_self_attn_layers} decoder layers")
            for lid, dec_layer in enumerate(self.decoder.layers):
                if lid in rm_self_attn_layers:
                    dec_layer.rm_self_attn_modules()

        self.rm_detach = rm_detach
        if self.rm_detach:
            assert isinstance(rm_detach, list)
            assert any([i in ['enc_ref', 'enc_tgt', 'dec'] for i in rm_detach])
        self.decoder.rm_detach = rm_detach

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = mx.sum(~mask[:, :, 0], axis=1)
        valid_W = mx.sum(~mask[:, 0, :], axis=1)
        valid_ratio_h = valid_H.astype(mx.float32) / H
        valid_ratio_w = valid_W.astype(mx.float32) / W
        valid_ratio = mx.stack([valid_ratio_w, valid_ratio_h], axis=-1)
        return valid_ratio

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, 4)
        if self.random_refpoints_xy:
            nn.init.uniform(self.refpoint_embed.weight[:, :2], 0, 1)
            self.refpoint_embed.weight[:, :2] = inverse_sigmoid(
                self.refpoint_embed.weight[:, :2])
            self.refpoint_embed.weight[:, :2] = mx.stop_gradient(
                self.refpoint_embed.weight[:, :2])

    def __call__(self, srcs, masks, refpoint_embed, pos_embeds, tgt, attn_mask=None):
        src_flatten, mask_flatten, lvl_pos_embed_flatten, spatial_shapes = [], [], [], []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            src = src.transpose(0, 3, 1, 2)
            pos_embed = pos_embed.transpose(0, 3, 1, 2)
            bs, c, h, w = src.shape
            spatial_shape = [h, w]
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(0, 2, 1)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(0, 2, 1)
            if self.num_feature_levels > 1 and self.level_embed is not None:
                lvl_pos_embed = pos_embed + \
                    self.level_embed[lvl].reshape(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

        src_flatten = mx.concatenate(src_flatten, axis=1)
        mask_flatten = mx.concatenate(mask_flatten, axis=1)
        lvl_pos_embed_flatten = mx.concatenate(lvl_pos_embed_flatten, axis=1)
        spatial_shapes = mx.array(spatial_shapes, mx.int64)
        level_start_index = mx.concatenate(
            [mx.zeros((1,)), mx.cumsum(mx.array(spatial_shapes)[:-1].prod(1), axis=0)])
        valid_ratios = mx.stack([self.get_valid_ratio(m)
                                for m in masks], axis=1)

        enc_topk_proposals = enc_refpoint_embed = None
        memory, enc_intermediate_output, enc_intermediate_refpoints = self.encoder(
            src_flatten,
            pos=lvl_pos_embed_flatten,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            key_padding_mask=mask_flatten,
            ref_token_index=enc_topk_proposals,
            ref_token_coord=enc_refpoint_embed
        )
        if self.two_stage_type == 'standard':
            if self.two_stage_learn_wh:
                input_hw = self.two_stage_wh_embedding.weight[0]
            else:
                input_hw = None

            output_memory, output_proposals = gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes, input_hw)
            output_memory = self.enc_output_norm(
                self.enc_output(output_memory))

            if self.two_stage_pat_embed > 0:
                bs, nhw, _ = output_memory.shape
                output_memory = mx.repeat(
                    output_memory, self.two_stage_pat_embed, axis=1)
                _pats = mx.repeat(
                    self.pat_embed_for_2stage.weight[:, None], nhw, axis=0).squeeze()
                output_memory = output_memory + _pats
                output_proposals = mx.repeat(
                    output_proposals, self.two_stage_pat_embed, axis=1)

            if self.two_stage_add_query_num > 0:
                assert refpoint_embed is not None
                output_memory = mx.concatenate([output_memory, tgt], axis=1)
                output_proposals = mx.concatenate(
                    [output_proposals, refpoint_embed], axis=1)

            enc_outputs_class_unselected = self.enc_out_class_embed(
                output_memory)
            enc_outputs_coord_unselected = self.enc_out_bbox_embed(
                output_memory) + output_proposals
            topk = self.num_queries
            topk_proposals = mx.argpartition(
                enc_outputs_class_unselected.max(axis=-1) * -1, topk, axis=1)[:, :topk]
            refpoint_embed_undetach = enc_outputs_coord_unselected[mx.arange(
                bs)[:, None, None], topk_proposals[..., None], mx.arange(4)[None, None, :]]
            refpoint_embed_ = mx.stop_gradient(refpoint_embed_undetach)
            init_box_proposal = mx.sigmoid(output_proposals[mx.arange(
                bs)[:, None, None], topk_proposals[..., None], mx.arange(4)[None, None, :]])
            tgt_undetach = output_memory[mx.arange(
                bs)[:, None, None], topk_proposals[..., None], mx.arange(self.d_model)[None, None, :]]
            if self.embed_init_tgt:
                tgt_ = mx.repeat(
                    self.tgt_embed.weight[:, None, :], bs, axis=1).transpose(1, 0, 2)
            else:
                tgt_ = mx.stop_gradient(tgt_undetach)

            if refpoint_embed is not None:
                refpoint_embed = mx.concatenate(
                    [refpoint_embed, refpoint_embed_], axis=1)
                tgt = mx.concatenate([tgt, tgt_], axis=1)
            else:
                refpoint_embed, tgt = refpoint_embed_, tgt_

        elif self.two_stage_type == 'no':
            tgt_ = mx.repeat(
                self.tgt_embed.weight[:, None, :], bs, axis=1).transpose(1, 0, 2)
            refpoint_embed_ = mx.repeat(
                self.refpoint_embed.weight[:, None, :], bs, axis=1).transpose(1, 0, 2)

            if refpoint_embed is not None:
                refpoint_embed = mx.concatenate(
                    [refpoint_embed, refpoint_embed_], axis=1)
                tgt = mx.concatenate([tgt, tgt_], axis=1)
            else:
                refpoint_embed, tgt = refpoint_embed_, tgt_

            if self.num_patterns > 0:
                tgt_embed = mx.repeat(tgt, self.num_patterns, axis=1)
                refpoint_embed = mx.repeat(
                    refpoint_embed, self.num_patterns, axis=1)
                tgt_pat = mx.repeat(
                    self.patterns.weight[None, :, :], self.num_queries*self.num_patterns, axis=1)
                tgt = tgt_embed + tgt_pat

            init_box_proposal = mx.sigmoid(refpoint_embed_)

        else:
            raise NotImplementedError(
                f"unknown two_stage_type {self.two_stage_type}")
        hs, references = self.decoder(
            tgt=tgt.transpose(1, 0, 2),
            memory=memory.transpose(1, 0, 2),
            memory_key_padding_mask=mask_flatten,
            pos=lvl_pos_embed_flatten.transpose(1, 0, 2),
            refpoints_unsigmoid=refpoint_embed.transpose(1, 0, 2),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios, tgt_mask=attn_mask
        )

        if self.two_stage_type == 'standard':
            if self.two_stage_keep_all_tokens:
                hs_enc = output_memory[None, ...]
                ref_enc = enc_outputs_coord_unselected[None, ...]
                init_box_proposal = output_proposals
            else:
                hs_enc = tgt_undetach[None, ...]
                ref_enc = mx.sigmoid(refpoint_embed_undetach)[None, ...]
        else:
            hs_enc = ref_enc = None

        return hs, references, hs_enc, ref_enc, init_box_proposal


def build_deformable_transformer(args):
    decoder_query_perturber = None
    if args.decoder_layer_noise:
        from .utils import RandomBoxPerturber
        decoder_query_perturber = RandomBoxPerturber(
            x_noise_scale=args.dln_xy_noise, y_noise_scale=args.dln_xy_noise,
            w_noise_scale=args.dln_hw_noise, h_noise_scale=args.dln_hw_noise)

    use_detached_boxes_dec_out = False
    try:
        use_detached_boxes_dec_out = args.use_detached_boxes_dec_out
    except:
        use_detached_boxes_dec_out = False

    return DeformableTransformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_queries=args.num_queries,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_unicoder_layers=args.unic_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        query_dim=args.query_dim,
        activation=args.transformer_activation,
        num_patterns=args.num_patterns,
        modulate_hw_attn=True,

        deformable_encoder=True,
        deformable_decoder=True,
        num_feature_levels=args.num_feature_levels,
        enc_n_points=args.enc_n_points,
        dec_n_points=args.dec_n_points,
        use_deformable_box_attn=args.use_deformable_box_attn,
        box_attn_type=args.box_attn_type,

        learnable_tgt_init=True,
        decoder_query_perturber=decoder_query_perturber,

        add_channel_attention=args.add_channel_attention,
        add_pos_value=args.add_pos_value,
        random_refpoints_xy=args.random_refpoints_xy,

        # two stage
        two_stage_type=args.two_stage_type,  # ['no', 'standard', 'early']
        two_stage_pat_embed=args.two_stage_pat_embed,
        two_stage_add_query_num=args.two_stage_add_query_num,
        two_stage_learn_wh=args.two_stage_learn_wh,
        two_stage_keep_all_tokens=args.two_stage_keep_all_tokens,
        dec_layer_number=args.dec_layer_number,
        rm_self_attn_layers=None,
        key_aware_type=None,
        layer_share_type=None,

        rm_detach=None,
        decoder_sa_type=args.decoder_sa_type,
        module_seq=args.decoder_module_seq,

        embed_init_tgt=args.embed_init_tgt,
        use_detached_boxes_dec_out=use_detached_boxes_dec_out
    )
