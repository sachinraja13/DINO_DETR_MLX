# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
import sys

import copy
import math
from typing import List, Dict
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from util import box_ops
from util.misc import (nested_array_dict_array_list,
                       interpolate, inverse_sigmoid)
from .deformable_transformer import DeformableTransformer, build_deformable_transformer
from ..utils import MLP
from .dn_module import DNEncoder, dn_post_process

from ..backbone import build_backbone
from ..matchers import build_matcher
from ..loss_criteria import build_dino_loss_criterion
from ..postprocessors import build_postprocessors
from ..registry import MODULE_BUILD_FUNCS


class DINO(nn.Module):
    """ This is the Cross-Attention Detector module that performs object detection """

    def __init__(self, backbone, transformer, num_classes, num_queries,
                 aux_loss=False, iter_update=False,
                 query_dim=2,
                 random_refpoints_xy=False,
                 fix_refpoints_hw=-1,
                 num_feature_levels=1,
                 nheads=8,
                 # two stage
                 two_stage_type='no',  # ['no', 'standard']
                 two_stage_add_query_num=0,
                 dec_pred_class_embed_share=True,
                 dec_pred_bbox_embed_share=True,
                 two_stage_class_embed_share=True,
                 two_stage_bbox_embed_share=True,
                 decoder_sa_type='sa',
                 num_patterns=0,
                 dn_number=100,
                 dn_box_noise_scale=0.4,
                 dn_label_noise_ratio=0.5,
                 dn_labelbook_size=100,
                 ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.

            fix_refpoints_hw: -1(default): learn w and h for each box seperately
                                >0 : given fixed number
                                -2 : learn a shared w and h
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.label_enc = nn.Embedding(dn_labelbook_size + 1, hidden_dim)
        self.dn_encoder = DNEncoder(
            num_queries, num_classes, hidden_dim, self.label_enc)
        # setting query dim
        self.query_dim = query_dim
        assert query_dim == 4
        self.random_refpoints_xy = random_refpoints_xy
        self.fix_refpoints_hw = fix_refpoints_hw

        # for dn training
        self.num_patterns = num_patterns
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size
        self.use_dn = False
        if self.dn_number > 0:
            self.use_dn = True

        # prepare input projection layers
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = input_proj_list
        else:
            assert two_stage_type == 'no', "two_stage_type should be no if num_feature_levels=1 !!!"
            self.input_proj = [
                nn.Sequential(
                    nn.Conv2d(
                        backbone.num_channels[-1], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )]

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.box_pred_damping = box_pred_damping = None

        self.iter_update = iter_update
        assert iter_update, "Why not iter_update?"

        # prepare pred layers
        self.dec_pred_class_embed_share = dec_pred_class_embed_share
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        # prepare class & box embed
        _class_embed = nn.Linear(hidden_dim, num_classes)
        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # init the two embed layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        _class_embed.bias = mx.ones(self.num_classes) * bias_value
        nn.init.constant(0)(_bbox_embed.layers[-1].weight)
        nn.init.constant(0)(_bbox_embed.layers[-1].bias)

        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for i in range(
                transformer.num_decoder_layers)]
        else:
            box_embed_layerlist = [MLP(hidden_dim, hidden_dim, 4, 3)
                                   for i in range(transformer.num_decoder_layers)]
            for i in range(transformer.num_decoder_layers):
                nn.init.constant(0)(box_embed_layerlist[i].layers[-1].weight)
                nn.init.constant(0)(box_embed_layerlist[i].layers[-1].bias)
        if dec_pred_class_embed_share:
            class_embed_layerlist = [_class_embed for i in range(
                transformer.num_decoder_layers)]
        else:
            class_embed_layerlist = [nn.Linear(hidden_dim, num_classes) for i in range(
                transformer.num_decoder_layers)]
            for i in range(transformer.num_decoder_layers):
                class_embed_layerlist[i].bias = mx.ones(
                    self.num_classes) * bias_value
        self.bbox_embed = box_embed_layerlist
        self.class_embed = class_embed_layerlist
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        # two stage
        self.two_stage_type = two_stage_type
        self.two_stage_add_query_num = two_stage_add_query_num
        assert two_stage_type in [
            'no', 'standard'], "unknown param {} of two_stage_type".format(two_stage_type)
        if two_stage_type != 'no':
            if two_stage_bbox_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = MLP(
                    hidden_dim, hidden_dim, 4, 3)
                prior_prob = 0.01
                nn.init.constant(0)(
                    self.transformer.enc_out_bbox_embed.layers[-1].weight)
                nn.init.constant(0)(
                    self.transformer.enc_out_bbox_embed.layers[-1].bias)

            if two_stage_class_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = nn.Linear(
                    hidden_dim, num_classes)
                self.transformer.enc_out_class_embed.bias = mx.ones(
                    self.num_classes) * bias_value

            self.refpoint_embed = None
            if self.two_stage_add_query_num > 0:
                self.init_ref_points(two_stage_add_query_num)

        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ['sa', 'ca_label', 'ca_content']
        if decoder_sa_type == 'ca_label':
            self.label_embedding = nn.Embedding(num_classes, hidden_dim)
            for layer in self.transformer.decoder.layers:
                layer.label_embedding = self.label_embedding
        else:
            for layer in self.transformer.decoder.layers:
                layer.label_embedding = None
            self.label_embedding = None

    #     self._reset_parameters()

    # def _reset_parameters(self):
    #     # init input_proj
    #     for proj in self.input_proj:
    #         nn.init.he_uniform()(proj.weight)
    #         nn.init.constant(0)(proj.bias)

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)
        if self.random_refpoints_xy:

            nn.init.uniform()(self.refpoint_embed.weight[:, :2])
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(
                self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.freeze(recurse=False, keys="weight")

        if self.fix_refpoints_hw > 0:
            print("fix_refpoints_hw: {}".format(self.fix_refpoints_hw))
            assert self.random_refpoints_xy
            self.refpoint_embed.weight.data[:, 2:] = self.fix_refpoints_hw
            self.refpoint_embed.weight.data[:, 2:] = inverse_sigmoid(
                self.refpoint_embed.weight.data[:, 2:])
            self.refpoint_embed.freeze(recurse=False, keys="weight")
        elif int(self.fix_refpoints_hw) == -1:
            pass
        elif int(self.fix_refpoints_hw) == -2:
            print('learn a shared h and w')
            assert self.random_refpoints_xy
            self.refpoint_embed = nn.Embedding(use_num_queries, 2)
            nn.init.uniform()(self.refpoint_embed.weight[:, :2])
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(
                self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.freeze(recurse=False, keys="weight")
            self.hw_embed = nn.Embedding(1, 1)
        else:
            raise NotImplementedError(
                'Unknown fix_refpoints_hw {}'.format(self.fix_refpoints_hw))

    def __call__(self, samples: Dict[str, mx.array], targets: List = None):
        """ The forward expects a Dict[str, mx.array], which consists of:
               - samples['feature_map']: batched images, of shape [batch_size x 3 x H x W]
               - samples['mask']: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, mx.array)):
            samples = nested_array_dict_array_list(samples)
        features, poss = self.backbone(samples)
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src = feat['feature_map']
            mask = feat['mask']
            src = self.input_proj[l](src)
            srcs.append(src)
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1]['feature_map'])
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples['mask']
                interpolation = nn.Upsample(
                    scale_factor=(src.shape[1] / m.shape[1], src.shape[2] / m.shape[2]), mode="nearest"
                )
                mask = interpolation(m[..., None].astype(
                    mx.float32)).astype(mx.bool_).squeeze(-1)
                array_dict_additional = {'feature_map': src, 'mask': mask}
                pos_l = self.backbone.layers[1](
                    array_dict_additional).astype(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)
        if self.dn_number > 0 or targets is not None:
            num_q = self.num_queries * \
                self.num_patterns if self.num_patterns > 0 else self.num_queries
            input_query_label, input_query_bbox, attn_mask, dn_meta =\
                self.dn_encoder(dn_args=(targets, self.dn_number, self.dn_label_noise_ratio, self.dn_box_noise_scale),
                                training=self.training)
        else:
            assert targets is None
            input_query_bbox = input_query_label = attn_mask = dn_meta = None

        hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(
            srcs, masks, input_query_bbox, poss, input_query_label, attn_mask)
        # In case num object=0, we need to add a dummy object
        hs[0] += self.label_enc.weight[0, 0]*0.0
        # deformable-detr-like anchor update
        # reference_before_sigmoid = inverse_sigmoid(reference[:-1]) # n_dec, bs, nq, 4
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(zip(reference[:-1], self.bbox_embed, hs)):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + \
                inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = nn.sigmoid(layer_outputs_unsig)
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = mx.stack(outputs_coord_list)

        outputs_class = mx.stack([layer_cls_embed(layer_hs) for
                                  layer_cls_embed, layer_hs in zip(self.class_embed, hs)])
        if self.dn_number > 0 and dn_meta is not None:
            outputs_class, outputs_coord_list = \
                dn_post_process(outputs_class, outputs_coord_list,
                                dn_meta, self.aux_loss, self._set_aux_loss)
        out = {'pred_logits': outputs_class[-1],
               'pred_boxes': outputs_coord_list[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, outputs_coord_list)

        # for encoder output
        if hs_enc is not None:
            # prepare intermediate outputs
            interm_coord = ref_enc[-1]
            interm_class = self.transformer.enc_out_class_embed(hs_enc[-1])
            out['interm_outputs'] = {
                'pred_logits': interm_class, 'pred_boxes': interm_coord}
            out['interm_outputs_for_matching_pre'] = {
                'pred_logits': interm_class, 'pred_boxes': init_box_proposal}

            # prepare enc outputs
            if hs_enc.shape[0] > 1:
                enc_outputs_coord = []
                enc_outputs_class = []
                for layer_id, (layer_box_embed, layer_class_embed, layer_hs_enc, layer_ref_enc) in enumerate(zip(self.enc_bbox_embed, self.enc_class_embed, hs_enc[:-1], ref_enc[:-1])):
                    layer_enc_delta_unsig = layer_box_embed(layer_hs_enc)
                    layer_enc_outputs_coord_unsig = layer_enc_delta_unsig + \
                        inverse_sigmoid(layer_ref_enc)
                    layer_enc_outputs_coord = nn.sigmoid(
                        layer_enc_outputs_coord_unsig)
                    layer_enc_outputs_class = layer_class_embed(layer_hs_enc)
                    enc_outputs_coord.append(layer_enc_outputs_coord)
                    enc_outputs_class.append(layer_enc_outputs_class)

                out['enc_outputs'] = [
                    {'pred_logits': a, 'pred_boxes': b} for a, b in zip(enc_outputs_class, enc_outputs_coord)
                ]

        out['dn_meta'] = dn_meta

        return out

    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both an array and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


@MODULE_BUILD_FUNCS.register_with_name(module_name='dino')
def build_dino(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    # num_classes = 20 if args.dataset_file != 'coco' else 91
    # if args.dataset_file == "coco_panoptic":
    #     # for panoptic, we just add a num_classes that is large enough to hold
    #     # max_obj_id + 1, but the exact value doesn't really matter
    #     num_classes = 250
    # if args.dataset_file == 'o365':
    #     num_classes = 366
    # if args.dataset_file == 'vanke':
    #     num_classes = 51
    num_classes = args.num_classes

    backbone = build_backbone(args)

    transformer = build_deformable_transformer(args)

    try:
        match_unstable_error = args.match_unstable_error
        dn_labelbook_size = args.dn_labelbook_size
    except:
        match_unstable_error = True
        dn_labelbook_size = num_classes

    try:
        dec_pred_class_embed_share = args.dec_pred_class_embed_share
    except:
        dec_pred_class_embed_share = True
    try:
        dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share
    except:
        dec_pred_bbox_embed_share = True

    model = DINO(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=True,
        iter_update=True,
        query_dim=4,
        random_refpoints_xy=args.random_refpoints_xy,
        fix_refpoints_hw=args.fix_refpoints_hw,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        dec_pred_class_embed_share=dec_pred_class_embed_share,
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
        # two stage
        two_stage_type=args.two_stage_type,
        # box_share
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        decoder_sa_type=args.decoder_sa_type,
        num_patterns=args.num_patterns,
        dn_number=args.dn_number if args.use_dn else 0,
        dn_box_noise_scale=args.dn_box_noise_scale,
        dn_label_noise_ratio=args.dn_label_noise_ratio,
        dn_labelbook_size=dn_labelbook_size,
    )
    matcher = build_matcher(args)
    criterion = build_dino_loss_criterion(args, matcher)
    postprocessors = build_postprocessors(args)

    return model, criterion, postprocessors
