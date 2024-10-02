import copy
import math
from typing import List, Dict
import mlx.core as mx
import mlx.nn as nn
from util import box_ops
import numpy as np
from .two_stage_criterion import TwoStageCriterion
from ..utils import sigmoid_focal_loss


class DINOCriterion(TwoStageCriterion):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        losses=["class", "boxes"],
        eos_coef=0.1,
        loss_class_type="focal_loss_cost",
        focal_alpha=0.25,
        focal_gamma=2.0,
        pad_labels_to_n_max_ground_truths=False,
        n_max_ground_truths=500,
        two_stage_binary_cls=False,
        use_dn=False
    ):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__(
            num_classes, matcher, weight_dict, losses,
            eos_coef, loss_class_type, focal_alpha, focal_gamma,
            pad_labels_to_n_max_ground_truths, n_max_ground_truths
        )
        self.use_dn = use_dn

    def prep_for_dn(self, dn_meta):
        output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
        num_dn_groups, pad_size = dn_meta['num_dn_group'], dn_meta['pad_size']
        assert pad_size % num_dn_groups == 0
        single_pad = pad_size//num_dn_groups

        return output_known_lbs_bboxes, single_pad, num_dn_groups

    def get_dn_losses(self, outputs, targets, num_boxes, losses):
        # prepare for dn loss
        if self.use_dn and 'dn_meta' in outputs:
            dn_meta = outputs['dn_meta']

        if self.use_dn and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
            output_known_lbs_bboxes, single_pad, scalar = self.prep_for_dn(
                dn_meta)
            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)):
                if targets[i]['num_objects'] > 0:
                    t = mx.arange(
                        targets[i]['num_objects'] - 1).astype(mx.int32)
                    t = mx.tile(t[None, ...], (scalar, 1))
                    tgt_idx = t.flatten()
                    output_idx = (mx.arange(scalar) *
                                  single_pad).astype(mx.int32)[..., None] + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = mx.array([]).astype(mx.int64)

                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

            output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
            l_dict = {}
            for loss in self.losses:
                kwargs = {}
                if 'class' in loss:
                    kwargs = {'log': False}
                l_dict.update(self.get_loss(
                    loss, output_known_lbs_bboxes, targets, dn_pos_idx, num_boxes*scalar, **kwargs))

            if self.use_dn:
                l_dict = {k + f'_dn': v for k, v in l_dict.items()}
                losses.update(l_dict)
        else:
            if self.use_dn:
                l_dict = dict()
                l_dict['loss_bbox_dn'] = mx.array(0.)
                l_dict['loss_giou_dn'] = mx.array(0.)
                l_dict['loss_class_dn'] = mx.array(0.)
                l_dict['loss_xy_dn'] = mx.array(0.)
                l_dict['loss_hw_dn'] = mx.array(0.)
                l_dict['cardinality_error_dn'] = mx.array(0.)
                losses.update(l_dict)
        return losses, output_known_lbs_bboxes, dn_pos_idx, dn_neg_idx, scalar

    def get_aux_dn_losses(self, outputs, targets, num_boxes, losses,
                          output_known_lbs_bboxes, dn_pos_idx, dn_neg_idx, scalar):
        if self.use_dn and 'dn_meta' in outputs:
            dn_meta = outputs['dn_meta']
        if 'aux_outputs' in outputs:
            if type(outputs["aux_outputs"]) == list:
                for idx, aux_outputs in enumerate(outputs['aux_outputs']):
                    if self.use_dn and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
                        aux_outputs_known = output_known_lbs_bboxes['aux_outputs'][idx]
                        l_dict = {}
                        for loss in self.losses:
                            kwargs = {}
                            if 'labels' in loss:
                                kwargs = {'log': False}

                            l_dict.update(self.get_loss(loss, aux_outputs_known, targets, dn_pos_idx, num_boxes*scalar,
                                                        **kwargs))
                        if self.use_dn:
                            l_dict = {k + f'_dn_{idx}': v for k,
                                      v in l_dict.items()}
                            losses.update(l_dict)
                    else:
                        if self.use_dn:
                            l_dict = dict()
                            l_dict['loss_bbox_dn'] = mx.array(0.)
                            l_dict['loss_giou_dn'] = mx.array(0.)
                            l_dict['loss_class_dn'] = mx.array(0.)
                            l_dict['loss_xy_dn'] = mx.array(0.)
                            l_dict['loss_hw_dn'] = mx.array(0.)
                            l_dict['cardinality_error_dn'] = mx.array(0.)
                            l_dict = {k + f'_{idx}': v for k,
                                      v in l_dict.items()}
                            losses.update(l_dict)
            else:
                aux_outputs = outputs["aux_outputs"]
                if self.use_dn and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
                    aux_outputs_known = output_known_lbs_bboxes['aux_outputs']
                    l_dict = {}
                    for loss in self.losses:
                        kwargs = {}
                        if 'labels' in loss:
                            kwargs = {'log': False}

                        l_dict.update(self.get_loss(loss, aux_outputs_known, targets, dn_pos_idx, num_boxes*scalar,
                                                    **kwargs))
                    if self.use_dn:
                        l_dict = {k + f'_dn': v for k,
                                  v in l_dict.items()}
                        losses.update(l_dict)
                else:
                    if self.use_dn:
                        l_dict = dict()
                        l_dict['loss_bbox_dn'] = mx.array(0.)
                        l_dict['loss_giou_dn'] = mx.array(0.)
                        l_dict['loss_class_dn'] = mx.array(0.)
                        l_dict['loss_xy_dn'] = mx.array(0.)
                        l_dict['loss_hw_dn'] = mx.array(0.)
                        l_dict['cardinality_error_dn'] = mx.array(0.)
                        l_dict = {k: v for k,
                                  v in l_dict.items()}
                        losses.update(l_dict)
        return losses

    def get_interim_losses(self, outputs, targets, num_boxes, losses, indices_list, return_indices):
        # interm_outputs loss
        if 'interm_outputs' in outputs:
            interm_outputs = outputs['interm_outputs']
            indices = self.matcher(interm_outputs, targets)
            if return_indices:
                indices_list.append(indices)
            for loss in self.losses:
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs = {'log': False}
                l_dict = self.get_loss(
                    loss, interm_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)
        return indices_list, losses

    def forward(self, outputs, targets, return_indices=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of arrays, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc

             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        num_boxes = sum(t["num_objects"] for t in targets)
        num_boxes = mx.array([num_boxes], dtype=mx.float32)
        num_boxes = mx.clip(num_boxes, 1, None).item()

        losses, indices_list = super().forward(
            outputs, targets, return_indices=True)

        if self.use_dn and 'dn_meta' in outputs:
            dn_meta = outputs['dn_meta']

        if self.use_dn and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
            losses, output_known_lbs_bboxes, dn_pos_idx, dn_neg_idx, scalar = self.get_dn_losses(
                outputs, targets, num_boxes, losses)
            losses = self.get_aux_dn_losses(
                outputs, targets, num_boxes, losses,
                output_known_lbs_bboxes, dn_pos_idx, dn_neg_idx, scalar)

        losses = self.get_encoder_losses(outputs, targets, num_boxes, losses)
        indices_list, losses = self.get_interim_losses(
            outputs, targets, num_boxes, losses, indices_list, return_indices)

        if return_indices:
            return losses, indices_list

        return losses
