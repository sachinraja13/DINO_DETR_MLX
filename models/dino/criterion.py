import copy
import math
from typing import List, Dict
import mlx.core as mx
import mlx.nn as nn
from util import box_ops
import numpy as np
from .utils import sigmoid_focal_loss


class SetCriterion:
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
        losses,
        focal_alpha,
        use_dn,
        training=True,
        pad_labels_to_n_max_ground_truths=False,
        n_max_ground_truths=500
    ):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.use_dn = use_dn
        self.training = training
        self.pad_labels_to_n_max_ground_truths = pad_labels_to_n_max_ground_truths
        self.n_max_ground_truths = n_max_ground_truths

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing an array of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = mx.concatenate(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = mx.full(src_logits.shape[:2], self.num_classes,
                                 dtype=mx.int16)
        target_classes[idx] = target_classes_o

        target_classes_onehot = mx.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                         dtype=src_logits.dtype)

        bs, nq = target_classes.shape
        target_classes_onehot[mx.arange(bs)[:, None, None], mx.arange(
            nq)[None, :, None], target_classes[:, :, None]] = 1

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot,
                                     num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        # if log:
        #     # TODO this should probably be a separate loss, not hacked in this one here
        #     losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        tgt_lengths = mx.array([v["num_objects"] for v in targets])
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) !=
                     pred_logits.shape[-1] - 1).sum(1)
        card_err = nn.losses.l1_loss(card_pred.astype(
            mx.float32), tgt_lengths.astype(mx.float32))
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing an array of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = mx.concatenate(
            [t['boxes'][i] for t, (_, i) in zip(targets, indices)], axis=0)

        loss_bbox = nn.losses.l1_loss(
            src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        if src_boxes.shape[0] > 0 and target_boxes.shape[0] > 0:
            loss_giou = 1 - mx.diag(box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes)))
        else:
            loss_giou = mx.array(1.0)
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        # calculate the x,y and h,w loss

        losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
        losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = mx.concatenate([mx.full(src.shape, i)
                                   for i, (src, _) in enumerate(indices)])
        src_idx = mx.concatenate([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = mx.concatenate([mx.full(tgt.shape, i)
                                   for i, (_, tgt) in enumerate(indices)])
        tgt_idx = mx.concatenate([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, return_indices=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of arrays, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc

             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        outputs_without_aux = {k: v for k,
                               v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets)

        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(t["num_objects"] for t in targets)
        num_boxes = mx.array([num_boxes], dtype=mx.float32)
        num_boxes = mx.clip(num_boxes, 1, None).item()

        # Compute all the requested losses
        losses = {}

        # prepare for dn loss
        dn_meta = outputs['dn_meta']

        if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
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
                if 'labels' in loss:
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
                l_dict['loss_ce_dn'] = mx.array(0.)
                l_dict['loss_xy_dn'] = mx.array(0.)
                l_dict['loss_hw_dn'] = mx.array(0.)
                l_dict['cardinality_error_dn'] = mx.array(0.)
                losses.update(l_dict)

        for loss in self.losses:
            losses.update(self.get_loss(
                loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for idx, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

                if self.training and dn_meta and 'output_known_lbs_bboxes' in dn_meta:
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
                        l_dict['loss_ce_dn'] = mx.array(0.)
                        l_dict['loss_xy_dn'] = mx.array(0.)
                        l_dict['loss_hw_dn'] = mx.array(0.)
                        l_dict['cardinality_error_dn'] = mx.array(0.)
                        l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                        losses.update(l_dict)

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

        # enc output loss
        if 'enc_outputs' in outputs:
            for i, enc_outputs in enumerate(outputs['enc_outputs']):
                indices = self.matcher(enc_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(
                        loss, enc_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses

    def prep_for_dn(self, dn_meta):
        output_known_lbs_bboxes = dn_meta['output_known_lbs_bboxes']
        num_dn_groups, pad_size = dn_meta['num_dn_group'], dn_meta['pad_size']
        assert pad_size % num_dn_groups == 0
        single_pad = pad_size//num_dn_groups

        return output_known_lbs_bboxes, single_pad, num_dn_groups
