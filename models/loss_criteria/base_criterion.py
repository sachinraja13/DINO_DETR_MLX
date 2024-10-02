# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This is the original implementation of SetCriterion which will be deprecated in the next version.

We keep it here because our modified Criterion module is still under test.
"""

from typing import List
import mlx.core as mx
import mlx.nn as nn
from util import box_ops
import numpy as np
from ..utils import sigmoid_focal_loss


class BaseCriterion:
    """This class computes the loss for Conditional DETR.
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
        n_max_ground_truths=500
    ):
        """Create the criterion.
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
        self.focal_gamma = focal_gamma
        self.eos_coef = eos_coef
        self.loss_class_type = loss_class_type
        assert loss_class_type in [
            "ce_loss_cost",
            "focal_loss_cost",
        ], "only support ce loss and focal loss for computing classification loss"

        if self.loss_class_type == "ce_loss":
            self.empty_weight = torch.ones(self.num_classes + 1)
            self.empty_weight[0] = eos_coef

        self.pad_labels_to_n_max_ground_truths = pad_labels_to_n_max_ground_truths
        self.n_max_ground_truths = n_max_ground_truths

    def loss_labels(self, outputs, targets, indices, num_boxes, log=False):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = mx.concatenate(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = mx.full(src_logits.shape[:2], self.num_classes,
                                 dtype=mx.int16)
        target_classes[idx] = target_classes_o

        # Computation classification loss
        if self.loss_class_type == "ce_loss_cost":
            loss_class = nn.losses.cross_entropy(
                src_logits, target_classes, weights=self.empty_weight[target_classes])
        elif self.loss_class_type == "focal_loss_cost":
            # src_logits: (b, num_queries, num_classes) = (2, 300, 80)
            # target_classes_one_hot = (2, 300, 80)
            target_classes_onehot = mx.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                             dtype=src_logits.dtype)

            bs, nq = target_classes.shape
            target_classes_onehot[mx.arange(bs)[:, None, None], mx.arange(
                nq)[None, :, None], target_classes[:, :, None]] = 1

            target_classes_onehot = target_classes_onehot[:, :, :-1]
            loss_class = sigmoid_focal_loss(src_logits, target_classes_onehot,
                                            num_boxes, alpha=self.focal_alpha, gamma=self.focal_gamma) * src_logits.shape[1]

        losses = {"loss_class": loss_class}

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
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
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
            "class": self.loss_labels,
            'cardinality': self.loss_cardinality,
            "boxes": self.loss_boxes,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, return_indices=False):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc

             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        outputs_without_aux = {k: v for k,
                               v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
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
        for loss in self.losses:
            losses.update(self.get_loss(
                loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "loss_class_type: {}".format(self.loss_class_type),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "focal loss alpha: {}".format(self.alpha),
            "focal loss gamma: {}".format(self.gamma),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
