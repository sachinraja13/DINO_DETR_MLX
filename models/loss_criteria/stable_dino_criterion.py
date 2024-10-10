import copy
import math
from typing import List, Dict
import mlx.core as mx
import mlx.nn as nn
from util import box_ops
import numpy as np
from .two_stage_criterion import TwoStageCriterion
from .dino_criterion import DINOCriterion
from ..utils import sigmoid_focal_loss


class StableDINOCriterion(DINOCriterion):
    """This class computes the loss for DETR.
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
        eos_coef=None,
        loss_class_type="focal_loss_cost",
        focal_alpha=0.25,
        focal_gamma=2,
        ta_alpha=0.0,
        ta_beta=2.0,
        two_stage_binary_cls=False,
        use_dn=False,
        use_ce_loss_type="stable-dino",
        stg1_assigner=None,
        enc_kd_loss_weight=-1.0,
        enc_kd_loss_gamma=2.0,
        target_post_process="none",
        pad_labels_to_n_max_ground_truths=False,
        n_max_ground_truths=500,
    ):
        super().__init__(
            num_classes, matcher, weight_dict, losses,
            eos_coef, loss_class_type, focal_alpha, focal_gamma,
            pad_labels_to_n_max_ground_truths, n_max_ground_truths,
            two_stage_binary_cls, use_dn
        )
        self.two_stage_binary_cls = two_stage_binary_cls
        if self.two_stage_binary_cls:
            raise NotImplementedError

        # refer to task-aligned loss
        self.ta_alpha = ta_alpha
        self.ta_beta = ta_beta

        self.use_ce_loss_type = use_ce_loss_type

        self.stg1_assigner = stg1_assigner
        if stg1_assigner == "deta":
            raise NotImplementedError
        else:
            self.stg1_assigner_func = None

        self.enc_kd_loss_weight = enc_kd_loss_weight
        self.enc_kd_loss_gamma = enc_kd_loss_gamma

        self.target_post_process = target_post_process

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]  # bs, nq, 80
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = mx.concatenate(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = mx.full(src_logits.shape[:2], self.num_classes,
                                 dtype=mx.int32)
        target_classes[idx] = target_classes_o
        target_classes_onehot = mx.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                         dtype=src_logits.dtype)

        bs, nq = target_classes.shape
        target_classes_onehot[mx.arange(bs)[:, None, None], mx.arange(
            nq)[None, :, None], target_classes[:, :, None]] = 1

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        out_prob = mx.sigmoid(src_logits)
        bs, nq = src_logits.shape[:2]
        if self.use_ce_loss_type in ['stable-dino']:
            src_boxes = outputs["pred_boxes"][idx]  # (nbox, 4)
            tgt_bbox = mx.concatenate(
                [t['boxes'][i] for t, (_, i) in zip(targets, indices)], axis=0)
            tgt_labels = mx.concatenate(
                [t['labels'][i] for t, (_, i) in zip(targets, indices)], axis=0)
            iou = box_ops.box_iou(box_ops.box_cxcywh_to_xyxy(
                src_boxes), box_ops.box_cxcywh_to_xyxy(tgt_bbox))[0].diag()
            _s = out_prob
            _u = mx.zeros_like(_s)
            _u[idx[0], idx[1], tgt_labels] = iou
            _t = mx.power(_s, self.ta_alpha) * mx.power(_u, self.ta_beta)
            # (b, num_queries, num_classes)
            # p**alpha * u**beta if pos, 0 if neg
            if self.target_post_process == "exp":
                _t = (_t.exp() - 1) / (np.e - 1)
            elif self.target_post_process == "sin":
                _t = (_t * np.pi / 2).sin()
            else:
                assert self.target_post_process == "none", self.target_post_process

        # get loss
        if self.use_ce_loss_type in ['stable-dino']:
            # refer to: Tal loss
            # we first shift the quality _t to larger than prob
            # follow the paper: TOOD: Task-aligned One-stage Object Detection.
            # link: https://readpaper.com/paper/3201828441
            ngt_in_batch = [(t["num_objects"]) for t in targets]
            norm_t = mx.zeros_like(_t)
            all_out_bbox = outputs["pred_boxes"]
            all_tgt_bbox = mx.concatenate(
                [v["boxes"][0: v['num_objects']] for v in targets])  # nbox
            all_iou = box_ops.box_iou(box_ops.box_cxcywh_to_xyxy(all_out_bbox.flatten(
                0, 1)), box_ops.box_cxcywh_to_xyxy(all_tgt_bbox))[0].reshape(bs, nq, -1)  # (b*num_queries, ngt)
            cum = 0
            for i in range(bs):
                if self.use_ce_loss_type == 'stable-dino':
                    max_iou = 1.0
                    # import ipdb; ipdb.set_trace()
                else:
                    if ngt_in_batch[i] == 0:
                        max_iou = 1.0
                    else:
                        max_iou = all_iou[i, :, cum:cum+ngt_in_batch[i]].max()
                normalizer = max(
                    (max_iou / mx.stop_gradient(_t[i].max() + 1e-8)), 1)
                norm_t[i] = _t[i] * normalizer
                cum += ngt_in_batch[i]
            norm_t = mx.stop_gradient(norm_t)
            neg_loss = (1 - self.focal_alpha) * (out_prob**self.focal_gamma) * \
                (1 - target_classes_onehot) * (-(1 - out_prob + 1e-8).log())
            pos_loss = target_classes_onehot * (
                self.focal_alpha * ((norm_t - out_prob)**self.focal_gamma) *
                (-norm_t * out_prob.log() - (1 - norm_t) * (1 - out_prob + 1e-8).log())
            )
            loss_class = (pos_loss + neg_loss).sum() / num_boxes

        elif self.use_ce_loss_type == "none":
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
        else:
            assert self.use_ce_loss_type in [
                'none', 'stable-dino'], "use_ce_loss_type should be none or stable-dino"
        losses = {"loss_class": loss_class}
        return losses

    def forward(self, outputs, targets, return_indices=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of arrays, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc

             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        if return_indices:
            losses, indices_list = super().forward(
                outputs, targets, return_indices)
        else:
            losses = super().forward(outputs, targets, return_indices)

        if return_indices:
            return losses, indices_list

        return losses
