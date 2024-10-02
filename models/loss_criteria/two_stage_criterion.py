
from typing import List
import mlx.core as mx
import mlx.nn as nn
from util import box_ops
import numpy as np
from ..utils import sigmoid_focal_loss
from .base_criterion import BaseCriterion


class TwoStageCriterion(BaseCriterion):
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
        two_stage_binary_cls=False,
        pad_labels_to_n_max_ground_truths=False,
        n_max_ground_truths=500,
    ):
        super().__init__(
            num_classes, matcher, weight_dict, losses,
            eos_coef, loss_class_type, focal_alpha, focal_gamma,
            pad_labels_to_n_max_ground_truths, n_max_ground_truths
        )
        self.two_stage_binary_cls = two_stage_binary_cls

    def get_encoder_losses(self, outputs, targets, num_boxes, losses):
        # enc output loss
        if 'enc_outputs' in outputs:
            if type(outputs["enc_outputs"]) == list:
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
                        l_dict = {k + f'_enc_{i}': v for k,
                                  v in l_dict.items()}
                        losses.update(l_dict)
            else:
                enc_outputs = outputs["enc_outputs"]
                if self.two_stage_binary_cls:
                    for bt in targets:
                        bt["labels"] = mx.zeros_like(bt["labels"])
                indices = self.matcher(enc_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(
                        loss, enc_outputs, targets, indices, num_boxes)
                    l_dict = {k + "_enc": v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses

    def forward(self, outputs, targets, return_indices=False):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc

             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.

        """
        num_boxes = sum(t["num_objects"] for t in targets)
        num_boxes = mx.array([num_boxes], dtype=mx.float32)
        num_boxes = mx.clip(num_boxes, 1, None).item()

        if return_indices:
            losses, indices_list = super().forward(
                outputs, targets, return_indices)
        else:
            losses = super().forward(outputs, targets, return_indices)
        losses = self.get_encoder_losses(outputs, targets, num_boxes, losses)

        if return_indices:
            return losses, indices_list

        return losses
