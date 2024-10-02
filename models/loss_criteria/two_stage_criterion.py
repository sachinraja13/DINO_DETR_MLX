
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
        pad_labels_to_n_max_ground_truths=False,
        n_max_ground_truths=500,
        two_stage_binary_cls=False,
    ):
        super().__init__(
            num_classes, matcher, weight_dict, losses,
            eos_coef, loss_class_type, focal_alpha, focal_gamma,
            pad_labels_to_n_max_ground_truths, n_max_ground_truths
        )
        self.two_stage_binary_cls = two_stage_binary_cls

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
        indices = self.matcher(outputs_without_aux, targets)
        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(t["num_objects"] for t in targets)
        num_boxes = mx.array([num_boxes], dtype=mx.float32)
        num_boxes = mx.clip(num_boxes, 1, None).item()
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

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses
