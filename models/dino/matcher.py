# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modules to compute the matching cost and solve the corresponding LSAP.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------


import os
from scipy.optimize import linear_sum_assignment
import numpy as np
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
import mlx.core as mx


class HungarianMatcher:
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        focal_alpha=0.25,
        pad_labels_to_n_max_ground_truths=False,
        n_max_ground_truths=500
    ):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"
        self.focal_alpha = focal_alpha
        self.pad_labels_to_n_max_ground_truths = pad_labels_to_n_max_ground_truths
        self.n_max_ground_truths = n_max_ground_truths

    @staticmethod
    def compute_l1_distance(src_boxes, tgt_boxes):
        src_boxes = src_boxes[:, None, :]  # [batch_size * num_queries, 1, 4]
        tgt_boxes = tgt_boxes[None, :, :]  # [1, num_targets, 4]
        # [batch_size * num_queries, num_targets]
        return mx.sum(mx.abs(src_boxes - tgt_boxes), axis=-1)

    def __call__(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Array of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Array of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Array of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Array of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        # [batch_size * num_queries, num_classes]
        out_prob = mx.sigmoid(outputs["pred_logits"].flatten(0, 1))
        out_bbox = outputs["pred_boxes"].flatten(
            0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = mx.concatenate(
            [v["labels"][0: v['num_objects']] for v in targets])
        tgt_bbox = mx.concatenate(
            [v["boxes"][0: v['num_objects']] for v in targets])

        # Compute the classification cost.
        alpha = self.focal_alpha
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * \
            (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * \
            ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = self.compute_l1_distance(out_bbox, tgt_bbox)
        # Compute the giou cost betwen boxes
        cost_giou = - \
            generalized_box_iou(box_cxcywh_to_xyxy(
                out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * \
            cost_class + self.cost_giou * cost_giou
        C = np.asarray(C.reshape(bs, num_queries, -1))

        sizes = [v["num_objects"] for v in targets]
        indices = []
        for i, c in enumerate(C):
            # print(c.shape)
            if i == 0:
                start_index = 0
                end_index = start_index + sizes[i]
            else:
                start_index = sizes[i-1]
                end_index = start_index + sizes[i]
            # print(i, start_index, end_index)
            cost_matrix = c[:, start_index:end_index]
            # print(cost_matrix.shape)
            indices.append(linear_sum_assignment(cost_matrix))
        return [(mx.array(i, dtype=mx.int64), mx.array(j, dtype=mx.int64)) for i, j in indices]


class SimpleMinsumMatcher:
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        focal_alpha=0.25,
        pad_labels_to_n_max_ground_truths=False,
        n_max_ground_truths=500
    ):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"
        self.focal_alpha = focal_alpha
        self.pad_labels_to_n_max_ground_truths = pad_labels_to_n_max_ground_truths
        self.n_max_ground_truths = n_max_ground_truths

    @staticmethod
    def compute_l1_distance(src_boxes, tgt_boxes):
        src_boxes = src_boxes[:, None, :]  # [batch_size * num_queries, 1, 4]
        tgt_boxes = tgt_boxes[None, :, :]  # [1, num_targets, 4]
        # [batch_size * num_queries, num_targets]
        return mx.sum(mx.abs(src_boxes - tgt_boxes), axis=-1)

    def __call__(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Array of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Array of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Array of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Array of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        # [batch_size * num_queries, num_classes]
        out_prob = mx.sigmoid(outputs["pred_logits"].flatten(0, 1))
        out_bbox = outputs["pred_boxes"].flatten(
            0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = mx.concatenate(
            [v["labels"][0: v['num_objects']] for v in targets])
        tgt_bbox = mx.concatenate(
            [v["boxes"][0: v['num_objects']] for v in targets])

        # Compute the classification cost.
        alpha = self.focal_alpha
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * \
            (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * \
            ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = self.compute_l1_distance(out_bbox, tgt_bbox)

        # Compute the giou cost betwen boxes
        cost_giou = - \
            generalized_box_iou(box_cxcywh_to_xyxy(
                out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * \
            cost_class + self.cost_giou * cost_giou
        C = np.asarray(C.reshape(bs, num_queries, -1))

        sizes = [v["num_objects"] for v in targets]
        indices = []

        for i, c in enumerate(C):
            # print(c.shape)
            if i == 0:
                start_index = 0
                end_index = start_index + sizes[i]
            else:
                start_index = sizes[i-1]
                end_index = start_index + sizes[i]
            cost_matrix = c[:, start_index:end_index]
            size_ = cost_matrix.shape[1]
            idx_i = cost_matrix.argmin(0)
            idx_j = mx.arange(size_)
            indices.append((idx_i, idx_j))

        return [(mx.array(i, dtype=mx.int64), mx.array(j, dtype=mx.int64)) for i, j in indices]


def build_matcher(args):
    assert args.matcher_type in [
        'HungarianMatcher', 'SimpleMinsumMatcher'], "Unknown args.matcher_type: {}".format(args.matcher_type)
    if args.matcher_type == 'HungarianMatcher':
        return HungarianMatcher(
            cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou,
            focal_alpha=args.focal_alpha, pad_labels_to_n_max_ground_truths=args.pad_labels_to_n_max_ground_truths,
            n_max_ground_truths=args.n_max_ground_truths
        )
    elif args.matcher_type == 'SimpleMinsumMatcher':
        return SimpleMinsumMatcher(
            cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou,
            focal_alpha=args.focal_alpha, pad_labels_to_n_max_ground_truths=args.pad_labels_to_n_max_ground_truths,
            n_max_ground_truths=args.n_max_ground_truths
        )
    else:
        raise NotImplementedError(
            "Unknown args.matcher_type: {}".format(args.matcher_type))
