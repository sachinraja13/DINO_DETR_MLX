import os
from scipy.optimize import linear_sum_assignment
import numpy as np
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
import mlx.core as mx


class StableHungarianMatcher:
    """HungarianMatcher which computes an assignment between targets and predictions.

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).

    Args:
        cost_class (float): The relative weight of the classification error
            in the matching cost. Default: 1.
        cost_bbox (float): The relative weight of the L1 error of the bounding box
            coordinates in the matching cost. Default: 1.
        cost_giou (float): This is the relative weight of the giou loss of
            the bounding box in the matching cost. Default: 1.
        cost_class_type (str): How the classification error is calculated.
            Choose from ``["ce_cost", "focal_loss_cost"]``. Default: "focal_loss_cost".
        alpha (float): Weighting factor in range (0, 1) to balance positive vs
            negative examples in focal loss. Default: 0.25.
        gamma (float): Exponent of modulating factor (1 - p_t) to balance easy vs
            hard examples in focal loss. Default: 2.
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        cost_class_type: str = "focal_loss_cost",
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        cec_beta: float = -1.0,
        pad_labels_to_n_max_ground_truths=False,
        n_max_ground_truths=500
    ):
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_class_type = cost_class_type
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.cec_beta = cec_beta  # ce constraint beta
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"
        assert cost_class_type in {
            "ce_loss_cost",
            "focal_loss_cost",
            "focal_loss_cost2"
        }, "only support ce loss or focal loss for computing class cost"

    @staticmethod
    def compute_l1_distance(src_boxes, tgt_boxes):
        src_boxes = src_boxes[:, None, :]  # [batch_size * num_queries, 1, 4]
        tgt_boxes = tgt_boxes[None, :, :]  # [1, num_targets, 4]
        # [batch_size * num_queries, num_targets]
        return mx.sum(mx.abs(src_boxes - tgt_boxes), axis=-1)

    def __call__(self, outputs, targets, repeat_times=1):
        """Forward function for `HungarianMatcher` which performs the matching.

        Args:
            outputs (Dict[str, torch.Tensor]): This is a dict that contains at least these entries:

                - ``"pred_logits"``: Tensor of shape (bs, num_queries, num_classes) with the classification logits.
                - ``"pred_boxes"``: Tensor of shape (bs, num_queries, 4) with the predicted box coordinates.

            targets (List[Dict[str, torch.Tensor]]): This is a list of targets (len(targets) = batch_size),
                where each target is a dict containing:

                - ``"labels"``: Tensor of shape (num_target_boxes, ) (where num_target_boxes is the number of ground-truth objects in the target) containing the class labels.  # noqa
                - ``"boxes"``: Tensor of shape (num_target_boxes, 4) containing the target box coordinates.

        Returns:
            list[torch.Tensor]: A list of size batch_size, containing tuples of `(index_i, index_j)` where:

                - ``index_i`` is the indices of the selected predictions (in order)
                - ``index_j`` is the indices of the corresponding selected targets (in order)

            For each batch element, it holds: `len(index_i) = len(index_j) = min(num_queries, num_target_boxes)`
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        if self.cost_class_type == "ce_loss_cost":
            out_prob = mx.softmax(outputs["pred_logits"].flatten(0, 1))
        else:
            out_prob = mx.sigmoid(outputs["pred_logits"].flatten(0, 1))
        out_bbox = outputs["pred_boxes"].flatten(
            0, 1)  # [batch_size * num_queries, 4]
        tgt_ids = mx.concatenate(
            [v["labels"][0: v['num_objects']] for v in targets])
        tgt_bbox = mx.concatenate(
            [v["boxes"][0: v['num_objects']] for v in targets])
        if repeat_times > 1:
            tgt_ids = mx.repeat(tgt_ids, repeat_times)
            tgt_bbox = mx.repeat(tgt_bbox, repeat_times, axis=0)
        if self.cec_beta > 0:
            giou = (generalized_box_iou(box_cxcywh_to_xyxy(
                out_bbox), box_cxcywh_to_xyxy(tgt_bbox)) + 1) / 2
            _s = out_prob
            _u = mx.zeros_like(_s)
            _u[:, tgt_ids] = giou
            _uv = mx.reshape(_u, (bs, num_queries, -1))
            _u_max = _uv.flatten(1, 2).max(-1)
            scalar = 1 / (mx.array(_u_max) + 1e-8)
            scalar = mx.maximum(scalar, mx.ones_like(scalar))
            _uv = _uv * scalar[:, None, None]
            _u = _uv.reshape(bs * num_queries, -1)
            out_prob = _s * mx.power(_u, self.cec_beta)
            # ce constraint beta

        # Compute the classification cost.
        if self.cost_class_type == "ce_loss_cost":
            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]
        elif self.cost_class_type == "focal_loss_cost":
            alpha = self.focal_alpha
            gamma = self.focal_gamma
            neg_cost_class = (1 - alpha) * (out_prob**gamma) * \
                (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * \
                ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - \
                neg_cost_class[:, tgt_ids]

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

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_bbox: {}".format(self.cost_bbox),
            "cost_giou: {}".format(self.cost_giou),
            "cost_class_type: {}".format(self.cost_class_type),
            "focal cost alpha: {}".format(self.alpha),
            "focal cost gamma: {}".format(self.gamma),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
