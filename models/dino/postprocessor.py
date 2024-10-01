import copy
import math
from typing import List, Dict
import mlx.core as mx
import mlx.nn as nn
from util import box_ops
import numpy as np
from .utils import sigmoid_focal_loss, MLP


class PostProcessor:
    """ This module converts the model's output into the format expected by the coco api"""

    def __init__(self, num_select=100, nms_iou_threshold=-1) -> None:
        super().__init__()
        self.num_select = num_select
        self.nms_iou_threshold = nms_iou_threshold

    def __call__(self, outputs, target_sizes, not_to_xyxy=False, test=False):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: array of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        num_select = self.num_select
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = mx.sigmoid(out_logits)
        # topk_values = mx.topk(prob.reshape(out_logits.shape[0], -1), num_select, axis=1)
        topk_indexes = mx.argpartition(prob.reshape(
            out_logits.shape[0], -1) * -1, num_select, axis=1)[:, :num_select]
        prob = prob.reshape(out_logits.shape[0], -1)
        topk_values = prob[mx.arange(out_logits.shape[0])[
            :, None], topk_indexes]
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        if not_to_xyxy:
            boxes = out_bbox
        else:
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
            boxes = mx.clip(boxes, 0, 1)
        if test:
            assert not not_to_xyxy
            boxes[:, :, 2:] = boxes[:, :, 2:] - boxes[:, :, :2]
        boxes = boxes[mx.arange(topk_boxes.shape[0])[
            :, None, None], topk_boxes[:, :, None], mx.arange(4)[None, None, :]]

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes[..., 0], target_sizes[..., 1]
        scale_fct = mx.stack([img_w, img_h, img_w, img_h], axis=1)
        boxes = boxes * scale_fct[:, None, :]

        # if self.nms_iou_threshold > 0:
        #     item_indices = [nms(b, s, iou_threshold=self.nms_iou_threshold) for b,s in zip(boxes, scores)]

        # results = [{'scores': s[i], 'labels': l[i], 'boxes': b[i]} for s, l, b, i in zip(scores, labels, boxes, item_indices)]
        # else:
        results = [{'scores': s, 'labels': l, 'boxes': b}
                   for s, l, b in zip(scores, labels, boxes)]

        return results
