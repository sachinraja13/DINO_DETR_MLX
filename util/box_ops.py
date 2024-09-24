import os
import mlx.core as mx




def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return mx.stack(b, axis=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return mx.stack(b, axis=-1)


def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by their
    (x1, y1, x2, y2) coordinates.

    Args:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format with
            ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Returns:
        Tensor[N]: the area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# Compute the IoU and return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = mx.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = mx.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = mx.clip((rb - lt), 0, None)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-6)
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    assert (boxes1[..., 2:] >= boxes1[..., :2]).all()
    assert (boxes2[..., 2:] >= boxes2[..., :2]).all()

    iou, union = box_iou(boxes1, boxes2)

    lt = mx.minimum(boxes1[..., None, :2], boxes2[..., :2])
    rb = mx.maximum(boxes1[..., None, 2:], boxes2[..., 2:])

    wh = mx.clip((rb - lt), 0, None)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / (area + 1e-6)


# Pairwise IoU and return the union
def box_iou_pairwise(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = mx.maximum(boxes1[:, :2], boxes2[:, :2])  # [N,2]
    rb = mx.minimum(boxes1[:, 2:], boxes2[:, 2:])  # [N,2]

    wh = mx.clip((rb - lt), 0, None)  # [N,2]
    inter = wh[:, 0] * wh[:, 1]  # [N]

    union = area1 + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou_pairwise(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    Input:
        - boxes1, boxes2: N,4
    Output:
        - giou: N, 4
    """
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    assert boxes1.shape == boxes2.shape

    iou, union = box_iou_pairwise(boxes1, boxes2)  # N, 4

    lt = mx.minimum(boxes1[:, :2], boxes2[:, :2])
    rb = mx.maximum(boxes1[:, 2:], boxes2[:, 2:])

    wh = mx.clip((rb - lt), 0, None)  # [N,2]
    area = wh[:, 0] * wh[:, 1]

    return iou - (area - union) / area

