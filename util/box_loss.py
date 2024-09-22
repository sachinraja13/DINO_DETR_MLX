# borrow from https://github.com/Zzh-tju/CIoU/blob/master/layers/modules/multibox_loss.py

import math
import mlx.core as mx



def ciou(bboxes1, bboxes2):
    bboxes1 = mx.sigmoid(bboxes1)
    bboxes2 = mx.sigmoid(bboxes2)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    cious = mx.zeros((rows, cols))
    if rows * cols == 0:
        return cious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        cious = mx.zeros((cols, rows))
        exchange = True
    w1 = mx.exp(bboxes1[:, 2])
    h1 = mx.exp(bboxes1[:, 3])
    w2 = mx.exp(bboxes2[:, 2])
    h2 = mx.exp(bboxes2[:, 3])
    area1 = w1 * h1
    area2 = w2 * h2
    center_x1 = bboxes1[:, 0]
    center_y1 = bboxes1[:, 1]
    center_x2 = bboxes2[:, 0]
    center_y2 = bboxes2[:, 1]

    inter_l = mx.maximum(center_x1 - w1 / 2, center_x2 - w2 / 2)
    inter_r = mx.minimum(center_x1 + w1 / 2, center_x2 + w2 / 2)
    inter_t = mx.maximum(center_y1 - h1 / 2, center_y2 - h2 / 2)
    inter_b = mx.minimum(center_y1 + h1 / 2, center_y2 + h2 / 2)
    inter_area = mx.clip((inter_r - inter_l), 0, None) * mx.clip((inter_b - inter_t), 0, None)

    c_l = mx.minimum(center_x1 - w1 / 2, center_x2 - w2 / 2)
    c_r = mx.maximum(center_x1 + w1 / 2, center_x2 + w2 / 2)
    c_t = mx.minimum(center_y1 - h1 / 2, center_y2 - h2 / 2)
    c_b = mx.maximum(center_y1 + h1 / 2, center_y2 + h2 / 2)

    inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
    c_diag = mx.clip((c_r - c_l), 0, None) ** 2 + mx.clip((c_b - c_t), 0, None) ** 2

    union = area1 + area2 - inter_area
    u = inter_diag / c_diag
    iou = inter_area / union
    v = (4 / (math.pi ** 2)) * mx.power((mx.arctan(w2 / h2) - mx.arctan(w1 / h1)), 2)
    S = (iou > 0.5).astype(mx.float32)
    alpha = S * v / (1 - iou + v)
    cious = iou - u - alpha * v
    cious = mx.clip(cious, -1.0, 1.0)
    if exchange:
        cious = cious.T
    return 1 - cious


def diou(bboxes1, bboxes2):
    bboxes1 = mx.sigmoid(bboxes1)
    bboxes2 = mx.sigmoid(bboxes2)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    dious = mx.zeros((rows, cols))
    if rows * cols == 0:
        return dious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        dious = mx.zeros((cols, rows))
        exchange = True
    w1 = mx.exp(bboxes1[:, 2])
    h1 = mx.exp(bboxes1[:, 3])
    w2 = mx.exp(bboxes2[:, 2])
    h2 = mx.exp(bboxes2[:, 3])
    area1 = w1 * h1
    area2 = w2 * h2
    center_x1 = bboxes1[:, 0]
    center_y1 = bboxes1[:, 1]
    center_x2 = bboxes2[:, 0]
    center_y2 = bboxes2[:, 1]

    inter_l = mx.maximum(center_x1 - w1 / 2, center_x2 - w2 / 2)
    inter_r = mx.minimum(center_x1 + w1 / 2, center_x2 + w2 / 2)
    inter_t = mx.maximum(center_y1 - h1 / 2, center_y2 - h2 / 2)
    inter_b = mx.minimum(center_y1 + h1 / 2, center_y2 + h2 / 2)
    inter_area = mx.clip((inter_r - inter_l), 0, None) * mx.clip((inter_b - inter_t), 0, None)

    c_l = mx.minimum(center_x1 - w1 / 2, center_x2 - w2 / 2)
    c_r = mx.maximum(center_x1 + w1 / 2, center_x2 + w2 / 2)
    c_t = mx.minimum(center_y1 - h1 / 2, center_y2 - h2 / 2)
    c_b = mx.maximum(center_y1 + h1 / 2, center_y2 + h2 / 2)

    inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
    c_diag = mx.clip((c_r - c_l), 0, None) ** 2 + mx.clip((c_b - c_t), 0, None) ** 2

    union = area1 + area2 - inter_area
    u = inter_diag / c_diag
    iou = inter_area / union
    dious = iou - u
    dious = mx.clip(dious, -1.0, 1.0)
    if exchange:
        dious = dious.T
    return 1 - dious


