# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import os
import sys
from typing import Iterable
from functools import partial
from util.utils import slprint
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import util.misc as utils
from datasets.coco_eval import CocoEvaluator


def train_one_epoch(model: nn.Module, criterion,
                    data_loader: Iterable, optimizer: optim.Optimizer, epoch: int, max_norm: float = 0, 
                    wo_class_error=False, args=None, logger=None, print_freq=100,
                    compile_forward=False, compile_backward=False):
    
    model.train()
    state = [model.state, optimizer.state, mx.random.state]
    mx.eval(state)
    def loss_fn(array_dict, targets, need_tgt_for_training=False, return_outputs=False):
        if compile_forward and not compile_backward:
            model_forward = mx.compile(model, inputs=state, outputs=state)
        else:
            model_forward = model
        if need_tgt_for_training:
            outputs = model_forward(array_dict, targets)
        else:
            outputs = model_forward(array_dict)
        loss_dict = criterion.forward(outputs, targets)
        weight_dict = criterion.weight_dict
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        if return_outputs:
            return loss, loss_dict, outputs
        return loss, loss_dict
    
    def step(array_dict, targets, need_tgt_for_training=False, return_outputs=False):
        train_step_fn = nn.value_and_grad(model, loss_fn)
        (loss_value, loss_dict), grads = train_step_fn(samples, targets, need_tgt_for_training, return_outputs=False)
        grads, total_norm = optim.clip_grad_norm(grads, max_norm=max_norm)
        optimizer.update(model, grads)
        return loss_value, loss_dict
    
    if compile_forward and compile_backward:
        step = mx.compile(step, inputs=state, outputs=state)
    
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    _cnt = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):
        loss_value, loss_dict = step(samples, targets, need_tgt_for_training, return_outputs=False)
        mx.eval(state)
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        metric_logger.update(loss=loss_value, **loss_dict)
        if 'class_error' in loss_dict:
            metric_logger.update(class_error=loss_dict['class_error'])
        metric_logger.update(lr=optimizer.learning_rate)

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    # gather the stats from all processes
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    return resstat



def evaluate(model, criterion, postprocessors, data_loader, base_ds, output_dir, wo_class_error=False, args=None, logger=None):
    model.eval()
    state = [model.state, mx.random.state]
    mx.eval(state)
    def loss_fn(array_dict, targets, need_tgt_for_training=False, return_outputs=False):
        if need_tgt_for_training:
            outputs = model(array_dict, targets)
        else:
            outputs = model(array_dict)
        loss_dict = criterion.forward(outputs, targets)
        weight_dict = criterion.weight_dict
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        if return_outputs:
            return loss, loss_dict, outputs
        return loss, loss_dict
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('bbox') if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
    coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]


    _cnt = 0
    output_state_dict = {} # for debug only
    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):

        loss_value, loss_dict, outputs = loss_fn(samples, targets, need_tgt_for_training, return_outputs=False)
        weight_dict = criterion.weight_dict
        mx.eval(model)
        metric_logger.update(loss=loss_value, **loss_dict)

        if 'class_error' in loss_dict:
            metric_logger.update(class_error=loss_dict['class_error'])

        orig_target_sizes = mx.stack([t["orig_size"] for t in targets], axis=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        if coco_evaluator is not None:
            coco_evaluator.update(res)

        
        if args.save_results:
            res_score = outputs['res_score']
            res_label = outputs['res_label']
            res_bbox = outputs['res_bbox']
            res_idx = outputs['res_idx']


            for i, (tgt, res, outbbox) in enumerate(zip(targets, results, outputs['pred_boxes'])):
                """
                pred vars:
                    K: number of bbox pred
                    score: array(K),
                    label: list(len: K),
                    bbox: array(K, 4)
                    idx: list(len: K)
                tgt: dict.

                """
                # compare gt and res (after postprocess)
                gt_bbox = tgt['boxes']
                gt_label = tgt['labels']
                gt_info = mx.concatenate((gt_bbox, gt_label.unsqueeze(-1)), 1)
                
                # img_h, img_w = tgt['orig_size'].unbind()
                # scale_fct = mx.stack([img_w, img_h, img_w, img_h], axis=0)
                # _res_bbox = res['boxes'] / scale_fct
                _res_bbox = outbbox
                _res_prob = res['scores']
                _res_label = res['labels']
                res_info = mx.concatenate((_res_bbox, _res_prob[..., None], _res_label[..., None]), 1)
                # import ipdb;ipdb.set_trace()

                if 'gt_info' not in output_state_dict:
                    output_state_dict['gt_info'] = []
                output_state_dict['gt_info'].append(gt_info)

                if 'res_info' not in output_state_dict:
                    output_state_dict['res_info'] = []
                output_state_dict['res_info'].append(res_info)

            # # for debug only
            # import random
            # if random.random() > 0.7:
            #     print("Now let's break")
            #     break

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if args.save_results:
        import os.path as osp
        
        # output_state_dict['gt_info'] = mx.concatenate(output_state_dict['gt_info'])
        # output_state_dict['res_info'] = mx.concatenate(output_state_dict['res_info'])
        savepath = osp.join(args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
        print("Saving res to {}".format(savepath))
        import pickle
        with open(savepath, 'wb') as f:
            pickle.dump(output_state_dict, f)


    print("Averaged stats:", metric_logger)

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
    return stats, coco_evaluator



def test(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    # if not wo_class_error:
    #     metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]


    final_res = []
    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):

        outputs = model(samples)
        mx.eval(model)
        metric_logger.update(loss=sum(loss_dict.values()),
                             **loss_dict)
                            
        if 'class_error' in loss_dict:
            metric_logger.update(class_error=loss_dict['class_error'])

        orig_target_sizes = mx.stack([t["orig_size"] for t in targets], axis=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes, not_to_xyxy=True)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        for image_id, outputs in res.items():
            _scores = outputs['scores'].tolist()
            _labels = outputs['labels'].tolist()
            _boxes = outputs['boxes'].tolist()
            for s, l, b in zip(_scores, _labels, _boxes):
                assert isinstance(l, int)
                itemdict = {
                        "image_id": int(image_id), 
                        "category_id": l, 
                        "bbox": b, 
                        "score": s,
                        }
                final_res.append(itemdict)

    if args.output_dir:
        import json
        with open(args.output_dir + f'/results{args.rank}.json', 'w') as f:
            json.dump(final_res, f)        

    return final_res