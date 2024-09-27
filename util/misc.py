# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import os
import random
import subprocess
import time
from collections import OrderedDict, defaultdict, deque
import datetime
import pickle
from typing import Optional, List
from mlx.utils import tree_flatten, tree_unflatten
import json
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import colorsys
from .pytorch_weights_to_mlx import load_mlx_model_with_pytorch_weights
pad_all_images_to_same_size = False
image_array_fixed_size = [1024, 1024, 3]


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = np.asarray(list(self.deque))
        if d.shape[0] == 0:
            return 0
        return np.median(d)

    @property
    def avg(self):
        d = np.asarray(list(self.deque), dtype=np.float32)
        return np.mean(d)

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, mx.array):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            # print(name, str(meter))
            # import ipdb;ipdb.set_trace()
            if meter.count > 0:
                loss_str.append(
                    "{}: {}".format(name, str(meter))
                )
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, logger=None):
        if logger is None:
            print_func = print
        else:
            print_func = logger.info

        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = self.delimiter.join([
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj

            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                print_func(log_msg.format(
                    i, len(iterable), eta=eta_string,
                    meters=str(self),
                    time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print_func('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def collate_fn(batch):

    batch = list(zip(*batch))
    batch[0] = nested_array_dict_array_list(batch[0])
    return tuple(batch)


def find_smallest_i_divisible_by_n(nums, n=32):
    return [((k + n - 1) // n) * n for k in nums[:-1]]


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    maxes_div_n = find_smallest_i_divisible_by_n(maxes, n=64)
    maxes_div_n.append(maxes[-1])
    return maxes_div_n


def nested_array_dict_array_list(array_list: List[mx.array], pad_imgs=pad_all_images_to_same_size, img_fixed_size=image_array_fixed_size):
    # TODO make this more general
    if array_list[0].ndim == 3:
        # TODO make it support different-sized images
        if not pad_imgs or img_fixed_size is None or len(img_fixed_size) != 3:
            max_size = _max_by_axis([list(img.shape) for img in array_list])
        else:
            max_size = img_fixed_size
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(array_list)] + max_size
        b, h, w, c = batch_shape
        dtype = array_list[0].dtype
        feature_map = mx.zeros(batch_shape, dtype=dtype)
        mask = mx.ones((b, h, w), dtype=mx.bool_)
        for i in range(b):
            img = array_list[i]
            feature_map[i, : img.shape[0],
                        : img.shape[1], : img.shape[2]] = img
            mask[i, : img.shape[0], : img.shape[1]] = False
    else:
        raise ValueError('not supported')
    return {
        'feature_map': feature_map,
        'mask': mask
    }


def interpolate(input, scale_factor=None, mode="nearest", align_corners=False):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    return nn.UpSample(scale_factor, mode, align_corners)(input)


class color_sys():
    def __init__(self, num_colors) -> None:
        self.num_colors = num_colors
        colors = []
        for i in np.arange(0., 360., 360. / num_colors):
            hue = i/360.
            lightness = (50 + np.random.rand() * 10)/100.
            saturation = (90 + np.random.rand() * 10)/100.
            colors.append(
                tuple([int(j*255) for j in colorsys.hls_to_rgb(hue, lightness, saturation)]))
        self.colors = colors

    def __call__(self, idx):
        return self.colors[idx]


def inverse_sigmoid_np(x, eps=1e-3):
    x = np.clip(x, 0, 1)
    x1 = np.clip(x, eps, None)
    x2 = np.clip((1 - x), eps, None)
    return np.log(x1/x2)


def inverse_sigmoid(x, eps=1e-3):
    x = mx.clip(x, 0, 1)
    x1 = mx.clip(x, eps, None)
    x2 = mx.clip((1 - x), eps, None)
    return mx.log(x1/x2)


def clean_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    return new_state_dict


def get_state_dict(model, optimizer, args, epoch=None):
    if epoch is not None:
        args.last_epoch = epoch
    state_dict = {
        'model': model,
        'optimizer_state': optimizer.state,
        'args': args
    }
    return state_dict


def get_state_path_dict(checkpoint_path):
    save_type_dict = {
        'model': 'model.safetensors',
        'optimizer_state': 'optimizer_state.safetensors',
        'args': 'args.json'
    }
    path_dict = {}
    for k in save_type_dict:
        path_dict[k] = str(checkpoint_path / save_type_dict[k])
    return path_dict


def load_complete_state(path_dict):
    model = None
    optimizer_state = None
    args = None
    assert 'model' in path_dict and 'model' in state_dict, "model key not found while saving"
    assert 'optimizer_state' in path_dict and 'optimizer_state' in state_dict, "model key not found while saving"
    assert 'args' in path_dict and 'args' in state_dict, "model key not found while saving"
    try:
        with open(path_dict['args'], 'r') as f:
            args = json.load(f)
    except:
        print("Unable to load arguments namespace")
    try:
        model = mx.load(path_dict['model'])
    except:
        print("Unable to load MLX model")
    try:
        optimizer_state = tree_unflatten(
            list(mx.load(path_dict['optimizer_state']).items()))
    except:
        print("Unable to load MLX optimizer state")
    return model, optimizer_state, args


def save_complete_state(path_dict, state_dict):
    assert 'model' in path_dict and 'model' in state_dict, "model key not found while saving"
    assert 'optimizer_state' in path_dict and 'optimizer_state' in state_dict, "model key not found while saving"
    assert 'args' in path_dict and 'args' in state_dict, "model key not found while saving"
    try:
        with open(path_dict['args'], 'w') as f:
            json.dump(vars(state_dict['args']), f)
    except:
        print("Unable to save arguments namespace")
    try:
        state_dict['model'].save(path_dict['model'])
    except:
        print("Unable to save MLX model")
    try:
        state = tree_flatten(state_dict['optimizer_state'])
        mx.save_safetensors(path_dict['optimizer_state'], dict(state))
    except:
        print("Unable to save MLX optimizer state")
