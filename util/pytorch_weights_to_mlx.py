import torch
from typing import Dict, List
from mlx.utils import tree_flatten, tree_unflatten
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from pathlib import Path
import torch
import pickle
import pprint


def get_keys_list(flattened_tree):
    param_keys = {}
    for i in range(len(flattened_tree)):
        param_key = flattened_tree[i][0]
        param_keys[param_key] = None
    return param_keys


def check_common_keys(mlx_flattened_model, pytorch_flattened_params):
    mlx_model_param_keys = get_keys_list(mlx_flattened_model)
    pytorch_parameters_param_keys = get_keys_list(pytorch_flattened_params)
    common_keys = {}
    mlx_model_keys_not_in_saved_pytorch_params = {}
    torch_param_keys_not_in_mlx_model = {}
    for k in mlx_model_param_keys:
        if k in pytorch_parameters_param_keys:
            common_keys[k] = None
        elif 'params.' not in k:
            mlx_model_keys_not_in_saved_pytorch_params[k] = None
    for k in pytorch_parameters_param_keys:
        if k in mlx_model_param_keys:
            common_keys[k] = None
        else:
            torch_param_keys_not_in_mlx_model[k] = None
    return common_keys, mlx_model_keys_not_in_saved_pytorch_params, torch_param_keys_not_in_mlx_model


def generate_resnet_backbone_key_mapping(backbone_keys, key_mapping):
    for original_key in backbone_keys:
        key_tokens = original_key.split('.')
        new_key = ''
        first_int = True
        for key_token in key_tokens:
            if len(new_key) > 0:
                new_key = new_key + '.'
            try:
                key_token_integer = int(key_token)
                new_key = new_key + 'layers.' + str(key_token_integer)
            except:
                if 'layer' in key_token:
                    print(key_token)
                    int_layer_token = key_token.split('layer')[1]
                    # try:
                    int_layer_token = int(int_layer_token)
                    new_key = new_key + 'layers.' + str(int_layer_token-1)
                    print(new_key)
                    # except:
                    #     new_key = new_key + key_token
                else:
                    new_key = new_key + key_token
        key_mapping[original_key] = new_key
    return key_mapping


def generate_swin_backbone_key_mapping(backbone_keys, key_mapping):
    direct_replace_strings = {
        'backbone.0': 'backbone.layers.0.body',
        'mlp.fc1': 'mlp.layers.0',
        'mlp.fc2': 'mlp.layers.3',
        'patch_embed.proj': 'patch_embed.layers.0',
        'patch_embed.norm': 'patch_embed.layers.2',
        'layers.0.blocks': 'layers.0.layers',
        'layers.0.downsample': 'layers.1',
        'layers.1.blocks': 'layers.2.layers',
        'layers.1.downsample': 'layers.3',
        'layers.2.blocks': 'layers.4.layers',
        'layers.2.downsample': 'layers.5',
        'layers.3.blocks': 'layers.6.layers',
    }

    for original_key in backbone_keys:
        new_key = original_key
        for replace_string in direct_replace_strings:
            new_key = new_key.replace(
                replace_string, direct_replace_strings[replace_string])
        key_mapping[original_key] = new_key
    key_mapping['backbone.0.norm1.weight'] = 'backbone.layers.0.body.norm_layers.0.weight'
    key_mapping['backbone.0.norm2.weight'] = 'backbone.layers.0.body.norm_layers.1.weight'
    key_mapping['backbone.0.norm3.weight'] = 'backbone.layers.0.body.norm_layers.2.weight'
    key_mapping['backbone.0.norm1.bias'] = 'backbone.layers.0.body.norm_layers.0.bias'
    key_mapping['backbone.0.norm2.bias'] = 'backbone.layers.0.body.norm_layers.1.bias'
    key_mapping['backbone.0.norm3.bias'] = 'backbone.layers.0.body.norm_layers.2.bias'
    return key_mapping


def generate_input_proj_key_mapping(input_proj_keys, key_mapping):
    for original_key in input_proj_keys:
        key_tokens = original_key.split('.')
        first_int = True
        new_key = ''
        for key_token in key_tokens:
            if len(new_key) > 0:
                new_key = new_key + '.'
            try:
                key_token_integer = int(key_token)
                if first_int:
                    new_key = new_key + key_token
                    first_int = False
                else:
                    new_key = new_key + 'layers.' + str(key_token_integer)
            except:
                new_key = new_key + key_token
        key_mapping[original_key] = new_key
    return key_mapping


def generate_key_mapping(flattened_tree, backbone):
    module_groups = {'backbone': [], 'input_proj': []}
    key_mapping = {}
    for i in range(len(flattened_tree)):
        weight_tuple = flattened_tree[i]
        weight_key_to_replace = weight_tuple[0]
        key_type_found = False
        for module_type in module_groups:
            if module_type in weight_key_to_replace:
                key_type_found = True
                module_groups[module_type].append(weight_key_to_replace)
        if not key_type_found:
            key_mapping[weight_key_to_replace] = weight_key_to_replace
    if 'resnet' in backbone:
        key_mapping = generate_resnet_backbone_key_mapping(
            module_groups['backbone'], key_mapping)
    elif 'swin' in backbone:
        key_mapping = generate_swin_backbone_key_mapping(
            module_groups['backbone'], key_mapping)
    key_mapping = generate_input_proj_key_mapping(
        module_groups['input_proj'], key_mapping)
    return key_mapping


def update_flattened_tree_keys(flattened_tree, key_mapping):
    replaced_flattened_tree = []
    for i in range(len(flattened_tree)):
        weight_tuple = flattened_tree[i]
        weight_key = weight_tuple[0]
        weight_value = weight_tuple[1]
        mapped_weight_key = key_mapping[weight_key]
        replaced_flattened_tree.append((mapped_weight_key, weight_value))
    return replaced_flattened_tree


def handle_in_proj_weight(name, param):
    l = param.shape[0]
    qkv_dim = l // 3
    q = param[:qkv_dim]
    k = param[qkv_dim:qkv_dim*2]
    v = param[qkv_dim*2:]
    q_name = name
    k_name = name
    v_name = name
    q_name = q_name.replace("in_proj_weight", "query_proj.weight")
    k_name = k_name.replace("in_proj_weight", "key_proj.weight")
    v_name = v_name.replace("in_proj_weight", "value_proj.weight")
    out_dict = {
        q_name: q,
        k_name: k,
        v_name: v
    }
    return out_dict


def handle_in_proj_bias(name, param):
    l = param.shape[0]
    qkv_dim = l // 3
    q = param[:qkv_dim]
    k = param[qkv_dim:qkv_dim*2]
    v = param[qkv_dim*2:]
    q_name = name
    k_name = name
    v_name = name
    q_name = q_name.replace("in_proj_bias", "query_proj.bias")
    k_name = k_name.replace("in_proj_bias", "key_proj.bias")
    v_name = v_name.replace("in_proj_bias", "value_proj.bias")
    out_dict = {
        q_name: q,
        k_name: k,
        v_name: v
    }
    return out_dict


def load_mlx_model_with_pytorch_weights(
    mlx_model,
    pytorch_weights_path,
    backbone,
    logger=None
):
    pytorch_weights_flattened = torch.load(
        pytorch_weights_path, map_location=torch.device('cpu'))
    pytorch_weights_flattened = tree_flatten(
        pytorch_weights_flattened['model'])
    processed_pytorch_params = []
    conv_layers = []
    for k, v in pytorch_weights_flattened:
        v = mx.array(v.detach().cpu().numpy())
        if 'in_proj_weight' in k:
            in_proj_weight_dict = handle_in_proj_weight(k, v)
            for key in in_proj_weight_dict:
                processed_pytorch_params.append(
                    (key, in_proj_weight_dict[key]))
            continue
        if 'in_proj_bias' in k:
            in_proj_bias_dict = handle_in_proj_bias(k, v)
            for key in in_proj_bias_dict:
                processed_pytorch_params.append((key, in_proj_bias_dict[key]))
            continue
        if len(v.shape) == 4 and 'input_proj' in k:
            v = v.transpose(0, 2, 3, 1)
        if len(v.shape) == 4 and 'backbone' in k and 'downsample' in k:
            v = v.transpose(0, 2, 3, 1)
        if 'conv' in k and len(v.shape) == 4:
            v = v.transpose(0, 2, 3, 1)
        if len(v.shape) == 4 and 'patch_embed' in k:
            v = v.transpose(0, 2, 3, 1)
        processed_pytorch_params.append((k, v))

    processed_flattened_tree_torch = tree_unflatten(processed_pytorch_params)
    key_mapping = generate_key_mapping(processed_pytorch_params, backbone)
    if logger is not None:
        logger.info("key_mapping:" + pprint.pformat(key_mapping, indent=4))
    processed_mapped_flattened_tree_torch = update_flattened_tree_keys(
        processed_pytorch_params, key_mapping)
    common_keys, model_keys_not_in_saved_params, param_keys_not_in_model = check_common_keys(
        tree_flatten(mlx_model), processed_mapped_flattened_tree_torch)
    if logger is not None:
        logger.info("common_keys:" +
                    pprint.pformat(list(common_keys.keys()), indent=4))
    if logger is not None:
        logger.info(
            "model_keys_not_in_saved_params:" + pprint.pformat(list(model_keys_not_in_saved_params.keys()), indent=4))
    if logger is not None:
        logger.info(
            "param_keys_not_in_model:" + pprint.pformat(list(param_keys_not_in_model.keys()), indent=4))

    mapped_weights_tree = tree_unflatten(processed_mapped_flattened_tree_torch)
    pre_weights = {
        k: np.array(v.tolist())
        for k, v in tree_flatten(mlx_model.children())
        if isinstance(v, mx.array)
    }
    mlx_model.update(mapped_weights_tree)
    post_weights = {
        k: np.array(v.tolist())
        for k, v in tree_flatten(mlx_model.children())
        if isinstance(v, mx.array)
    }
    for k, v in pre_weights.items():
        if k in post_weights:
            if 'params.' in k:
                continue
            if np.linalg.norm(v - post_weights[k]) < 0.01:
                if logger is not None:
                    logger.info("Weight: " + str(k) +
                                " very close to the original value ")
    return mlx_model
