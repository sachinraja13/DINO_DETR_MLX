import numpy as np
import mlx.core as mx
import mlx.nn as mnn
import torch.nn as nn
import torch
from mlx.utils import tree_flatten, tree_unflatten
# Set random seed for reproducibility
np.random.seed(0)
torch.manual_seed(0)

def apply(dst, parameters, param_key = ''):
    if isinstance(parameters, dict):
        for k in parameters:
            if k in dst:
                current_value = dst[k]
                new_value = parameters[k]
                # print("Key: ", k, " FOUND")
                if isinstance(current_value, mx.array):
                    if current_value.shape == new_value.shape:
                        dst[k] = new_value
                        param_key = param_key + k
                    else:
                        print("Not updated : " , param_key)
                        print(current_value.shape, new_value.shape)
                elif isinstance(current_value, mnn.Module):
                    param_key = param_key + k + "."
                    apply(current_value, new_value, param_key)
                elif isinstance(current_value, (dict, list)):
                    param_key = param_key + k + "."
                    apply(current_value, new_value, param_key)
            else:
                pass
                # print("Key: ", k, " NOT FOUND")
    elif isinstance(parameters, list):
        for i in range(len(parameters)):
            current_value = dst[i]
            new_value = parameters[i]
            if isinstance(current_value, mx.array):
                print("Updated : " + param_key)
                dst[i] = new_value
            elif isinstance(current_value, mnn.Module):
                apply(current_value, new_value, param_key)
            elif isinstance(current_value, (dict, list)):
                apply(current_value, new_value, param_key)



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
        q_name : q,
        k_name : k,
        v_name : v
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
        q_name : q,
        k_name : k,
        v_name : v
    }
    return out_dict

def initialize_numpy_arrays(model):
    numpy_arrays = {}
    for name, param in model.named_parameters():
        if 'params.' in name:
            continue
        print(name, param.shape)
        param_shape = param.shape
        np_array = np.random.randn(*param_shape).astype(np.float32) / param_shape[-1]**0.5
        if 'in_proj_weight' in name:
            in_proj_weight_dict = handle_in_proj_weight(name, np_array)
            numpy_arrays.update(in_proj_weight_dict)
        elif 'in_proj_bias' in name:
            in_proj_bias_dict = handle_in_proj_bias(name, np_array)
            numpy_arrays.update(in_proj_bias_dict)
        else:
            numpy_arrays[name] = np_array
        param.data = torch.from_numpy(np_array)
    return model, numpy_arrays

def initialize_mlx_model(mlx_model, numpy_arrays):
    new_params = []
    keys_mapped = {}
    for name, param in tree_flatten(mlx_model):
        if 'params.' in name:
            continue
        print(name)
        try:
            new_params.append((name, mx.array(numpy_arrays[name])))
            keys_mapped[name] = True
        except:
            print("Could not initialize : " + name)
    mlx_model = tree_unflatten(new_params)
    for name in numpy_arrays:
        if name not in keys_mapped:
            print("Key not found in mlx model: ", name)
    return mlx_model

def initialize_models(pt_model, mlx_model):
    # Initialize parameters with random NumPy arrays
    pt_model, numpy_arrays = initialize_numpy_arrays(pt_model)
    mlx_params = initialize_mlx_model(mlx_model, numpy_arrays)
    apply(mlx_model.children(), mlx_params)

