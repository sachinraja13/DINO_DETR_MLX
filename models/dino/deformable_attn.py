# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math
import mlx.core as mx 
import numpy as np
import timeit
import random
import mlx.nn as nn

def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


@mx.custom_function
def grid_sample(x, grid):

    assert x.ndim == 4, "`x` must be 4D."
    assert grid.ndim == 4, "`grid` must be 4D."

    B, _, _, C = x.shape
    _, gN, gM, D = grid.shape
    out_shape = (B, gN, gM, C)

    assert D == 2, "Last dim of `grid` must be size 2."

    source = """
        uint elem = thread_position_in_grid.x;
        int H = x_shape[1];
        int W = x_shape[2];
        int C = x_shape[3];
        int gH = grid_shape[1];
        int gW = grid_shape[2];

        int w_stride = C;
        int h_stride = W * w_stride;
        int b_stride = H * h_stride;

        uint grid_idx = elem / C * 2;
        float ix = ((grid[grid_idx] + 1) * W - 1) / 2;
        float iy = ((grid[grid_idx + 1] + 1) * H - 1) / 2;

        int ix_nw = floor(ix);
        int iy_nw = floor(iy);

        int ix_ne = ix_nw + 1;
        int iy_ne = iy_nw;

        int ix_sw = ix_nw;
        int iy_sw = iy_nw + 1;

        int ix_se = ix_nw + 1;
        int iy_se = iy_nw + 1;

        T nw = (ix_se - ix)    * (iy_se - iy);
        T ne = (ix    - ix_sw) * (iy_sw - iy);
        T sw = (ix_ne - ix)    * (iy    - iy_ne);
        T se = (ix    - ix_nw) * (iy    - iy_nw);

        int batch_idx = elem / C / gH / gW * b_stride;
        int channel_idx = elem % C;
        int base_idx = batch_idx + channel_idx;

        T I_nw = x[base_idx + iy_nw * h_stride + ix_nw * w_stride];
        T I_ne = x[base_idx + iy_ne * h_stride + ix_ne * w_stride];
        T I_sw = x[base_idx + iy_sw * h_stride + ix_sw * w_stride];
        T I_se = x[base_idx + iy_se * h_stride + ix_se * w_stride];

        I_nw = iy_nw >= 0 && iy_nw <= H - 1 && ix_nw >= 0 && ix_nw <= W - 1 ? I_nw : 0;
        I_ne = iy_ne >= 0 && iy_ne <= H - 1 && ix_ne >= 0 && ix_ne <= W - 1 ? I_ne : 0;
        I_sw = iy_sw >= 0 && iy_sw <= H - 1 && ix_sw >= 0 && ix_sw <= W - 1 ? I_sw : 0;
        I_se = iy_se >= 0 && iy_se <= H - 1 && ix_se >= 0 && ix_se <= W - 1 ? I_se : 0;

        out[elem] = nw * I_nw + ne * I_ne + sw * I_sw + se * I_se;
    """
    kernel = mx.fast.metal_kernel(
        name="grid_sample",
        input_names=["x", "grid"],
        output_names=["out"],
        source=source,
    )
    outputs = kernel(
        inputs=[x, grid],
        template=[("T", x.dtype)],
        output_shapes=[out_shape],
        output_dtypes=[x.dtype],
        grid=(np.prod(out_shape), 1, 1),
        threadgroup=(256, 1, 1),
    )
    return outputs[0]

@grid_sample.vjp
def grid_sample_vjp(primals, cotangent, _):
    x, grid = primals
    B, _, _, C = x.shape
    _, gN, gM, D = grid.shape

    assert D == 2, "Last dim of `grid` must be size 2."

    source = """
        uint elem = thread_position_in_grid.x;
        int H = x_shape[1];
        int W = x_shape[2];
        int C = x_shape[3];
        // Pad C to the nearest larger simdgroup size multiple
        int C_padded = ceildiv(C, threads_per_simdgroup) * threads_per_simdgroup;

        int gH = grid_shape[1];
        int gW = grid_shape[2];

        int w_stride = C;
        int h_stride = W * w_stride;
        int b_stride = H * h_stride;

        uint grid_idx = elem / C_padded * 2;
        float ix = ((grid[grid_idx] + 1) * W - 1) / 2;
        float iy = ((grid[grid_idx + 1] + 1) * H - 1) / 2;

        int ix_nw = floor(ix);
        int iy_nw = floor(iy);

        int ix_ne = ix_nw + 1;
        int iy_ne = iy_nw;

        int ix_sw = ix_nw;
        int iy_sw = iy_nw + 1;

        int ix_se = ix_nw + 1;
        int iy_se = iy_nw + 1;

        T nw = (ix_se - ix)    * (iy_se - iy);
        T ne = (ix    - ix_sw) * (iy_sw - iy);
        T sw = (ix_ne - ix)    * (iy    - iy_ne);
        T se = (ix    - ix_nw) * (iy    - iy_nw);

        int batch_idx = elem / C_padded / gH / gW * b_stride;
        int channel_idx = elem % C_padded;
        int base_idx = batch_idx + channel_idx;

        T gix = T(0);
        T giy = T(0);
        if (channel_idx < C) {
            int cot_index = elem / C_padded * C + channel_idx;
            T cot = cotangent[cot_index];
            if (iy_nw >= 0 && iy_nw <= H - 1 && ix_nw >= 0 && ix_nw <= W - 1) {
                int offset = base_idx + iy_nw * h_stride + ix_nw * w_stride;
                atomic_fetch_add_explicit(&x_grad[offset], nw * cot, memory_order_relaxed);

                T I_nw = x[offset];
                gix -= I_nw * (iy_se - iy) * cot;
                giy -= I_nw * (ix_se - ix) * cot;
            }
            if (iy_ne >= 0 && iy_ne <= H - 1 && ix_ne >= 0 && ix_ne <= W - 1) {
                int offset = base_idx + iy_ne * h_stride + ix_ne * w_stride;
                atomic_fetch_add_explicit(&x_grad[offset], ne * cot, memory_order_relaxed);

                T I_ne = x[offset];
                gix += I_ne * (iy_sw - iy) * cot;
                giy -= I_ne * (ix - ix_sw) * cot;
            }
            if (iy_sw >= 0 && iy_sw <= H - 1 && ix_sw >= 0 && ix_sw <= W - 1) {
                int offset = base_idx + iy_sw * h_stride + ix_sw * w_stride;
                atomic_fetch_add_explicit(&x_grad[offset], sw * cot, memory_order_relaxed);

                T I_sw = x[offset];
                gix -= I_sw * (iy - iy_ne) * cot;
                giy += I_sw * (ix_ne - ix) * cot;
            }
            if (iy_se >= 0 && iy_se <= H - 1 && ix_se >= 0 && ix_se <= W - 1) {
                int offset = base_idx + iy_se * h_stride + ix_se * w_stride;
                atomic_fetch_add_explicit(&x_grad[offset], se * cot, memory_order_relaxed);

                T I_se = x[offset];
                gix += I_se * (iy - iy_nw) * cot;
                giy += I_se * (ix - ix_nw) * cot;
            }
        }

        T gix_mult = W / 2;
        T giy_mult = H / 2;

        // Reduce across each simdgroup first.
        // This is much faster than relying purely on atomics.
        gix = simd_sum(gix);
        giy = simd_sum(giy);

        if (thread_index_in_simdgroup == 0) {
            atomic_fetch_add_explicit(&grid_grad[grid_idx], gix * gix_mult, memory_order_relaxed);
            atomic_fetch_add_explicit(&grid_grad[grid_idx + 1], giy * giy_mult, memory_order_relaxed);
        }
    """
    kernel = mx.fast.metal_kernel(
        name="grid_sample_grad",
        input_names=["x", "grid", "cotangent"],
        output_names=["x_grad", "grid_grad"],
        source=source,
        atomic_outputs=True,
    )
    # pad the output channels to simd group size
    # so that our `simd_sum`s don't overlap.
    simdgroup_size = 32
    C_padded = (C + simdgroup_size - 1) // simdgroup_size * simdgroup_size
    grid_size = B * gN * gM * C_padded
    outputs = kernel(
        inputs=[x, grid, cotangent],
        template=[("T", x.dtype)],
        output_shapes=[x.shape, grid.shape],
        output_dtypes=[x.dtype, x.dtype],
        grid=(grid_size, 1, 1),
        threadgroup=(256, 1, 1),
        init_value=0,
    )
    return outputs[0], outputs[1]




def ms_deform_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights):
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape

    level_indices = [H_ * W_ for H_, W_ in value_spatial_shapes]
    split_indices = []
    prev = 0
    for i in range(len(level_indices)):
        split_indices.append(prev + level_indices[i])
        prev = split_indices[-1]
    split_indices = split_indices[:-1]
    value_list = mx.split(value, split_indices, axis=1)

    sampling_grids = 2 * sampling_locations - 1

    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_, W_ -> N_*M_, D_, H_, W_
        value_l_ = mx.reshape(mx.transpose(mx.reshape(value_list[lid_], (N_, H_ * W_, M_, D_)), (0, 2, 1, 3)), (N_ * M_, H_, W_, D_))
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = mx.reshape(mx.transpose(sampling_grids[:, :, :, lid_, :], (0, 2, 1, 3, 4)), (N_ * M_, Lq_, P_, 2))
        sampling_value_l_ = grid_sample(value_l_, sampling_grid_l_)
        sampling_value_list.append(mx.transpose(sampling_value_l_, (0, 3, 1, 2)))
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = mx.reshape(mx.transpose(attention_weights, (0, 2, 1, 3, 4)), (N_ * M_, 1, Lq_, L_ * P_))
    sampling_value_list = mx.stack(sampling_value_list, axis=-2).flatten(-2)
    output = (sampling_value_list * attention_weights).sum(-1).reshape((N_, M_ * D_, Lq_))
    return mx.transpose(output, (0, 2, 1))

class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        self.params = locals()
        del self.params['self']
        del self.params['__class__']
        if d_model % n_heads != 0:
            raise ValueError(f'd_model must be divisible by n_heads, but got {d_model} and {n_heads}')
        _d_per_head = d_model // n_heads
        # if not _is_power_of_2(_d_per_head):
        #     warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
        #                   "which is more efficient in our implementation.")

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)


    def __call__(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        # reference_points = mx.stop_gradient(reference_points)
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in
        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value * (1 - input_padding_mask[..., None].astype(value.dtype))
        value = value.reshape(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).reshape(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)

        attention_weights = self.attention_weights(query).reshape(N, Len_q, self.n_heads, self.n_levels * self.n_points)

        attention_weights = nn.softmax(attention_weights, axis=-1).reshape(N, Len_q, self.n_heads, self.n_levels, self.n_points)

        if reference_points.shape[-1] == 2:
            input_spatial_shapes_mx = mx.array(input_spatial_shapes)
            input_spatial_shapes_mx = mx.stop_gradient(input_spatial_shapes_mx)
            offset_normalizer = mx.stack([input_spatial_shapes_mx[:, 1], input_spatial_shapes_mx[:, 0]], axis=-1)
            sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(f'Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]} instead.')
        # sampling_locations = mx.stop_gradient(sampling_locations)
        output = ms_deform_attn_core(value, input_spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)
        # print(output.shape)
        return output
    
