from functools import partial
import mlx.nn as nn
from .swin_transformer import PatchMergingV2, SwinTransformer, SwinTransformerBlockV2


def swin_tiny_patch4_window7(num_classes: int = 1000) -> SwinTransformer:
    embed_dim = 96
    return SwinTransformer(
        patch_size=[4, 4],
        embed_dim=embed_dim,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[7, 7],
        interim_layer_channels=[embed_dim, 2 *
                                embed_dim, 4*embed_dim, 8*embed_dim],
        stochastic_depth_prob=0.2,
    )


def swin_small_patch4_window7(num_classes: int = 1000) -> SwinTransformer:
    embed_dim = 96
    return SwinTransformer(
        patch_size=[4, 4],
        embed_dim=embed_dim,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[7, 7],
        interim_layer_channels=[embed_dim, 2 *
                                embed_dim, 4*embed_dim, 8*embed_dim],
        stochastic_depth_prob=0.2,
        num_classes=num_classes,
    )


def swin_base_patch4_window7(num_classes: int = 1000) -> SwinTransformer:
    embed_dim = 128
    return SwinTransformer(
        patch_size=[4, 4],
        embed_dim=embed_dim,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=[7, 7],
        interim_layer_channels=[embed_dim, 2 *
                                embed_dim, 4*embed_dim, 8*embed_dim],
        stochastic_depth_prob=0.2,
        num_classes=num_classes,
    )


def swin_v2_tiny_patch4_window8(num_classes: int = 1000) -> SwinTransformer:
    embed_dim = 96
    return SwinTransformer(
        patch_size=[4, 4],
        embed_dim=embed_dim,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[8, 8],
        interim_layer_channels=[embed_dim, 2 *
                                embed_dim, 4*embed_dim, 8*embed_dim],
        stochastic_depth_prob=0.2,
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
        num_classes=num_classes,
    )


def swin_v2_base_patch4_window8(num_classes: int = 1000) -> SwinTransformer:
    embed_dim = 128
    return SwinTransformer(
        patch_size=[4, 4],
        embed_dim=embed_dim,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=[8, 8],
        interim_layer_channels=[embed_dim, 2 *
                                embed_dim, 4*embed_dim, 8*embed_dim],
        stochastic_depth_prob=0.2,
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
        num_classes=num_classes,
    )


def swin_v2_small_patch4_window8(num_classes: int = 1000) -> SwinTransformer:
    embed_dim = 96
    return SwinTransformer(
        patch_size=[4, 4],
        embed_dim=embed_dim,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[8, 8],
        interim_layer_channels=[embed_dim, 2 *
                                embed_dim, 4*embed_dim, 8*embed_dim],
        stochastic_depth_prob=0.3,
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
        num_classes=num_classes,
    )


def swin_large_patch4_window7(num_classes: int = 1000) -> SwinTransformer:
    embed_dim = 192
    return SwinTransformer(
        patch_size=[4, 4],
        embed_dim=embed_dim,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=[7, 7],
        interim_layer_channels=[embed_dim, 2 *
                                embed_dim, 4*embed_dim, 8*embed_dim],
        stochastic_depth_prob=0.3,
        num_classes=num_classes,
    )


def swin_v2_large_patch4_window12(num_classes: int = 1000) -> SwinTransformer:
    embed_dim = 192
    return SwinTransformer(
        patch_size=[4, 4],
        embed_dim=embed_dim,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=[12, 12],
        interim_layer_channels=[embed_dim, 2 *
                                embed_dim, 4*embed_dim, 8*embed_dim],
        stochastic_depth_prob=0.3,
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
        num_classes=num_classes,
    )
