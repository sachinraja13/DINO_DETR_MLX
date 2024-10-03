from .resnet import BasicBlock, Bottleneck, ResNet
from ..utils import FrozenBatchNorm2d


def resnet18(num_classes: int = 1000, replace_stride_with_dilation=[False, False, False], norm_layer=FrozenBatchNorm2d) -> ResNet:
    return ResNet(block=BasicBlock, layers=[2, 2, 2, 2], interim_layer_channels=[64, 128, 256, 512],
                  num_classes=num_classes, norm_layer=norm_layer,
                  replace_stride_with_dilation=replace_stride_with_dilation)


def resnet34(num_classes: int = 1000, replace_stride_with_dilation=[False, False, False], norm_layer=FrozenBatchNorm2d) -> ResNet:
    return ResNet(block=BasicBlock, layers=[3, 4, 6, 3],
                  interim_layer_channels=[64, 128, 256, 512],
                  num_classes=num_classes, norm_layer=norm_layer,
                  replace_stride_with_dilation=replace_stride_with_dilation)


def resnet50(num_classes: int = 1000, replace_stride_with_dilation=[False, False, False], norm_layer=FrozenBatchNorm2d) -> ResNet:
    return ResNet(block=Bottleneck, layers=[3, 4, 6, 3], interim_layer_channels=[256, 512, 1024, 2048],
                  num_classes=num_classes, norm_layer=norm_layer,
                  replace_stride_with_dilation=replace_stride_with_dilation)


def resnet101(num_classes: int = 1000, replace_stride_with_dilation=[False, False, False], norm_layer=FrozenBatchNorm2d) -> ResNet:
    return ResNet(block=Bottleneck, layers=[3, 4, 23, 3], interim_layer_channels=[256, 512, 1024, 2048],
                  num_classes=num_classes, norm_layer=norm_layer,
                  replace_stride_with_dilation=replace_stride_with_dilation)


def resnet152(num_classes: int = 1000, replace_stride_with_dilation=[False, False, False], norm_layer=FrozenBatchNorm2d) -> ResNet:
    return ResNet(block=Bottleneck, layers=[3, 8, 36, 3], interim_layer_channels=[256, 512, 1024, 2048],
                  num_classes=num_classes, norm_layer=norm_layer,
                  replace_stride_with_dilation=replace_stride_with_dilation)


def wide_resnet50_2(num_classes: int = 1000, replace_stride_with_dilation=[False, False, False], norm_layer=FrozenBatchNorm2d) -> ResNet:
    return ResNet(block=Bottleneck, layers=[3, 4, 6, 3], interim_layer_channels=[512, 1024, 2048, 4096],
                  num_classes=num_classes, width_per_group=64 * 2, norm_layer=norm_layer,
                  replace_stride_with_dilation=replace_stride_with_dilation)


def wide_resnet101_2(num_classes: int = 1000, replace_stride_with_dilation=[False, False, False], norm_layer=FrozenBatchNorm2d) -> ResNet:
    return ResNet(block=Bottleneck, layers=[3, 4, 23, 3], interim_layer_channels=[512, 1024, 2048, 4096],
                  num_classes=num_classes, width_per_group=64 * 2, norm_layer=norm_layer,
                  replace_stride_with_dilation=replace_stride_with_dilation)


# TODO: waiting for groups and dilation support
# def resnext50_32x4d(num_classes: int = 1000) -> ResNet:
#     return ResNet(block=Bottleneck, layers=[3, 4, 6, 3], num_classes=num_classes, groups=32, width_per_group=4)

# def resnext101_32x8d(num_classes: int = 1000) -> ResNet:
#     return ResNet(block=Bottleneck, layers=[3, 4, 23, 3], num_classes=num_classes, groups=32, width_per_group=8)

# def resnext101_64x4d(num_classes: int = 1000) -> ResNet:
#     return ResNet(block=Bottleneck, layers=[3, 4, 23, 3], num_classes=num_classes, groups=64, width_per_group=4)
