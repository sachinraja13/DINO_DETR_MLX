from typing import Dict, List
import mlx.nn as nn
import mlx.core as mx
from . import resnet
from . import swin_transformer as swin
from .positional_embedding import build_position_encoding
from .utils import FrozenBatchNorm2d


class BackboneBase(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        train_backbone: bool,
        return_interm_layers: List[int],
    ):
        super().__init__()
        self.body = backbone
        self.return_interm_layers = return_interm_layers
        if not train_backbone:
            self.body.freeze()

    def __call__(self, array_dict: Dict[str, mx.array]):
        xs = self.body.get_features(array_dict['feature_map'])
        out: Dict[str, Dict[str, mx.array]] = {}
        for i in self.return_interm_layers:
            x = xs[i]
            m = array_dict['mask']
            assert m is not None
            interpolation = nn.Upsample(
                scale_factor=(x.shape[1] / m.shape[1], x.shape[2] / m.shape[2]), mode="nearest"
            )
            m = (
                interpolation(m[..., None].astype(mx.float32))
                .astype(mx.bool_)
                .squeeze(-1)
            )
            out[i] = {
                'feature_map': x,
                'mask': m
            }
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(
        self,
        name: str,
        train_backbone: bool,
        return_interm_layers: List[int],
        dilation: bool,
        batch_norm: nn.Module = FrozenBatchNorm2d,
    ):
        self.num_channels = []
        if name in ['resnet50', 'resnet101']:
            backbone = getattr(resnet, name)(
                replace_stride_with_dilation=[False, False, dilation],
                norm_layer=batch_norm,
            )
            all_num_channels = backbone.interim_layer_channels
            assert name not in (
                'resnet18', 'resnet34'), "Only resnet50 and resnet101 are available."
            assert return_interm_layers in [[0, 1, 2, 3], [1, 2, 3], [3]]
            for i in return_interm_layers:
                self.num_channels.append(all_num_channels[i])

        elif 'swin' in name:
            backbone = getattr(swin, name)()
            all_num_channels = backbone.interim_layer_channels
            assert return_interm_layers in [[0, 1, 2, 3], [1, 2, 3], [3]]
            for i in return_interm_layers:
                self.num_channels.append(all_num_channels[i])

        super().__init__(
            backbone=backbone,
            train_backbone=train_backbone,
            return_interm_layers=return_interm_layers,
        )


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.num_channels = backbone.num_channels

    def __call__(self, array_dict: Dict[str, mx.array]):
        xs = self.layers[0](array_dict)
        out: List[Dict[str, mx.array]] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            p = self.layers[1](x)
            pos.append(p)
        return out, pos


def build_backbone(args):
    """
    Useful args:
        - backbone: backbone name
        - lr_backbone
        - dilation
        - return_interm_indices: available:  [1,2,3]
        - backbone_freeze_keywords: 
        - use_checkpoint: for swin only for now

    """
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr > 0
    if not train_backbone:
        raise ValueError("Please set lr_backbone > 0")
    return_interm_indices = args.return_interm_indices
    assert return_interm_indices in [[0, 1, 2, 3], [1, 2, 3], [2, 3], [3]]
    backbone_freeze_keywords = args.backbone_freeze_keywords
    use_checkpoint = getattr(args, 'use_checkpoint', False)

    if args.backbone in ['resnet50', 'resnet101']:
        backbone = Backbone(
            name=args.backbone,
            train_backbone=train_backbone,
            dilation=args.dilation,
            return_interm_layers=return_interm_indices,
            batch_norm=FrozenBatchNorm2d
        )
        bb_num_channels = backbone.num_channels
    elif 'swin' in args.backbone:

        backbone = Backbone(
            name=args.backbone,
            train_backbone=train_backbone,
            dilation=args.dilation,
            return_interm_layers=return_interm_indices,
            batch_norm=FrozenBatchNorm2d
        )
        bb_num_channels = backbone.num_channels

    #     # freeze some layers
    #     if backbone_freeze_keywords is not None:
    #         for name, parameter in backbone.named_parameters():
    #             for keyword in backbone_freeze_keywords:
    #                 if keyword in name:
    #                     parameter.requires_grad_(False)
    #                     break
    #     if "backbone_dir" in args:
    #         pretrained_dir = args.backbone_dir
    #         PTDICT = {
    #             'swin_T_224_1k': 'swin_tiny_patch4_window7_224.pth',
    #             'swin_B_384_22k': 'swin_base_patch4_window12_384.pth',
    #             'swin_L_384_22k': 'swin_large_patch4_window12_384_22k.pth',
    #         }
    #         pretrainedpath = os.path.join(pretrained_dir, PTDICT[args.backbone])
    #         checkpoint = torch.load(pretrainedpath, map_location='cpu')['model']
    #         from collections import OrderedDict
    #         def key_select_function(keyname):
    #             if 'head' in keyname:
    #                 return False
    #             if args.dilation and 'layers.3' in keyname:
    #                 return False
    #             return True
    #         _tmp_st = OrderedDict({k:v for k, v in clean_state_dict(checkpoint).items() if key_select_function(k)})
    #         _tmp_st_output = backbone.load_state_dict(_tmp_st, strict=False)
    #         print(str(_tmp_st_output))
    else:
        raise NotImplementedError("Unknown backbone {}".format(args.backbone))

    assert len(bb_num_channels) == len(
        return_interm_indices), f"len(bb_num_channels) {len(bb_num_channels)} != len(return_interm_indices) {len(return_interm_indices)}"

    model = Joiner(backbone, position_embedding)
    model.num_channels = bb_num_channels
    assert isinstance(bb_num_channels, List), "bb_num_channels is expected to be a List but {}".format(
        type(bb_num_channels))
    return model
