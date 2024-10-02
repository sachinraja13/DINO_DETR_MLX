from .base_criterion import BaseCriterion
from .two_stage_criterion import TwoStageCriterion
from .dino_criterion import DINOCriterion
import copy


def build_base_criterion(args, matcher):
    num_classes = args.num_classes
    weight_dict = {'loss_class': args.cls_loss_coef,
                   'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef

    clean_weight_dict = copy.deepcopy(weight_dict)

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update(
                {k + f'_{i}': v for k, v in clean_weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['class', 'boxes', 'cardinality']
    criterion = BaseCriterion(
        num_classes, matcher=matcher, weight_dict=weight_dict, losses=losses,
        eos_coef=args.eos_coef, loss_class_type=args.cost_class_type,
        focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma,
        pad_labels_to_n_max_ground_truths=args.pad_labels_to_n_max_ground_truths,
        n_max_ground_truths=args.n_max_ground_truths
    )
    return criterion


def build_two_stage_criterion(args, matcher):
    num_classes = args.num_classes
    weight_dict = {'loss_class': args.cls_loss_coef,
                   'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef

    clean_weight_dict = copy.deepcopy(weight_dict)

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update(
                {k + f'_{i}': v for k, v in clean_weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['class', 'boxes', 'cardinality']
    criterion = TwoStageCriterion(
        num_classes, matcher=matcher, weight_dict=weight_dict, losses=losses,
        eos_coef=args.eos_coef, loss_class_type=args.cost_class_type,
        focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma,
        pad_labels_to_n_max_ground_truths=args.pad_labels_to_n_max_ground_truths,
        n_max_ground_truths=args.n_max_ground_truths,
        two_stage_binary_cls=args.two_stage_binary_cls
    )
    return criterion


def build_dino_loss_criterion(args, matcher):
    # prepare weight dict
    num_classes = args.num_classes
    weight_dict = {'loss_class': args.cls_loss_coef,
                   'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    clean_weight_dict_wo_dn = copy.deepcopy(weight_dict)

    # for DN training
    if args.use_dn:
        weight_dict['loss_class_dn'] = args.cls_loss_coef
        weight_dict['loss_bbox_dn'] = args.bbox_loss_coef
        weight_dict['loss_giou_dn'] = args.giou_loss_coef

    clean_weight_dict = copy.deepcopy(weight_dict)

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update(
                {k + f'_{i}': v for k, v in clean_weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    if args.two_stage_type != 'no':
        interm_weight_dict = {}
        try:
            no_interm_box_loss = args.no_interm_box_loss
        except:
            no_interm_box_loss = False
        _coeff_weight_dict = {
            'loss_class': 1.0,
            'loss_bbox': 1.0 if not no_interm_box_loss else 0.0,
            'loss_giou': 1.0 if not no_interm_box_loss else 0.0,
        }
        try:
            interm_loss_coef = args.interm_loss_coef
        except:
            interm_loss_coef = 1.0
        interm_weight_dict.update({k + f'_interm': v * interm_loss_coef *
                                  _coeff_weight_dict[k] for k, v in clean_weight_dict_wo_dn.items()})
        weight_dict.update(interm_weight_dict)

    losses = ['class', 'boxes', 'cardinality']
    criterion = DINOCriterion(
        num_classes, matcher=matcher, weight_dict=weight_dict, losses=losses,
        eos_coef=args.eos_coef, loss_class_type=args.cost_class_type,
        focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma,
        pad_labels_to_n_max_ground_truths=args.pad_labels_to_n_max_ground_truths,
        n_max_ground_truths=args.n_max_ground_truths,
        two_stage_binary_cls=args.two_stage_binary_cls,
        use_dn=args.use_dn
    )
    return criterion
