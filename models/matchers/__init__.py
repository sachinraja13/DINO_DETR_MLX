from .simple_minsum_matcher import SimpleMinsumMatcher
from .hungarian_matcher import HungarianMatcher
from .stable_hungarian_matcher import StableHungarianMatcher


def build_matcher(args):
    assert args.matcher_type in [
        'StableHungarianMatcher', 'HungarianMatcher', 'SimpleMinsumMatcher'], "Unknown args.matcher_type: {}".format(args.matcher_type)
    if args.matcher_type == 'StableHungarianMatcher':
        return StableHungarianMatcher(
            cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou, cost_class_type=args.cost_class_type,
            alpha=args.focal_alpha, gamma=args.focal_cost_gamma, cec_beta=args.cec_beta,
            pad_labels_to_n_max_ground_truths=args.pad_labels_to_n_max_ground_truths,
            n_max_ground_truths=args.n_max_ground_truths
        )
    elif args.matcher_type == 'HungarianMatcher':
        return HungarianMatcher(
            cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou,
            focal_alpha=args.focal_alpha, pad_labels_to_n_max_ground_truths=args.pad_labels_to_n_max_ground_truths,
            n_max_ground_truths=args.n_max_ground_truths
        )
    elif args.matcher_type == 'SimpleMinsumMatcher':
        return SimpleMinsumMatcher(
            cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou,
            focal_alpha=args.focal_alpha,
            pad_labels_to_n_max_ground_truths=args.pad_labels_to_n_max_ground_truths,
            n_max_ground_truths=args.n_max_ground_truths
        )
    else:
        raise NotImplementedError(
            "Unknown args.matcher_type: {}".format(args.matcher_type))
