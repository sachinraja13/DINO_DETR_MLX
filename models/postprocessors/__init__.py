from .bbox_postprocessor import BBoxPostProcessor


def build_postprocessors(args):
    postprocessors = {'bbox': BBoxPostProcessor(
        num_select=args.num_select, nms_iou_threshold=args.nms_iou_threshold)}
    return postprocessors
