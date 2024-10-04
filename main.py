# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
import sys
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os
import sys
import numpy as np
import pprint
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, BatchSampler
from datasets.custom_dataset_loader import CustomDataLoader
from util.logger import setup_logger
from util.slconfig import DictAction, SLConfig
from util.utils import BestMetricHolder
import util.misc as utils
import pickle
import datasets
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch, test


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Set transformer detector', add_help=False)
    parser.add_argument('--config_file', '-c', type=str, required=True)
    parser.add_argument('--options',
                        nargs='+',
                        action=DictAction,
                        help='override some settings in the used config, the key-value pair '
                        'in xxx=yyy format will be merged into config file.')

    # dataset parameters
    # parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str,
                        default='/comp_robot/cv_public_dataset/COCO2017/')
    parser.add_argument('--remove_difficult', action='store_true')

    # training parameters
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--note', default='',
                        help='add some notes to the experiment')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path',
                        help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_log', action='store_true')

    return parser


def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors


def main(args):
    # load cfg file and update the args
    print("Loading config file from {}".format(args.config_file))
    cfg = SLConfig.fromfile(args.config_file)
    cfg.merge_from_dict(args.options)
    save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
    cfg.dump(save_cfg_path)
    save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
    with open(save_json_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k, v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    # update some new args temporally
    if not getattr(args, 'use_ema', None):
        args.use_ema = False
    if not getattr(args, 'debug', None):
        args.debug = False
    if not getattr(args, 'pad_all_images_to_same_size', False):
        args.pad_all_images_to_same_size = False
    if not getattr(args, 'image_array_fixed_size', [1024, 1024, 3]):
        args.image_array_fixed_size = [1024, 1024, 3]

    # setup logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(output=os.path.join(
        args.output_dir, 'info.txt'), distributed_rank=0, color=False, name="dino_detr")
    logger.info("Command: "+' '.join(sys.argv))
    save_json_path = os.path.join(args.output_dir, "config_args_all.json")
    with open(save_json_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    logger.info("Full config saved to {}".format(save_json_path))
    logger.info("args namespace: " + str(args) + '\n')
    logger.info("args dict:")
    logger.info(pprint.pformat(vars(args), indent=4))

    # fix the seed for reproducibility
    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model, criterion, postprocessors = build_model_main(args)

    if args.eval:
        args.pad_all_images_to_same_size = False
        args.pad_labels_to_n_max_ground_truths = False

    if not args.compile_computation_graph:
        mx.disable_compile()

    if args.device == 'cpu':
        mx.set_default_device(mx.cpu)
    else:
        mx.set_default_device(mx.gpu)

    if args.load_pytorch_weights:
        model = utils.load_mlx_model_with_pytorch_weights(
            model, args.pytorch_weights_path, args.backbone, logger)

    if args.precision == 'half':
        logger.info("Changing weights to half precision")
        model.apply(lambda x: x.astype(mx.bfloat16))

    if args.quantize_model:
        logger.info("Quantizing model")
        logger.info("Quantizing n groups: " + str(args.quantize_groups))
        logger.info("Quantizing n bits: " + str(args.quantize_bits))
        nn.quantize(model, group_size=args.quantize_groups, bits=args.quantize_bits,
                    class_predicate=lambda _, m: isinstance(m, nn.Linear))

    wo_class_error = False

    trainable_params = model.trainable_parameters()
    # Count the total number of trainable parameters
    n_parameters = sum(p.size for _, p in tree_flatten(trainable_params))
    logger.info('number of params:'+str(n_parameters))
    logger.info("params:\n"+json.dumps({n: p.size for n,
                p in tree_flatten(trainable_params)}, indent=2))

    lr_schedule = optim.step_decay(
        args.lr, args.lr_drop_factor, args.lr_drop_steps)
    optimizer = optim.AdamW(learning_rate=lr_schedule,
                            weight_decay=args.weight_decay)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    data_loader_train = None
    data_loader_val = None
    if not args.use_custom_dataloader:
        sampler_train = RandomSampler(dataset_train)
        sampler_val = SequentialSampler(dataset_val)

        batch_sampler_train = BatchSampler(
            sampler_train, args.batch_size, drop_last=True)

        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                       collate_fn=utils.collate_fn, num_workers=args.num_workers)
        data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
                                     drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    elif not args.reinstantiate_dataloader_every_epoch:
        data_loader_train = CustomDataLoader(
            dataset_train, args.batch_size, shuffle=True, collate_fn=utils.collate_fn)
        data_loader_val = CustomDataLoader(
            dataset_val, 1, shuffle=False, collate_fn=utils.collate_fn)

    if args.dataset_file == 'coco':
        base_ds = get_coco_api_from_dataset(dataset_val)
    else:
        base_ds = None
    args.base_ds = base_ds

    if args.frozen_weights is not None:
        frozen_checkpoint_path = Path(os.path.join(
            args.output_dir, args.frozen_weights))
        model_weights, optimizer_state, args_json = utils.load_complete_state(
            utils.get_state_path_dict(frozen_checkpoint_path))
        model.update(model_weights)
        model_without_ddp = model
        logger.info("Loaded model weights from {}".format(
            str(frozen_checkpoint_path)))
        if not args.eval:
            optimizer.state = optimizer_state
            logger.info("Loaded optimizer state from {}".format(
                str(frozen_checkpoint_path)))
            last_epoch = args_json['last_epoch']
            logger.info(
                "Last epoch {}".format(last_epoch))

    output_dir = Path(args.output_dir)

    if args.resume and len(args.resume) > 0:
        args.resume_checkpoint = args.resume

    args.resume_checkpoint_complete_path = None

    if args.resume_checkpoint and os.path.exists(os.path.join(args.output_dir, args.resume_checkpoint)):
        args.resume_checkpoint_complete_path = os.path.join(
            args.output_dir, args.resume_checkpoint)
    if args.resume_checkpoint_complete_path:
        checkpoint_path = Path(args.resume_checkpoint_complete_path)
        model_weights, optimizer_state, args_json = utils.load_complete_state(
            utils.get_state_path_dict(checkpoint_path))
        model.update(model_weights)
        model_without_ddp = model
        logger.info("Loaded model weights from {}".format(
            str(checkpoint_path)))
        if not args.eval:
            optimizer.state = optimizer_state
            logger.info("Loaded optimizer state from {}".format(
                str(checkpoint_path)))
            args.start_epoch = args_json['last_epoch'] + 1
            logger.info(
                "Starting training from epoch {}".format(args.start_epoch))

    coco_evaluator = None
    if args.eval and args.base_ds is not None:
        os.environ['EVAL_FLAG'] = 'TRUE'
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, args.output_dir,
                                              wo_class_error=wo_class_error, args=args, logger=logger,
                                              print_freq=args.print_freq, print_loss_dict_freq=args.print_loss_dict_freq,
                                              max_iterations=args.max_eval_iterations,
                                              compile_forward=args.compile_forward, compile_loss_computation=args.compile_backward)
        if args.output_dir:
            savepath = os.path.join(
                args.output_dir, 'eval.pkl')
            logger.info("Saving res to {}".format(savepath))
            import pickle
            with open(savepath, 'wb') as f:
                pickle.dump(coco_evaluator.coco_eval["bbox"].eval, f)

        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}}
        if args.output_dir:
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        return

    print("Start training")
    start_time = time.time()
    best_map_holder = BestMetricHolder()
    for epoch in range(args.start_epoch, args.epochs):

        if args.use_custom_dataloader and args.reinstantiate_dataloader_every_epoch:
            data_loader_train = CustomDataLoader(
                dataset_train, args.batch_size, shuffle=True, collate_fn=utils.collate_fn)
            data_loader_val = CustomDataLoader(
                dataset_val, 1, shuffle=False, collate_fn=utils.collate_fn)

        epoch_start_time = time.time()

        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, epoch,
            args.clip_max_norm, wo_class_error=wo_class_error, args=args,
            logger=(logger if args.save_log else None),
            print_freq=args.print_freq,
            print_loss_dict_freq=args.print_loss_dict_freq,
            compile_forward=args.compile_forward,
            compile_backward=args.compile_backward)

        if args.output_dir:
            checkpoint_paths = [Path(output_dir / 'checkpoint')]

        if args.output_dir:
            checkpoint_paths = [Path(output_dir / 'checkpoint')]
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop_epochs == 0 or (epoch + 1) % args.save_checkpoint_interval == 0:
                checkpoint_paths.append(
                    Path(output_dir / f'checkpoint{epoch:04}'))
            for checkpoint_path in checkpoint_paths:
                path_dict = utils.get_state_path_dict(checkpoint_path)
                state_dict = utils.get_state_dict(
                    model, optimizer, args, epoch)
                utils.save_complete_state(path_dict, state_dict)
                logger.info(f"Saved checkpoint: {str(path_dict)}")

        # eval
        if args.base_ds is not None:
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, args.output_dir,
                wo_class_error=wo_class_error, args=args, logger=(
                    logger if args.save_log else None)
            )
            map_regular = test_stats['coco_eval_bbox'][0]
            _isbest = best_map_holder.update(map_regular, epoch, is_ema=False)
            if _isbest:
                checkpoint_path = Path(output_dir / 'checkpoint_best_regular')
                path_dict = utils.get_state_path_dict(checkpoint_path)
                state_dict = utils.get_state_dict(
                    model, optimizer, args, epoch)
                utils.save_complete_state(path_dict, state_dict)
            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
            }
            log_stats.update(best_map_holder.summary())
        else:
            log_stats = {}
        ep_paras = {
            'epoch': epoch,
            'n_parameters': n_parameters
        }
        log_stats.update(ep_paras)
        try:
            log_stats.update({'now_time': str(datetime.datetime.now())})
        except:
            pass

        epoch_time = time.time() - epoch_start_time
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
        log_stats['epoch_time'] = epoch_time_str

        if args.output_dir:
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.p']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.p')
                    for name in filenames:
                        with open(filename, 'wb') as f:
                            pickle.dump(
                                coco_evaluator.coco_eval["bbox"].eval, f)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # remove the copied files.
    copyfilelist = vars(args).get('copyfilelist', None)
    if copyfilelist:
        from datasets.data_util import remove
        for filename in copyfilelist:
            print("Removing: {}".format(filename))
            remove(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
