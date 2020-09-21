#!/usr/bin/env python
""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import argparse
import time
import yaml
import os, sys
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm.data import Dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import create_model, resume_checkpoint, convert_splitbn_model
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

import json
from taowei.timer import Timer
from taowei.torch2.utils.logging import initialize_logger, initialize_tb_writer
from taowei.torch2.utils.classif import print_torch_info, post_collate_fn
from taowei.torch2.utils.distributed import compute_precise_bn_stats

writer = None # NOTE: do not use args.writer, because args.writer can not be pickled

torch.backends.cudnn.benchmark = True
# _logger = logging.getLogger('train')


def add_bool_group(parser, flag, default=False, help=None):
    """Add --<flag> and --no-<flag> to parser."""

    # Set help to flag name if None
    help = flag if help is None else help
    # Replace '-' to '_' for flag name since '-' is not allowed for variable name
    _flag = flag.replace('-', '_')
    # Add a mutex group with --<flag> and --no-<flag>
    flag_parser = parser.add_mutually_exclusive_group(required=False)
    flag_parser.add_argument('--' + flag, dest=_flag, action='store_true',
        help='enable {}'.format(help))
    flag_parser.add_argument('--no-' + flag, dest=_flag, action='store_false',
        help='disable {}'.format(help))
    # Set default (flag is set to False by default)
    if default is True:
        parser.set_defaults(**{_flag: True})


# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset / Model parameters
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--model', default='resnet101', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('--model-kwargs', metavar='KWARGS', default='{}',
                    help='model architecture keyword arguments')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes', type=int, default=1000, metavar='N',
                    help='number of label classes (default: 1000)')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-vb', '--validation-batch-size-multiplier', type=int, default=1, metavar='N',
                    help='ratio of validation batch size to training batch size (default: 1)')

# Optimizer parameters
parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: 1e-8)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.0001,
                    help='weight decay (default: 0.0001)')

# Learning rate schedule parameters
parser.add_argument('--sched', default='step', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Augmentation & regularization parameters
parser.add_argument('--data-aug', default='',
                    help='Data augmentation strategy by name, e.g. fbresearch etc.')
parser.add_argument('--no-aug', action='store_true', default=False,
                    help='Disable all training augmentation, override other train aug args')
parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
parser.add_argument('--ratio', type=float, nargs='+', default=[3./4., 4./3.], metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
parser.add_argument('--hflip', type=float, default=0.5,
                    help='Horizontal flip training aug probability')
parser.add_argument('--vflip', type=float, default=0.,
                    help='Vertical flip training aug probability')
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--color-jitter-hue', type=float, default=0.0, metavar='PCT',
                    help='Color jitter hue factor (default: 0.0)')
parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
parser.add_argument('--aug-splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
parser.add_argument('--jsd', action='store_true', default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                    help='Random erase prob (default: 0.)')
parser.add_argument('--remode', type=str, default='const',
                    help='Random erase mode (default: "const")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
parser.add_argument('--mixup', type=float, default=0.0,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix', type=float, default=0.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')

# Batch norm parameters (only works with gen_efficientnet based models currently)
parser.add_argument('--bn-tf', action='store_true', default=False,
                    help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--dist-bn', type=str, default='',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
parser.add_argument('--split-bn', action='store_true',
                    help='Enable separate BN layers per augmentation split.')

# Model Exponential Moving Average
parser.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')

# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
# parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    # help='disable fast prefetcher')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')

add_bool_group(parser, 'prefetcher', default=True, help='fast prefetcher')
add_bool_group(parser, 'filter-bias-and-bn', default=True, help='filter bias and bn for optimizer')
add_bool_group(parser, 'auto-resume', default=True, help='auto resume model from last checkpoint')
add_bool_group(parser, 'eval-first', default=False, help='evaluate first before training')
add_bool_group(parser, 'use-precise-bn-stats', default=False, help='use precise bn stats')


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    # setup_default_logging()
    args, args_text = _parse_args()

    # args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        if args.distributed and args.num_gpu > 1:
            logging.warning(
                'Using more than one GPU per process in distributed mode is not allowed.Setting num_gpu to 1.')
            args.num_gpu = 1

    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.num_gpu = 1
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
    assert args.rank >= 0

    if args.color_jitter_hue > 0.0:
        args.color_jitter = tuple([args.color_jitter] * 3 + [args.color_jitter_hue])

    # output dir
    args.log_file = 'stdout.log.txt'
    args.output_dir = '.' if not args.output else args.output
    if args.local_rank == 0:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    initialize_logger(os.path.join(args.output_dir, args.log_file), distributed=args.distributed)
    print('Cmd:\n{}'.format(' '.join(sys.argv)))
    print(json.dumps(args.__dict__, indent=4))
    print_torch_info()
    global writer
    writer = initialize_tb_writer(os.path.join(args.output_dir, 'runs'), distributed=args.distributed)

    if args.distributed:
        logging.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
    else:
        logging.info('Training with a single process on %d GPUs.' % args.num_gpu)

    torch.manual_seed(args.seed + args.rank)

    if '.' in args.model:
        from taowei.torch2.models import load_network
        model = load_network(args.model, **eval(args.model_kwargs))
        logging.info('Creating custom model {} with kwargs {}. Dropout, DropPath, DropBlock may not be supported'.format(args.model, args.model_kwargs))
    else:
        model = create_model(
            args.model,
            pretrained=args.pretrained,
            num_classes=args.num_classes,
            drop_rate=args.drop,
            drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
            drop_path_rate=args.drop_path,
            drop_block_rate=args.drop_block,
            global_pool=args.gp,
            bn_tf=args.bn_tf,
            bn_momentum=args.bn_momentum,
            bn_eps=args.bn_eps,
            checkpoint_path=args.initial_checkpoint)

    # Modify BN momentum and eps 
    bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    for bn in bns:
        if args.bn_momentum:
            bn.momentum = args.bn_momentum
        if args.bn_eps:
            bn.eps = args.bn_eps

    if args.local_rank == 0:
        logging.info('Model %s created, param count: %d' %
                     (args.model, sum([m.numel() for m in model.parameters()])))

    data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)

    # print model summary and plot network
    try:
        from taowei.torch2.utils.viz import print_summary, plot_network
        model.eval() # NOTE: avoid batch_norm buffer being changed
        if 'input_size' in data_config:
            data_shape = (1, ) + tuple(data_config['input_size'])
        else:
            data_shape = (1, 3, 224, 224)
        print_summary(model, data_shape=data_shape)
        if args.local_rank == 0:
            plot_network(model, data_shape=data_shape).save(os.path.join(args.output_dir, args.model) + '.gv')
    except Exception as e:
        print(e)
    model_strs = str(model).split('\n')
    model_strs = model_strs[:25] + ['... ...'] + model_strs[-25:] if len(model_strs) > 50 else model_strs
    print('\n'.join(model_strs)) # INFO: for DataParallel or DistributedDataParallel, BatchNorm momentum etc.

    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    use_amp = None
    if args.amp:
        # for backwards compat, `--amp` arg tries apex before native amp
        if has_apex:
            args.apex_amp = True
        elif has_native_amp:
            args.native_amp = True
    if args.apex_amp and has_apex:
        use_amp = 'apex'
    elif args.native_amp and has_native_amp:
        use_amp = 'native'
    elif args.apex_amp or args.native_amp:
        logging.warning("Neither APEX or native Torch AMP is available, using float32. "
                        "Install NVIDA apex or upgrade to PyTorch 1.6")

    if args.num_gpu > 1:
        if use_amp == 'apex':
            logging.warning(
                'Apex AMP does not work well with nn.DataParallel, disabling. Use DDP or Torch AMP.')
            use_amp = None
        model = nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
        assert not args.channels_last, "Channels last not supported with DP, use DDP."
    else:
        model.cuda()
        if args.channels_last:
            model = model.to(memory_format=torch.channels_last)

    optimizer = create_optimizer(args, model, filter_bias_and_bn=args.filter_bias_and_bn)
    print('Optimizer: {}'.format(optimizer))

    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if args.local_rank == 0:
            logging.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            logging.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if args.local_rank == 0:
            logging.info('AMP not enabled. Training in float32.')

    # optionally resume from a checkpoint
    resume_epoch = None
    # auto resume for last.pth.tar, resume is strict (can not mix distributed and multi-gpu)
    if args.auto_resume:
        checkpoint_file = os.path.join(args.output_dir, 'last.pth.tar')
        if os.path.exists(checkpoint_file):
            args.resume = checkpoint_file
    if args.resume:
        resume_epoch = resume_checkpoint(
            model, args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.local_rank == 0)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume=args.resume)

    if args.distributed:
        if args.sync_bn:
            assert not args.split_bn
            try:
                if has_apex and use_amp != 'native':
                    # Apex SyncBN preferred unless native amp is activated
                    model = convert_syncbn_model(model)
                else:
                    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                if args.local_rank == 0:
                    logging.info(
                        'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                        'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')
            except Exception as e:
                logging.error('Failed to enable Synchronized BatchNorm. Install Apex or Torch >= 1.1')
        if has_apex and use_amp != 'native':
            # Apex DDP preferred unless native amp is activated
            if args.local_rank == 0:
                logging.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if args.local_rank == 0:
                logging.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[args.local_rank])  # can use device str in Torch >= 1.1
        # NOTE: EMA model does not need to be wrapped by DDP

    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    print('LR Scheduler:\n{}'.format(lr_scheduler))
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    args.start_epoch = start_epoch

    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        logging.info('Scheduled epochs: {}'.format(num_epochs))

    train_dir = os.path.join(args.data, 'train')
    if not os.path.exists(train_dir):
        logging.error('Training folder does not exist at: {}'.format(train_dir))
        exit(1)
    dataset_train = Dataset(train_dir)

    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        if args.prefetcher:
            assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        data_aug=args.data_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        use_multi_epochs_loader=args.use_multi_epochs_loader
    )
    try:
        print('Data Config:\n{}'.format(data_config)) # For Transform
        print('Train Dataset:\n{}'.format(loader_train.dataset))
        print('Train Transform:\n{}'.format(loader_train.dataset.transform))
    except:
        pass

    eval_dir = os.path.join(args.data, 'val')
    if not os.path.isdir(eval_dir):
        eval_dir = os.path.join(args.data, 'validation')
        if not os.path.isdir(eval_dir):
            logging.error('Validation folder does not exist at: {}'.format(eval_dir))
            exit(1)
    dataset_eval = Dataset(eval_dir)

    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size_multiplier * args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        data_aug=args.data_aug,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
    )
    try:
        print('Val Dataset:\n{}'.format(loader_eval.dataset))
        print('Val Transform:\n{}'.format(loader_eval.dataset.transform))
    except:
        pass

    if args.jsd:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing).cuda()
    elif mixup_active:
        # smoothing is handled with mixup target transform
        train_loss_fn = SoftTargetCrossEntropy().cuda()
    elif args.smoothing:
        train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing).cuda()
    else:
        train_loss_fn = nn.CrossEntropyLoss().cuda()
    print('Loss Function:\n{}'.format(train_loss_fn))
    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    # output_dir = ''
    output_dir = args.output_dir
    if args.local_rank == 0:
        # output_base = args.output if args.output else './output'
        # exp_name = '-'.join([
        #     datetime.now().strftime("%Y%m%d-%H%M%S"),
        #     args.model,
        #     str(data_config['input_size'][-1])
        # ])
        # output_dir = get_outdir(output_base, 'train', exp_name)
        decreasing = True if eval_metric == 'loss' else False
        saver = CheckpointSaver(
            model=model, optimizer=optimizer, args=args, model_ema=model_ema, amp_scaler=loss_scaler,
            checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    try:
        # compute precise bn stats before training, for quick debugging (e.g. no .batch_size)
        if args.use_precise_bn_stats:
            print('Computing precise bn stats')
            compute_precise_bn_stats(model, loader_train, args.world_size)
        if args.eval_first:
            eval_metrics = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)
        for epoch in range(start_epoch, num_epochs):
            args.epoch = epoch
            if args.distributed:
                loader_train.sampler.set_epoch(epoch)

            train_metrics = train_epoch(
                epoch, model, loader_train, optimizer, train_loss_fn, args,
                lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                amp_autocast=amp_autocast, loss_scaler=loss_scaler, model_ema=model_ema, mixup_fn=mixup_fn)

            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if args.local_rank == 0:
                    logging.info("Distributing BatchNorm running means and vars")
                distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

            if args.use_precise_bn_stats:
                print('Computing precise bn stats')
                compute_precise_bn_stats(model, loader_train, args.world_size)

            eval_metrics = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)

            if model_ema is not None and not args.model_ema_force_cpu:
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')
                ema_eval_metrics = validate(
                    model_ema.ema, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast, log_suffix=' (EMA)')
                eval_metrics = ema_eval_metrics

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            # update_summary(
                # epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                # write_header=best_metric is None)

            if saver is not None and args.local_rank == 0:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        logging.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))


def train_epoch(
        epoch, model, loader, optimizer, loss_fn, args,
        lr_scheduler=None, saver=None, output_dir='', amp_autocast=suppress,
        loss_scaler=None, model_ema=None, mixup_fn=None):

    global writer
    from taowei.torch2.utils.classif import ProgressMeter
    batch_size = args.batch_size * args.world_size if args.distributed else args.batch_size
    progress = ProgressMeter(iters_per_epoch=len(loader),
        epoch=epoch, epochs=args.epochs, split='train', writer=writer, batch_size=batch_size)

    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    # batch_time_m = AverageMeter()
    # data_time_m = AverageMeter()
    # losses_m = AverageMeter()

    model.train()

    # end = time.time()
    timer = Timer()
    timer.tic()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        # data_time_m.update(time.time() - end)
        progress.update('data_time', timer.toc(from_last_toc=True))

        if not torch.is_tensor(input) or not torch.is_tensor(target):
            with torch.no_grad():
                input, target = post_collate_fn((input, target))
        
        if batch_idx  == 0:
            print('Inputs: {}'.format({'mean': input.mean().item(), 'std': input.std().item(),
                'min': input.min().item(), 'max': input.max().item(), 'shape': input.shape}))
            print('Target: {} shape={}'.format(target, target.shape))

        if not args.prefetcher:
            input, target = input.cuda(), target.cuda()
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        with amp_autocast():
            output = model(input)
            loss = loss_fn(output, target)
            if target.ndim == 1:
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
        progress.update('forward_time', timer.toc(from_last_toc=True))

        # if not args.distributed:
            # losses_m.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(loss, optimizer)
            progress.update('backupdate_time', timer.toc(from_last_toc=True))
        else:
            loss.backward()
            progress.update('backward_time', timer.toc(from_last_toc=True))
            optimizer.step()
            progress.update('update_time', timer.toc(from_last_toc=True))

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data, args.world_size)
            if target.ndim == 1:
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
        else:
            reduced_loss = loss.data

        torch.cuda.synchronize()

        n = output.size(0) * args.world_size if args.distributed else output.size(0)
        progress.update('loss', loss.item(), n)
        if target.ndim == 1:
            progress.update('top1', acc1.item(), n)
            progress.update('top5', acc5.item(), n)

        # torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
        num_updates += 1

        # measure elapsed time
        progress.update('batch_time', timer.toctic())

        # batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            # lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            # lr = sum(lrl) / len(lrl)

            # if args.distributed:
            #     reduced_loss = reduce_tensor(loss.data, args.world_size)
            #     losses_m.update(reduced_loss.item(), input.size(0))

            if args.local_rank == 0:
                progress.log_iter_stats(iter=batch_idx, batch_size=n, lr=optimizer.param_groups[0]['lr'])
                # logging.info(
                #     'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                #     'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                #     'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                #     '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                #     'LR: {lr:.3e}  '
                #     'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                #         epoch,
                #         batch_idx, len(loader),
                #         100. * batch_idx / last_idx,
                #         loss=losses_m,
                #         batch_time=batch_time_m,
                #         rate=input.size(0) * args.world_size / batch_time_m.val,
                #         rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                #         lr=lr,
                #         data_time=data_time_m))

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True)

        # if saver is not None and args.recovery_interval and (
        #         last_batch or (batch_idx + 1) % args.recovery_interval == 0):
        #     saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=progress.stats['loss'].avg) # losses_m.avg)

        # end = time.time()
        # end for

    progress.log_epoch_stats(lr=optimizer.param_groups[0]['lr'])
    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    # return OrderedDict([('loss', losses_m.avg)])
    return OrderedDict([('loss', progress.stats['loss'].avg)])


def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix=''):
    # batch_time_m = AverageMeter()
    # losses_m = AverageMeter()
    # top1_m = AverageMeter()
    # top5_m = AverageMeter()

    global writer
    from taowei.torch2.utils.classif import ProgressMeter
    epoch = args.start_epoch - 1 if 'epoch' not in args else args.epoch
    batch_size = args.batch_size * args.world_size if args.distributed else args.batch_size
    progress = ProgressMeter(iters_per_epoch=len(loader),
        epoch=epoch, split='val', writer=writer, batch_size=batch_size)

    model.eval()

    timer = Timer()
    timer.tic()
    # end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            progress.update('data_time', timer.toc(from_last_toc=True))

            if batch_idx  == 0:
                print('Inputs: {}'.format({'mean': input.mean().item(), 'std': input.std().item(),
                    'min': input.min().item(), 'max': input.max().item(), 'shape': input.shape}))
                print('Target: {} shape={}'.format(target, target.shape))

            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]
            progress.update('forward_time', timer.toc(from_last_toc=True))

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            # losses_m.update(reduced_loss.item(), input.size(0))
            # top1_m.update(acc1.item(), output.size(0))
            # top5_m.update(acc5.item(), output.size(0))
            n = output.size(0) * args.world_size if args.distributed else output.size(0)
            progress.update('loss', loss.item(), n)
            progress.update('top1', acc1.item(), n)
            progress.update('top5', acc5.item(), n)

            # measure elapsed time
            progress.update('batch_time', timer.toctic())

            # batch_time_m.update(time.time() - end)
            # end = time.time()
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                progress.log_iter_stats(iter=batch_idx, batch_size=n)
                # log_name = 'Test' + log_suffix
                # logging.info(
                #     '{0}: [{1:>4d}/{2}]  '
                #     'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                #     'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                #     'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                #     'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                #         log_name, batch_idx, last_idx, batch_time=batch_time_m,
                #         loss=losses_m, top1=top1_m, top5=top5_m))

    progress.log_epoch_stats()
    # metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])
    metrics = OrderedDict([('loss', progress.stats['loss'].avg), ('top1', progress.stats['top1'].avg),
        ('top5', progress.stats['top5'].avg)])

    return metrics


if __name__ == '__main__':
    main()
