# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import numpy as np
from git import Repo

import _init_paths
from core.config import config
from core.config import update_config
from core.config import update_dir
from core.config import get_model_name
from core.loss import JointsMSELoss
from core.function import train
from core.function import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint, load_checkpoint
from utils.utils import create_logger
import dataset
import models

from dataset.totalcapture_collate import totalcapture_collate


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    parser.add_argument(
        '--frequent',
        help='frequency of logging',
        default=config.PRINT_FREQ,
        type=int)
    parser.add_argument('--gpus', help='gpus', type=str)
    parser.add_argument('--workers', help='num of dataloader workers', type=int)

    parser.add_argument(
        '--modelDir', help='model directory', type=str, default='')
    parser.add_argument('--logDir', help='log directory', type=str, default='')
    parser.add_argument(
        '--dataDir', help='data directory', type=str, default='')
    parser.add_argument(
        '--data-format', help='data format', type=str, default='')
    args = parser.parse_args()
    update_dir(args.modelDir, args.logDir, args.dataDir)
    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.data_format:
        config.DATASET.DATA_FORMAT = args.data_format
    if args.workers:
        config.WORKERS = args.workers


def main():
    args = parse_args()
    reset_config(config, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    # print code version info
    repo = Repo('')
    repo_git = repo.git
    working_tree_diff_head = repo_git.diff('HEAD')
    this_commit_hash = repo.commit()
    cur_branches = repo_git.branch('--list')
    logger.info('Current Code Version is {}'.format(this_commit_hash))
    logger.info('Current Branch Info :\n{}'.format(cur_branches))
    logger.info('Working Tree diff with HEAD: \n{}'.format(working_tree_diff_head))

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    backbone_model = eval('models.' + config.BACKBONE_MODEL + '.get_pose_net')(
        config, is_train=True)
    model = models.multiview_pose_net.get_multiview_pose_net(backbone_model, config)
    # logger.info(pprint.pformat(model))

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # dump_input = torch.rand(
    #     (config.TRAIN.BATCH_SIZE, 3,  # config.NETWORK.NUM_JOINTS,
    #      config.NETWORK.IMAGE_SIZE[1], config.NETWORK.IMAGE_SIZE[0]))
    # writer_dict['writer'].add_graph(model, dump_input)

    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    criterion = JointsMSELoss(
        use_target_weight=config.LOSS.USE_TARGET_WEIGHT).cuda()
    # criterion_fuse = JointsMSELoss(use_target_weight=True).cuda()

    optimizer = get_optimizer(config, model)
    start_epoch = config.TRAIN.BEGIN_EPOCH
    if config.TRAIN.RESUME:
        start_epoch, model, optimizer, ckpt_perf = load_checkpoint(model, optimizer,
                                                                   final_output_dir)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR)

    # Data loading
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = eval('dataset.' + config.DATASET.TRAIN_DATASET)(
        config, config.DATASET.TRAIN_SUBSET, True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    valid_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
        config, config.DATASET.TEST_SUBSET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE * len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        collate_fn=totalcapture_collate,
        pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        collate_fn=totalcapture_collate,
        pin_memory=True)

    best_perf = ckpt_perf
    best_epoch = -1
    best_model = False
    for epoch in range(start_epoch, config.TRAIN.END_EPOCH):
        lr_scheduler.step()
        extra_param = dict()
        # extra_param['loss2'] = criterion_fuse
        train(config, train_loader, model, criterion, optimizer, epoch,
              final_output_dir, writer_dict, **extra_param)

        perf_indicator = validate(config, valid_loader, valid_dataset, model,
                                  criterion, final_output_dir, writer_dict, **extra_param)

        logger.info('=> perf indicator at epoch {} is {}. old best is {} '.format(epoch, perf_indicator, best_perf))

        if perf_indicator > best_perf:
            best_perf = perf_indicator
            best_model = True
            best_epoch = epoch
            logger.info('====> find new best model at end of epoch {}. (start from 0)'.format(epoch))
        else:
            best_model = False
        logger.info('epoch of best validation results is {}'.format(best_epoch))

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': get_model_name(config),
            'state_dict': model.module.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

        # save final state at every epoch
        final_model_state_file = os.path.join(final_output_dir,
                                              'final_state_ep{}.pth.tar'.format(epoch))
        logger.info('saving final model state to {}'.format(final_model_state_file))
        torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
