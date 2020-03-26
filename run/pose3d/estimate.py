# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import h5py
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.multiprocessing
from tqdm import tqdm
import pandas

import _init_paths
from core.config import config
from core.config import update_config, update_dir
from utils.utils import create_logger
from multiviews.pictorial import rpsm, RpsmFunc
from multiviews.cameras import camera_to_world_frame
import dataset
import models
from utils.pose_utils import PoseUtils


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate 3D Pose Estimation')
    parser.add_argument(
        '--cfg', help='configuration file name', required=True, type=str)
    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    parser.add_argument(
        '--modelDir', help='model directory', type=str, default='')
    parser.add_argument('--logDir', help='log directory', type=str, default='')
    parser.add_argument(
        '--dataDir', help='data directory', type=str, default='')
    parser.add_argument(
        '--withIMU', help='use bone orientation in 3d',
        type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=False)

    args = parser.parse_args()
    update_dir(args.modelDir, args.logDir, args.dataDir)
    return args


def main():
    args = parse_args()

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'test3d')

    prediction_path = os.path.join(final_output_dir,
                                   config.TEST.HEATMAP_LOCATION_FILE)
    # prediction_path = os.path.join(final_output_dir, 'image_only_heatmaps.h5')
    logger.info(prediction_path)
    test_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
        config, config.DATASET.TEST_SUBSET, False)

    if config.DATASET.TRAIN_DATASET == 'multiview_h36m':
        from multiviews.h36m_body import HumanBody
        from dataset.heatmap_dataset_h36m import HeatmapDataset, no_mix_collate_fn
    elif config.DATASET.TRAIN_DATASET == 'totalcapture':
        from multiviews.totalcapture_body import HumanBody
        from dataset.heatmap_dataset import HeatmapDataset, no_mix_collate_fn

    all_heatmaps = h5py.File(prediction_path)['heatmaps']
    all_heatmaps = all_heatmaps[()]  # load all heatmaps into ram to avoid a h5 multi-threading bug

    pairwise_file = os.path.join(config.DATA_DIR, config.PICT_STRUCT.PAIRWISE_FILE)
    with open(pairwise_file, 'rb') as f:
        pairwise = pickle.load(f)['pairwise_constrain']
    if config.DATASET.TRAIN_DATASET == 'multiview_h36m':
        # convert sparse mat to int64 mat
        no_sparse_pairwise = dict()
        for k in pairwise:
            no_sparse_pairwise[k] = pairwise[k].toarray().astype(np.int64)
        pairwise = no_sparse_pairwise

    # mp = torch.multiprocessing.get_context('spawn')
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    gpus = [0]  # todo write in config rather than hard code
    do_bone_vectors = args.withIMU
    logger.info('Whether use IMU Bone Orientation: {}'.format(str(do_bone_vectors)))
    dev = torch.device('cuda:{}'.format(gpus[0]))
    b_align_mpjpe = False

    grouping = test_dataset.grouping
    db = test_dataset.db
    mpjpes = []
    aligned_mpjpes = []
    body = HumanBody()
    body_joints = []
    for j in body.skeleton:
        body_joints.append(j['name'])
    heatmap_dataset = HeatmapDataset(all_heatmaps, db, grouping, body)
    heatmap_loader = torch.utils.data.DataLoader(heatmap_dataset,
                                                 batch_size=1*len(gpus),
                                                 shuffle=False,
                                                 num_workers=0,  # 1*len(gpus),
                                                 pin_memory=True,
                                                 collate_fn=no_mix_collate_fn)

    pairwise_tensor = dict()
    for edge in pairwise:
        edge_pairwise = pairwise[edge].astype(np.int64)
        edge_pairwise = torch.as_tensor(edge_pairwise, dtype=torch.float32)
        pairwise_tensor[edge] = edge_pairwise.to(dev)
        # pairwise_tensor[edge] = edge_pairwise
    pairwise_tensor = pairwise_tensor

    if do_bone_vectors:
        # load orient grid for psm 4096bins
        orient_grid_file = os.path.join(config.DATA_DIR, 'data/pict/orient_grid.npy')
        with open(orient_grid_file, 'rb') as f:
            orient_pairwise = np.load(f)
        orient_pairwise_tensor = torch.as_tensor(orient_pairwise, dtype=torch.float32).to(dev)

    rpsm_nn_kwargs = dict()
    rpsm_nn_kwargs['do_bone_vectors'] = do_bone_vectors
    if do_bone_vectors:
        rpsm_nn_kwargs['orient_pairwise'] = orient_pairwise_tensor
    rpsm_nn = RpsmFunc(pairwise_tensor, body, **rpsm_nn_kwargs)

    # data parallel !!!
    # replicate module
    device_ids = gpus  # make sure gpus is a list of int rather than string
    replicas = nn.parallel.replicate(rpsm_nn, device_ids)

    # modify df definition if add extra metrics to report
    results_df = pandas.DataFrame(columns=['imgid','subject', 'action', 'subaction', 'mpjpe'] + body_joints)
    aligned_results_df = pandas.DataFrame(columns=['imgid', 'subject', 'action', 'subaction', 'mpjpe'] + body_joints)
    pose_utils = PoseUtils()
    p3dpose = []
    p3dpose_align = []
    p3dpose_gt = []
    for i, items in tqdm(enumerate(heatmap_loader)):
        input_params_all_devices = []
        for item in items:
            # item = items[0]
            heatmaps = item['heatmaps']
            datum = item['datum']
            boxes = item['boxes']
            poses = item['poses']
            cameras = item['cameras']
            limb_length = item['limb_length']
            bone_vectors = item['bone_vectors']

            grid_center = poses[0]

            input_params = dict()
            input_params['cams'] = cameras
            input_params['heatmaps'] = heatmaps
            input_params['boxes'] = boxes
            input_params['grid_center'] = grid_center
            input_params['limb_length'] = limb_length
            input_params['config'] = config

            input_params['do_bone_vectors'] = do_bone_vectors
            input_params['bone_vectors'] = bone_vectors

            input_params_all_devices.append(input_params)

        # replicas = nn.parallel.replicate(rpsm_nn, device_ids)
        # data parallel !!!
        # data
        inputs = []
        for dev_idx, dev in enumerate(device_ids):
            if len(input_params_all_devices) > dev_idx:  # in case last batch has fewer data
                param = input_params_all_devices[dev_idx]
                device = torch.device('cuda:{}'.format(dev))
                # data/param to dev if needed
                # d['k1'] = d['k1'].to(device)
                inputs.append(param)

        replicas_valid = replicas[:len(inputs)]  # if last batch have fewer data
        outputs = nn.parallel.parallel_apply(replicas_valid, inputs=[None]*len(inputs), kwargs_tup=inputs)

        # gather
        outputs_cat = []
        # out_dev = torch.device('cuda:{}'.format(output_device))
        for out in outputs:
            outputs_cat.append(out)

        for idx_datum, prediction in enumerate(outputs_cat):
            datum = items[idx_datum]['datum']
            gt_poses = datum['joints_gt']
            # mpjpe = np.mean(np.sqrt(np.sum((prediction - gt_poses) ** 2, axis=1)))
            p3dpose.append(prediction)
            p3dpose_gt.append(gt_poses)

            metric = get_one_grouping_metric(datum, prediction, results_df)
            mpjpe = metric['mpjpe']
            mpjpes.append(mpjpe)

            logger.info('idx_{:0>6d} sub_{} act_{} subact_{} mpjpe is {}'.format(
                datum['image_id'], datum['subject'], datum['action'], datum['subaction'], mpjpe))

            if b_align_mpjpe:
                d, Z, tform = pose_utils.procrustes(gt_poses, prediction)
                aligned_pjpe = np.sqrt(np.sum((Z - gt_poses) ** 2, axis=1))
                aligned_mpjpe = np.mean(aligned_pjpe)
                aligned_metric = get_one_grouping_metric(datum, Z, aligned_results_df)
                aligned_mpjpe = aligned_metric['mpjpe']
                aligned_mpjpes.append(aligned_mpjpe)
                logger.info(aligned_mpjpe)
                p3dpose_align.append(Z)

    logger.info('avg mpjpes on {} val samples is: {}'.format(len(grouping), np.mean(mpjpes)))
    flag_orient = 'with' if do_bone_vectors else 'without'
    backbone_time = os.path.split(tb_log_dir)[1]
    results_df_save_path = os.path.join(final_output_dir, 'estimate3d_{}_imu_{}.csv'.format(flag_orient, backbone_time))
    results_df.to_csv(results_df_save_path)
    p3dpose = np.array(p3dpose)
    p3dpose_path = os.path.join(final_output_dir, 'estimate3d_{}_imu_{}.npy'.format(flag_orient, backbone_time))
    np.save(p3dpose_path, p3dpose)

    p3dpose_gt = np.array(p3dpose_gt)
    p3dpose_path = os.path.join(final_output_dir, 'estimate3d_gt_{}_imu_{}.npy'.format(flag_orient, backbone_time))
    np.save(p3dpose_path, p3dpose_gt)

    if b_align_mpjpe:
        logger.info('aligned avg mpjpes on {} val samples is: {}'.format(len(grouping), np.mean(aligned_mpjpes)))
        results_df_save_path = os.path.join(final_output_dir,
                                            'estimate3d_aligned_{}_imu_{}.csv'.format(flag_orient, backbone_time))
        aligned_results_df.to_csv(results_df_save_path)
        p3dpose_align = np.array(p3dpose_align)
        p3dpose_path = os.path.join(final_output_dir, 'estimate3d_align_{}_imu_{}.npy'.format(flag_orient, backbone_time))
        np.save(p3dpose_path, p3dpose_align)


def get_one_grouping_metric(datum, pred, results_df):
    gt_poses = datum['joints_gt']
    # gt_poses = torch.as_tensor(gt_poses, device=torch.device('cuda'))
    # pred_cuda = torch.as_tensor(pred, device=torch.device('cuda'))
    # pjpe = torch.sqrt(torch.sum(torch.pow((pred_cuda - gt_poses), 2), dim=1))
    # mpjpe = torch.mean(pjpe)
    pjpe = np.sqrt(np.sum((pred - gt_poses) ** 2, axis=1))
    mpjpe = np.mean(pjpe)
    results_df.loc[len(results_df)] = [datum['image_id'], datum['subject'], datum['action'], datum['subaction'], mpjpe] + pjpe.tolist()
    return results_df.loc[len(results_df)-1]


if __name__ == '__main__':
    main()
