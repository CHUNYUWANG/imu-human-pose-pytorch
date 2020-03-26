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
from easydict import EasyDict as edict
import pandas

import _init_paths
from core.config import config
from core.config import update_config, update_dir
from utils.utils import create_logger
from multiviews.pictorial import rpsm, RpsmFunc
# from multiviews.body import HumanBody
from multiviews.cameras import camera_to_world_frame
# from dataset.heatmap_dataset import HeatmapDataset, no_mix_collate_fn
import dataset
import models
from multiviews.triangulate import triangulate_poses
from core.inference import get_max_preds, get_final_preds


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
    # with open('/data/extra/zhe/projects/multiview-pose-github/all_heatmaps.pkl', 'rb') as f:  # todo
    #     all_heatmaps = pickle.load(f)

    # pairwise_file = os.path.join(config.DATA_DIR, config.PICT_STRUCT.PAIRWISE_FILE)
    # with open(pairwise_file, 'rb') as f:
    #     pairwise = pickle.load(f)['pairwise_constrain']

    # mp = torch.multiprocessing.get_context('spawn')
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # gpus = [0]  # todo write in config rather than hard code
    # do_bone_vectors = args.withIMU
    # logger.info('Whether use IMU Bone Orientation: {}'.format(str(do_bone_vectors)))
    # dev = torch.device('cuda:{}'.format(gpus[0]))

    grouping = test_dataset.grouping
    db = test_dataset.db
    mpjpes = []
    body = HumanBody()
    body_joints = []
    for j in body.skeleton:
        body_joints.append(j['name'])
    heatmap_dataset = HeatmapDataset(all_heatmaps, db, grouping, body)
    heatmap_loader = torch.utils.data.DataLoader(heatmap_dataset,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 num_workers=1,
                                                 pin_memory=True,
                                                 collate_fn=no_mix_collate_fn)

    # modify df definition if add extra metrics to report
    results_df = pandas.DataFrame(columns=['imgid','subject', 'action', 'subaction', 'mpjpe'] + body_joints)
    # for i, items in tqdm(enumerate(heatmap_loader)):
    for i, items in enumerate(heatmap_loader):
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

        # preds = []
        # maxvs = []
        nview = heatmaps.shape[0]
        njoints = heatmaps.shape[1]
        # for idv in range(nview):  # nview
        #     hm = heatmaps[idv]
        #     center = boxes[idv]['center']
        #     scale = boxes[idv]['scale']
        #     pred, maxv = get_final_preds(config, hm, center, scale)
        #     preds.append(pred)
        #     maxvs.append(maxv)

        centers = [boxes[i]['center'] for i in range(nview)]
        scales = [boxes[i]['scale'] for i in range(nview)]
        preds, maxvs = get_final_preds(config, heatmaps, centers, scales)

        # obtain joint vis from maxvs by a threshold
        vis_thresh = 0.3
        joints_vis = np.greater(maxvs, vis_thresh)

        # if not np.all(joints_vis):  # for debug
        #     print(maxvs)

        # check if at least two views available for each joints
        valid_views = np.swapaxes(joints_vis, 0, 1).sum(axis=1).reshape(-1)
        # print(valid_views)
        if np.any(valid_views < 2):
            # print(maxvs)
            maxvs_t = np.swapaxes(maxvs, 0, 1).reshape(njoints, nview)  # (njoints, nview)
            sorted_index = np.argsort(maxvs_t, axis=1)
            top2_index = sorted_index[:, ::-1][:, :2]  # large to fewer, select top 2
            top2_vis = np.zeros((njoints, nview), dtype=np.bool)
            for j in range(njoints):
                for ind_view in top2_index[j]:
                    top2_vis[j, ind_view] = True
            top2_vis_reshape = np.transpose(top2_vis).reshape(nview, njoints, 1)
            joints_vis = np.logical_or(joints_vis, top2_vis_reshape)
            logger.info('idx_{:0>6d} sub_{} act_{} subact_{} has some joints whose valid view < 2'.format(
                datum['image_id'], datum['subject'], datum['action'], datum['subaction']))

        poses2ds = np.array(preds)
        pose3d = np.squeeze(triangulate_poses(cameras, poses2ds, joints_vis))

        # for idx_datum, prediction in enumerate(outputs_cat):
        datum = items[0]['datum']
        # gt_poses = datum['joints_gt']
        # mpjpe = np.mean(np.sqrt(np.sum((prediction - gt_poses) ** 2, axis=1)))

        metric = get_one_grouping_metric(datum, pose3d, results_df)
        mpjpe = metric['mpjpe']
        mpjpes.append(mpjpe)

        logger.info('idx_{:0>6d} sub_{} act_{} subact_{} mpjpe is {}'.format(
            datum['image_id'], datum['subject'], datum['action'], datum['subaction'], mpjpe))
            # logger.info(prediction[0], )

        # prediction = rpsm_nn(**input_params)

    logger.info('avg mpjpes on {} val samples is: {}'.format(len(grouping), np.mean(mpjpes)))
    # flag_orient = 'with' if do_bone_vectors else 'without'
    backbone_time = os.path.split(tb_log_dir)[1]
    results_df_save_path = os.path.join(final_output_dir, 'estimate3d_triangulate_{}.csv'.format(backbone_time))
    results_df.to_csv(results_df_save_path)


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
