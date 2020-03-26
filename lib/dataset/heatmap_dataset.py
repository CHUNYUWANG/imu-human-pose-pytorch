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
from torch.utils.data import Dataset


def no_mix_collate_fn(data):
    return data


class HeatmapDataset(Dataset):
    def __init__(self, heatmaps, annot_db, grouping, human_body):
        super().__init__()
        self.heatmaps = heatmaps
        self.annot_db = annot_db
        self.grouping = grouping
        self.nviews = len(self.grouping[0])
        self.body = human_body

    def __len__(self,):
        return len(self.grouping)

    def __getitem__(self, idx):
        items = self.grouping[idx]
        heatmaps = []
        boxes = []
        poses = []
        cameras = []
        heatmap_start_idx = self.nviews * idx
        for itm_offset, itm in enumerate(items):
            datum = self.annot_db[itm]
            camera = datum['camera']
            cameras.append(camera)
            poses.append(datum['joints_gt'])
            box = dict()
            box['scale'] = np.array(datum['scale'])
            box['center'] = np.array(datum['center'])
            boxes.append(box)
            heatmaps.append(self.heatmaps[heatmap_start_idx+itm_offset])
        heatmaps = np.array(heatmaps)

        poses = poses[0]  # 4 poses are identical
        bone_vec_tc = datum['bone_vec']
        # child - parent, parent is more close to root

        # use HumanBody.imubone as bone_vectors dict key
        imubone_mapping = {'Head': 8, 'Pelvis': 2, 'L_UpArm': 11, 'R_UpArm': 13, 'L_LowArm': 12, 'R_LowArm': 14,
                           'L_UpLeg': 5, 'R_UpLeg': 3, 'L_LowLeg': 6, 'R_LowLeg': 4}   # todo read from body
        bone_vectors = dict()
        for bone_name in imubone_mapping:
            bone_vectors[imubone_mapping[bone_name]] = bone_vec_tc[bone_name]

        limb_length = self.compute_limb_length(self.body, poses)

        # todo transform some vars into tensor
        # bone_vectors = torch.as_tensor(bone_vectors, dtype=torch.float32)

        return {'heatmaps': heatmaps, 'datum': datum,
                'boxes': boxes, 'poses': poses, 'cameras': cameras,
                'limb_length': limb_length,
                'bone_vectors': bone_vectors}

    def compute_limb_length(self, body, pose):
        limb_length = {}
        skeleton = body.skeleton
        for node in skeleton:
            idx = node['idx']
            children = node['children']

            for child in children:
                length = np.linalg.norm(pose[idx] - pose[child])
                limb_length[(idx, child)] = length
        return limb_length
