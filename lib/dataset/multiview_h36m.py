# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
import pickle
import collections

from dataset.joints_dataset import JointsDataset
from multiviews.h36m_body import HumanBody


class MultiViewH36M(JointsDataset):

    def __init__(self, cfg, image_set, is_train, transform=None):
        super().__init__(cfg, image_set, is_train, transform)
        self.actual_joints = {
            0: 'root',
            1: 'rhip',
            2: 'rkne',
            3: 'rank',
            4: 'lhip',
            5: 'lkne',
            6: 'lank',
            7: 'belly',
            8: 'neck',
            9: 'nose',
            10: 'head',
            11: 'lsho',
            12: 'lelb',
            13: 'lwri',
            14: 'rsho',
            15: 'relb',
            16: 'rwri'
        }

        self.u2a_mapping = super().get_mapping()

        grouping_db_pickle_file = osp.join(self.root, 'h36m', 'quickload',
                                           'totalcapture_quickload_{}.pkl'
                                           .format(image_set))
        if osp.isfile(grouping_db_pickle_file):
            with open(grouping_db_pickle_file, 'rb') as f:
                grouping_db = pickle.load(f)
                self.grouping = grouping_db['grouping']
                self.db = grouping_db['db']
        else:
            anno_file = osp.join(self.root, 'h36m', 'annot',
                                 'h36m_{}.pkl'.format(image_set))
            self.db = self.load_db(anno_file)

            self.u2a_mapping = super().get_mapping()
            super().do_mapping()

            self.grouping = self.get_group(self.db)
            grouping_db_to_dump = {'grouping': self.grouping, 'db': self.db}
            with open(grouping_db_pickle_file, 'wb') as f:
                pickle.dump(grouping_db_to_dump, f)

        self.group_size = len(self.grouping)

        self.body = HumanBody()
        self.imubone_mapping = {'Pelvis': 2, 'L_UpArm': 11, 'R_UpArm': 13, 'L_LowArm': 12, 'R_LowArm': 14,
                                'L_UpLeg': 5, 'R_UpLeg': 3, 'L_LowLeg': 6, 'R_LowLeg': 4}  # todo read from body

    def index_to_action_names(self):
        return {
            2: 'Direction',
            3: 'Discuss',
            4: 'Eating',
            5: 'Greet',
            6: 'Phone',
            7: 'Photo',
            8: 'Pose',
            9: 'Purchase',
            10: 'Sitting',
            11: 'SittingDown',
            12: 'Smoke',
            13: 'Wait',
            14: 'WalkDog',
            15: 'Walk',
            16: 'WalkTwo'
        }

    def load_db(self, dataset_file):
        with open(dataset_file, 'rb') as f:
            dataset = pickle.load(f)
            return dataset

    def get_group(self, db):
        grouping = {}
        nitems = len(db)
        for i in range(nitems):
            keystr = self.get_key_str(db[i])
            camera_id = db[i]['camera_id']
            if keystr not in grouping:
                grouping[keystr] = [-1, -1, -1, -1]
            grouping[keystr][camera_id] = i

        filtered_grouping = []
        for _, v in grouping.items():
            if np.all(np.array(v) != -1):
                filtered_grouping.append(v)

        if self.is_train:
            filtered_grouping = filtered_grouping[::5]
        else:
            filtered_grouping = filtered_grouping[::64]

        return filtered_grouping

    def __getitem__(self, idx):
        input, target, weight, meta = [], [], [], []
        items = self.grouping[idx]
        for item in items:
            i, t, w, m = super().__getitem__(item)
            # data type convert to float32
            m['scale'] = m['scale'].astype(np.float32)
            m['center'] = m['center'].astype(np.float32)
            m['rotation'] = int(m['rotation'])
            if 'name' in m['camera']:
                del m['camera']['name']
            for k in m['camera']:
                m['camera'][k] = m['camera'][k].astype(np.float32)

            # add bone_vec and bone_vector into meta
            # bone_vectors = child - parent
            m['bone_vectors'] = dict()
            imu_edges_reverse = self.body.imu_edges_reverse
            for boneid in imu_edges_reverse:
                parent = imu_edges_reverse[boneid][0]
                child = imu_edges_reverse[boneid][1]
                bonevec = m['joints_gt'][child] - m['joints_gt'][parent]
                m['bone_vectors'][boneid] = bonevec.astype(np.float32)

            input.append(i)
            target.append(t)
            weight.append(w)
            meta.append(m)
        return input, target, weight, meta

    def __len__(self):
        return self.group_size

    def get_key_str(self, datum):
        return 's_{:02}_act_{:02}_subact_{:02}_imgid_{:06}'.format(
            datum['subject'], datum['action'], datum['subaction'],
            datum['image_id'])

    def evaluate(self, pred, *args, **kwargs):
        pred = pred.copy()
        nview = 4
        # headsize = self.image_size[0] / 10.0
        # threshold = 0.8
        if 'threshold' in kwargs:
            threshold = kwargs['threshold']
        else:
            threshold = 0.075  # use threshold x 2000mm is threshold to decide if joint is detected
        # default box length is 2000mm, head 300mm, half head 150mm, threshold set to 0.075
        # other option 100mm -> 0.05, 50mm -> 0.025, 25mm -> 0.0125

        u2a = self.u2a_mapping
        a2u = {v: k for k, v in u2a.items() if v != '*'}
        a = list(a2u.keys())
        u = list(a2u.values())
        indexes = list(range(len(a)))
        indexes.sort(key=a.__getitem__)
        sa = list(map(a.__getitem__, indexes))
        su = np.array(list(map(u.__getitem__, indexes)))

        gt = []
        flat_items = []
        box_lengthes = []
        for items in self.grouping:
            for item in items:
                gt.append(self.db[item]['joints_2d'][su, :2])
                flat_items.append(self.db[item])
                boxsize = np.array(self.db[item]['scale']).sum() * 100.0  # crop img pixels
                box_lengthes.append(boxsize)
        gt = np.array(gt)
        pred = pred[:, su, :2]
        detection_threshold = np.array(box_lengthes).reshape((-1, 1)) * threshold

        distance = np.sqrt(np.sum((gt - pred)**2, axis=2))
        detected = (distance <= detection_threshold)

        joint_detection_rate = np.sum(detected, axis=0) / np.float(gt.shape[0])

        name_values = collections.OrderedDict()
        joint_names = self.actual_joints
        for i in range(len(a2u)):
            name_values[joint_names[sa[i]]] = joint_detection_rate[i]
        detected_int = detected.astype(np.int)
        # detected_int[joint_validity == False] = -1
        nsamples, njoints = detected.shape
        per_grouping_detected = detected_int.reshape(nsamples // nview, nview * njoints)
        # per_grouping_detected of shape (n_groupings, njoints*nview), -1: not-vis, 0: not detected, 1: detected
        return name_values, np.mean(joint_detection_rate), per_grouping_detected
