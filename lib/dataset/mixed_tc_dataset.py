# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from core.config import config
from dataset.joints_dataset import JointsDataset
from dataset.multiview_h36m import MultiViewH36M
from dataset.mpii import MPIIDataset
from dataset.totalcapture import TotalCaptureDataset


class MixedTotalCaptureDataset(JointsDataset):

    def __init__(self, cfg, image_set, is_train, transform=None):
        super().__init__(cfg, image_set, is_train, transform)
        mpii = MPIIDataset(cfg, image_set, is_train, transform)
        tc = TotalCaptureDataset(cfg, image_set, is_train, transform)
        self.mpii = mpii
        self.tc = tc

        self.tc_size = len(tc.db)
        self.tc_grouping_size = len(tc.grouping)
        self.db = tc.db + mpii.db

        self.grouping = tc.grouping + self.mpii_grouping(mpii.db, start_frame=len(tc.db))

        self.group_size = len(self.grouping)

    def __len__(self):
        return self.group_size

    def __getitem__(self, idx):
        input, target, weight, meta = [], [], [], []
        # use data_source to determine what info in meta
        if idx < self.tc_grouping_size:
            data_source = 'totalcapture'
            return self.tc.__getitem__(idx)
        else:
            data_source = 'mpii'
            items = self.grouping[idx]
            # self.mpii.__getitem__(idx-self.tc_size)  # mpii does not support __getitem__()
            for item in items:
                i, t, w, m = super().__getitem__(item, source=data_source)  # parent class JointsDataset
                input.append(i)
                target.append(t)
                weight.append(w)
                meta.append(m)
            return input, target, weight, meta

    def mpii_grouping(self, db, start_frame=1):
        mpii_grouping = []
        mpii_length = len(db)
        for i in range(mpii_length // 4):
            mini_group = []
            for j in range(4):
                index = i * 4 + j
                mini_group.append(index + start_frame)
            mpii_grouping.append(mini_group)
        return mpii_grouping
