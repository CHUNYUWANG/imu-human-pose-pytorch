# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.orn import CamFusionModule, get_inv_cam, get_inv_affine_transform


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class MultiViewPose(nn.Module):
    def __init__(self, PoseResNet, CFG):
        super(MultiViewPose, self).__init__()
        self.config = CFG
        if self.config.DATASET.TRAIN_DATASET == 'multiview_h36m':
            from multiviews.h36m_body import HumanBody
            selected_bones = [3, 4, 5, 6, 12, 13, 14, 15]  # todo as param
            general_joint_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: '*', 9: 8, 10: '*', 11: 9, 12: 10,
                            13: '*', 14: 11, 15: 12, 16: 13, 17: 14, 18: 15, 19: 16}
            imu_related_joint_in_hm = [1, 2, 3, 4, 5, 6, 14, 15, 16, 17, 18, 19]
        elif self.config.DATASET.TRAIN_DATASET == 'totalcapture':
            from multiviews.totalcapture_body import HumanBody
            selected_bones = [3, 4, 5, 6, 11, 12, 13, 14]  # todo as param
            general_joint_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: '*', 9: 8, 10: '*', 11: 9,
                                     12: '*',
                                     13: '*', 14: 10, 15: 11, 16: 12, 17: 13, 18: 14, 19: 15}
            imu_related_joint_in_hm = [1, 2, 3, 4, 5, 6, 14, 15, 16, 17, 18, 19]
        self.resnet = PoseResNet
        self.b_in_view_fusion = self.config.CAM_FUSION.IN_VIEW_FUSION
        self.b_xview_self_fusion = self.config.CAM_FUSION.XVIEW_SELF_FUSION
        self.b_xview_fusion = self.config.CAM_FUSION.XVIEW_FUSION

        njoints = self.config.NETWORK.NUM_JOINTS
        h = int(self.config.NETWORK.HEATMAP_SIZE[0])
        w = int(self.config.NETWORK.HEATMAP_SIZE[1])
        self.selected_views = self.config.SELECTED_VIEWS
        nview = len(self.selected_views)
        body = HumanBody()
        depth = torch.logspace(2.7, 3.9, steps=100)  # 8m
        ndepth = depth.shape[0]
        self.h = h
        self.w = w
        self.njoints = njoints
        self.nview = nview

        joint_channel_mask = torch.zeros(1, 20, 1, 1)  # no imu fusion joints set to 0
        for sj in imu_related_joint_in_hm:
            joint_channel_mask[0, sj, 0, 0] = 1.0
        self.register_buffer('joint_channel_mask', joint_channel_mask)
        if self.b_in_view_fusion or self.b_xview_fusion:
            self.cam_fusion_module = CamFusionModule(nview, njoints, h, w, body, depth,
                                                     general_joint_mapping, selected_bones, self.config)

    def forward(self, inputs, **kwargs):
        dev = inputs.device
        meta = dict()
        for kk in kwargs:
            meta[kk] = self.merge_first_two_dims(kwargs[kk])

        batch = inputs.shape[0]
        nview = inputs.shape[1]
        inputs = inputs.view(batch*nview, *inputs.shape[2:])
        hms, feature_before_final = self.resnet(inputs)

        if self.b_in_view_fusion or self.b_xview_fusion or self.b_xview_self_fusion:
            cam_R = meta['camera_R'].to(dev)
            cam_T = meta['camera_T'].to(dev)
            cam_Intri = meta['camera_Intri'].to(dev)
            inv_cam_Intri, inv_cam_R, inv_cam_T = get_inv_cam(cam_Intri, cam_R, cam_T)
            affine_trans = meta['affine_trans'].to(dev)
            inv_affine_trans = meta['inv_affine_trans'].to(dev)
            bones_tensor = meta['bone_vectors_tensor']

            if self.b_in_view_fusion or self.b_xview_fusion:
                inview_hm, _, xview_hm = self.cam_fusion_module.forward(hms, affine_trans, cam_Intri, cam_R, cam_T, inv_affine_trans,
                                                           bones_tensor)

        extra = dict()

        extra['joint_channel_mask'] = self.joint_channel_mask
        extra['origin_hms'] = hms
        b_imu_fuse = False
        if self.b_xview_fusion:
            extra['fused_hms'] = xview_hm
            b_imu_fuse = True
        if self.b_in_view_fusion:
            extra['fused_hms'] = inview_hm
            b_imu_fuse = True
        extra['imu_fuse'] = b_imu_fuse

        if b_imu_fuse:
            imu_joint_mask = extra['joint_channel_mask'][0]
            non_imu_joint_mask = -0.5 * imu_joint_mask + 1.0
            output_hms = (extra['fused_hms'] * 0.5) + (extra['origin_hms'] * non_imu_joint_mask)
        else:
            output_hms = extra['origin_hms']

        return output_hms, extra

    def merge_first_two_dims(self, tensor):
        dim0 = tensor.shape[0]
        dim1 = tensor.shape[1]
        left = tensor.shape[2:]
        return tensor.view(dim0 * dim1, *left)


def get_multiview_pose_net(resnet, CFG):
    model = MultiViewPose(resnet, CFG)
    return model
