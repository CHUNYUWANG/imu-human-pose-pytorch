# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Written by Zhe Zhang (v-zhaz@microsoft.com)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools


tv1, tv2, _ = torch.__version__.split('.')
tv = int(tv1) * 10 + int(tv2) * 1
if tv >= 13:  # api change since 1.3.0 for grid_sample
    grid_sample = functools.partial(F.grid_sample, align_corners=True)
else:
    grid_sample = F.grid_sample


def gen_hm_grid_coords(h, w, dev=None):
    """

    :param h:
    :param w:
    :param dev:
    :return: (3, h*w) each col is (u, v, 1)^T
    """
    if not dev:
        dev = torch.device('cpu')
    h = int(h)
    w = int(w)
    h_s = torch.linspace(0, h - 1, h).to(dev)
    w_s = torch.linspace(0, w - 1, w).to(dev)
    hm_cords = torch.meshgrid(h_s, w_s)
    flat_cords = torch.stack(hm_cords, dim=0).view(2, -1)
    out_grid = torch.ones(3, h*w, device=dev)
    out_grid[0] = flat_cords[1]
    out_grid[1] = flat_cords[0]
    return out_grid


def batch_uv_to_global_with_multi_depth(uv1, inv_affine_t, inv_cam_intrimat, inv_cam_extri_R, inv_cam_extri_T,
                                        depths, nbones):
    """

    :param uv1:
    :param inv_affine_t: hm -> uv
    :param inv_cam_intrimat: uv -> norm image frame
    :param inv_cam_extri_R: transpose of cam_extri_R
    :param inv_cam_extri_T: same as cam_extri_T
    :param depths:
    :param nbones:
    :return:
    """
    dev = uv1.device
    nview_batch = inv_affine_t.shape[0]
    h = int(torch.max(uv1[1]).item()) + 1
    w = int(torch.max(uv1[0]).item()) + 1
    depths = torch.as_tensor(depths, device=dev).view(-1,1,1)
    ndepth = depths.shape[0]

    # uv1 copy
    coords_hm_frame = uv1.view(1, 3, h*w, 1).expand(nview_batch, -1, -1, nbones*2).contiguous().view(nview_batch, 3, -1)

    # uv to image frame
    inv_cam_intrimat = inv_cam_intrimat.view(nview_batch, 3, 3)
    inv_affine_t = inv_affine_t.view(nview_batch, 3, 3)
    synth_trans = torch.bmm(inv_cam_intrimat, inv_affine_t)
    coords_img_frame = torch.bmm(synth_trans, coords_hm_frame)

    # image frame to 100 depth cam frame
    coords_img_frame = coords_img_frame.permute(1,0,2).contiguous().view(1, 3,-1)  # (1, 3, nview*batch * h*w * 2nbones)
    coords_img_frame_all_depth = coords_img_frame.expand(ndepth, -1, -1)
    coords_img_frame_all_depth = torch.mul(coords_img_frame_all_depth, depths)  # (ndepth, 3, nview*batch *h*w *2nbones)

    # cam frame to global frame
    coords_img_frame_all_depth = coords_img_frame_all_depth.view(ndepth, 3, nview_batch, -1).permute(2, 1, 0, 3)\
        .contiguous().view(nview_batch, 3, -1)
    inv_cam_extri_R = inv_cam_extri_R.view(-1, 3, 3)
    inv_cam_extri_T = inv_cam_extri_T.view(-1, 3, 1)
    coords_global = torch.bmm(inv_cam_extri_R, coords_img_frame_all_depth) + inv_cam_extri_T
    coords_global = coords_global.view(nview_batch, 3, ndepth, h*w, 2*nbones)
    return coords_global


def apply_bone_offset(coords_global, bone_vectors):
    """

    :param coords_global: (nview*batch, 3, ndepth, h*w, 2*nbones)
    :param bone_vectors: (nview*batch, 3, 1, 2*nbones)
    :return:
    """
    nview_batch, coords_dim, ndepth, hw, nbones2 = coords_global.shape
    coords_global = coords_global.view(nview_batch, coords_dim, ndepth*hw, nbones2)
    res = coords_global + bone_vectors
    return res.view(nview_batch, coords_dim, ndepth, hw, nbones2)


def batch_global_to_uv(coords_global, affine_t, cam_intrimat, cam_extri_R, cam_extri_T):
    nview_batch, coords_dim, ndepth, hw, nbones2 = coords_global.shape
    coords_global_flat = coords_global.view(nview_batch, coords_dim, -1)

    cam_extri_R = cam_extri_R.view(-1, 3, 3)
    cam_extri_T = cam_extri_T.view(-1, 3, 1)
    coords_cam = torch.bmm(cam_extri_R, coords_global_flat - cam_extri_T)

    # divide z to obtain norm image frame coords
    coords_cam_norm = coords_cam / coords_cam[:, 2:3]

    cam_intrimat = cam_intrimat.view(nview_batch, 3, 3)
    affine_t = affine_t.view(nview_batch, 3, 3)
    synth_trans = torch.bmm(affine_t, cam_intrimat)
    coords_uv_hm = torch.bmm(synth_trans, coords_cam_norm)
    return coords_uv_hm.view(nview_batch, 3, ndepth, hw, nbones2)


def get_inv_cam(intri_mat, extri_R, extri_T):
    """
    all should be in (nview*batch, x, x)
    :param intri_mat:
    :param extri_R:
    :param extri_T:
    :return:
    """
    # camera_to_world  torch.mm(torch.t(R), x) + T
    # world_to_camera  torch.mm(R, x - T)
    # be aware of that: extri T is be add and minus in cam->world and reverse
    return torch.inverse(intri_mat), extri_R.permute(0,2,1).contiguous(), extri_T


def get_inv_affine_transform(affine_t):
    """

    :param affine_t: (3x3) mat instead of 2x3 mat. shape of (nview*batch, 3, 3)
    :return:
    """
    return torch.inverse(affine_t)


def get_bones_vector_tensor(bone_vectors, selected_bones, nview=4):
    """

    :param bone_vectors: list of len batch,  of dict {bone_idx: (3,1)}
    :return:
    """
    bone_vec_tensors = []
    for bv in bone_vectors:
        bvs = torch.stack(list(map(bv.get, selected_bones)), dim=0)
        bone_vec_tensors.append(torch.stack((-bvs, bvs), dim=1).view(2*len(selected_bones), 3, 1).permute(1,2,0))
    bone_vec_out = torch.stack(bone_vec_tensors, dim=0)
    bone_vec_out = bone_vec_out.unsqueeze(dim=0).expand(nview, *bone_vec_out.shape)\
        .contiguous().view(-1, *bone_vec_out.shape[1:])  # copy nview times
    # (nview*batch, 2*nbones, 3, 1)
    return bone_vec_out


def get_bone_vector_meta(selected_bones, body, dataset_joint_mapping):
    reverse_joint_mapping = dict()  # totalcapture: general
    for k in dataset_joint_mapping.keys():
        v = dataset_joint_mapping[k]
        if v != '*':
            reverse_joint_mapping[v] = k

    bone_vec_meta_out_ref = []
    bone_vec_meta_out_cur = []
    bone_joint_map = body.get_imu_edges_reverse()
    for bone in selected_bones:
        par, child = bone_joint_map[bone]
        bone_vec_meta_out_ref.append(reverse_joint_mapping[par])
        bone_vec_meta_out_ref.append(reverse_joint_mapping[child])
        bone_vec_meta_out_cur.append(reverse_joint_mapping[child])
        bone_vec_meta_out_cur.append(reverse_joint_mapping[par])

    return bone_vec_meta_out_cur, bone_vec_meta_out_ref


class CamFusionModule(nn.Module):
    def __init__(self, nview, njoint, h, w, body, depth, joint_hm_mapping, selected_bones, config):
        super().__init__()
        self.nview = nview
        # self.batch = batch
        self.njoint = njoint  # njoint in heatmap, normally 20
        self.h = h  # h of heatmap
        self.w = w  # w of heatmap
        # self.dev = dev  # computing on what device
        self.selected_bones = selected_bones  # idx of body's joint
        self.body = body  # HumanBody

        self.nbones = len(self.selected_bones)
        self.depth = depth
        self.ndepth = depth.shape[0]
        self.joint_hm_mapping = joint_hm_mapping
        self.bone_vectors_meta = get_bone_vector_meta(selected_bones, body, joint_hm_mapping)
        #  (cur joint, ref joint)

        self.config = config
        self.b_inview_fusion = config.CAM_FUSION.IN_VIEW_FUSION
        self.b_xview_self_fusion = config.CAM_FUSION.XVIEW_SELF_FUSION
        self.b_xview_fusion = config.CAM_FUSION.XVIEW_FUSION

        self.onehm = gen_hm_grid_coords(h, w)
        self.grid_norm_factor = (torch.tensor([h - 1, w - 1, njoint - 1], dtype=torch.float32) / 2.0)
        self.imu_bone_norm_factor = torch.ones(20, 1, 1)
        if config.DATASET.TRAIN_DATASET == 'totalcapture':
            self.imu_bone_norm_factor[2,0,0] = 2.0
            self.imu_bone_norm_factor[5, 0, 0] = 2.0
            self.imu_bone_norm_factor[15, 0, 0] = 2.0
            self.imu_bone_norm_factor[18, 0, 0] = 2.0
        elif config.DATASET.TRAIN_DATASET == 'multiview_h36m':
            self.imu_bone_norm_factor[2, 0, 0] = 2.0
            self.imu_bone_norm_factor[5, 0, 0] = 2.0
            self.imu_bone_norm_factor[15, 0, 0] = 2.0
            self.imu_bone_norm_factor[18, 0, 0] = 2.0

    def forward(self, heatmaps, affine_trans, cam_Intri, cam_R, cam_T, inv_affine_trans, bone_vectors_tensor):
        dev = heatmaps.device
        batch = heatmaps.shape[0] // self.nview
        self.grid_norm_factor = self.grid_norm_factor.to(dev)
        self.onehm = self.onehm.to(dev)

        cam_Intri = cam_Intri.to(dev)
        cam_R = cam_R.to(dev)
        cam_T = cam_T.to(dev)
        affine_trans = affine_trans.to(dev)
        inv_affine_trans = inv_affine_trans.to(dev)
        bone_vectors_tensor = bone_vectors_tensor.to(dev)
        imu_bone_norm_factor = self.imu_bone_norm_factor.to(dev)

        inv_cam_Intri, inv_cam_R, inv_cam_T = get_inv_cam(cam_Intri, cam_R, cam_T)
        out_grid_g_all_joints = batch_uv_to_global_with_multi_depth(self.onehm, inv_affine_trans, inv_cam_Intri, inv_cam_R,
                                                         inv_cam_T,
                                                         self.depth, self.njoint // 2)

        heatmaps_5d = heatmaps.view(self.nview * batch, 1, self.njoint, self.h, self.w)
        inview_fused = None
        xview_self_fused = None
        xview_fused = None
        if self.b_inview_fusion:
            out_grid_g = out_grid_g_all_joints[:,:,:,:,:16]
            ref_coords = apply_bone_offset(out_grid_g, bone_vectors_tensor)

            # project to hm
            ref_coords_hm = batch_global_to_uv(ref_coords, affine_trans, cam_Intri, cam_R, cam_T)

            ref_bone_meta = torch.as_tensor(self.bone_vectors_meta[1])  # todo
            ref_bone_meta_3 = torch.zeros(len(ref_bone_meta), 3)
            ref_bone_meta_3[:, 2] = ref_bone_meta
            ref_bone_meta_3 = ref_bone_meta_3.to(dev)
            ref_bone_meta = ref_bone_meta_3.view(1, self.nbones*2, 1, 1, 3)
            ref_bone_meta_expand = ref_bone_meta.expand(self.nview*batch, self.nbones*2, self.ndepth, self.h*self.w, 3)

            ref_coords_hm = ref_coords_hm.permute(0, 4, 2, 3, 1).contiguous()
            ref_coords_hm[:,:,:,:,2] = 0.0
            ref_coords_hm = ref_coords_hm + ref_bone_meta_expand

            # normalize ref_corords_hm to [-1,1]
            ref_coords_flow = ref_coords_hm / self.grid_norm_factor - 1.0
            sampled_hm = grid_sample(input=heatmaps_5d, grid=ref_coords_flow, mode='nearest')

            sum_sampled_hm_over_depth = torch.max(sampled_hm, dim=3)[0].view(self.nview * batch * self.nbones * 2, self.h, self.w)
            fusion_hm = torch.zeros(self.nview * batch, self.njoint, self.h, self.w, device=dev)

            # inview fusion
            idx_view_batch = torch.linspace(0, self.nview * batch - 1, self.nview * batch).type(torch.long).to(dev)
            idx_dst_bones = torch.tensor(self.bone_vectors_meta[0], dtype=torch.long).to(dev)
            idx_put_grid_h, idx_put_grid_w = torch.meshgrid(idx_view_batch, idx_dst_bones)
            idx_put_grid_flat = (idx_put_grid_h.contiguous().view(-1), idx_put_grid_w.contiguous().view(-1))
            fusion_hm.index_put_(idx_put_grid_flat, sum_sampled_hm_over_depth, accumulate=True)
            inview_fused = fusion_hm / imu_bone_norm_factor

        if self.b_xview_self_fusion:
            assert 1 == 0, 'Not implemented !'

        if self.b_xview_fusion:
            out_grid_g = out_grid_g_all_joints[:, :, :, :, :16]
            ref_coords = apply_bone_offset(out_grid_g, bone_vectors_tensor)  # in global frame

            ref_bone_meta = torch.as_tensor(self.bone_vectors_meta[1])  # todo
            ref_bone_meta_3 = torch.zeros(len(ref_bone_meta), 3)
            ref_bone_meta_3[:, 2] = ref_bone_meta
            ref_bone_meta_3 = ref_bone_meta_3.to(dev)
            ref_bone_meta = ref_bone_meta_3.view(1, self.nbones * 2, 1, 1, 3)
            ref_bone_meta_expand = ref_bone_meta.expand(self.nview * batch, self.nbones * 2, self.ndepth,
                                                        self.h * self.w, 3)

            # re-org affine and cam parameters for 4 views
            affine_trans_bv = affine_trans.view(batch, self.nview, *affine_trans.shape[1:])
            cam_Intri_bv = cam_Intri.view(batch, self.nview, *cam_Intri.shape[1:])
            cam_R_bv = cam_R.view(batch, self.nview, *cam_R.shape[1:])
            cam_T_bv = cam_T.view(batch, self.nview, *cam_T.shape[1:])

            xview_hm_sumed = torch.zeros(batch*self.nview, self.njoint, self.h, self.w, device=dev)
            for offset in range(self.nview):  # offset means how many views to roll/shift
                affine_trans_shift = self.roll_on_dim1(affine_trans_bv, offset=offset)
                cam_Intri_shift = self.roll_on_dim1(cam_Intri_bv, offset=offset)
                cam_R_shift = self.roll_on_dim1(cam_R_bv, offset=offset)
                cam_T_shift = self.roll_on_dim1(cam_T_bv, offset=offset)
                ref_coords_hm = batch_global_to_uv(ref_coords, affine_trans_shift,
                                                   cam_Intri_shift, cam_R_shift, cam_T_shift)

                heatmaps_5d_shift = heatmaps.view(batch, self.nview, 1, self.njoint, self.h, self.w)
                heatmaps_5d_shift = self.roll_on_dim1(heatmaps_5d_shift, offset=offset)
                heatmaps_5d_shift = heatmaps_5d_shift.view(batch * self.nview, 1, self.njoint, self.h, self.w)

                ref_coords_hm = ref_coords_hm.permute(0, 4, 2, 3, 1).contiguous()
                ref_coords_hm[:, :, :, :, 2] = 0.0
                ref_coords_hm = ref_coords_hm + ref_bone_meta_expand

                ref_coords_flow = ref_coords_hm / self.grid_norm_factor - 1.0
                sampled_hm = grid_sample(input=heatmaps_5d_shift, grid=ref_coords_flow, mode='nearest')
                sum_sampled_hm_over_depth = torch.max(sampled_hm, dim=3)[0]. \
                    view(batch*self.nview*self.nbones * 2, self.h, self.w)
                fusion_hm = torch.zeros(batch, self.nview, self.njoint, self.h, self.w, device=dev)

                # fusion
                idx_batch = torch.linspace(0, batch - 1, batch).type(torch.long).to(dev)
                idx_view = (torch.linspace(0, self.nview - 1, self.nview).type(torch.long).to(dev) + offset) % self.nview
                idx_dst_bones = torch.tensor(self.bone_vectors_meta[0], dtype=torch.long).to(dev)
                idx_put_grid_b, idx_put_grid_v, idx_put_grid_j = torch.meshgrid(idx_batch, idx_view, idx_dst_bones)
                idx_put_grid_flat = (idx_put_grid_b.contiguous().view(-1), idx_put_grid_v.contiguous().view(-1), idx_put_grid_j.contiguous().view(-1))
                fusion_hm.index_put_(idx_put_grid_flat, sum_sampled_hm_over_depth, accumulate=True)

                fusion_hm = torch.zeros(batch*self.nview, self.njoint, self.h, self.w, device=dev)
                idx_view_batch = torch.linspace(0, self.nview * batch - 1, self.nview * batch).type(torch.long).to(dev)
                idx_dst_bones = torch.tensor(self.bone_vectors_meta[0], dtype=torch.long).to(dev)
                idx_put_grid_h, idx_put_grid_w = torch.meshgrid(idx_view_batch, idx_dst_bones)
                idx_put_grid_flat = (idx_put_grid_h.contiguous().view(-1), idx_put_grid_w.contiguous().view(-1))
                fusion_hm.index_put_(idx_put_grid_flat, sum_sampled_hm_over_depth, accumulate=True)
                fusion_hm = fusion_hm/imu_bone_norm_factor

                xview_hm_sumed += fusion_hm
            xview_fused = xview_hm_sumed / self.nview

        return inview_fused, xview_self_fused, xview_fused

    def roll_on_dim1(self, tensor, offset, maxoffset=None):
        if maxoffset is None:
            maxoffset = self.nview
        offset = offset % maxoffset

        part1 = tensor[:, :offset]
        part2 = tensor[:, offset:]
        res = torch.cat((part2, part1), dim=1).contiguous()
        return res
