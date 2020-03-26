# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np

import multiviews.cameras_cuda_col as cameras_utils


def totalcapture_collate(data):
    batch = len(data)
    nview = len(data[0][0])
    if data[0][3][0]['source'] == 'h36m':
        selected_bones = [3, 4, 5, 6, 12, 13, 14, 15]
    else:  # totalcapture
        selected_bones = [3, 4, 5, 6, 11, 12, 13, 14]  # todo maybe write to dataset definition
    h_hm = data[0][1][0].shape[1]
    w_hm = data[0][1][0].shape[2]

    inputs = []
    targets = []
    weights = []
    metas = []
    for b in range(batch):
        inputs.extend(data[b][0])
        targets.extend(data[b][1])
        weights.extend(data[b][2])
        metas.extend(data[b][3])

    inputs = torch.stack(inputs, dim=0)  # (batch*nview, 3, h_im, w_im)
    targets = torch.stack(targets, dim=0)
    weights = torch.stack(weights, dim=0)

    nitems = batch * nview
    # deal with metas
    centers = []
    scales = []
    joints_vis = []
    joints_2d_transformed = []
    cam_intri = []
    cam_R = []
    cam_T = []
    affine_trans = []
    inv_affine_trans = []
    bone_vec_tensors = []

    for bv in range(nitems):
        m = metas[bv]
        centers.append(m['center'])
        scales.append(m['scale'])
        joints_vis.append(m['joints_vis'])
        joints_2d_transformed.append(m['joints_2d_transformed'])

        # deal with camera
        intri = torch.eye(3,3)
        cam = m['camera']
        intri[0, 0] = float(cam['fx'])
        intri[1, 1] = float(cam['fy'])
        intri[0, 2] = float(cam['cx'])
        intri[1, 2] = float(cam['cy'])
        cam_intri.append(intri)
        cam_R.append(cam['R'])
        cam_T.append(cam['T'])

        # affine transform between origin img and heatmap
        aff_tran_3x3 = torch.eye(3, dtype=torch.float32)
        aff_tran = cameras_utils.get_affine_transform(m['center'], m['scale'], patch_size=(h_hm, w_hm), inv=0)
        aff_tran_3x3[0:2] = torch.as_tensor(aff_tran, dtype=torch.float32)
        inv_aff_tran_3x3 = torch.eye(3, dtype=torch.float32)
        inv_aff_tran = cameras_utils.get_affine_transform(m['center'], m['scale'], patch_size=(h_hm, w_hm), inv=1)
        inv_aff_tran_3x3[0:2] = torch.as_tensor(inv_aff_tran, dtype=torch.float32)
        affine_trans.append(aff_tran_3x3)
        inv_affine_trans.append(inv_aff_tran_3x3)

        # bone vectors
        bone_vec_tensor = get_bone_vector_tensor(m['bone_vectors'], selected_bones)
        bone_vec_tensors.append(bone_vec_tensor)

    metas_collate = dict()
    metas_collate['center'] = torch.as_tensor(np.array(centers))
    metas_collate['scale'] = torch.as_tensor(np.array(scales))
    metas_collate['joints_vis'] = torch.as_tensor(np.array(joints_vis))
    metas_collate['joints_2d_transformed'] = torch.as_tensor(np.array(joints_2d_transformed))
    metas_collate['camera_Intri'] = torch.stack(cam_intri, dim=0)
    metas_collate['camera_R'] = torch.as_tensor(np.array(cam_R))
    metas_collate['camera_T'] = torch.as_tensor(np.array(cam_T))
    metas_collate['affine_trans'] = torch.stack(affine_trans, dim=0)
    metas_collate['inv_affine_trans'] = torch.stack(inv_affine_trans, dim=0)
    metas_collate['bone_vectors_tensor'] = torch.stack(bone_vec_tensors, dim=0)

    # reshape to (batch, nview, ...)
    inputs = inputs.view(batch, nview, *inputs.shape[1:])
    targets = targets.view(batch, nview, *targets.shape[1:])
    weights = weights.view(batch, nview, *weights.shape[1:])

    metas_collate_batch_nview = dict()
    for kk in metas_collate:
        metas_collate_batch_nview[kk] = metas_collate[kk].view(batch, nview, *metas_collate[kk].shape[1:])

    return inputs, targets, weights, metas_collate_batch_nview


def get_bone_vector_tensor(bone_vec, selected_bones):
    """
    sort bone_vec in the order of selected_bones
    :param bone_vec: dict {bone_idx: (3,1)}
    :param selected_bones: list or tuple
    :return: (3, 1, 2*nbones)
    """
    bone_vecs = torch.as_tensor(list(map(bone_vec.get, selected_bones)))
    bone_vecs_2x = torch.stack((-bone_vecs, bone_vecs), dim=1).view(-1, 1, 3)
    bone_vecs_out = bone_vecs_2x.permute(2,1,0).contiguous()
    return bone_vecs_out


def dict_to_dev(d, dev):
    for k in d:
        d[k] = d[k].to(dev)
    return d


def ndlist_to_dev(l, dev):
    # get list dimension, support 1d, 2d list
    list_dim = 0
    if isinstance(l, (list, tuple)):
        list_dim = 1
        if isinstance(l[0], (list, tuple)):
            list_dim = 2
    if list_dim == 0 and isinstance(l, torch.FloatTensor):
        return l.to(dev)
    if list_dim == 1:
        out_list = []
        for ele in l:
            if isinstance(ele, dict):
                out_list.append(dict_to_dev(ele, dev))
            else:
                out_list.append(ele.to(dev))
        return out_list
    if list_dim == 2:
        out_list = []
        for subl in l:
            subl_list = []
            for ele in subl:
                if isinstance(ele, dict):
                    subl_list.append(dict_to_dev(ele, dev))
                else:
                    subl_list.append(ele.to(dev))
            out_list.append(subl_list)
        return out_list
