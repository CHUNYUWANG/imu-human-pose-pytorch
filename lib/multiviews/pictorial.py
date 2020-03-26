# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import multiviews.cameras as cameras
from utils.transforms import get_affine_transform, affine_transform, affine_transform_pts


tv1, tv2, _ = torch.__version__.split('.')
tv = int(tv1) * 10 + int(tv2) * 1
if tv >= 13:  # api change since 1.3.0 for grid_sample
    grid_sample = functools.partial(F.grid_sample, align_corners=True)
else:
    grid_sample = F.grid_sample


def infer(unary, pairwise, body, config, **kwargs):
    """
    Args:
        unary: a list of unary terms for all JOINTS
        pairwise: a list of pairwise terms of all EDGES
        body: tree structure human body
    Returns:
        pose3d_as_cube_idx: 3d pose as cube index
    """
    # current_device = torch.device('cuda:{}'.format(pairwise.items()[0].get_device()))
    current_device = kwargs['current_device']
    skeleton = body.skeleton
    skeleton_sorted_by_level = body.skeleton_sorted_by_level
    root_idx = config.DATASET.ROOTIDX
    nbins = len(unary[root_idx])
    states_of_all_joints = {}
    # print('dev {} id unary: {}'.format(current_device, id(unary)))

    # zhe 20190104 replace torch with np
    for node in skeleton_sorted_by_level:
        # energy = []
        children_state = []
        unary_current = unary[node['idx']]
        # unary_current = torch.tensor(unary_current, dtype=torch.float32).to(current_device)
        if len(node['children']) == 0:
            energy = unary[node['idx']].squeeze()
            children_state = [[-1]] * len(energy)
        else:
            children = node['children']
            for child in children:
                child_energy = states_of_all_joints[child][
                    'Energy'].squeeze()
                pairwise_mat = pairwise[(node['idx'], child)]
                # if type(pairwise_mat) == scipy.sparse.csr.csr_matrix:
                #     pairwise_mat = pairwise_mat.toarray()
                # unary_child = child_energy
                unary_child = torch.tensor(child_energy, dtype=torch.float32).to(current_device).expand_as(pairwise_mat)
                # unary_child_with_pairwise = np.multiply(pairwise_mat, unary_child)
                # unary_child_with_pairwise = ne.evaluate('pairwise_mat*unary_child')
                unary_child_with_pairwise = torch.mul(pairwise_mat, unary_child)
                # max_i = np.argmax(unary_child_with_pairwise, axis=1)
                # max_v = np.max(unary_child_with_pairwise, axis=1)
                max_v, max_i = torch.max(unary_child_with_pairwise, dim=1)
                unary_current = torch.mul(unary_current, max_v)

                # unary_current = np.multiply(unary_current, max_v)
                # children_state.append(max_i)
                children_state.append(max_i.detach().cpu().numpy())

            # rearrange children_state
            children_state = np.array(children_state).T  # .tolist()

        res = {'Energy': unary_current.detach().cpu().numpy(), 'State': children_state}
        states_of_all_joints[node['idx']] = res
    # end here 20181225

    pose3d_as_cube_idx = []
    energy = states_of_all_joints[root_idx]['Energy']
    cube_idx = np.argmax(energy)
    pose3d_as_cube_idx.append([root_idx, cube_idx])

    queue = pose3d_as_cube_idx.copy()
    while queue:
        joint_idx, cube_idx = queue.pop(0)
        children_state = states_of_all_joints[joint_idx]['State']
        state = children_state[cube_idx]

        children_index = skeleton[joint_idx]['children']
        if -1 not in state:
            for joint_idx, cube_idx in zip(children_index, state):
                pose3d_as_cube_idx.append([joint_idx, cube_idx])
                queue.append([joint_idx, cube_idx])

    pose3d_as_cube_idx.sort()
    return pose3d_as_cube_idx


def get_loc_from_cube_idx(grid, pose3d_as_cube_idx):
    """
    Estimate 3d joint locations from cube index.

    Args:
        grid: a list of grids 
        pose3d_as_cube_idx: a list of tuples (joint_idx, cube_idx)
    Returns:
        pose3d: 3d pose 
    """
    njoints = len(pose3d_as_cube_idx)
    pose3d = np.zeros(shape=[njoints, 3])
    is_single_grid = len(grid) == 1
    for joint_idx, cube_idx in pose3d_as_cube_idx:
        gridid = 0 if is_single_grid else joint_idx
        pose3d[joint_idx] = grid[gridid][cube_idx]
    return pose3d


def compute_grid(boxSize, boxCenter, nBins):
    grid1D = np.linspace(-boxSize / 2, boxSize / 2, nBins)
    gridx, gridy, gridz = np.meshgrid(
        grid1D + boxCenter[0],
        grid1D + boxCenter[1],
        grid1D + boxCenter[2],
    )
    dimensions = gridx.shape[0] * gridx.shape[1] * gridx.shape[2]
    gridx, gridy, gridz = np.reshape(gridx, (dimensions, -1)), np.reshape(
        gridy, (dimensions, -1)), np.reshape(gridz, (dimensions, -1))
    grid = np.concatenate((gridx, gridy, gridz), axis=1)
    return grid


def compute_pairwise_constrain(skeleton, limb_length, grid, tolerance, **kwargs):
    do_bone_vectors = False
    if 'do_bone_vectors' in kwargs:
        if kwargs['do_bone_vectors']:
            do_bone_vectors = True
            bone_vectors = kwargs['bone_vectors']

    pairwise_constrain = {}
    for node in skeleton:
        current = node['idx']
        children = node['children']
        if do_bone_vectors:
            bone_index = node['imubone']

        for idx_child, child in enumerate(children):
            expect_length = limb_length[(current, child)]
            if do_bone_vectors:
                if bone_index[idx_child] >= 0:  # if certain bone has imu
                    expect_orient_vector = bone_vectors[bone_index[idx_child]]
                    norm_expect_orient_vector = expect_orient_vector / (np.linalg.norm(expect_orient_vector)+1e-9)
            nbin_current = len(grid[current])
            nbin_child = len(grid[child])
            constrain_array = np.zeros((nbin_current, nbin_child), dtype=np.float32)

            for i in range(nbin_current):
                for j in range(nbin_child):
                    actual_length = np.linalg.norm(grid[current][i] -
                                                   grid[child][j]) + 1e-9
                    offset = np.abs(actual_length - expect_length)
                    if offset <= tolerance:
                        constrain_array[i, j] = 1

                    if do_bone_vectors and bone_index[idx_child] >= 0:
                        acutal_orient_vector = (grid[current][i] - grid[child][j]) / actual_length
                        cos_theta = np.dot(-norm_expect_orient_vector, acutal_orient_vector)
                        # notice norm_expect_orient_vector is child - parent
                        # while acutal_orient_vector is parent - child
                        constrain_array[i, j] *= cos_theta

            pairwise_constrain[(current, child)] = constrain_array

    return pairwise_constrain


def compute_unary_term(heatmap, grid, bbox2D, cam, imgSize, **kwargs):
    """
    Args:
        heatmap: array of size (n * k * h * w)
                -n: number of views,  -k: number of joints
                -h: heatmap height,   -w: heatmap width
        grid: list of k ndarrays of size (nbins * 3)
                    -k: number of joints; 1 when the grid is shared in PSM
                    -nbins: number of bins in the grid
        bbox2D: bounding box on which heatmap is computed
    Returns:
        unary_of_all_joints: a list of ndarray of size nbins
    """

    n, k = heatmap.shape[0], heatmap.shape[1]
    h, w = heatmap.shape[2], heatmap.shape[3]
    nbins = grid[0].shape[0]
    current_device = torch.device('cuda:{}'.format(heatmap.get_device()))

    # unary_of_all_joints = []
    # for j in range(k):
    #     unary = np.zeros(nbins, dtype=np.float32)
    #     for c in range(n):
    #
    #         grid_id = 0 if len(grid) == 1 else j
    #         xy = cameras.project_pose(grid[grid_id], cam[c])
    #         trans = get_affine_transform(bbox2D[c]['center'],
    #                                      bbox2D[c]['scale'], 0, imgSize)
    #
    #         xy = affine_transform_pts(xy, trans) * np.array([w, h]) / imgSize
    #         # for i in range(nbins):
    #         #     xy[i] = affine_transform(xy[i], trans) * np.array([w, h]) / imgSize
    #
    #         hmap = heatmap[c, j, :, :]
    #         point_x, point_y = np.arange(hmap.shape[0]), np.arange(
    #             hmap.shape[1])
    #         rgi = RegularGridInterpolator(
    #             points=[point_x, point_y],
    #             values=hmap.transpose(),
    #             bounds_error=False,
    #             fill_value=0)
    #         score = rgi(xy)
    #         unary = unary + np.reshape(score, newshape=unary.shape)
    #     unary_of_all_joints.append(unary)
    # return unary_of_all_joints

    # torch version
    # heatmaps = torch.tensor(heatmap, dtype=torch.float32)
    heatmaps = heatmap
    grid_cords = np.zeros([n, k, nbins, 2], dtype=np.float32)
    for c in range(n):
        for j in range(k):
            grid_id = 0 if len(grid) == 1 else j
            xy = cameras.project_pose(grid[grid_id], cam[c])
            trans = get_affine_transform(bbox2D[c]['center'],
                                         bbox2D[c]['scale'], 0, imgSize)
            xy = affine_transform_pts(xy, trans) * np.array([w, h]) / imgSize
            # xy of shape (4096,2)
            # xy is cord of certain view and certain joint
            if len(grid) == 1:  # psm 4096bins
                grid_cords[c, 0, :, :] = xy/np.array([h-1, w-1], dtype=np.float32) * 2.0 - 1.0
                for j in range(1, k):
                    grid_cords[c, j, :, :] = grid_cords[c, 0, :, :]
                break  # since all joints share same grid, no need computing for each joint, just copy it
            else:
                grid_cords[c, j, :, :] = xy/np.array([h-1, w-1], dtype=np.float32) * 2.0 - 1.0

    grid_cords_tensor = torch.as_tensor(grid_cords).to(current_device)
    unary_all_views_joints = grid_sample(heatmaps, grid_cords_tensor)
    # unary_all_views_joints -> shape(4,16,16,4096)
    unary_all_views = torch.zeros(n,k,nbins).to(current_device)
    for j in range(k):
        unary_all_views[:,j,:] = unary_all_views_joints[:, j, j, :]
    unary_tensor = torch.zeros(k, nbins).to(current_device)
    for una in unary_all_views:
        unary_tensor = torch.add(unary_tensor, una)

    return unary_tensor


def recursive_infer(initpose, cams, heatmaps, boxes, img_size, heatmap_size,
                    body, limb_length, grid_size, nbins, tolerance, config, **kwargs):
    current_device = kwargs['current_device']
    k = initpose.shape[0]
    grids = []
    for i in range(k):
        point = initpose[i]
        grid = compute_grid(grid_size, point, nbins)
        grids.append(grid)

    unary = compute_unary_term(heatmaps, grids, boxes, cams, img_size)

    skeleton = body.skeleton
    pairwise_constrain = compute_pairwise_constrain(skeleton, limb_length,
                                                    grids, tolerance, **kwargs)
    pairwise_tensor = dict()
    for edge in pairwise_constrain:
        # edge_pairwise = pairwise_constrain[edge].astype(np.int64)
        edge_pairwise = torch.as_tensor(pairwise_constrain[edge], dtype=torch.float32)
        pairwise_tensor[edge] = edge_pairwise.to(current_device)
    pairwise_constrain = pairwise_tensor

    kwargs_infer = kwargs
    pose3d_cube = infer(unary, pairwise_constrain, body, config, **kwargs_infer)
    pose3d = get_loc_from_cube_idx(grids, pose3d_cube)

    return pose3d


def rpsm(cams, heatmaps, boxes, grid_center, limb_length, pairwise_constraint,
         config, **kwargs):
    """
    Args:
        cams : camera parameters for each view
        heatmaps: 2d pose heatmaps (n, k, h, w)
        boxes: on which the heatmaps are computed; n dictionaries
        grid_center: 3d location of the root
        limb_length: template limb length
        pairwise_constrain: pre-computed pairwise terms (iteration 0 psm only)
    Returns:
        pose3d: 3d pose
    """
    image_size = config.NETWORK.IMAGE_SIZE
    heatmap_size = config.NETWORK.HEATMAP_SIZE
    first_nbins = config.PICT_STRUCT.FIRST_NBINS
    recur_nbins = config.PICT_STRUCT.RECUR_NBINS
    recur_depth = config.PICT_STRUCT.RECUR_DEPTH
    grid_size = config.PICT_STRUCT.GRID_SIZE
    tolerance = config.PICT_STRUCT.LIMB_LENGTH_TOLERANCE

    # Iteration 1: discretizing 3d space
    # body = HumanBody()
    # current_device = torch.device('cuda:{}'.format(pairwise_constraint.values()[0].get_device()))
    current_device = kwargs['current_device']
    body = kwargs['human_body']
    grid = compute_grid(grid_size, grid_center, first_nbins)

    heatmaps = torch.as_tensor(heatmaps, dtype=torch.float32).to(current_device)  # todo: do this in dataloader
    extra_kwargs = kwargs

    # PSM
    do_bone_vectors = False
    if 'do_bone_vectors' in kwargs:
        if kwargs['do_bone_vectors']:
            do_bone_vectors = True
            bone_vectors = kwargs['bone_vectors']
    if do_bone_vectors:
        # merge limb length pairwise and bone orientation/vector pairwise term
        orient_pairwise = kwargs['orient_pairwise']
        new_pairwise_constrain = {}
        for node in body.skeleton:
            current = node['idx']
            children = node['children']
            bone_index = node['imubone']

            for idx_child, child in enumerate(children):
                constrain_array = pairwise_constraint[(current, child)]
                if bone_index[idx_child] >= 0:  # if certain bone has imu
                    expect_orient_vector = bone_vectors[bone_index[idx_child]]
                    expect_orient_vector = torch.as_tensor(expect_orient_vector, dtype=torch.float32).to(current_device)
                    norm_expect_orient_vector = expect_orient_vector / (torch.norm(expect_orient_vector) + 1e-9)
                    norm_expect_orient_vector = norm_expect_orient_vector.view(-1)  # (3,)
                    acutal_orient_vector = orient_pairwise  # (4096, 4096, 3)
                    cos_theta = torch.matmul(acutal_orient_vector, -norm_expect_orient_vector)
                    # todo we can add cos_theta activation func here
                    # acutal_orient_vector refer to 2 bin direction
                    # norm_expect_orient_vector refer to groundtruth direction
                    constrain_array = torch.mul(constrain_array, cos_theta)

                new_pairwise_constrain[(current, child)] = constrain_array
        pairwise_constraint = new_pairwise_constrain
    unary = compute_unary_term(heatmaps, [grid], boxes, cams, image_size)
    pose3d_as_cube_idx = infer(unary, pairwise_constraint, body, config, **extra_kwargs)
    pose3d = get_loc_from_cube_idx([grid], pose3d_as_cube_idx)

    cur_grid_size = grid_size / first_nbins
    for i in range(recur_depth):
        pose3d = recursive_infer(pose3d, cams, heatmaps, boxes, image_size,
                                 heatmap_size, body, limb_length, cur_grid_size,
                                 recur_nbins, tolerance, config, **extra_kwargs)
        cur_grid_size = cur_grid_size / recur_nbins

    return pose3d


class RpsmFunc(nn.Module):
    def __init__(self, pairwise_constraint, human_body, **kwargs):
        super().__init__()
        # self.pairwise_constraint = pairwise_constraint
        # self.register_parameter('pairwise_constraint', pairwise_constraint)  # auto to dev when replicating

        # register pairwise constraint in buff
        self.current_device = None
        self.pairwise_constraint = dict()
        for idx, k in enumerate(pairwise_constraint):
            buff_name = 'pairwise_constraint_{}'.format(idx)
            self.register_buffer(buff_name, pairwise_constraint[k])
            self.pairwise_constraint[k] = self.__getattr__(buff_name)
        self.human_body = human_body

        self.do_bone_vectors = kwargs['do_bone_vectors']
        if self.do_bone_vectors:
            orient_pairwise = kwargs['orient_pairwise']
            self.register_buffer('orient_pairwise', orient_pairwise)

    def __call__(self, *args, **kwargs):
        if self.current_device is None:
            self.current_device = torch.device('cuda:{}'.format(list(self.pairwise_constraint.values())[0].get_device()))

        extra_kwargs = dict()
        extra_kwargs['human_body'] = self.human_body
        extra_kwargs['current_device'] = self.current_device
        if self.do_bone_vectors:
            extra_kwargs['orient_pairwise'] = self.orient_pairwise
            # do_bone_vectors has already been in kwargs
        return rpsm(pairwise_constraint=self.pairwise_constraint, **kwargs, **extra_kwargs)
