# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np


class HumanBody(object):

    def __init__(self):
        self.skeleton = self.get_skeleton()
        self.skeleton_sorted_by_level = self.sort_skeleton_by_level(
            self.skeleton)
        self.imu_edges = self.get_imu_edges()
        self.imu_edges_reverse = self.get_imu_edges_reverse()

    def get_skeleton(self):
        # joint_names = [
        #     'root', 'rhip', 'rkne', 'rank', 'lhip', 'lkne', 'lank', 'belly',
        #     'neck', 'nose', 'lsho', 'lelb', 'lwri', 'rsho', 'relb',  # use nose here instead of head
        #     'rwri'
        # ]
        # children = [[1, 4, 7], [2], [3], [], [5], [6], [], [8], [9, 10, 13],
        #             [], [11], [12], [], [14], [15], []]
        # imubone = [[-1, -1, -1], [3], [4], [], [5], [6], [], [-1], [-1, -1, -1],
        #             [], [11], [12], [], [13], [14], []]
        joint_names = [
            'root', 'rhip', 'rkne', 'rank', 'lhip', 'lkne', 'lank', 'belly',
            'neck', 'nose', 'head', 'lsho', 'lelb', 'lwri', 'rsho', 'relb',
            'rwri'
        ]
        children = [[1, 4, 7], [2], [3], [], [5], [6], [], [8], [9, 11, 14],
                    [10], [], [12], [13], [], [15], [16], []]
        imubone = [[-1, -1, -1], [3], [4], [], [5], [6], [], [-1], [-1, -1, -1],
                   [-1], [], [12], [13], [], [14], [15], []]

        skeleton = []
        for i in range(len(joint_names)):
            skeleton.append({
                'idx': i,
                'name': joint_names[i],
                'children': children[i],
                'imubone': imubone[i]
            })
        return skeleton

    def sort_skeleton_by_level(self, skeleton):
        njoints = len(skeleton)
        level = np.zeros(njoints)

        queue = [skeleton[0]]
        while queue:
            cur = queue[0]
            for child in cur['children']:
                skeleton[child]['parent'] = cur['idx']
                level[child] = level[cur['idx']] + 1
                queue.append(skeleton[child])
            del queue[0]

        desc_order = np.argsort(level)[::-1]
        sorted_skeleton = []
        for i in desc_order:
            skeleton[i]['level'] = level[i]
            sorted_skeleton.append(skeleton[i])
        return sorted_skeleton

    def get_imu_edges(self):
        imu_edges = dict()
        for joint in self.skeleton:
            for idx_child, child in enumerate(joint['children']):
                if joint['imubone'][idx_child] >= 0:
                    one_edge_name = (joint['idx'], child)
                    bone_idx = joint['imubone'][idx_child]
                    imu_edges[one_edge_name] = bone_idx
        return imu_edges

    def get_imu_edges_reverse(self):
        imu_edges = self.imu_edges
        imu_edges_reverse = {imu_edges[k]:k for k in imu_edges}
        return imu_edges_reverse


if __name__ == '__main__':
    hb = HumanBody()
    print(hb.skeleton)
    print(hb.skeleton_sorted_by_level)
