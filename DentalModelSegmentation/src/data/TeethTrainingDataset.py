import math

import torch
import torch.utils.data as data
import numpy as np
import h5py
import tqdm
import random
import open3d as o3d
from sklearn.neighbors import KDTree

random.seed(20000228)


class TeethTrainingDataset(data.Dataset):
    """
    Dataset for training AllToothSegNet
    """
    def __init__(self, data_list, train=True, require_dicts=False, remove_curvature=False):
        super().__init__()
        self.require_dicts = require_dicts
        self.data_list = data_list
        self.train = train

        data_list_file = open(self.data_list, 'r')
        all_files = [line.strip() for line in data_list_file.readlines() if line.strip() != '']
        data_list_file.close()

        self.all_data, self.all_labels, self.all_centroids, self.all_dicts, self.all_real_sizes = [], [], [], [], []
        self.all_gd = []

        for file in tqdm.tqdm(all_files[:1200]):
            dic = h5py.File(file, 'r')
            if not dic.keys().__contains__('flat_indices'):
                continue
            origin_data = dic['data'][:]
            labels = dic['labels'][:]

            # 1. Remove braces
            id_selected = dic['id_selected'][:]
            origin_data = origin_data[id_selected]
            labels = labels[id_selected]

            # 2. Normalize
            m = float(np.asanyarray(dic['m']))
            origin_data[:, 0:3] = (origin_data[:, 0:3] - np.array(dic['c'])) / dic['m']

            # 3. FPS
            fps_indices = dic['fps_indices'][:]

            self.all_real_sizes.append(fps_indices.shape[0])
            # if fps_indices.shape[0] < 2048 * 16:
            #     repeat_times = (2048 * 16 // fps_indices.shape[0]) + 1
            #     fps_indices = fps_indices.repeat(repeat_times)[0:2048 * 16]

            origin_data = origin_data[fps_indices]

            # 4. Add flat indices
            flat_indices = np.expand_dims(dic['flat_indices'][:], 1)
            origin_data = np.concatenate((origin_data, flat_indices[fps_indices]), axis=1)

            labels = labels[fps_indices]

            #####################################################################################
            # 扣掉牙窝里的曲率信息
            #####################################################################################

            if remove_curvature:
                pred_seg_int = np.array(labels)
                pred_seg_int[pred_seg_int > 0.5] = 1
                pred_seg_int[pred_seg_int < 1] = 0
                pred_seg_int = np.asarray(pred_seg_int, dtype=np.int32)

                # 缩小一圈pred_seg
                kdtree = KDTree(origin_data[:, 0:3])
                indices = kdtree.query(origin_data[pred_seg_int > 0, 0:3], 10, return_distance=False)
                neighbour_zero_count = np.sum(pred_seg_int[indices], axis=1) - 1
                indices = np.zeros(labels.shape, dtype=np.int32)
                indices[pred_seg_int > 0] = neighbour_zero_count

                origin_data[indices > 5, 6] = 0
                # origin_data[:, 6] /= (np.max(origin_data[:, 6], axis=-1) + 1e-9)
            #####################################################################################

            # Compute centroids
            centroids = np.zeros((len(origin_data), 3))
            for tooth_id in np.unique(labels):
                if tooth_id > 0:
                    centroids[labels == tooth_id] = np.mean(origin_data[labels == tooth_id, 0:3], axis=0)
                else:
                    # Due to normalization, coordinate smaller than -1 is illegal,
                    # so -100 is chosen to represent non-tooth vertex
                    centroids[labels == tooth_id] = np.array([-100, -100, -100])
            self.all_centroids.append(centroids)
            self.all_data.append(origin_data)
            self.all_labels.append(labels)
            # self.all_gd.append(dic['gd'][:][fps_indices])
            self.all_dicts.append({
                # 'id': str(dic['id'].asstr()[...]),
                # 'jaw': str(dic['jaw'].asstr()[...]),
                'c': np.array(dic['c']),
                'm': m,
                'id_selected': id_selected,
                'fps_indices': fps_indices
            })

            # self.data_augmentation_rotation(origin_data, labels, dic, m, id_selected, fps_indices)
            # self.data_augmentation_scale_gauss_rotation(origin_data, labels, dic, m, id_selected, fps_indices)

    def data_augmentation_rotation(self, origin_data, labels, dic, m, id_selected, fps_indices):
        # Data augmentation
        origin_data_aug = np.copy(origin_data)
        mesh = o3d.geometry.PointCloud()
        mesh.points = o3d.utility.Vector3dVector(origin_data_aug[:, 0:3])
        mesh.normals = o3d.utility.Vector3dVector(origin_data_aug[:, 3:6])
        angle_x = math.pi / 3 * (random.random() - 0.5)
        angle_y = math.pi / 3 * (random.random() - 0.5)
        angle_z = math.pi / 3 * (random.random() - 0.5)
        mesh.rotate(mesh.get_rotation_matrix_from_xyz((angle_x, angle_y, angle_z)))
        origin_data_aug[:, 0:6] = np.concatenate((np.asarray(mesh.points), np.asarray(mesh.normals)), axis=-1)

        centroids_aug = np.zeros((len(origin_data_aug), 3))
        for tooth_id in np.unique(labels):
            if tooth_id > 0:
                centroids_aug[labels == tooth_id] = np.mean(origin_data_aug[labels == tooth_id, 0:3], axis=0)
            else:
                # Due to normalization, coordinate smaller than -1 is illegal,
                # so -100 is chosen to represent non-tooth vertex
                centroids_aug[labels == tooth_id] = np.array([-100, -100, -100])
        self.all_centroids.append(centroids_aug)
        self.all_data.append(origin_data_aug)
        self.all_labels.append(labels)
        self.all_dicts.append({
            'c': np.array(dic['c']),
            'm': m,
            'id_selected': id_selected,
            'fps_indices': fps_indices
        })
        self.all_real_sizes.append(fps_indices.shape[0])

    def data_augmentation_scale_gauss_rotation(self, origin_data, labels, dic, m, id_selected, fps_indices):
        # Data augmentation 2
        origin_data_aug = np.copy(origin_data)
        origin_data_aug[:, 0:3] += np.random.normal(0.001, 0.005, origin_data_aug[:, 0:3].shape)
        origin_data_aug[:, 0:3] *= (0.3 * (random.random() - 0.5))
        mesh = o3d.geometry.PointCloud()
        mesh.points = o3d.utility.Vector3dVector(origin_data_aug[:, 0:3])
        mesh.normals = o3d.utility.Vector3dVector(origin_data_aug[:, 3:6])
        angle_x = math.pi / 9 * (random.random() - 0.5)
        angle_y = math.pi / 9 * (random.random() - 0.5)
        angle_z = math.pi / 9 * (random.random() - 0.5)
        mesh.rotate(mesh.get_rotation_matrix_from_xyz((angle_x, angle_y, angle_z)))

        centroids_aug = np.zeros((len(origin_data_aug), 3))
        for tooth_id in np.unique(labels):
            if tooth_id > 0:
                centroids_aug[labels == tooth_id] = np.mean(origin_data_aug[labels == tooth_id, 0:3], axis=0)
            else:
                # Due to normalization, coordinate smaller than -1 is illegal,
                # so -100 is chosen to represent non-tooth vertex
                centroids_aug[labels == tooth_id] = np.array([-100, -100, -100])
        self.all_centroids.append(centroids_aug)
        origin_data_aug[:, 0:6] = np.concatenate((np.asarray(mesh.points), np.asarray(mesh.normals)), axis=-1)
        self.all_data.append(origin_data_aug)
        self.all_labels.append(labels)
        self.all_dicts.append({
            'c': np.array(dic['c']),
            'm': m,
            'id_selected': id_selected,
            'fps_indices': fps_indices
        })
        self.all_real_sizes.append(fps_indices.shape[0])

    def __getitem__(self, index):
        if self.require_dicts:
            return self.all_data[index], \
                   self.all_labels[index], \
                   self.all_centroids[index], self.all_real_sizes[index]#, self.all_gd[index]
        else:
            return self.all_data[index], \
                   self.all_labels[index], \
                   self.all_centroids[index], self.all_real_sizes[index]

    def __len__(self):
        return len(self.all_data)
