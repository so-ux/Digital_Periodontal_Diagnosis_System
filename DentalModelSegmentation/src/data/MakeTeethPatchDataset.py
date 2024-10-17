"""
对牙齿模型根据标签进行Patch拆分

Stage 2
"""
import os

import numpy as np
import torch
from sklearn.neighbors import KDTree
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.cluster import DBSCAN

from pointnet2_utils import furthest_point_sample
from src.data.TeethTrainingDataset import TeethTrainingDataset
from src.models import AllToothSegNet

import h5py


def chamfer_distance(p1, p2):
    """
    Calculate Chamfer Distance between two point sets

    :param p1: size[N, D]
    :param p2: size[M, D]
    :param debug: Whether you need to output debug info
    :return: Sum of Chamfer Distance of two point sets
    """
    p1 = np.expand_dims(p1, 0)
    p2 = np.expand_dims(p2, 0)

    p1 = p1.repeat(p2.shape[1], 0)      # [M, N, D]
    p1 = p1.transpose(1, 0, 2)             # [N, M, D]
    p2 = p2.repeat(p1.shape[0], 0)
    dist = p1 - p2
    dist = np.min(np.linalg.norm(dist, axis=2), axis=1)
    indices = np.argsort(dist)
    return indices


def chamfer_distance_gpu(p1, p2):
    """
    Calculate Chamfer Distance between two point sets

    :param p1: size[N, D]
    :param p2: size[M, D]
    """
    p1 = np.expand_dims(p1, 0)
    p2 = np.expand_dims(p2, 0)

    p1 = torch.from_numpy(p1).to('cuda', non_blocking=True)
    p2 = torch.from_numpy(p2).to('cuda', non_blocking=True)
    p1 = p1.repeat(p2.size(1), 1, 1)

    p1 = p1.transpose(0, 1)

    p2 = p2.repeat(p1.size(0), 1, 1)

    dist = torch.add(p1, torch.neg(p2))

    dist, _ = torch.min(torch.norm(dist, 2, dim=2), dim=1)

    indices = torch.argsort(dist).detach().cpu().numpy()
    return indices


class MakeTeethPatchDataset:
    def __init__(self, data_dir, file_part):
        super().__init__()
        self.file_part = file_part
        self.data_dir = data_dir

        all_seg_model = AllToothSegNet(3)
        all_seg_model.load_state_dict(torch.load('/home/zsj/PycharmProjects/miccai-3d-teeth-seg/model/model_all_tooth_seg.aug.0725'))
        all_seg_model.cuda()
        all_seg_model.eval()

        all_data, all_labels, all_indices, all_seg, all_cls, all_centroids = [], [], [], [], [], []
        all_filenames = []

        dset = TeethTrainingDataset(os.path.join(data_dir, file_part), train=file_part.startswith('train'),
                                    require_dicts=True, remove_curvature=False)
        dataloader = DataLoader(
            dset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=12
        )
        for batch_data in tqdm(dataloader):
            points, labels, _, real_size = batch_data

            points_tensor = points.to(dtype=torch.float32, device='cuda')
            pred_labels = torch.argmax(all_seg_model(points_tensor[:, :, 0:6]), dim=1)
            pred_labels = pred_labels[0].detach().cpu().numpy()

            labels = labels[0].numpy()
            points = points[0].numpy()

            points = self.remove_curvatures_on_tooth(points, pred_labels)

            patch, label, indices, centroids = self.split_patches_one_model_by_gt(self, points, labels)

            # Split seg & cls into patches
            for patch_id in range(patch.shape[0]):
                all_data.append(patch[patch_id])
                all_labels.append(label[patch_id])
                # all_indices.append(indices[patch_id])
                # all_filenames.append('{}_{}'.format(dicts['id'][0], dicts['jaw'][0]))
                all_centroids.append(centroids[patch_id])

        all_data = np.stack(all_data)
        all_labels = np.stack(all_labels)
        # all_indices = np.stack(all_indices)
        # all_filenames = np.stack(all_filenames)
        all_centroids = np.stack(all_centroids)
        self.make_h5(all_data, all_labels, all_indices, all_filenames, all_centroids)

    # Deprecated
    @staticmethod
    def split_patches_one_model(points, labels, pred_cls, n_points=4096):
        """
        Split patches

        Split one model according to predicted class labels. Gather the nearest 4096 points

        :param n_points: Number of points in one patch, default [4096]
        :type n_points: int
        :param points: Points [N, 6] numpy array
        :type points: numpy.ndarray
        :param labels: Labels [N] numpy array
        :type labels: numpy.ndarray
        :param pred_cls: Predicted class labels [N]
        :type pred_cls: numpy.ndarray
        :return: Patches [P, 4096, 3], corresponding labels [P, 4096], indices [P, 4096]
        :rtype: (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)
        """
        p_points, p_labels, p_indices, p_centroids = [], [], [], []
        for tooth_id in np.unique(pred_cls):
            if tooth_id > 0.5:
                # print(pred_cls.shape, tooth_id, points[pred_cls == tooth_id, 0:3].shape, points.shape, np.unique(points, axis=0).shape)
                cls_points = points[pred_cls == tooth_id, 0:3]    # [N', 3]
                this_tooth_labels = labels[pred_cls == tooth_id]

                this_tooth_labels = this_tooth_labels[this_tooth_labels > 0]
                if this_tooth_labels.shape[0] == 0:
                    continue

                unique_this_tooth_labels, counts = np.unique(this_tooth_labels, return_counts=True)
                this_tooth_label = unique_this_tooth_labels[np.argmax(counts)]

                # Remove some small connected components
                db = DBSCAN(eps=0.01, min_samples=1)
                clustering = db.fit(cls_points)
                label_id = clustering.labels_
                unique_label_id, counts = np.unique(label_id, return_counts=True)


                for s_label_id in np.argwhere(counts > 50):
                    # max_label_id = np.argmax(counts)
                    max_label = unique_label_id[s_label_id]
                    s_cls_points = cls_points[label_id == max_label]

                    # Find the nearest 4096 points
                    sorted_indices = chamfer_distance_gpu(points[:, 0:3], s_cls_points)[:n_points]
                    p_indices.append(sorted_indices)
                    p_points.append(points[sorted_indices, :])

                    nn_labels = labels[sorted_indices]
                    nn_labels[nn_labels != this_tooth_label] = 0
                    p_labels.append(nn_labels)

                    # Extract a centroid according to labels
                    p_centroids.append(np.mean(s_cls_points, axis=0))

        return np.array(p_points), np.array(p_labels), np.array(p_indices), np.array(p_centroids)

    @staticmethod
    def split_patches_one_model_by_gt(self, points, labels, n_points=4096):
        def fps(xyz, n=3):
            results = np.zeros((n,), dtype=np.int32)
            distance = np.ones((xyz.shape[0], )) * 1e10
            farthest = 0
            for i in range(n):
                results[i] = farthest
                centroid = xyz[farthest, :]
                dist = np.sum((xyz - centroid) ** 2, -1)
                mask = dist < distance
                distance[mask] = dist[mask]
                farthest = np.argmax(distance)
            return results

        kdtree = KDTree(points[:, 0:3])
        p_points, p_labels, p_indices, p_centroids = [], [], [], []
        for tooth_id in np.unique(labels):
            if tooth_id > 0.5:
                labels_mask = np.zeros_like(labels)
                labels_mask[labels == tooth_id] = 1
                cls_points = points[labels == tooth_id, 0:3]    # [N', 3]
                if cls_points.shape[0] > 3500:
                    print(tooth_id, cls_points.shape)
                    continue

                # Find edge
                nn_indices = kdtree.query(cls_points, 20, return_distance=False)
                edge_indices = np.sum(labels_mask[nn_indices], axis=1) < 20
                labels_mask_nn = labels_mask[labels == tooth_id]
                labels_mask_nn[edge_indices] = 2
                labels_mask[labels == tooth_id] = labels_mask_nn

                # Find the nearest 4096 points
                sorted_indices = chamfer_distance_gpu(points[:, 0:3], cls_points)[:n_points]

                # 为质心添加偏移
                centroids = [np.mean(cls_points, axis=0)]
                fps_3_indices = fps(cls_points, n=2)[1:]
                for fps_index in fps_3_indices:
                    new_centroid = centroids[0] + (cls_points[fps_index] - centroids[0]) * 0.8
                    centroids.append(new_centroid)

                for c in centroids:
                    p_indices.append(sorted_indices)
                    p_points.append(points[sorted_indices, :])

                    nn_labels = labels_mask[sorted_indices]
                    # nn_labels[nn_labels != tooth_id] = 0
                    p_labels.append(nn_labels)

                    # Extract a centroid according to labels
                    p_centroids.append(c)

                # 构造偏移数据
                # 否则目标牙齿均处于Patch正中央，dist_heatmap特征会被弱化
                # 网络会更倾向于分割Patch中心位置的牙齿
                for id_offset in [-1, 1]:
                    p, l, c, idx = self.get_neighbour_label_points(points, labels, labels_mask, tooth_id, id_offset)
                    if p is None:
                        continue
                    p_points.append(p)
                    p_labels.append(l)
                    p_centroids.append(c)
                    p_indices.append(idx)

        return np.array(p_points), np.array(p_labels), np.array(p_indices), np.array(p_centroids)

    @staticmethod
    def get_neighbour_label_points(points, labels, labels_mask, cur_tooth_id, id_offset=1, n_points=4096):
        """
        通过当前牙齿的邻居牙齿构造偏移数据

        通过邻居牙齿质心与当前牙齿质心的线性组合，得到新的裁剪中心，并裁剪n_points个点作为新的Patch，打上当前牙齿的对应标签
        """
        # 计算编号
        if cur_tooth_id == 0:
            return None, None, None, None
        jaw = cur_tooth_id // 10
        neighbour_tooth_id = cur_tooth_id % 10 + id_offset
        if neighbour_tooth_id == 0:
            # 当前牙齿ID为1，加上了-1变为0，变换分区
            if jaw == 1:
                jaw = 2
            elif jaw == 2:
                jaw = 1
            elif jaw == 3:
                jaw = 4
            elif jaw == 4:
                jaw = 3
            neighbour_tooth_id = 1
        neighbour_tooth_id = jaw * 10 + neighbour_tooth_id

        # 判断有无编号为neighbour_tooth_id的牙齿存在
        neighbour_tooth_mask = labels == neighbour_tooth_id
        if np.sum(neighbour_tooth_mask) == 0:
            return None, None, None, None

        # 获取牙齿质心
        neighbour_tooth_centroid = np.mean(points[neighbour_tooth_mask, 0:3], axis=0)
        current_tooth_centroid = np.mean(points[labels == cur_tooth_id, 0:3], axis=0)

        # 与当前牙齿质心加权得到裁剪中心
        crop_centroid = (current_tooth_centroid - neighbour_tooth_centroid) * 0.2 + neighbour_tooth_centroid
        crop_centroid = np.expand_dims(crop_centroid, 0)

        # Find the nearest 4096 points
        sorted_indices = chamfer_distance_gpu(points[:, 0:3], crop_centroid)[:n_points]
        nn_labels = labels_mask[sorted_indices]
        # nn_labels[nn_labels != cur_tooth_id] = 0
        return points[sorted_indices, :], nn_labels, current_tooth_centroid, sorted_indices

    def make_h5(self, patches, labels, indices, dicts, centroids):
        # str_dtype = h5py.special_dtype(vlen=str)
        h5 = h5py.File('/run/media/zsj/DATA/Data/miccai/h5_patches/{}.h5'
                       .format(os.path.basename(self.file_part).replace('.list', '')), 'w')
        h5['patches'] = patches
        h5['labels'] = labels
        # h5['indices'] = indices
        # filename_ds = h5.create_dataset('filename', dicts.shape, dtype=str_dtype)
        # filename_ds[:] = dicts
        h5['centroids'] = centroids
        h5.close()

        # Vis
        # color = np.ones((4096, 4))
        # for i in range(len(patches[0])):
        #     color[i, 0:3] = patches[0, i, 6] * np.array([1, 0, 0]) + (1 - patches[0, i, 6]) * np.array([0.5, 0.5, 0.5])
        # np.savetxt('/run/media/zsj/DATA/Data/miccai/h5_patches/demo_{}.txt'.format(os.path.basename(self.file_part).replace('.list', '')), np.concatenate([patches[0, :, 0:6], color], axis=-1))

    def remove_curvatures_on_tooth(self, all_data, labels):
        """
        移除牙窝和牙齿表面的曲率信息

        :return:
        :rtype:
        """
        pred_seg_int = np.array(labels)
        pred_seg_int[pred_seg_int > 0.5] = 1
        pred_seg_int[pred_seg_int < 1] = 0
        pred_seg_int = np.asarray(pred_seg_int, dtype=np.int32)

        nd_data = all_data[:, :]
        # k = 32
        # 缩小一圈pred_seg
        kdtree = KDTree(nd_data[:, 0:3])
        indices = kdtree.query(nd_data[pred_seg_int > 0, 0:3], 10, return_distance=False)
        neighbour_zero_count = np.sum(pred_seg_int[indices], axis=1) - 1
        indices = np.zeros(labels.shape, dtype=np.int32)
        indices[pred_seg_int > 0] = neighbour_zero_count
        # indices = kdtree.query(nd_data[pred_seg_int > 0, 0:3], 10, return_distance=False)
        # neighbour_zero_count = np.sum(pred_seg_int[indices], axis=1) - 1
        # indices = np.zeros(labels.shape, dtype=np.int32)
        # indices[pred_seg_int > 0] = neighbour_zero_count

        # indices = kdtree.query(nd_data[:, 0:3], k, return_distance=False)
        # neighbour_pred_count = np.sum(pred_seg_int[indices], axis=1).astype(np.float32)
        # neighbour_pred_count /= k
        # # indices = ((neighbour_pred_count < 0.1) + (neighbour_pred_count > 0.9)) > 0
        # indices = neighbour_pred_count < 0.1

        all_data[indices > 0, 6] = 0
        # all_data[:, 6] /= (np.max(all_data[:, 6], axis=-1) + 1e-9)
        #
        # # Vis
        # color = np.ones((len(all_data), 4))
        # color_seg = np.ones((len(all_data), 4))
        # for i in range(len(all_data)):
        #     color[i, 0:3] = all_data[i, 6] * np.array([1, 0, 0])
        #     color_seg[i, 0:3] = np.array([1, 1, 1]) * pred_seg_int[i]
        #
        # np.savetxt('/run/media/zsj/DATA/Data/miccai/h5_patches/demo_seg_{}.txt'.format(os.path.basename(self.file_part).replace('.list', '')), np.concatenate([all_data[:, 0:6], color_seg], axis=-1))
        # np.savetxt('/run/media/zsj/DATA/Data/miccai/h5_patches/demo_{}.txt'.format(os.path.basename(self.file_part).replace('.list', '')), np.concatenate([all_data[:, 0:6], color], axis=-1))
        # exit(0)

        return all_data


# Make patch dataset in h5 format
if __name__ == '__main__':
    import glob
    file_list = glob.glob('/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/h5_patches/*.*.list')

    for train in file_list:
        print(train)
        MakeTeethPatchDataset(
            '/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/h5_patches',
            train
        )

    # for i in range(23):
    #     MakeTeethPatchDataset(
    #         '/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/h5_patches',
    #         f'train.{i}.list'
    #     )
