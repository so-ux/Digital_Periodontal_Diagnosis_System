import os

import numpy as np
import torch
from sklearn.neighbors import KDTree
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data.TeethTrainingDataset import TeethTrainingDataset
from src.pointnet2.pointnet2_utils import furthest_point_sample
from src.utils.pc_normalize import pc_normalize

import h5py

sample_size = 4096


class MakeTeethClassDataset:
    """
    Make teeth classification training dataset

    {
      data: [B, N, 6],
      labels: [B, T],
      seg: [B, T, M],
      filenames: [B, ]
    }
    """

    def __init__(self, data_dir, file_part):
        super().__init__()
        self.file_part = file_part
        self.data_dir = data_dir

        all_data, all_labels, all_seg, all_sizes = [], [], [], []
        all_filenames = []
        all_resamples = []

        dset = TeethTrainingDataset(os.path.join(data_dir, file_part), train=file_part.startswith('train'),
                                    require_dicts=True)
        dataloader = DataLoader(
            dset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=12
        )

        for batch_data in tqdm(dataloader):
            points, labels, _, real_size = batch_data

            size = 0

            points = points[0]
            labels = labels[0]
            dicts = []

            data_tensor = points.unsqueeze(0).to('cuda', dtype=torch.float32, non_blocking=True)

            fps_indices = furthest_point_sample(data_tensor, 16384).detach().cpu().numpy()[0]

            labels[labels > 0] = (torch.div(labels[labels > 0], 10, rounding_mode='floor') - 1) * 8 + labels[
                labels > 0] % 10

            labels = labels.numpy()
            points = points.numpy()
            points = points[fps_indices]
            labels = labels[fps_indices]

            kdtree = KDTree(points[:, 0:3])

            data_labels = np.ones((25, )) * -1
            data_seg = np.ones((25, sample_size)) * -1
            data_resample = np.zeros((25, sample_size, 7))

            # 添加Mask
            try:
                for label_id in np.unique(labels):
                    if label_id > 0:
                        masked_labels = np.array(labels)
                        masked_labels[masked_labels != label_id] = 0
                        masked_labels[masked_labels > 0] = 1

                        data_seg_indices = np.squeeze(np.argwhere(masked_labels == 1))
                        data_labels[size] = label_id
                        data_seg[size, :data_seg_indices.shape[0]] = data_seg_indices

                        centroid = np.mean(points[data_seg_indices][:, 0:3], axis=0)

                        nn_idx = kdtree.query([centroid], sample_size, return_distance=False)[0]
                        nn_mask = np.zeros((sample_size, ), dtype=np.int32)
                        nn_mask[masked_labels[nn_idx] > 0] = 1
                        data_resample[size, :, 0:6] = points[nn_idx, 0:6]
                        data_resample[size, :, 6] = nn_mask
                        data_resample[size, :, 0:3], _, _ = pc_normalize(data_resample[size, :, 0:3])

                        size += 1

                # Add negative sample
                indices = self.crop_negative_samples(points, labels)[0]
                if indices is not None:
                    data_labels[size] = 0
                    # data_seg[size, :indices.shape[0]] = indices

                    centroid = np.mean(points[indices[indices > -1]][:, 0:3], axis=0)

                    nn_idx = kdtree.query([centroid], sample_size, return_distance=False)[0]
                    data_seg[size, :nn_idx.shape[0]] = nn_idx
                    data_resample[size, :, 0:6] = points[nn_idx, 0:6]
                    data_resample[size, :, 6] = 1
                    data_resample[size, labels[nn_idx] > 0.5, 6] = 0
                    data_resample[size, :, 0:3], _, _ = pc_normalize(data_resample[size, :, 0:3])

                    size += 1

                # Add fore-gum negative sample
                indices = self.crop_fore_gum_negative_samples(points, labels)
                if indices is not None:
                    for index in indices:
                        data_labels[size] = 0
                        # data_seg[size, :index.shape[0]] = index

                        centroid = np.mean(points[index[index > -1]][:, 0:3], axis=0)

                        nn_idx = kdtree.query([centroid], sample_size, return_distance=False)[0]
                        data_seg[size, :nn_idx.shape[0]] = nn_idx
                        data_resample[size, :, 0:6] = points[nn_idx, 0:6]
                        data_resample[size, :, 6] = 1
                        data_resample[size, labels[nn_idx] > 0.5, 6] = 0
                        data_resample[size, :, 0:3], _, _ = pc_normalize(data_resample[size, :, 0:3])

                        size += 1

                all_data.append(points)
                # all_filenames.append('{}_{}'.format(dicts['id'][0], dicts['jaw'][0]))
                all_filenames.append('')
                all_labels.append(data_labels)
                all_seg.append(data_seg)
                all_sizes.append(size)
                all_resamples.append(data_resample)
            except:
                continue

        all_data = np.stack(all_data)
        all_labels = np.array(all_labels)
        all_seg = np.array(all_seg)
        all_filenames = np.stack(all_filenames)
        all_sizes = np.stack(all_sizes)
        all_resamples = np.stack(all_resamples)
        self.make_h5(all_data, all_labels, all_seg, all_filenames, all_sizes, all_resamples)

    def make_h5(self, patches, labels, segs, dicts, sizes, resamples):
        # str_dtype = h5py.special_dtype(vlen=str)
        h5 = h5py.File('/run/media/zsj/DATA/Data/miccai/h5_cls/{}.h5'
                       .format(os.path.basename(self.file_part).replace('.list', '')), 'w')
        h5['patches'] = patches
        h5['labels'] = labels
        h5['segs'] = segs

        # filename_ds = h5.create_dataset('filename', dicts.shape, dtype=str_dtype)
        # filename_ds[:] = dicts
        h5['sizes'] = sizes
        h5['resamples'] = resamples
        h5.close()

    def crop_negative_samples(self, points, labels):
        # 获取球内的点
        def crop_ball(ver, center, nor, r=0.3):
            ind = np.squeeze(np.argwhere(np.sum((center - ver[:, 0:3]) ** 2, axis=1) <= r))
            inner_pointcloud = ver[ind, :]
            inner_pointcloud_normals = nor[ind, :]
            outer_pointcloud = ver[np.sum((center - ver[:, 0:3]) ** 2, axis=1) > r, :]
            return ind, inner_pointcloud, outer_pointcloud, inner_pointcloud_normals

        center_of_pointcloud = np.mean(points[:, 0:3], axis=0)

        crop_ball_ind, after_crop_pointcloud, outer_pointcloud, after_crop_pointcloud_normals = \
            crop_ball(points, center_of_pointcloud, points[:, 3:6])

        if crop_ball_ind.shape[0] == 0:
            return None, None, None, None

        length, _ = after_crop_pointcloud.shape
        random_index = np.random.randint(0, length - 1)

        xyz_of_centre_point = after_crop_pointcloud[random_index, 0:3]

        dis_np = np.sum((after_crop_pointcloud[:, 0:3] - xyz_of_centre_point) ** 2, axis=1)
        knn_index = np.argsort(dis_np)[0:sample_size]

        # Remove teeth
        for i in range(knn_index.shape[0]):
            if labels[crop_ball_ind[knn_index[i]]] > 0:
                knn_index[i] = -1
        knn_index = knn_index[knn_index > -1]

        knn_cloudpoint = after_crop_pointcloud[knn_index, :]
        knn_cloudpoint_normals = after_crop_pointcloud_normals[knn_index, :]

        return crop_ball_ind[knn_index], knn_cloudpoint, outer_pointcloud, knn_cloudpoint_normals

    def crop_fore_gum_negative_samples(self, points, labels):
        center_of_pointcloud = np.mean(points[:, 0:3], axis=0)

        points_with_centroid = np.concatenate((points[:, 0:3], [center_of_pointcloud]), axis=0)

        def farthest_point_sample(xyz, npoint):
            """
            Input:
                xyz: pointcloud data, [B, N, C]
                npoint: number of samples
            Return:
                centroids: sampled pointcloud index, [B, npoint]
            """
            if isinstance(xyz, list):
                xyz = np.array(xyz)
            xyz = np.reshape(xyz, (1, xyz.shape[0], xyz.shape[1]))
            xyz = torch.from_numpy(xyz.astype(np.float32)).cuda()

            B, N, C = xyz.shape
            centroids = torch.zeros([B, npoint], dtype=torch.long).cuda()
            distance = torch.ones(B, N).cuda() * 1e10
            farthest = torch.randint(N - 1, N, (B,), dtype=torch.long).cuda()
            batch_indices = torch.arange(B, dtype=torch.long).cuda()
            for j in range(npoint):
                centroids[:, j] = farthest
                centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
                dist = torch.sum((xyz - centroid) ** 2, -1)
                mask = dist < distance
                distance[mask] = dist[mask]
                farthest = torch.max(distance, -1)[1]
            return centroids.cpu().numpy()

        indices = farthest_point_sample(points_with_centroid, 6)[0, 1:]
        results = []
        for center in indices:
            knn = np.argsort(np.sum((points[:, 0:3] - points[center, 0:3]) ** 2, axis=1))[:sample_size]
            for i in range(knn.shape[0]):
                if labels[knn[i]] > 0:
                    knn[i] = -1
            # Remove teeth
            results.append(knn)
        return results


# Make patch dataset in h5 format
if __name__ == '__main__':
    import glob

    file_list = glob.glob('/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/h5_cls/*.list')

    for train in file_list:
        MakeTeethClassDataset(
            '/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/h5',
            train
        )
