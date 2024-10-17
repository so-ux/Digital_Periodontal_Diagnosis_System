import os

import torch.utils.data as data
import numpy as np
import h5py


class TeethClassDataset(data.Dataset):
    """
    Dataset for training ClsNet
    """
    def __init__(self, data_dir, train=True, require_dicts=False):
        super().__init__()
        self.require_dicts = require_dicts
        self.data_dir = data_dir
        self.train = train

        data_file = os.path.join(data_dir, 'train.h5' if train else 'test.h5')
        h5file = h5py.File(data_file)

        self.all_data = h5file['patches'][:]
        self.all_data = np.delete(self.all_data, 6, axis=2)
        self.all_labels = h5file['labels'][:]
        # self.all_labels_dense = h5file['labels_dense'][:]
        # self.all_filenames = h5file['filename'][:]
        self.all_segs = h5file['segs'][:]
        self.all_sizes = h5file['sizes'][:]
        self.all_resamples = h5file['resamples'][:]
        self.patches_length = np.sum(self.all_sizes)

        # 将Label还原至FDI
        # all_labels[all_labels > 0] = ((all_labels[all_labels > 0] - 1) // 8 + 1) * 10 \
        #                              + ((all_labels[all_labels > 0] - 1) % 8 + 1)
        # self.all_labels = all_labels

        # Normalize
        # centroids = np.mean(self.all_resamples[:, :, 0:3], axis=-1, keepdims=True)
        # max_dis = np.sqrt(np.sum((self.all_resamples[:, :, 0:3] - centroids) ** 2, axis=-1, keepdims=True))
        # self.all_resamples[:, :, 0:3] = (self.all_resamples[:, :, 0:3] - centroids) / (max_dis + 1e-9)

        self.patch_indices = []
        for i in range(self.all_sizes.shape[0]):
            size = self.all_sizes[i]
            patch_indices = np.arange(size)
            self.patch_indices.append(patch_indices)

        # 建立有序数组，存储data[i]的结束数据index
        # dataloader获取元素时，通过二分法可以在O(logn)的时间内寻找到对应的数据
        self.sorted_data_len = np.zeros((self.all_data.shape[0], ))
        self.sorted_data_len[0] = self.all_sizes[0] - 1
        for i in range(1, self.all_data.shape[0]):
            self.sorted_data_len[i] = \
                self.sorted_data_len[i - 1] + self.all_sizes[i]

    def find_real_data_index(self, index, l, r):
        if l == r:
            patch_index = index if l == 0 else index - (self.sorted_data_len[l - 1] + 1)
            return l, int(patch_index)

        mid = (l + r) // 2
        if index < self.sorted_data_len[mid]:
            return self.find_real_data_index(index, l, mid)
        elif index > self.sorted_data_len[mid]:
            return self.find_real_data_index(index, mid + 1, r)
        else:
            return mid, int(index) if mid == 0 else int(index - (self.sorted_data_len[mid - 1] + 1))

    def __getitem__(self, index):
        data_index, patch_index = self.find_real_data_index(index, 0, self.all_data.shape[0] - 1)
        points = self.all_data[data_index]

        # Add mask to data
        mask = self.all_segs[data_index, patch_index]
        mask = np.asarray(mask[mask > -1], dtype=np.int32)

        data = np.zeros((points.shape[0], points.shape[1] + 1))
        data[:, :points.shape[1]] = points
        data[mask, points.shape[1]] = 1

        labels_dense = 0
        # labels_dense = self.all_labels_dense[data_index]

        # if self.require_dicts:
        #     return data, self.all_labels[data_index, patch_index], self.all_filenames[data_index]
        # else:
        return data, self.all_labels[data_index, patch_index], self.all_resamples[data_index, patch_index], labels_dense

    def __len__(self):
        return self.patches_length
