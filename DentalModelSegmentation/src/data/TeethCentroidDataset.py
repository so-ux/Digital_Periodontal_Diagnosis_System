import os.path

import torch.utils.data as data
import numpy as np
import h5py


class TeethCentroidDataset(data.Dataset):
    def __init__(self, data_dir, train=True):
        super().__init__()
        self.data_list = data_dir
        self.train = train

        data_file = os.path.join(data_dir, 'train.h5' if train else 'test.h5')
        h5file = h5py.File(data_file)

        self.all_data = h5file['data'][:]
        self.all_filenames = h5file['filename'].asstr()[:]

        self.all_centroids = h5file['centroids'][:]

        # Normalize
        # self.c = np.mean(self.all_data[:, :, 0:3], axis=1, keepdims=True)
        # self.m = np.max(np.sqrt(np.sum((self.all_data[:, :, 0:3] - self.c) ** 2, axis=2)), axis=1)
        # self.m = np.expand_dims(self.m, (1, 2))
        # self.all_data[:, :, 0:3] = (self.all_data[:, :, 0:3] - self.c) / self.m
        # self.all_centroids[:, :, 0:3] = (self.all_centroids[:, :, 0:3] - self.c) / self.m

    def __getitem__(self, index):
        return self.all_data[index], \
               self.all_centroids[index], \
               self.all_filenames[index]

    def __len__(self):
        return len(self.all_data)

