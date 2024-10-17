import os.path

import torch.utils.data as data
import numpy as np
import h5py
import glob
import tqdm


class TeethPatchDataset(data.Dataset):
    def __init__(self, data_dir, train=True):
        super().__init__()
        self.data_list = data_dir
        self.train = train

        all_data, all_labels, all_indices, all_filenames, all_centroids = [], [], [], [], []
        data_files = glob.glob(os.path.join(data_dir, 'train' if train else 'test') + '.*.h5')

        for data_file in tqdm.tqdm(data_files):
            num = int(data_file.split('.')[-2])
            if num > 0:
                continue
            h5file = h5py.File(data_file, 'r')
            all_data.append(h5file['patches'][:])
            all_labels.append(h5file['labels'][:])
            #all_indices.append(h5file['indices'][:])
            #all_filenames.append(h5file['filename'][:])
            all_centroids.append(h5file['centroids'][:])
            h5file.close()

        all_data = np.concatenate(all_data, axis=0)
        all_data = all_data[:, :, 0:7]
        self.all_data = np.zeros((all_data.shape[0], all_data.shape[1], 6 + 1 + 1))
        self.all_labels = np.concatenate(all_labels, axis=0)
        all_labels = None
        #self.all_indices = np.concatenate(all_indices, axis=0)
        #all_indices = None
        #self.all_filenames = np.concatenate(all_filenames, axis=0)
        #all_filenames = None

        self.all_centroids = np.expand_dims(np.concatenate(all_centroids, axis=0), 1)
        all_centroids = None
        print("Data loaded")

        # Normalize
        self.c = np.mean(all_data[:, :, 0:3], axis=1, keepdims=True)
        self.m = np.max(np.sqrt(np.sum((all_data[:, :, 0:3] - self.c) ** 2, axis=2)), axis=1)
        self.m = np.expand_dims(self.m, (1, 2))
        all_data[:, :, 0:3] = (all_data[:, :, 0:3] - self.c) / self.m
        self.all_centroids[:, :, 0:3] = (self.all_centroids[:, :, 0:3] - self.c) / self.m
        self.c = None
        self.m = None
        print("Normalize complete")

        centroid_pointer_tensor = all_data[:, :, 0:3] - self.all_centroids

        dist_heatmap = np.exp(-2 * np.sum((all_data[:, :, 0:3] - self.all_centroids) ** 2, axis=2))
        dist_heatmap = np.expand_dims(dist_heatmap, 2)
        print("Dist heatmap complete")

        #self.all_data = np.concatenate((self.all_data, dist_heatmap, centroid_pointer_tensor), axis=2)
        self.all_data[:, :, 0:7] = all_data
        self.all_data[:, :, 7:8] = dist_heatmap
        #self.all_data[:, :, 7:] = centroid_pointer_tensor


        print('Load complete', self.all_data.shape)

    def __getitem__(self, index):
        return self.all_data[index], \
               self.all_labels[index], self.all_centroids[index]
               #self.all_indices[index], \
               #self.all_filenames[index], \
               #self.all_centroids[index]

    def __len__(self):
        return len(self.all_data)