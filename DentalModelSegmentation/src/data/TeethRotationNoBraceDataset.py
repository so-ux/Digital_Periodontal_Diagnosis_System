import h5py
import torch.utils.data as data
import random
from torch.utils.data.dataset import T_co
import torch
from sklearn.cluster import DBSCAN
import numpy as np

from pointnet2_utils import furthest_point_sample

random.seed(20000228)


def vis(filename, data, axes):
    with open(filename, 'w') as fp:
        for p in data:
            fp.write(f'v {p[0]} {p[1]} {p[2]} 0.1 0.5 0.5\n')
        fp.write(f'v 0 0 0 0 1 0\n')
        for p in axes * 50:
            fp.write(f'v {p[0]} {p[1]} {p[2]} 1 0 0\n')
        for i in range(len(axes)):
            fp.write(f'l {len(data) + 1} {len(data) + i + 2}\n')
        fp.close()


def remove_braces(verts, min_samples=10):
    db = DBSCAN(eps=1, min_samples=min_samples).fit(verts[:, 0:3])
    indices, counts = np.unique(db.labels_, return_counts=True)
    which = np.argmax(counts)
    selected = np.argwhere(db.labels_ == indices[which]).squeeze()
    return selected


class TeethRotationDataset(data.Dataset):
    def __init__(self, train):
        super(TeethRotationDataset, self).__init__()
        h5_file = '/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/data_rot/h5/train.h5' \
            if train else '/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/data_rot/h5/test.h5'
        h5 = h5py.File(h5_file, 'r')
        all_data = h5['data'][:]
        all_labels = h5['labels'][:]
        self.all_axes = h5['axes'][:]

        self.all_data = []
        self.all_labels = []

        for i in range(len(all_data)):
            data_ = all_data[i]
            labels_ = all_labels[i]
            id_selected = remove_braces(data_)
            if len(id_selected) < 8000:
                continue
            data_ = data_[id_selected, :]
            labels_ = labels_[id_selected]


            data_tensor = torch.Tensor(np.array([data_[:, 0:3]])).cuda()

            # For each tooth cropping area, sample 2048 points
            # Assume that there are 16 teeth on each model
            fps_indices = furthest_point_sample(data_tensor, 8000).detach().cpu().numpy()[0]
            data_ = data_[fps_indices]
            c = np.mean(data_, axis=0)
            m = np.sqrt(np.max(np.sum((data_ - c) ** 2, axis=-1)))
            data_ = (data_ - c) / m
            self.all_data.append(data_)
            self.all_labels.append(labels_[fps_indices])

    def __getitem__(self, index) -> T_co:
        return self.all_data[index], self.all_axes[index], self.all_labels[index]

    def __len__(self):
        return len(self.all_data)
