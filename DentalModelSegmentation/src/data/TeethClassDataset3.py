import os

import torch.utils.data as data
import numpy as np
import h5py


class TeethClassDataset(data.Dataset):
    """
    Dataset for training ClsNet
    """
    def __init__(self, data_dir, train=True):
        super().__init__()
        self.data_dir = data_dir
        self.train = train

        data_file = os.path.join(data_dir, 'train.h5' if train else 'test.h5')
        h5file = h5py.File(data_file)

        self.all_data = h5file['data'][:][:, :, 0:6]
        self.all_labels = h5file['labels'][:]
        self.all_segs = h5file['segs'][:]
        self.all_masks = h5file['masks'][:]
        self.all_sizes = h5file['sizes'][:]

    def __getitem__(self, index):
        size = self.all_sizes[index]
        pc = np.zeros((self.all_data[index].shape[0], 6 + 3 + 1))
        pc[:, 0:6] = self.all_data[index]

        labels = self.all_labels[index, :size]
        segs = np.asarray(self.all_segs[index, :size, :], dtype=np.int32)
        masks = self.all_masks[index, :size, :]
        pred_labels = np.arange(0, size) + 1
        patches = np.zeros((size, 4096, 7))
        patches[:, :, 0:6] = pc[:, 0:6][segs]
        patches[:, :, 6] = masks
        centroids = np.zeros((size, 3))
        for i in range(size):
            centroids[i] = np.mean(patches[i, patches[i, :, -1] > 0, 0:3], axis=0)
            pc[segs[i], 6:9] = centroids[i]
        return pc, patches, labels, pred_labels, centroids

    def __len__(self):
        return self.all_data.shape[0]


if __name__ == '__main__':
    dset = TeethClassDataset('/run/media/zsj/DATA/Data/miccai/h5_cls2/', train=False)

    # Vis
    pc, patches, labels, pred_labels, centroids = dset.__getitem__(5)
    print('label', labels)
    with open('/home/zsj/PycharmProjects/miccai-3d-teeth-seg/temp/cls.obj', 'w') as f:
        for i in range(pc.shape[0]):
            color = [255, 0, 100] if pc[i, 9] > 0.9 else [0, 200, 200]
            f.write(f'v {pc[i, 0]} {pc[i, 1]} {pc[i, 2]} {color[0]} {color[1]} {color[2]}\n')
            f.write(f'vn {pc[i, 3]} {pc[i, 4]} {pc[i, 5]}\n')

        delta = pc.shape[0] + 1
        for i in range(pc.shape[0]):
            if np.sum(pc[i, 6:9]) != 0:
                f.write(f'v {pc[i, 6]} {pc[i, 7]} {pc[i, 8]} 0 255 0\n')
                f.write(f'l {delta} {i + 1}\n')
                delta += 1
