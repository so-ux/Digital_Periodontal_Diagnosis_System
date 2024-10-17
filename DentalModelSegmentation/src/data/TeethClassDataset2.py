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

        all_data = h5file['data'][:][:, :, 0:6]
        all_labels = h5file['labels'][:]
        all_segs = h5file['segs'][:]
        all_masks = h5file['masks'][:]
        all_sizes = h5file['sizes'][:]

        self.all_pc = []
        self.all_patches = []
        self.all_labels = []
        self.all_masks = []
        self.all_pcid = []

        for index in range(all_data.shape[0]):
            size = all_sizes[index]
            pc = np.zeros((all_data[index].shape[0], 6 + 3 + 1))
            pc[:, 0:6] = all_data[index]

            labels = all_labels[index, :size]
            segs = np.asarray(all_segs[index, :size, :], dtype=np.int32)
            masks = all_masks[index, :size, :]
            patches = np.zeros((size, 4096, 7))
            patches[:, :, 0:6] = pc[:, 0:6][segs]
            patches[:, :, 6] = masks

            centroids = np.zeros((size, 3))
            for i in range(size):
                centroids[i] = np.mean(patches[i, patches[i, :, -1] > 0, 0:3], axis=0)
                mask = segs[i]
                mask = mask[patches[i, :, 6] > 0]
                pc[mask, 6:9] = centroids[i]

            for i in range(size):
                mask = segs[i]
                mask = mask[patches[i, :, 6] > 0]
                self.all_masks.append(mask)
                self.all_pcid.append(index)

            self.all_pc.append(pc)
            self.all_labels.append(labels)

        self.all_labels = np.concatenate(self.all_labels, axis=0)

    def __getitem__(self, index):
        pcid = self.all_pcid[index]
        pc = self.all_pc[pcid]
        mask = self.all_masks[index]
        pc[mask, 9] = 1
        return pc, self.all_labels[index]

    def __len__(self):
        return len(self.all_pcid)


if __name__ == '__main__':
    dset = TeethClassDataset('/run/media/zsj/DATA/Data/miccai/h5_cls2/', train=False)

    # Vis
    pc, labels = dset.__getitem__(5)
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
