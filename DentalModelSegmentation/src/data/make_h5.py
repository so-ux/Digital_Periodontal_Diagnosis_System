import glob
import json
import os

import h5py
import numpy as np

from TeethDataLoaderWithFaces import TeethDataset
from src.utils import geodesic
from src.utils.input import read_obj

# output_dir = '/home/zsj/Projects/data/miccai/results/gt_small_tooth/'
output_dir = '/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/h5/'


def make_h5(id, jaw, data, labels, c, m, id_selected, fps_indices, flat_indices, gd, fps_indices_16k):
    """
    Make h5 data file for faster training, avoid repeated data initialization

    :param id: Patient id
    :type id: str
    :param jaw: Upper / lower jaw
    :type jaw: str
    :param data: Full point cloud data, including normals
    :type data: ndarray
    :param labels: Labels for each vertex
    :type labels: ndarray
    :param c: Normalization - centroid
    :type c: ndarray
    :param m: Normalization - max distance
    :type m: float
    :param id_selected: Selected id after braces removed
    :type id_selected: ndarray
    :param fps_indices: Farthest point sampling indices
    :type fps_indices: ndarray
    :return: None
    :rtype: None
    """
    file = h5py.File(os.path.join(output_dir, '%s_%s.h5' % (id, jaw)), 'w')
    file['id'] = id
    file['jaw'] = jaw
    file['data'] = data
    file['labels'] = labels
    file['c'] = c
    file['m'] = m
    file['id_selected'] = id_selected
    file['fps_indices'] = fps_indices
    file['flat_indices'] = flat_indices
    file['gd'] = gd
    file['fps_indices_16k'] = fps_indices_16k
    file.close()


def make_tempdir(id, jaw, data, labels, c, m, id_selected, fps_indices):
    """
    Make temp directory for faster training, avoid repeated data initialization

    :param id: Patient id
    :type id: str
    :param jaw: Upper / lower jaw
    :type jaw: str
    :param data: Full point cloud data, including normals
    :type data: ndarray
    :param labels: Labels for each vertex
    :type labels: ndarray
    :param c: Normalization - centroid
    :type c: ndarray
    :param m: Normalization - max distance
    :type m: float
    :param id_selected: Selected id after braces removed
    :type id_selected: ndarray
    :return: None
    :rtype: None
    """
    folder = os.path.join(output_dir, '%s_%s' % (id, jaw))
    os.makedirs(folder, exist_ok=True)

    np.savetxt(os.path.join(folder, 'data.xyz'), data)
    np.savetxt(os.path.join(folder, 'labels.txt'), labels)
    np.savetxt(os.path.join(folder, 'id_selected.txt'), id_selected)
    np.savetxt(os.path.join(folder, 'c.txt'), c)
    np.savetxt(os.path.join(folder, 'fps_indices.txt'), fps_indices)

    with open(os.path.join(folder, 'm.txt'), 'w') as f:
        f.write('{}\n'.format(m))
        f.close()


# Debug test method
if __name__ == '__main__':
    import torch
    from src.utils.remove_braces import model_flat_indices, remove_braces_mesh
    from src.utils.pc_normalize import pc_normalize
    from src.pointnet2.pointnet2_utils import furthest_point_sample

    # dataset = TeethDataset('/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data')
    # for i in range(len(dataset)):
    #     id, jaw, origin_data, labels, faces = dataset.__getitem__(i)
    #     print('Processing %s - %s...' % (id, jaw))
    #
    #     # Remove braces
    #     # selected = remove_braces(origin_data)
    #     selected, faces = remove_braces_mesh(origin_data, faces)
    #     data = origin_data[selected]
    #
    #     # Normalize
    #     data, c, m = pc_normalize(data)
    #     gd = geodesic.gt_mesh_geodesic_field(data, faces, labels[selected])
    #
    #     data_tensor = torch.Tensor(np.array([data[:, 0:3]])).cuda()
    #
    #     # For each tooth cropping area, sample 2048 points
    #     # Assume that there are 16 teeth on each model
    #     fps_indices = furthest_point_sample(data_tensor, 2048 * 16).detach().cpu().numpy()[0]
    #     fps_indices_16k = furthest_point_sample(data_tensor, 1024 * 16).detach().cpu().numpy()[0]
    #     flat_indices = model_flat_indices(data[fps_indices])
    #     make_h5(id, jaw, origin_data, labels, c, m, selected, fps_indices, flat_indices, gd, fps_indices_16k)
    #
    #     data_tensor = data_tensor.detach().cpu()

    datalist = glob.glob('/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/3D_scans_per_patient_obj_files/*/*.obj')\
                + glob.glob('/home/zsj/Projects/data/miccai/results/gt_small_tooth/*.obj')
    for i in range(len(datalist)):
        id = os.path.basename(datalist[i])
        id, jaw = id.replace('.obj', '').split('_')
        print(id, jaw)
        v, f, n = read_obj(datalist[i], True)
        v = np.concatenate((v, n), axis=-1)

        try:
            with open(datalist[i].replace('.obj', '.json'), 'r') as fp:
                labels = np.array(json.load(fp)['labels'], dtype=np.int32)
        except:
            with open(f'/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/ground-truth_labels_instances/{id}/{id}_{jaw}.json', 'r') as fp:
                labels = np.array(json.load(fp)['labels'], dtype=np.int32)

        # Remove braces
        selected, faces = remove_braces_mesh(v, f)
        data = v[selected]

        # labels = np.zeros((v.shape[0], ), dtype=np.int32)
        # selected = np.arange(0, v.shape[0])
        # data = np.copy(v)
        data, c, m = pc_normalize(data)

        data_tensor = torch.Tensor(np.array([data[:, 0:3]])).cuda()

        # For each tooth cropping area, sample 2048 points
        # Assume that there are 16 teeth on each model
        fps_indices = furthest_point_sample(data_tensor, 2048 * 16).detach().cpu().numpy()[0]
        fps_indices_16k = furthest_point_sample(data_tensor, 1024 * 16).detach().cpu().numpy()[0]
        flat_indices = model_flat_indices(data)
        make_h5(id, jaw, v, labels, c, m, selected, fps_indices, flat_indices, [], fps_indices_16k)

        data_tensor = data_tensor.detach().cpu()