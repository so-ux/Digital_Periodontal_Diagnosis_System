import h5py
import numpy as np
import os
import open3d as o3d
import glob

import torch
import tqdm

from pointnet2_utils import furthest_point_sample
from src.utils import geodesic
from src.utils.input import read_obj
from src.utils.pc_normalize import pc_normalize
from src.utils.remove_braces import remove_braces, model_flat_indices, remove_braces_mesh
from src.vis.anatomy_colors import AnatomyColors

base = '/run/media/zsj/DATA/Data/Gordon/'
# base = '/run/media/zsj/DATA/Data/miccai/gordon_16k/'
output_dir = '/run/media/zsj/DATA/Data/miccai/gordon/'

standard_pcd, standard_faces, standard_norms = read_obj(
    '/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/3D_scans_per_patient_obj_files/0EJBIPTC/0EJBIPTC_lower.obj',
    True)
standard_mesh = o3d.geometry.PointCloud()
standard_mesh.points = o3d.utility.Vector3dVector(standard_pcd)
standard_mesh.normals = o3d.utility.Vector3dVector(standard_norms)

colors = AnatomyColors()


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
    file['fps_indices_16k'] = fps_indices_16k
    file['flat_indices'] = flat_indices
    file['gd'] = gd
    file.close()


def read_off(filename):
    file = open(filename, 'r')
    header = file.readline().strip()
    if header not in ['OFF', 'COFF']:
        print('Not a valid OFF header')
    n_verts, n_faces, _ = tuple([int(s) for s in file.readline().strip().split(' ') if s != ''])
    verts = [[float(s) for s in file.readline().strip().split(' ')[0:3] if s != ''] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ') if s != ''][1:4] for i_face in range(n_faces)]
    verts = np.vstack(verts)
    faces = np.vstack(faces)
    # Compute normal
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals)
    return verts, faces, normals


def register(pc, norm):
    trans_init = np.asarray([[1, 0, 0, 0],  # 4x4 identity matrix，这是一个转换矩阵，
                             [0, 1, 0, 0],  # 象征着没有任何位移，没有任何旋转，我们输入
                             [0, 0, 1, 0],  # 这个矩阵为初始变换
                             [0, 0, 0, 1]])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.normals = o3d.utility.Vector3dVector(norm)
    print(pcd, standard_mesh)
    reg = o3d.pipelines.registration.registration_icp(pcd, standard_mesh, 0.2, trans_init,
                                                      o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                      o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=3000))
    print(reg)
    pcd = pcd.transform(reg.transformation)
    return np.concatenate((
        np.asarray(pcd.points), np.asarray(pcd.normals)
    ), axis=1)


def transform(data, upper=True):
    if upper:
        # X rotate 180deg
        cos_gamma = -1
        sin_gamma = 0
        rot_x = np.array([
            [1, 0, 0],
            [0, cos_gamma, sin_gamma],
            [0, -sin_gamma, cos_gamma]
        ])
        data[:, 0:3] = np.matmul(data[:, 0:3], rot_x)
        data[:, 3:6] = np.matmul(data[:, 3:6], rot_x)
    # Z rotate 270deg
    cos_gamma = 0
    sin_gamma = -1
    rot_z = np.array([
        [cos_gamma, sin_gamma, 0],
        [-sin_gamma, cos_gamma, 0],
        [0, 0, 1]
    ])
    data[:, 0:3] = np.matmul(data[:, 0:3], rot_z)
    data[:, 3:6] = np.matmul(data[:, 3:6], rot_z)
    return data


for model in tqdm.tqdm(os.listdir(base)):
    print('model', model)
    lower = glob.glob(os.path.join(base, model) + '/*_L_*.off') + glob.glob(os.path.join(base, model) + '/*_L_*.txt') + ['L']
    upper = glob.glob(os.path.join(base, model) + '/*_U_*.off') + glob.glob(os.path.join(base, model) + '/*_U_*.txt') + ['U']
    all_data = []
    if len(lower) == 3:
        all_data.append(lower)
    if len(upper) == 3:
        all_data.append(upper)
    for g in all_data:
        try:
            v, f, n = read_off(g[0])
        except:
            continue
        labels = np.loadtxt(g[1], dtype=int)[1:]
        lgtz = labels[labels > 0]
        if np.sum(lgtz < 11) > 0:
            print('Wrong labels 1@ ', g[0], np.unique(lgtz))
            continue
        lgtz = lgtz % 10
        if (np.sum(lgtz == 0) + np.sum(lgtz > 8)) > 0:
            print('Wrong labels 2@ ', g[0], np.unique(lgtz))
            continue

        labels[labels < 0] = 0
        origin_data = np.concatenate((v, n), axis=1)
        # h5 = h5py.File(os.path.join(base, model))
        # origin_data = h5['data'][:]
        # labels = h5['labels'][:]
        # origin_data = transform(origin_data, model.__contains__('_U'))
        origin_data = transform(origin_data, g[2] == 'U')

        # color = np.ones((origin_data.shape[0], 4))
        # color[:, 0:3] *= 0.5

        # for pid in range(origin_data.shape[0]):
        #     if (labels[pid] // 10 == 1) or (labels[pid] // 10 == 3):
        #         color[pid, 0:3] = colors.get_tooth_color(labels[pid])

        print('Processing %s...' % (model))

        # Remove braces
        selected, faces = remove_braces_mesh(origin_data, f)
        data = origin_data[selected]

        # Normalize
        data, c, m = pc_normalize(data)
        # gd = geodesic.gt_mesh_geodesic_field(data, faces, labels[selected])

        data_tensor = torch.Tensor(np.array([data[:, 0:3]])).cuda()

        # For each tooth cropping area, sample 2048 points
        # Assume that there are 16 teeth on each model
        fps_indices = furthest_point_sample(data_tensor, 2048 * 16).detach().cpu().numpy()[0]
        fps_indices_16k = furthest_point_sample(data_tensor, 1024 * 16).detach().cpu().numpy()[0]

        # model_id = model[:model.index('_', 6)]
        flat_indices = model_flat_indices(data)
        make_h5(model, g[2], origin_data, labels, c, m, selected, fps_indices, flat_indices, 0, fps_indices_16k)
