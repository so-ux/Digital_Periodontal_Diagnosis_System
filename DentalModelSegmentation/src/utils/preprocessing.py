import numpy as np
from scipy import linalg
from sklearn.cluster import DBSCAN

from src.utils.pca import pca
import torch
import os


normal_models = []
for i in range(1, 5):
    normal_models.append(np.loadtxt(os.path.join(os.path.dirname(__file__), f'./normal_{i}.xy')))
    c = np.mean(normal_models[i - 1], axis=0)
    m = np.max(np.sqrt(np.sum((normal_models[i - 1] - c) ** 2, axis=-1)))


def chamfer_distance_without_batch(p1, p2, debug=False):
    """
    Calculate Chamfer Distance between two point sets

    :param p1: size[1, N, D]
    :param p2: size[1, M, D]
    :param debug: Whether you need to output debug info
    :return: Sum of Chamfer Distance of two point sets
    """

    p1 = torch.Tensor(np.array([p1])).cuda()
    p2 = torch.Tensor(np.array([p2])).cuda()

    assert p1.size(0) == 1 and p2.size(0) == 1
    assert p1.size(2) == p2.size(2)

    if debug:
        print(p1[0][0])
    p1 = p1.repeat(p2.size(1), 1, 1)

    p1 = p1.transpose(0, 1)

    p2 = p2.repeat(p1.size(0), 1, 1)

    dist = torch.add(p1, torch.neg(p2))

    dist = torch.norm(dist, 2, dim=2)

    dist_sort, indices = torch.sort(dist)
    dist1 = dist_sort[:, 0]
    # dist1_2 = dist_sort[:, 1]
    # index_min_dis = indices[:, 0]

    return torch.mean(dist1).item()


def align_model(data):
    standard_up_vector = np.array([0, 0, 1])
    normal_vectors = data[:, 3:6]

    eigen_values, eigen_vectors = pca(data[:, 0:3])

    max_cnt_ratio = 0
    up_vector = eigen_vectors[0]
    for eigen_vector in eigen_vectors:
        cnts = []
        for co in [1, -1]:
            dots = np.sum(normal_vectors * eigen_vector * co, axis=-1)
            cnts.append(np.sum(dots < 0))
        if np.max(cnts) / (np.min(cnts) + 1e-9) > max_cnt_ratio:
            max_cnt_ratio = np.max(cnts) / (np.min(cnts) + 1e-9)
            up_vector = eigen_vector * [1, -1][np.argmin(cnts)]

    # up_vector = np.round(up_vector)
    # print('Up vector', up_vector)
    # theta > 60deg, cos_theta < 0.5
    cos_theta = np.sum(standard_up_vector * up_vector) / np.linalg.norm(up_vector)
    rotated = False
    rot_matrix = np.eye(3)
    if cos_theta < 2.2 and np.sum(np.abs(up_vector - standard_up_vector)) > 1e-2:
        rotated = True
        # print('ROTATE')
        # Rotate
        rot_axis = np.cross(up_vector, standard_up_vector)
        rot_angle = np.arccos(np.clip(cos_theta, -1, 1))
        if abs(np.pi - rot_angle) < 1e-4:
            rot_axis = np.array([0, 1, 0])
        # print('AXIS=', rot_axis, '  ANGLE=', rot_angle)
        rot_matrix = linalg.expm(np.cross(np.eye(3), rot_axis / linalg.norm(rot_axis) * rot_angle))
        data[:, 0:3] = np.matmul(rot_matrix, data[:, 0:3, np.newaxis])[:, :, 0]
        data[:, 3:6] = np.matmul(rot_matrix, data[:, 3:6, np.newaxis])[:, :, 0]

    return data[:, 0:6], rotated, rot_matrix


def align_model_on_xy(positive_points):
    centroid = np.mean(positive_points, axis=0)
    max_dis = np.max(np.sqrt(np.sum((positive_points - centroid) ** 2, axis=-1)))
    norm_points = (positive_points[:, 0:2] - c) / max_dis
    eigen_values, eigen_vectors = pca(norm_points)

    min_cd = 1e9
    min_rot_angle = None
    normal_directions = np.array([
        [1, 0], [0, -1], [-1, 0], [0, 1]
    ])
    # 选择最小倒角距离的一项旋转
    # eigen_vectors是正交的，因此选择一个即可
    vec = np.round(eigen_vectors[0])
    print('Direction vec', vec)
    for direction in normal_directions:
        cos_theta = np.sum(direction * vec) / np.linalg.norm(vec)

        points_ = np.copy(norm_points)

        # Rotate
        rot_angle = np.arccos(np.clip(cos_theta, -1, 1))
        rot_matrix = np.array([
            [np.cos(rot_angle), -np.sin(rot_angle)],
            [np.sin(rot_angle), np.cos(rot_angle)]
        ]).T
        points_[:, 0:2] = np.matmul(points_[:, 0:2], rot_matrix)

        cd = 1e9
        for normal_model in normal_models:
            # todo change a better algorithm
            cd = min(cd, chamfer_distance_without_batch(normal_model, points_) +
                     chamfer_distance_without_batch(points_, normal_model))
        print('Direction', direction, '  cd', cd)
        if cd < min_cd:
            min_cd = cd
            min_rot_angle = rot_angle
    return min_rot_angle

# if __name__ == '__main__':
#     import glob
#     import trimesh
#
#     for file in glob.glob('/run/media/zsj/DATA/Data/miccai/3D_scans_per_patient_obj_files/*/*.obj'):
#         mesh = trimesh.load(file, process=False)
#         verts = np.asarray(mesh.vertices)
#         normals = np.asarray(mesh.vertex_normals)
#         data = np.concatenate((verts, normals), axis=-1)
#         align_model(data)

# def farthest_point_sample(xy, npoint):
#     """
#     Input:
#         xyz: pointcloud data, [B, N, C]
#         npoint: number of samples
#     Return:
#         centroids: sampled pointcloud index, [B, npoint]
#     """
#     if isinstance(xy, list):
#         xy = np.array(xy)
#
#     N, C = xy.shape
#     centroids = np.zeros([npoint], dtype=np.int64)
#     distance = np.ones((N,)) * 1e10
#     farthest = np.random.randint(0, N, (1,), dtype=np.int64)
#     for i in range(npoint):
#         centroids[i] = farthest
#         centroid = xy[farthest, :]
#         dist = np.sum((xy - centroid) ** 2, -1)
#         mask = dist < distance
#         distance[mask] = dist[mask]
#         farthest = np.argmax(distance, -1)
#     return centroids
#
#
# if __name__ == '__main__':
#     for i in range(1, 5):
#         data = np.loadtxt(f'normal{i}.xy')
#         id_list = farthest_point_sample(data, 2000)
#         data = data[id_list]
#         c = np.mean(data, axis=0)
#         m = np.max(np.sqrt(np.sum((data - c) ** 2, axis=-1)))
#         data = (data - c) / m
#         np.savetxt(f'normal_{i}.xy', data)


