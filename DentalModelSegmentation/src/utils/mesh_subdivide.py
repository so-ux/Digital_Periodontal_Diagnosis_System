"""
网格细分
用于较为稀疏的网格

建议预测时使用
"""
import math

import numpy as np

from src.utils.input import read_obj
from src.utils.output import write_obj
import open3d as o3d
from loguru import logger


def face_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = b - a
    ac = c - a
    cos_theta = np.dot(ab, ac) / (np.linalg.norm(ab) * np.linalg.norm(ac))
    sin_theta = np.sqrt(1 - cos_theta ** 2)
    return np.linalg.norm(ab) * np.linalg.norm(ac) * sin_theta / 2


def do_subdivide(v, f, fi, mid, feats):
    triangle = f[fi]
    mid_points = np.zeros((3, 3))
    for vi_list in [[0, 1], [1, 2], [2, 0]]:
        a = triangle[vi_list[0]]
        b = triangle[vi_list[1]]
        mid_index = mid[a].keys().__contains__(b)
        if not mid_index:
            mid_point = np.expand_dims((v[a] + v[b]) / 2, 0)
            mid[a][b] = len(v)
            mid[b][a] = len(v)
            v = np.concatenate((v, mid_point), axis=0)
            if feats is not None:
                new_feat = min(feats[a], feats[b])
                feats = np.concatenate((feats, [new_feat]), axis=0)
        mid_points[vi_list[0], vi_list[1]] = mid[a][b]
        mid_points[vi_list[1], vi_list[0]] = mid[a][b]
    m_ab = mid_points[0, 1]
    m_ac = mid_points[0, 2]
    m_bc = mid_points[1, 2]
    f_ext = np.array([
        [m_ab, triangle[1], m_bc],
        [m_ac, m_bc, triangle[2]],
        [m_ac, m_ab, m_bc]
    ], dtype=np.int32)
    f[fi] = np.array([
        triangle[0], m_ab, m_ac
    ])
    return v, np.concatenate((f, f_ext), axis=0), feats


def iterate(v, f, feats=None):
    new_f = np.array(f, dtype=np.int32)
    new_v = np.array(v)
    if feats is not None:
        new_feats = np.array(feats)
    else:
        new_feats = None
    mid = []
    for i in range(len(v)):
        mid.append({})

    i = 0
    for face in f:
        area = face_area(v[face[0]], v[face[1]], v[face[2]])
        if 0.1 < area < 3:
            new_v, new_f, new_feats = do_subdivide(new_v, new_f, i, mid, new_feats)
        i += 1
    return new_v, new_f, new_feats


def infer_mesh_subdivide(v, f):
    """
    用于预测时细分网格

    :return: v, f, n
    """
    if len(v) < 60000:
        iters = math.ceil(60000 / len(v))
        logger.info('This model has {} vertices, will run subdivide {} times', len(v), iters)
        for i in range(iters):
            v, f, _ = iterate(v, f, None)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.triangles = o3d.utility.Vector3iVector(f)
    mesh.compute_vertex_normals()
    return v, f, np.asarray(mesh.vertex_normals)


def infer_mesh_subdivide_with_features(v, f, feats):
    """
    用于训练时细分网格

    :return: v, f, n, feats
    """
    if len(v) < 32768:
        iters = math.ceil(32768 / len(v))
        logger.info('This model has {} vertices, will run subdivide {} times', len(v), iters)
        for i in range(iters):
            v, f, feats = iterate(v, f, feats)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.triangles = o3d.utility.Vector3iVector(f)
    mesh.compute_vertex_normals()
    return v, f, np.asarray(mesh.vertex_normals), feats


if __name__ == '__main__':
    v, f, _ = read_obj('/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/3D_scans_per_patient_obj_files/Y9WQHQMT/Y9WQHQMT_lower.obj', True)
    v, f, n = infer_mesh_subdivide(v, f)
    write_obj('/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/3D_scans_per_patient_obj_files/Y9WQHQMT/d.obj', v, f, None)
