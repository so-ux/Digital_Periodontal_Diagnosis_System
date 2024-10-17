import numpy as np
from scipy import linalg
from scipy.spatial.transform import Rotation as R


def apply_rotation(nd_data, axis_up, axis_forward):
    # rot_axis = np.cross(axis_up, np.array([0, 0, 1]))
    # cos_theta = np.sum(axis_up * np.array([0, 0, 1])) / np.linalg.norm(axis_up)
    # rot_angle = np.arccos(np.clip(cos_theta, -1, 1))
    # rot_matrix = linalg.expm(np.cross(np.eye(3), rot_axis / linalg.norm(rot_axis) * rot_angle))
    # nd_data[:, 0:3] = np.matmul(rot_matrix, nd_data[:, 0:3, np.newaxis])[:, :, 0]
    # nd_data[:, 3:6] = np.matmul(rot_matrix, nd_data[:, 3:6, np.newaxis])[:, :, 0]
    # axis_up = np.matmul(rot_matrix, axis_up)
    # axis_forward = np.matmul(rot_matrix, axis_forward)

    rot_axis = np.cross(np.array([0, -1, 0]), axis_forward)
    cos_theta = np.sum(axis_forward * np.array([0, -1, 0])) / np.linalg.norm(axis_forward)
    rot_angle = np.arccos(np.clip(cos_theta, -1, 1))
    print(np.cross(axis_forward, np.array([0, -1, 0])))
    rot_matrix = linalg.expm(np.cross(np.eye(3), rot_axis / linalg.norm(rot_axis) * rot_angle))
    nd_data[:, 0:3] = np.matmul(rot_matrix, nd_data[:, 0:3, np.newaxis])[:, :, 0]
    nd_data[:, 3:6] = np.matmul(rot_matrix, nd_data[:, 3:6, np.newaxis])[:, :, 0]
    axis_forward = np.matmul(rot_matrix, axis_forward)

    return nd_data, axis_up, axis_forward


data = np.loadtxt('/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/data_rot/test_12_0.txt')
data = data[:4096, :6]

cur_axis = np.array([0, -1, 0])
target_axis = np.array([-1, 0, 0])

data, _, _ = apply_rotation(data, None, target_axis)
print(data)
np.savetxt('/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/data_rot/test_12_00.xyz', data)
