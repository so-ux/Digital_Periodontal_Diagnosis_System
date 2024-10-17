import numpy as np
import random
import open3d as o3d
import torch
import json


# 读取obj文件
def read_obj(file_path):
    # o3d.io.read_triangle_mesh vertex indices are not identical to the original file's
    # mesh = o3d.geometry.TriangleMesh()
    # Read obj
    f = open(file_path, 'r')
    verts, faces, normals = [], [], []
    for line in f.readlines():
        if line.startswith('v'):
            verts.append([float(p) for p in line[1:].strip().split(' ')])
        elif line.startswith('f'):
            faces.append([int(p) for p in line[1:].strip().split(' ')])
    # mesh.vertices = o3d.utility.Vector3dVector(np.array(verts))
    # mesh.faces = o3d.utility.Vector3iVector(np.array(faces))
    return np.array(verts)[:, 0:3], np.array(faces)


# 读取labels 的json文件
def read_labels(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        labels = np.array(data["labels"])
    return labels


# 连接vertices和labels
def cat_vertices_labels(ver, labels):
    labels = np.expand_dims(labels, 1)
    target = np.concatenate((ver, labels), axis=1)
    return target


def metric(pred, target):
    """
    Args:
        pred: [B, N, C]
        target: [B, N, C]


    """
    output_list = []
    for i in range(pred.shape[0]):
        label_of_pred = pred[i, :, 3]
        label_of_target = target[i, :, 3]
        index_list = list(range(11, 29)) + list(range(31, 49))
        distance_list = []
        count = 0
        size = 0
        for tooth_index in index_list:  # 找相同标签的
            xyz_of_target = target[i, label_of_target == tooth_index, 0:3]  # target中是该标签的所有点的前三位用于计算距离
            if xyz_of_target.shape[0] == 0:
                continue

            xyz_of_pred = pred[i, label_of_pred == tooth_index, 0:3]  # predict中是该标签的所有点的前三位用于计算距离

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz_of_target)

            obb = pcd.get_oriented_bounding_box()

            # obb.color =  np.array([0, 0, 0])
            # o3d.visualization.draw_geometries([
            #     pcd, obb
            # ])
            size_of_teeth = np.sqrt(np.sum((obb.get_max_bound() - obb.get_min_bound()) ** 2))
            # print(size_of_teeth)

            # 点云质心距离
            center_of_pred = torch.mean(xyz_of_pred, axis=0)
            # print(center_of_pred)
            center_of_target = torch.mean(xyz_of_target, axis=0)
            # print(center_of_target)
            distance = torch.sqrt(torch.sum((center_of_pred - center_of_target) ** 2))
            # print(distance)
            # normalize

            normalized_distance = distance / size_of_teeth

            distance_list.append(normalized_distance)
            if (normalized_distance < (size_of_teeth / 2)):
                count += 1
            size += 1
        # mean_distance = torch.mean(torch.Tensor(np.array(distance_list)))
        output_list.append(count / size)
        # print(output_list)
    output = np.array(output_list)
    return output
