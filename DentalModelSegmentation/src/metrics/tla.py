import numpy as np
import open3d as o3d
import json


# 读取obj文件
import torch


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

    Returns: Teeth localization accuracy (mean of normalized Euclidean distance between
        ground truth (GT) teeth centroids and the closest localized teeth centroid)

    """
    output_list = []
    for i in range(pred.shape[0]):
        label_of_pred = pred[i, :, 3]
        label_of_target = target[i, :, 3]
        index_list = list(range(11, 29)) + list(range(31, 49))
        distance_list = []
        for tooth_index in index_list:
            xyz_of_target = target[i, label_of_target == tooth_index, 0:3]
            if xyz_of_target.shape[0] == 0:
                continue

            xyz_of_pred = pred[i, label_of_pred == tooth_index, 0:3]
            if xyz_of_pred.shape[0] == 0:
                distance_list.append(5)
                continue

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz_of_target)

            obb = pcd.get_oriented_bounding_box()

            size_of_teeth = np.sqrt(np.sum((obb.get_max_bound() - obb.get_min_bound()) ** 2))

            # 点云质心距离
            center_of_pred = torch.mean(xyz_of_pred, dim=0)
            center_of_target = torch.mean(xyz_of_target, dim=0)

            distance = torch.sqrt(torch.sum((center_of_pred - center_of_target) ** 2))

            # normalize
            normalized_distance = distance.item() / size_of_teeth

            distance_list.append(normalized_distance)
        mean_distance = np.mean(np.array(distance_list))
        output_list.append(mean_distance)
    output = np.array(output_list)
    return output

#
# vertices, _ = read_obj(r"E:\MICCAI_learning\3D_scans_per_patient_obj_files\0EJBIPTC\0EJBIPTC_lower.obj")
# target_labels = read_labels(r"E:\MICCAI_learning\ground-truth_labels_instances\0EJBIPTC\0EJBIPTC_lower.json")
# target = cat_vertices_labels(vertices, target_labels)
# target = np.expand_dims(target, 0)
# target = torch.Tensor(target)
#
# pred_labels = []
# for i in range(len(vertices)):
#     # 随机赋予顶点0（非牙齿）或[11, 12, ..., 17, 21, 22, ...27, ..., 41, 42, ..., 47]的标签
#     # 上颌为[11~27]，下颌为[31~47]
#     if random.random() > 0.5:
#         pred_labels.append(random.randint(1, 2) * 10 + random.randint(1, 7)+20)
#     else:
#         pred_labels.append(0)
# pred = cat_vertices_labels(vertices, pred_labels)
# pred = np.expand_dims(pred, 0)
# pred = torch.Tensor(pred)
# pred_labels = torch.Tensor(np.array(pred_labels))
# print(metric(pred, target))
