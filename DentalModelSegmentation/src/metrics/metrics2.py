import math

import numpy as np
import torch

import src.metrics.tsa as tsa
import src.metrics.tir as tir
import src.metrics.tla as tla
import open3d as o3d

def teeth_localization_accuracy(pred: torch.Tensor, target: torch.Tensor):
    """
    Teeth localization accuracy (TLA)

    Mean of normalized Euclidean distance between ground truth (GT) teeth
    centroids and the closest localized teeth centroid. Each computed Euclidean distance is normalized by the
    size of the corresponding GT tooth. In case of no centroid (e.g. algorithm crashes or missing output
    for a given scan) a nominal penalty of 5 per GT tooth will be given. This corresponds to a distance 5 times
    the actual GT tooth size. As the number of teeth per patient may be variable, here the mean is computed
    over all gathered GT Teeth in the two testing sets.

    :param pred: [B, N, 3]
    :param target: [B, N, 3]
    :return: Accuracy value
    :rtype: float
    """
    pred_labels = torch.unique(pred[0, :, -1])
    target_labels = torch.unique(target[0, :, -1])

    pred_centroids = []
    scores = []

    for i in range(pred_labels.shape[0]):
        if pred_labels[i] > 0:
            pred_centroids.append(torch.mean(pred[0, pred[0, :, -1] == pred_labels[i], 0:3], dim=0))
    pred_centroids = torch.stack(pred_centroids, dim=0)

    for i in range(target_labels.shape[0]):
        if target_labels[i] > 0:
            if target[0, target[0, :, -1] == target_labels[i], 0:3].shape[0] < 4:
                scores.append(1e-5)
                continue
            cent = torch.mean(target[0, target[0, :, -1] == target_labels[i], 0:3], dim=0)
            dists = torch.sqrt(torch.sum((pred_centroids - cent) ** 2, dim=1))
            min_dist = torch.min(dists)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(target[0, target[0, :, -1] == target_labels[i], 0:3])
            obb = pcd.get_oriented_bounding_box()
            size_of_teeth = np.sqrt(np.sum((obb.get_max_bound() - obb.get_min_bound()) ** 2))
            min_dist /= size_of_teeth
            scores.append(min_dist)

    return np.sum(scores) / (target_labels.shape[0] - 1), (target_labels.shape[0] - 1)


def teeth_identification_rate(pred: torch.Tensor, target: torch.Tensor):
    """
    Teeth identification rate (TIR)

    Is computed as the percentage of true identification cases relatively to all GT teeth in the two testing sets.
    A true identification is considered when for a given GT Tooth, the closest detected tooth centroid :
    is localized at a distance under half of the GT tooth size, and is attributed the same label as the GT tooth.

    :return: Accuracy value
    :rtype: float
    """
    true_cnt = 0

    pred_labels = torch.unique(pred[0, :, -1])
    pred_labels = pred_labels[pred_labels > 0]

    target_labels = torch.unique(target[0, :, -1])
    target_labels = target_labels[target_labels > 0]

    pred_centroids = []

    for i in range(pred_labels.shape[0]):
        pred_centroids.append(torch.mean(pred[0, pred[0, :, -1] == pred_labels[i], 0:3], dim=0))
    pred_centroids = torch.stack(pred_centroids, dim=0)

    for i in range(target_labels.shape[0]):
        cent = torch.mean(target[0, target[0, :, -1] == target_labels[i], 0:3], dim=0)
        dists = torch.sqrt(torch.sum((pred_centroids - cent) ** 2, dim=1))
        arg_min_dist = torch.argmin(dists)
        min_dist = torch.min(dists)

        if target[0, target[0, :, -1] == target_labels[i], 0:3].shape[0] < 4:
            min_dist = 0.1
        else:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(target[0, target[0, :, -1] == target_labels[i], 0:3])
            obb = pcd.get_oriented_bounding_box()
            size_of_teeth = np.sqrt(np.sum((obb.get_max_bound() - obb.get_min_bound()) ** 2))
            min_dist /= size_of_teeth

        if min_dist < 0.5 and target_labels[i] == pred_labels[arg_min_dist]:
            true_cnt += 1

    return true_cnt / target_labels.shape[0], target_labels.shape[0]


def teeth_segmentation_accuracy(pred: torch.Tensor, target: torch.Tensor):
    """
    Teeth segmentation accuracy (TSA)

    Is computed as the average F1-score over all instances of teeth point clouds. The F1-score of each tooth
    instance is measured as: F1=2*(precision * recall)/(precision+recall).

    :param pred:
    :type pred:
    :param target:
    :type target:
    :return: Accuracy value
    :rtype: float
    """
    pred_ones = torch.zeros(pred.size())
    target_ones = torch.zeros(target.size())
    pred_ones[pred > 0] = 1
    target_ones[target > 0] = 1

    precision = torch.sum(pred_ones.view(-1) * target_ones.view(-1)) / torch.sum(pred_ones)
    recall = torch.sum(pred_ones.view(-1) * target_ones.view(-1)) / torch.sum(target_ones)
    f1 = 2 * precision * recall / (precision + recall)
    return f1
    # return mean_intersection_over_union(pred_ones, target_ones)


def global_ranking_score(tla: float, tir: float, tsa: float):
    score = (math.exp(-tla) + tir + tsa) / 3
    return score


def mean_intersection_over_union(pred, target, n_classes=1):
    """
    mIoU evaluation metric

    :param n_classes: Number of classes
    :type n_classes: int
    :param pred: [B, N]
    :type pred: torch.Tensor
    :param target: [B, N]
    :type target: torch.Tensor
    :return: mIoU value
    :rtype: float
    """
    pred = pred.view(-1)
    target = target.view(-1)
    value = 0
    for cls in range(1, n_classes + 1):
        intersections = torch.sum((pred == cls) * (target == cls))
        unions = torch.sum(pred == cls) + torch.sum(target == cls)
        if unions.item() == 0:
            continue
        value += 2 * intersections.item() / unions.item()
    return value


if __name__ == '__main__':
    import json
    import numpy as np
    from src.utils.input import read_obj

    input_obj_file = f'/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/3D_scans_per_patient_obj_files/0EAKT1CU/0EAKT1CU_lower.obj'
    label_json_file = f'/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/ground-truth_labels_instances/0EAKT1CU/0EAKT1CU_lower.json'

    verts, _, normals = read_obj(input_obj_file, True)
    verts = np.concatenate((verts, normals), -1)

    with open(label_json_file, 'r') as fp:
        labels = json.load(fp)
    labels = np.expand_dims(labels['labels'], 1)
    labels = np.concatenate((verts, labels), -1)
    labels_rm = np.copy(labels)

    labels = torch.Tensor(labels)
    labels_rm = torch.Tensor(labels_rm)

    labels_rm[labels_rm == 44] = 0

    labels = labels.unsqueeze(0)
    labels_rm = labels_rm.unsqueeze(0)

    print(f'Remove tooth 44, TLA: 1.0 -> {math.exp(-teeth_localization_accuracy(labels_rm, labels))}')
    print(teeth_segmentation_accuracy(labels_rm[:, :, -1], labels[:, :, -1]))


    input_obj_file = f'/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/3D_scans_per_patient_obj_files/Z83V9A9D/Z83V9A9D_lower.obj'
    label_json_file = f'/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/ground-truth_labels_instances/Z83V9A9D/Z83V9A9D_lower.json'

    verts, _, normals = read_obj(input_obj_file, True)
    verts = np.concatenate((verts, normals), -1)

    with open(label_json_file, 'r') as fp:
        labels = json.load(fp)
    labels = np.expand_dims(labels['labels'], 1)
    labels = np.concatenate((verts, labels), -1)
    labels_rm = np.copy(labels)

    labels = torch.Tensor(labels)
    labels_rm = torch.Tensor(labels_rm)

    labels_rm[labels_rm == 34] = 0

    labels = labels.unsqueeze(0)
    labels_rm = labels_rm.unsqueeze(0)

    print(f'Remove tooth 34, TLA: 1.0 -> {math.exp(-teeth_localization_accuracy(labels_rm, labels))}')
    print(teeth_segmentation_accuracy(labels_rm[:, :, -1], labels[:, :, -1]))
