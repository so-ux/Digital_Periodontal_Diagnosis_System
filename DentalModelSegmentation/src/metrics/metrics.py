import math

import torch

import src.metrics.tsa as tsa
import src.metrics.tir as tir
import src.metrics.tla as tla

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
    return tla.metric(pred, target)

def teeth_identification_rate(pred: torch.Tensor, target: torch.Tensor):
    """
    Teeth identification rate (TIR)

    Is computed as the percentage of true identification cases relatively to all GT teeth in the two testing sets.
    A true identification is considered when for a given GT Tooth, the closest detected tooth centroid :
    is localized at a distance under half of the GT tooth size, and is attributed the same label as the GT tooth.

    :return: Accuracy value
    :rtype: float
    """
    return tir.metric(pred, target)

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
    return tsa.metric(pred, target)


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
    a = torch.Tensor([
        [
            [1, 2, 3],
            [0, 0, 0],
            [4, 5, 6],
            [0, 0, 0],
            [0, 0, 1],
            [2, 5, 3]
        ],
        [
            [1, 2, 3],
            [0, 0, 0],
            [4, 5, 6],
            [0, .5, 0],
            [2, 0, 1],
            [2, 5, 3]
        ]
    ])
    b = torch.Tensor([
        [
            [1, 2, 4],
            [0, 0, 0],
            [4, 5, 5],
            [0, 0, 0],
            [0, 0, 0],
            [1.5, 5, 3.5]
        ],
        [
            [1, 2, 4],
            [0, 0, 0],
            [4, 5, 5],
            [0, 3, 0],
            [0, 0, 0],
            [1.5, 5, 3.5]
        ]
    ])
    print(teeth_localization_accuracy(a, b))
