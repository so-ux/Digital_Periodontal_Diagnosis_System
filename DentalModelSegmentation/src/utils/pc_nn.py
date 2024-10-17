import numpy as np
import torch


def pc_dist(p1, p2):
    """
    Calculate distances between two point sets

    :param p1: size[N, D]
    :param p2: size[M, D]
    """
    p1 = np.expand_dims(p1, 0)
    p2 = np.expand_dims(p2, 0)

    p1 = torch.from_numpy(p1).to('cuda', non_blocking=True)
    p2 = torch.from_numpy(p2).to('cuda', non_blocking=True)
    p1 = p1.repeat(p2.size(1), 1, 1)

    p1 = p1.transpose(0, 1)

    p2 = p2.repeat(p1.size(0), 1, 1)

    dist = torch.add(p1, torch.neg(p2))

    dist, _ = torch.min(torch.norm(dist, 2, dim=2), dim=1)

    indices = torch.argsort(dist).detach().cpu().numpy()
    return indices

