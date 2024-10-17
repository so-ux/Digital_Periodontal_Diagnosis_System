import numpy as np


def pc_normalize(nd_verts):
    """
    Normalize a
    :param nd_verts:
    :type nd_verts:
    :return:
    :rtype:
    """
    xyz = nd_verts[:, 0:3]
    c = np.mean(xyz, axis=0)
    m = np.max(np.sqrt(np.sum((xyz - c) ** 2, axis=1)))
    nd_verts[:, 0:3] = (xyz - c) / m
    return nd_verts, c, m


def pc_normalize_revert(nd_verts, c, m):
    nd_verts[:, 0:3] = nd_verts[:, 0:3] * m + c
    return nd_verts
