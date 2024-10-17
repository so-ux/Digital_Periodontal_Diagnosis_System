import math
import os
import measure.utils as utils
import glob
import numpy as np
from numpy import multiply
import trimesh

def pca(data, sort=True):
    mean = np.mean(data, axis=0)
    norm = data - mean
    H = np.dot(norm.T, norm)
    eigen_vectors, eigen_values, _ = np.linalg.svd(H)

    if sort:
        sort = eigen_values.argsort()[::-1]
        eigen_values = eigen_values[sort]
        eigen_vectors = eigen_vectors[:, sort]

    return eigen_values, eigen_vectors

def find_tooth_axis(mesh,tooth_axis_output, file):
    points = np.asarray(mesh.vertices)
    eigen_values, v = pca(points)
    eigen_vectors = v[:, 0]
    coordinates_squared = [x ** 2 for x in eigen_vectors]
    magnitude = math.sqrt(sum(coordinates_squared))
    eigen_vectors_reg = [x * (1. / magnitude) for x in eigen_vectors]
    center = np.mean(points, axis=0)
    axis = [center + multiply(eigen_vectors_reg, j) for j in range(-13, 17, 1)]
    gum_coords = np.array(axis)
    gum_line = trimesh.Trimesh(vertices=gum_coords)
    gum_line.export(os.path.join(tooth_axis_output, file))


if __name__ == '__main__':

    #tooth_axis
    dirs = glob.glob(os.path.join(utils.get_dataset(), '*'))

    for row in ['lower','upper']:#
        for d in dirs:
            print(d)
            cfg = utils.get_path(os.path.join(d, row))
            for file in os.listdir(cfg.path.cbct):
                mesh = trimesh.load(os.path.join(cfg.path.cbct, file), force='mesh', process=False)
                find_tooth_axis(mesh,cfg.path.tooth_axis)