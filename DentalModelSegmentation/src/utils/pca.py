import numpy as np
from sklearn.decomposition import PCA


def pca(data, sort=True):
    n_components = 2 if data.shape[-1] < 3 else 3
    _pca = PCA(n_components=n_components).fit(data)
    return np.asarray(_pca.explained_variance_), np.asarray(_pca.components_)
    mean = np.mean(data, axis=0)
    norm = data - mean
    H = np.dot(norm.T, norm)
    eigen_vectors, eigen_values, _ = np.linalg.svd(H)

    if sort:
        sort = eigen_values.argsort()[::-1]
        eigen_values = eigen_values[sort]
        eigen_vectors = eigen_vectors[:, sort]

    return eigen_values, eigen_vectors


# import open3d as o3d
# import trimesh
#
# fn = '/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/data_rot/gt/Q3NA514X_upper.obj'
# mesh = trimesh.load(fn, process=False)
# # points = np.loadtxt(fn)[:, 0:3]
# points = np.asarray(mesh.vertices)
#
# c = np.mean(points, axis=0)
# points -= c
#
# # rot_m = np.array([[ 0.98916014, -0.03927428, -0.14149114],
# #  [-0.03927428,  0.85770395, -0.51264165],
# #  [ 0.14149114,  0.51264165,  0.8468641 ]])
# # points[:, 0:3] = np.dot(points[:, 0:3], rot_m.T)
#
# eigen_values, eigen_vectors = pca(points)
# up = np.array([0.14149114, 0.51264165, 0.8468641])
# print(eigen_values, eigen_vectors)
# mesh = o3d.geometry.TriangleMesh()
# # axis = mesh.create_coordinate_frame(size=30).rotate(eigen_vectors, center=(0, 0, 0))
# # pc_view = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points))
# # o3d.visualization.draw_geometries([pc_view, axis])
#
# from sklearn.decomposition import PCA
# p = PCA(n_components=3)
# p.fit(points)
# print(p.components_)
# axis = mesh.create_coordinate_frame(size=30).rotate(p.components_, center=(0, 0, 0))
# pc_view = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points))
# o3d.visualization.draw_geometries([pc_view, axis])