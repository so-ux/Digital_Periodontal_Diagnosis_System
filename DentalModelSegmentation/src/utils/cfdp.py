"""
Clustering by fast search and find of density peaks
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pydpc import Cluster
from scipy.optimize import leastsq
from sklearn.neighbors import KDTree

from src.utils import geometry
from src.utils.pc_normalize import pc_normalize


def get_clustered_centroids(pred_centroids: np.ndarray) -> np.ndarray:
    """
    质心聚类

    :param pred_centroids:
    :type pred_centroids:
    :return:
    :rtype:
    """
    if len(pred_centroids) == 0:
        return pred_centroids
    points, c, m = pc_normalize(np.array(pred_centroids))
    clu = Cluster(points, autoplot=False)
    # clu.autoplot = True


    point_p = clu.density * clu.delta
    point_p_i = np.argsort(point_p)[-100:]

    temp_indices = []
    for i in range(1, len(point_p_i)):
        if np.isnan(point_p[point_p_i[i]]):
            pred_centroids = np.unique(pred_centroids, axis=0)
            return pred_centroids

        cur_delta = point_p[point_p_i[i]] - point_p[point_p_i[i - 1]]

        if point_p[point_p_i[i]] > point_p[point_p_i[i - 1]] * 2.5 or cur_delta > 0.01:
            temp_indices = point_p_i[i:]
            break
    return pred_centroids[temp_indices]

    # Multi-scale clustering
    # Smaller cluster counts indicate a more accurate shape

    # 2022 05 24 Deprecated
    # clu.assign(10, 0.1)
    # small_clusters = clu.clusters
    small_clusters = temp_indices
    small_pred_centroids_2d = points[temp_indices]
    small_pred_centroids_2d[:, 2] = 0
    plane = geometry.plane_approximate_3d(small_pred_centroids_2d)
    curve = geometry.curve_fitting(small_pred_centroids_2d, plane)
    # # _, dists = geometry.parameterize_points_on_curve(pred_centroids[small_clusters], plane, curve)
    # # _, dists = geometry.parameterize_points_on_curve_3d(pred_centroids[small_clusters], plane, curve)
    _, dists = geometry.parameterize_points_on_xy_plane(points[temp_indices])
    thres_3d = (2 * np.max(dists) + 0. * np.mean(dists)) / 2
    # print('dists3d', dists)
    # print(thres_3d)

    clu.assign(0.1, 0.05)
    big_clusters = clu.clusters
    # big_clusters = np.arange(0, points.shape[0])
    point_params, dists = geometry.parameterize_points_on_xy_plane(points[big_clusters], curve)
    # point_params, dists = geometry.parameterize_points_on_curve(points[big_clusters], plane, curve)
    # point_params, dists_3d = geometry.parameterize_points_on_curve_3d(points[big_clusters], plane, curve)
    # diff_big_clusters = big_clusters
    diff_big_clusters = np.setdiff1d(big_clusters, small_clusters)
    # print(diff_big_clusters)
    if len(diff_big_clusters) == 0:
        return pred_centroids[big_clusters]

    # print('dists', dists)

    final_indices = temp_indices
    for dif in diff_big_clusters:
        dist = np.squeeze(dists[big_clusters == dif])
        # dist_3d = np.squeeze(dists_3d[big_clusters == dif])
        # print('dist_2d', dist)
        index = big_clusters[big_clusters == dif]
        # if dist < 0.1:
        if dist < thres_3d:
            final_indices = np.concatenate((final_indices, index), axis=0)

    return pred_centroids[np.unique([*temp_indices, *final_indices])]


def get_clustered_centroids_with_seg(pred_centroids: np.ndarray, pred_seg: np.ndarray, teeth_points: np.ndarray) -> np.ndarray:
    """
    质心聚类

    :param pred_centroids:
    :type pred_centroids:
    :return:
    :rtype:
    """
    points, c, m = pc_normalize(np.array(pred_centroids))
    clu = Cluster(points, autoplot=False)

    point_p = clu.density * clu.delta
    point_p_i = np.argsort(point_p)[-20:]
    temp_indices = []
    for i in range(1, len(point_p_i)):
        cur_delta = point_p[point_p_i[i]] - point_p[point_p_i[i - 1]]
        if point_p[point_p_i[i]] > point_p[point_p_i[i - 1]] * 2.5 or cur_delta > 0.01:
            temp_indices = point_p_i[i:]
            break

    centroids_first = pred_centroids[temp_indices]

    kdtree = KDTree(pred_centroids)
    # Remove points around centroids
    nn_indices = kdtree.query_radius(centroids_first, 0.05, return_distance=False)
    nn_indices = np.unique(np.concatenate(nn_indices))
    mask = np.arange(0, pred_centroids.shape[0])
    mask[nn_indices] = -1
    mask = mask[mask > -1]
    points_remain = pred_centroids[mask]

    if points_remain.shape[0] == 0:
        return centroids_first

    clu = Cluster(points_remain, autoplot=False)
    point_p = clu.density * clu.delta
    point_p_i = np.argsort(point_p)[-20:]
    temp_indices = []
    for i in range(1, len(point_p_i)):
        cur_delta = point_p[point_p_i[i]] - point_p[point_p_i[i - 1]]
        if point_p[point_p_i[i]] > point_p[point_p_i[i - 1]] * 2.5 or cur_delta > 0.01:
            temp_indices = point_p_i[i:]
            break

    centroids_second = pred_centroids[temp_indices]

    kdtree = KDTree(teeth_points)
    centroids_second_final = []
    for point in centroids_second:
        nn_indices = kdtree.query_radius([point], 0.05, return_distance=False)[0]
        if np.sum(pred_seg[nn_indices]) > 5:
            centroids_second_final.append(point)

    if len(centroids_second_final) == 0:
        return centroids_first

    return np.concatenate((centroids_first, centroids_second_final), axis=0)


def sample_centroids(patch_points: np.ndarray, pred_seg: np.ndarray, i,
                     conf_threshold: float = 0.5, lmbd: float = 0.8) -> np.ndarray:
    """
    根据质心分割结果重新选取质心

    质心预测可能会产生偏差，例如落在上下颌平面，或落在两颗牙齿中间。此类情况会影响RefineNet分割结果。需要根据分割遮罩重新选取质心。

    策略：\n
    1.利用CFDP聚类选取密度峰值中心，一般位于牙冠曲率较大处，作为质心偏移参考；\n
    2.对已有分割结果质心作偏移。

    :param patch_points: [N, 3] Patch点云
    :type patch_points: np.ndarray
    :param pred_seg: [N, ] 分割结果
    :type pred_seg: np.ndarray
    :param conf_threshold: 分割置信度阈值
    :type conf_threshold: float
    :param lmbd: 预测质心偏移系数
    :type lmbd: float
    :return: [C, 3] 重新选取的质心
    :rtype: np.ndarray
    """
    new_centroids = []
    seg_points = patch_points[pred_seg > conf_threshold, 0:3]

    if seg_points.shape[0] == 0:
        return np.array(new_centroids)

    seg_points_centroid = np.mean(seg_points, axis=0)
    clu = Cluster(seg_points, autoplot=False)
    clu.assign(5, 0.25)

    if len(clu.clusters) == 0:
        new_centroids.append(seg_points_centroid)
    else:
        points_to_centroid_dist = np.linalg.norm(seg_points[clu.clusters] - seg_points_centroid, axis=1)
        weight = 0.4 * points_to_centroid_dist + 0.7
        # far_indices = clu.clusters[near_points > 0.3]
        # if len(clu.clusters) == 0:
        #     new_centroids.append(seg_points_centroid)
        for index, clu_id in enumerate(clu.clusters):
            new_centroids.append(seg_points_centroid + weight[index] * (seg_points[clu_id] - seg_points_centroid))

    return np.array(new_centroids)


def sample_centroids_mask(seg_points: np.ndarray,
                     conf_threshold: float = 0.5, lmbd: float = 0.8) -> np.ndarray:
    """
    根据质心分割结果重新选取质心

    质心预测可能会产生偏差，例如落在上下颌平面，或落在两颗牙齿中间。此类情况会影响RefineNet分割结果。需要根据分割遮罩重新选取质心。

    策略：\n
    1.利用CFDP聚类选取密度峰值中心，一般位于牙冠曲率较大处，作为质心偏移参考；\n
    2.对已有分割结果质心作偏移。

    :param patch_points: [N, 3] Patch点云
    :type patch_points: np.ndarray
    :param pred_seg: [N, ] 分割结果
    :type pred_seg: np.ndarray
    :param conf_threshold: 分割置信度阈值
    :type conf_threshold: float
    :param lmbd: 预测质心偏移系数
    :type lmbd: float
    :return: [C, 3] 重新选取的质心
    :rtype: np.ndarray
    """
    new_centroids = []

    if seg_points.shape[0] == 0:
        return np.array(new_centroids)

    seg_points = np.ascontiguousarray(seg_points[:, 0:3])
    seg_points, c, m = pc_normalize(seg_points)

    seg_points_centroid = np.mean(seg_points, axis=0)
    clu = Cluster(seg_points, autoplot=False)
    clu.assign(5, 0.25)

    if len(clu.clusters) == 0:
        new_centroids.append(seg_points_centroid)
    else:
        # Membership centroids
        mem_centroids = []
        for mem in np.unique(clu.membership):
            mem_centroids.append(np.mean(seg_points[clu.membership == mem, 0:3], axis=0))
        mem_centroids = np.array(mem_centroids)
        points_to_centroid_dist = np.linalg.norm(mem_centroids - seg_points_centroid, axis=1)
        mem_centroids = mem_centroids[points_to_centroid_dist > 0.2]
        if len(mem_centroids) > 0:
            return mem_centroids * m + c
        else:
            new_centroids.append(seg_points_centroid)

    return np.array(new_centroids) * m + c


def get_rotation_axis(pred_axis):
    clu = Cluster(pred_axis, autoplot=False)
    point_p = clu.density * clu.delta
    point_p_i = np.argsort(point_p)[-1]
    # temp_indices = []
    # for i in range(1, len(point_p_i)):
    #     cur_delta = point_p[point_p_i[i]] - point_p[point_p_i[i - 1]]
    #     if point_p[point_p_i[i]] > point_p[point_p_i[i - 1]] * 2.5 or cur_delta > 0.01:
    #         temp_indices = point_p_i[i:]
    #         break
    return pred_axis[point_p_i]


if __name__ == '__main__':
    # for i in range(1, 32):
    #     fn = f'/home/zsj/PycharmProjects/miccai-3d-teeth-seg/temp/validate/cent{i}.xyz'
    #     points = np.loadtxt(fn)
    #     mux = 1
    #     muy = 1
    #     # get_clustered_centroids(points)
    #
    #     clu = Cluster(points, autoplot=False)
    #     clu.assign(0.5, 0.1)
    #     print(clu.clusters, len(clu.clusters))
    #     plt.show()
    fn_list = [
        # '/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/test_vis/01FAXM1D_upper.obj.ct.xyz',
        # '/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/test_vis/WINM1RGX_lower.obj.ct.xyz',
        # '/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/test_vis/ZKJEPFDD_lower.obj.ct0.xyz',
        # '/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/test_vis/W82LGNOE_lower.obj.ct0.xyz',
        # '/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/test_vis/XBDVGZ7V_lower.obj.ct.xyz',
        # '/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/test_vis/01EDMPPZ_upper.obj.ct.xyz',
        # '/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/test_vis/0OTKQ5J9_lower.obj.ct.xyz',
        # '/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/test_vis/Q5GSR6N2_lower.obj.ct.xyz',
        # '/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/test_vis/019ZKUHV_upper.obj.ct.xyz',
        # '/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/test_vis/VY1R353X_lower.obj.ct0.xyz'
        # '/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/test_vis/01HMF5HV_lower.obj.ct1.xyz'
        # '/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/test_vis/0OTKQ5J9_lower.obj.ct.xyz'
        # '/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/test_vis/01J9K9S6_lower.obj.ct.xyz'
        # '/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/test_vis/XTF24UY3_upper.obj.ct.xyz'
        '/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/test_vis/Y8BV16EI_lower.obj.ct.xyz'
        # '/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/test_vis/43074710_shell_occlusion_l.out.obj.ct.xyz'
    ]
    for i in range(len(fn_list)):
        fn = fn_list[i]
        points = np.loadtxt(fn)
        points = np.ascontiguousarray(points[:, 0:3])
        points, _, _ = pc_normalize(points)
        pc = np.mean(points, axis=0)
        clu = Cluster(points, autoplot=False)
        clu.autoplot = True
        clu.assign(0.1, 0.05)

        fig, ax = plt.subplots(2, 2, figsize=(15, 15))
        ax[0][0].scatter(points[:, 0], points[:, 1], c=clu.membership, cmap=mpl.cm.cool, s=20)
        # ax[0][0].scatter(points[clu.clusters, 0], points[clu.clusters, 1], s=20)
        ax[0][0].scatter(points[clu.clusters, 0], points[clu.clusters, 1], s=30, c="red")

        point_p = np.sort(clu.density * clu.delta)[-30:]
        ax[0][1].scatter(np.arange(0, len(point_p)), point_p, s=10)

        # Membership centroids
        mem_centroids = []
        for mem in np.unique(clu.membership):
            mem_centroids.append(np.mean(points[clu.membership == mem, 0:3], axis=0))
        mem_centroids = np.array(mem_centroids)
        mem_centroid_dist = np.linalg.norm(mem_centroids - pc, axis=1)

        pred_centroids = get_clustered_centroids(points)
        ax[1][0].scatter(points[:, 0], points[:, 1], s=30)
        ax[1][0].scatter(pred_centroids[:, 0], pred_centroids[:, 1], s=30, c="red")

        clu.assign(10, 0.1)
        # ax[1][1].scatter(points[:, 0], points[:, 1], s=30)
        # ax[1][1].scatter(points[clu.clusters, 0], points[clu.clusters, 1], s=30, c="red")

        # small_clusters = clu.clusters
        # small_pred_centroids_2d = points[small_clusters]
        # small_pred_centroids_2d[:, 2] = 0
        small_pred_centroids_2d = get_clustered_centroids(points)
        small_pred_centroids_2d[:, 2] = 0
        params = geometry.parameterize_points_on_xy_plane(small_pred_centroids_2d)


        def parameterize_points_on_xy_plane(points, params=None):
            projected_points_2d = points[:, 0:2]

            # 最小二乘法拟合抛物线
            def func(params, x):
                a, b, c = params
                return a * x * x + b * x + c

            def error(params, x, y):
                return func(params, x) - y

            init_params = np.array([-1, 0, 0])
            if params is not None:
                a, b, c = params
            else:
                a, b, c = leastsq(error, init_params, args=(projected_points_2d[:, 0], projected_points_2d[:, 1]))[0]
            return a, b, c


        # plane = geometry.plane_approximate_3d(small_pred_centroids_2d)
        # curve = geometry.curve_fitting(small_pred_centroids_2d, plane)
        # projected_points_2d = geometry.project_points_to_plane_2d(points, plane)
        projected_points_2d = small_pred_centroids_2d[:, 0:2]
        print(projected_points_2d.tolist())
        # _, dists = geometry.parameterize_points_on_curve(points, plane, curve)
        curve = parameterize_points_on_xy_plane(projected_points_2d)
        ax[1][1].scatter(small_pred_centroids_2d[:, 0], small_pred_centroids_2d[:, 1], s=30, c='red')
        ax[1][1].scatter(small_pred_centroids_2d[:, 0], [-0.74] * small_pred_centroids_2d.shape[0], s=30, c='blue')
        X = np.linspace(-1, 1, 500)
        Y = curve[0] * X ** 2 + curve[1] * X + curve[2]
        ax[1][1].scatter(X, Y, s=10, c="green")

        for _ax in ax.flatten():
            _ax.set_xlabel(r"x / a.u.", fontsize=20)
            _ax.set_ylabel(r"y / a.u.", fontsize=20)
            _ax.tick_params(labelsize=15)
            _ax.set_aspect('equal')
        fig.tight_layout()

        print(clu.clusters, len(clu.clusters))
        plt.show()
