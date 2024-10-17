import numpy as np
from scipy.optimize import leastsq
from src.utils.ellipse import LsqEllipse
from src.utils import pca


def plane_approximate_3d(points):
    """
    根据散点拟合三维平面

    平面方程 Ax + By + Cz + D = 0
    转化为 z = ax + by + c, 其中 a = -A / C, b = -B / C, c = -D / C

    :param points: [N, 3]
    :type points: np.ndarray
    :return: [3, ] a, b, c
    :rtype: np.ndarray
    """
    A = np.array([
        [np.sum(points[:, 0] ** 2), np.sum(points[:, 0] * points[:, 1]), np.sum(points[:, 0])],
        [np.sum(points[:, 0] * points[:, 1]), np.sum(points[:, 1] ** 2), np.sum(points[:, 1])],
        [np.sum(points[:, 0]), np.sum(points[:, 1]), points.shape[0]]
    ])
    T = np.array([
        np.sum(points[:, 0] * points[:, 2]),
        np.sum(points[:, 1] * points[:, 2]),
        np.sum(points[:, 2])
    ])
    a = np.matmul(np.linalg.inv(A), T)
    return a


def project_points_to_plane_2d(points, plane):
    # 任取坐标原点(0, 0, z)
    A_C, B_C, D_C = -plane
    A2_B2_C2 = A_C ** 2 + B_C ** 2 + 1
    x_p = ((B_C ** 2 + 1) * points[:, 0] - A_C * (B_C * points[:, 1] + points[:, 2] + D_C)) / A2_B2_C2
    y_p = ((A_C ** 2 + 1) * points[:, 1] - B_C * (A_C * points[:, 0] + points[:, 2] + D_C)) / A2_B2_C2
    z_p = ((A_C ** 2 + B_C ** 2) * points[:, 2] - (A_C * points[:, 0] + B_C * points[:, 1] + D_C)) / A2_B2_C2
    projected_points = np.stack((x_p, y_p, z_p), axis=1)

    evalue, evector = pca.pca(projected_points)

    nx, ny, _ = -plane
    vz = [nx, ny, 1]
    vx = evector[0]
    vy = evector[1]
    origin = np.mean(projected_points, axis=0)
    T = np.stack((vx, vy, vz, origin), axis=1)
    T = np.concatenate((T, np.array([[0, 0, 0, 1]])), axis=0)
    projected_points_2d = np.matmul(np.linalg.inv(T),
                                    np.concatenate((projected_points, np.ones((projected_points.shape[0], 1))),
                                                   axis=1).T).T
    return projected_points_2d


def curve_fitting(points, plane):
    """
    散点投影到平面参数化，拟合抛物线

    :param points:
    :type points:
    :param plane:
    :type plane:
    :return:
    :rtype:
    """
    projected_points_2d = project_points_to_plane_2d(points, plane)

    # 最小二乘法拟合抛物线
    def func(params, x):
        a, b, c = params
        return a * x * x + b * x + c

    def error(params, x, y):
        return func(params, x) - y

    init_params = np.array([-1, 0, 0])
    params = leastsq(error, init_params, args=(projected_points_2d[:, 0], projected_points_2d[:, 1]))
    return params[0]


def parameterize_points_on_plane(points, plane):
    """
    散点投影到平面参数化，拟合抛物线参数

    :param points: [N, 3]
    :type points: np.ndarray
    :param plane: [3, ]，平面参数a, b, c
    :type plane: np.ndarray
    :return: [N, 2]，平面投影点集; [N] 距离
    :rtype: tuple(np.ndarray, np.ndarray)
    """
    projected_points_2d = project_points_to_plane_2d(points, plane)
    a, b, c = curve_fitting(points, plane)

    def func(params, x):
        a, b, c = params
        return a * x * x + b * x + c

    def point_on_curve(x, y, l=-1, r=1):
        # 求斜率和切线需要求解三次方程，不利于计算
        # 因此采用微分方法
        splits = 1000
        result_x = l

        min_dist = 10000
        current = l
        while current < r:
            current_y = func((a, b, c), current)
            if min_dist > ((current_y - y) ** 2 + (current - x) ** 2):
                min_dist = (current_y - y) ** 2 + (current - x) ** 2
                result_x = current
            current += (r - l) / splits

        return result_x, min_dist

    results = []
    dists = []
    for point_2d in projected_points_2d:
        point_on_curve_x, min_dist = point_on_curve(point_2d[0], point_2d[1])
        results.append(point_on_curve_x)
        dists.append(min_dist)
    return results, np.array(dists)


def parameterize_points_on_xy_plane(points, params=None):
    """
    散点投影到平面参数化，拟合抛物线参数

    :param points: [N, 3]
    :type points: np.ndarray
    :return: [N, 2]，平面投影点集; [N] 距离
    :rtype: tuple(np.ndarray, np.ndarray)
    """
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

    def point_on_curve(x, y, l=-1, r=1):
        # 求斜率和切线需要求解三次方程，不利于计算
        # 因此采用微分方法
        splits = 1000
        result_x = l

        min_dist = 10000
        current = l
        while current < r:
            current_y = func((a, b, c), current)
            if min_dist > ((current_y - y) ** 2 + (current - x) ** 2):
                min_dist = (current_y - y) ** 2 + (current - x) ** 2
                result_x = current
            current += (r - l) / splits

        return result_x, min_dist

    results = []
    dists = []
    for point_2d in projected_points_2d:
        point_on_curve_x, min_dist = point_on_curve(point_2d[0], point_2d[1])
        results.append(point_on_curve_x)
        dists.append(min_dist)
    return results, np.array(dists) + points[:, 2] ** 2


def parameterize_points_on_curve(points, plane, curve):
    """
    散点投影到平面参数化，拟合抛物线参数

    :param points: [N, 3]
    :type points: np.ndarray
    :param plane: [3, ]，平面参数a, b, c
    :type plane: np.ndarray
    :return: [N, 2]，平面投影点集; [N] 距离
    :rtype: tuple(np.ndarray, np.ndarray)
    """
    projected_points_2d = project_points_to_plane_2d(points, plane)
    a, b, c = curve

    def func(params, x):
        a, b, c = params
        return a * x * x + b * x + c

    def point_on_curve(x, y, l=-1, r=1):
        # 求斜率和切线需要求解三次方程，不利于计算
        # 因此采用微分方法
        splits = 1000
        result_x = l

        min_dist = 10000
        current = l
        while current < r:
            current_y = func((a, b, c), current)
            if min_dist > ((current_y - y) ** 2 + (current - x) ** 2):
                min_dist = (current_y - y) ** 2 + (current - x) ** 2
                result_x = current
            current += (r - l) / splits

        return result_x, min_dist

    results = []
    dists = []
    for point_2d in projected_points_2d:
        point_on_curve_x, min_dist = point_on_curve(point_2d[0], point_2d[1])
        results.append(point_on_curve_x)
        dists.append(min_dist)
    return results, np.array(dists)


def points_to_plane_distance(points, plane):
    project_points = project_points_to_plane_2d(points, plane)[:, 0:3]
    return np.sum((project_points - points) ** 2, axis=1)


def parameterize_points_on_curve_3d(points, plane, curve):
    params, dists = parameterize_points_on_curve(points, plane, curve)
    p2p_dists = points_to_plane_distance(points, plane)
    return params, dists + p2p_dists


def ellipse_fitting(points):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    X1, X2 = points[:, 0], points[:, 1]
    X = np.array(list(zip(X1, X2)))
    reg = LsqEllipse().fit(X)
    center, width, height, phi = reg.as_parameters()

    print(f'center: {center[0]:.3f}, {center[1]:.3f}')
    print(f'width: {width:.3f}')
    print(f'height: {height:.3f}')
    print(f'phi: {phi:.3f}')

    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot()
    ax.axis('equal')
    ax.plot(X1, X2, 'ro', zorder=1)
    ellipse = Ellipse(
        xy=center, width=2*width, height=2*height, angle=np.rad2deg(phi),
        edgecolor='b', fc='None', lw=2, label='Fit', zorder=2
    )
    ax.add_patch(ellipse)

    plt.xlabel('$X_1$')
    plt.ylabel('$X_2$')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    points = [[0.12116525, -0.43745947, 0.17122361],
              [0.26858726, -0.36107537, 0.15033586],
              [0.3927988, -0.21375327, 0.14342552],
              [0.41604894, -0.04430162, 0.1268331],
              [0.49132758, 0.10836203, 0.09417951],
              [0.53346145, 0.31454203, 0.03511949],
              [0.6163925, 0.5175281, -0.05625111],
              [-0.09409403, -0.468898, 0.16999017],
              [-0.2611, -0.4255544, 0.13781968],
              [-0.4110297, -0.3533067, 0.10513668],
              [-0.53501695, -0.01585807, 0.09083606],
              [-0.44956967, -0.16029993, 0.11906714],
              [-0.5931742, 0.183369, 0.04679934],
              [-0.68243045, 0.39553884, -0.02208549]]
    parameterize_points_on_plane(np.array(points), plane_approximate_3d(np.array(points)))
