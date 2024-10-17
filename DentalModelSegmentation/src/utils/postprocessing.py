import numpy as np
import pointnet2_utils
import torch
from sklearn.cluster import DBSCAN

from src.utils import geometry


def fast_cluster(preds_kp):
    # calculate
    clustering = DBSCAN(min_samples=1, eps=0.02).fit(preds_kp)
    label_id = clustering.labels_
    preds_kp_unique = []
    # for kp_id in range(1, max(label_id) + 1):
    #     preds_kp_temp = preds_kp[label_id == kp_id, :]
    #     for kp_temp_id in range(0, preds_kp_temp.shape[0]):
    #         if kp_temp_id < 5:
    #             preds_kp_unique.append(preds_kp_temp[kp_temp_id, :])
    #         if kp_temp_id >= 5:
    #             break
    for id in np.unique(clustering.labels_):
        selected = np.mean(preds_kp[label_id == id, :], axis=0)
        preds_kp_unique.append(selected)
    return np.asarray(preds_kp_unique)


def seg_nms(pred_seg, label):
    label_nms_count = []
    label_data_nms = []
    tooth_nms_id = []
    for label_id in range(pred_seg.shape[0]):
        if label_id == 0:
            tooth_nms_id.append(len(label_data_nms))
            label_nms_count.append(1)
            label_data_nms.append(pred_seg[label_id, :])
        if label_id > 0:
            flag = 0
            for overlap_id in range(len(label_data_nms)):
                intersection = np.sum((pred_seg[label_id, :] > 0.5) * (
                        label_data_nms[overlap_id] / label_nms_count[overlap_id] > 0.5)).astype(float)
                iou = 2 * intersection / (
                        np.sum(pred_seg[label_id, :] > 0.5) + np.sum(
                    label_data_nms[overlap_id] / label_nms_count[overlap_id] > 0.5)).astype(float)
                # Intersection over Self hhh
                ios = intersection / np.sum(pred_seg[label_id, :] > 0.5)
                # IoU > 0.15, consider as the same tooth
                if iou > 0.3 or ios > 0.9:
                    label_data_nms[overlap_id] = label_data_nms[overlap_id] + pred_seg[label_id, :]
                    label_nms_count[overlap_id] = label_nms_count[overlap_id] + 1
                    tooth_nms_id.append(overlap_id)
                    flag = 1
                    break
            if flag == 0:
                tooth_nms_id.append(len(label_data_nms))
                label_data_nms.append(pred_seg[label_id, :])
                label_nms_count.append(1)
    label_nms_count = np.asarray(label_nms_count, dtype=float)
    label_data_nms = np.asarray(label_data_nms, dtype=float)
    for tooth_id in range(label_nms_count.shape[0]):
        label_data_nms[tooth_id] = label_data_nms[tooth_id] / label_nms_count[tooth_id]
        label[label_data_nms[tooth_id] > 0.3] = tooth_id + 1
    return label, np.array(tooth_nms_id) + 1


def sample_to_points_cuda(pred_seg, data_samp, data_pts):
    dist, idx = pointnet2_utils.three_nn(data_pts.contiguous(), data_samp.contiguous())
    dist_recip = 1.0 / (dist + 1e-8)
    norm = torch.sum(dist_recip, dim=2, keepdim=True)
    weight = dist_recip / norm
    interpolated_feats = pointnet2_utils.three_interpolate(pred_seg.contiguous(), idx, weight)
    interpolated_feats = interpolated_feats[:, 0, :]
    return interpolated_feats


def infer_labels_denoise(patch_points, pred_seg):
    for patch_id in range(pred_seg.shape[0]):
        points = patch_points[patch_id, pred_seg[patch_id] > 0.5, :]
        pred_seg_patch = pred_seg[patch_id, pred_seg[patch_id] > 0.5]
        if points.shape[0] == 0:
            continue
        clu = DBSCAN(eps=0.05, min_samples=1).fit(points[:, 0:3])#基于密度的聚类
        clu_lbl, clu_counts = np.unique(clu.labels_, return_counts=True)

        # 丢弃散点
        for noise in clu_lbl[clu_counts < 50]:
            pred_seg_patch[clu.labels_ == noise] = 0

        pred_seg[patch_id, pred_seg[patch_id] > 0.5] = pred_seg_patch
    return pred_seg


def seg_voting_nms(pred_seg, pred_cls, n, is_fdi=False):
    """
    投票非极大值抑制

    逐点分类的投票，通过该分类总Patch数的1/2即通过

    :param n: 降采样全牙点数
    :type n: int
    :param is_fdi: 牙齿编号是否遵循FDI标准
    :type is_fdi: bool
    :param pred_seg: [B, N] 降采样全牙的分割遮罩
    :type pred_seg: np.ndarray
    :param pred_cls: [B, ] 降采样全牙的分类
    :type pred_cls: np.ndarray
    :return: Label [N, ] 降采样全牙标签
    :rtype: np.ndarray
    """
    label = np.zeros((n,))

    def tooth_fdi_to_id(fdi):
        if type(fdi) == np.ndarray:
            fdi[fdi > 0] = 8 * (fdi[fdi > 0] // 10 - 1) + fdi[fdi > 0] % 10 if is_fdi else fdi[fdi > 0]
            return fdi
        else:
            if fdi == 0:
                return 0
            return 8 * (fdi // 10 - 1) + fdi % 10 if is_fdi else fdi

    def tooth_id_to_fdi(id_array):
        id_array[id_array > 0] = np.ceil(id_array[id_array > 0] / 8) * 10 + (id_array[id_array > 0] - 1) % 8 + 1
        return id_array

    point_wise_voting = np.zeros((label.shape[0], 33))

    for patch_id in range(pred_seg.shape[0]):
        point_wise_voting[pred_seg[patch_id] > 0.5, tooth_fdi_to_id(pred_cls[patch_id])] += 1

    # 0-32
    pred_cls_unique, cls_counts = np.unique(pred_cls, return_counts=True)
    pred_cls_unique = tooth_fdi_to_id(pred_cls_unique)
    cls_cnt_33 = np.zeros((33,))

    for i in pred_cls_unique:
        cls_cnt_33[i] = cls_counts[pred_cls_unique == i]
    cls_cnt_33[cls_cnt_33 > 0] = 1 / cls_cnt_33[cls_cnt_33 > 0]

    point_wise_voting_rate = point_wise_voting * cls_cnt_33
    for point in range(label.shape[0]):
        point_cls_rate = point_wise_voting_rate[point]
        point_cls_rate[0] = 0

        possible_classes = np.sum(point_cls_rate > 0.5)
        # overlap_classes = np.sum(point_cls_rate > 0.5)
        if possible_classes > 0:
            label[point] = np.argmax(point_cls_rate)

    return tooth_id_to_fdi(label)


def rearrange_labels(points, pred_seg, pred_cls, id_flip=False, times=0):
    """
    重排牙齿分类标签

    尽管ToothClsNet的牙齿分类准确率可以达到97%甚至更高，但仍存在错误分类的情况；
    错误分类通常表现为相邻牙齿标签相同或整列牙齿标签产生位移。

    :param id_flip:
    :type id_flip:
    :param points:
    :type points:
    :param pred_seg: 预测的全牙Patch分割遮罩
    :type pred_seg: [B, N]
    :param pred_cls: 预测的Patch分类标签，按照FDI格式
    :type pred_cls: [B, ]
    :return: 重排后的分类标签
    :rtype: [B, N]
    """
    if times > 5:
        return rearrange_labels2(points, pred_seg, pred_cls)

    # 剔除分类为背景的Patch
    pred_seg = pred_seg[pred_cls > 0, :]
    pred_cls = pred_cls[pred_cls > 0]

    # 若最大置信度<0.6，则认为该Patch为负样本
    # 不进行该步处理，则有可能出现NaN centroid
    good_seg_id = np.squeeze(np.argwhere(np.max(pred_seg, axis=1) >= 0.6))
    pred_seg = pred_seg[good_seg_id, :]
    pred_cls = pred_cls[good_seg_id]

    # 确定属于上颌还是下颌
    # 基于多数的牙齿会被正确分类，且牙列呈单调的先验
    upper_cnt, lower_cnt = np.sum(pred_cls < 30), np.sum(pred_cls > 30)

    # 纠正上下颌分类错误
    # FDI标签，上下颌对应位置相差20
    if upper_cnt > lower_cnt > 0:
        pred_cls[pred_cls > 30] -= 20
    elif lower_cnt > upper_cnt > 0:
        pred_cls[pred_cls < 30] += 20

    seg_labels = np.zeros((pred_seg.shape[1],))
    # 使分类标签有序，从切牙到磨牙排列
    all_pred_cls_sort = np.argsort(pred_cls)
    # temp_labels为全牙去除overlap的逐点标签
    # tooth_nms_id为pred_cls元素对应的牙齿序号
    # 若pred_cls中相同元素在tooth_nms_id中编号不同，则需要调整pred_cls的值
    pred_seg = pred_seg[all_pred_cls_sort, :]
    pred_cls = pred_cls[all_pred_cls_sort]
    temp_labels, tooth_nms_id = seg_nms(pred_seg, seg_labels)

    # 将tooth_nms_id相同的归在一起
    all_pred_cls_sort = np.argsort(tooth_nms_id)
    pred_seg = pred_seg[all_pred_cls_sort, :]
    pred_cls = pred_cls[all_pred_cls_sort]
    tooth_nms_id = tooth_nms_id[all_pred_cls_sort]

    # 计算牙列质心，按照tooth_nms_id的顺序排列
    tooth_nms_centroids = []
    for tooth_id in tooth_nms_id:
        if points[temp_labels == tooth_id].shape[0] > 0:
            ct = np.mean(points[temp_labels == tooth_id], axis=0)
            tooth_nms_centroids.append(ct)

    tooth_nms_centroids = np.array(tooth_nms_centroids)

    arranged_pred_cls = np.array(pred_cls)

    # 22.05.25 Note: 投影方法牙弓曲线受到质心数量的影响较大，故直接使用XY平面的坐标进行拟合
    # 根据质心拟合牙弓平面和曲线
    # plane = geometry.plane_approximate_3d(tooth_nms_centroids)
    # # 求出每个质心点的参数
    # params = geometry.parameterize_points_on_plane(tooth_nms_centroids, plane)[0]
    point_on_curve = geometry.parameterize_points_on_xy_plane(tooth_nms_centroids)[0]
    point_on_curve = np.argsort(point_on_curve)
    params = np.zeros((len(point_on_curve),), dtype=np.int32)
    for i in range(len(point_on_curve)):
        params[point_on_curve[i]] = i + 1

    # 对于每个半边，params应该保持单调
    # 结合牙弓曲线的性质，两侧的params单调性不同
    # params 先减后增
    # if params[0] > params[-1]:
    #     params = params[::-1]

    # 保证相同牙齿有相同标签，否则会影响后续根据牙齿标签的排序
    for nms_id in np.unique(tooth_nms_id):
        if len(np.unique(arranged_pred_cls[tooth_nms_id == nms_id])) > 1:
            arranged_pred_cls[tooth_nms_id == nms_id] = np.min(arranged_pred_cls[tooth_nms_id == nms_id])

    def is_monotony(nums, up=True):
        for i in range(len(nums) - 1):
            if up:
                if nums[i] - nums[i + 1] > 0:
                    return False
            else:
                if nums[i] - nums[i + 1] < 0:
                    return False
        return True

    def possible_pred_cls(arranged_pred_cls, params, param):
        if param == 1:
            return arranged_pred_cls[params == 2]
        elif param == np.max(params):
            return arranged_pred_cls[params == param - 1]
        p_lt, p_gt = arranged_pred_cls[params == param - 1], arranged_pred_cls[params == param + 1]
        p = arranged_pred_cls[params == param]
        if p_lt == p:
            return p_gt
        elif p_gt == p:
            return p_lt

        if p_lt // 10 != p_gt // 10:
            # 选取个位数较大的
            if p_lt % 10 > p_gt % 10:
                return p_lt
            else:
                return p_gt
        else:
            return p_lt if p_lt != p else p_gt

    # Step 0 Fix id flip
    # 牙齿编号连续，但牙弓曲线参数出现不连续的情况
    # 左右分类出错
    # params 先减后增
    if id_flip:
        sign = -len(arranged_pred_cls) // 3
        # 记录连续相同牙齿的数量
        # same_nms_id_count = 0
        for i in range(1, len(arranged_pred_cls)):
            if tooth_nms_id[i] == tooth_nms_id[i - 1]:
                # same_nms_id_count += 1
                continue
            # Check the same part
            if (arranged_pred_cls[i] // 10) != (arranged_pred_cls[i - 1] // 10):
                sign = -sign
                # same_nms_id_count = 0
                continue

            # i starts from 0, params starts from 1
            param_i = params[i]
            param_i_1 = params[i - 1]
            if param_i - param_i_1 < sign < 0 or param_i - param_i_1 > sign > 0:
                # Error occurred
                # print('Error occurred, id flip', arranged_pred_cls[i], arranged_pred_cls, i, params, param_i, param_i_1,
                #       sign, tooth_nms_id)
                target_pred_cls = possible_pred_cls(arranged_pred_cls, params, param_i)
                if arranged_pred_cls[i] != target_pred_cls:
                    # for j in range(i - same_nms_id_count, i + 1):
                    #     # arranged_pred_cls[j] += 10
                    #     arranged_pred_cls[j] = target_pred_cls
                    arranged_pred_cls[tooth_nms_id == tooth_nms_id[i]] = target_pred_cls
                    return rearrange_labels(points, pred_seg, arranged_pred_cls, id_flip, times + 1)
            # elif param_i - param_i_1 > sign > 0:
            #     # Error occurred
            #     print('Error occurred, id flip', arranged_pred_cls[i], arranged_pred_cls, i, params, param_i, param_i_1,
            #           sign)
            #     target_pred_cls = possible_pred_cls(arranged_pred_cls, params, param_i)
            #     for j in range(i - same_nms_id_count, i + 1):
            #         # arranged_pred_cls[j] -= 10
            #         arranged_pred_cls[j] = target_pred_cls
            #     return rearrange_labels(points, pred_seg, arranged_pred_cls, id_flip)
            # same_nms_id_count = 0

        # 检查是否存在1号牙预测错误的情况
        # 判断：一侧存在>2号牙，不存在1号牙，且另一侧有至少2次1号牙非重叠预测
        for cls in [11, 21, 31, 41]:
            same_cls_patches_id = arranged_pred_cls == cls
            if len(same_cls_patches_id) == 0:
                continue
            if len(np.unique(tooth_nms_id[same_cls_patches_id])) > 1:
                # 根据param调整cls
                if (cls // 10) % 2 == 0:
                    if np.sum(arranged_pred_cls == cls - 10) > 0:
                        break
                    # 左侧颌，选param大的
                    max_nms_id = np.argmax(params[same_cls_patches_id])
                    arranged_pred_cls[tooth_nms_id == tooth_nms_id[same_cls_patches_id][max_nms_id]] = cls - 10
                    print(f'Flip {cls} => {cls - 10}')
                else:
                    if np.sum(arranged_pred_cls == cls + 10) > 0:
                        break
                    # 右侧颌，选param小的
                    min_nms_id = np.argmin(params[same_cls_patches_id])
                    arranged_pred_cls[tooth_nms_id == tooth_nms_id[same_cls_patches_id][min_nms_id]] = cls + 10
                    print(f'Flip {cls} => {cls + 10}')

        all_pred_cls_sort = np.argsort(arranged_pred_cls)
        pred_seg = pred_seg[all_pred_cls_sort, :]
        tooth_nms_id = tooth_nms_id[all_pred_cls_sort]
        params = params[all_pred_cls_sort]
        arranged_pred_cls = arranged_pred_cls[all_pred_cls_sort]

    # Step 1. Expand
    def do_expand(arranged_pred_cls, tooth_nms_id, params, first_round=True):
        hole_stack = []
        for i in range(1, len(arranged_pred_cls)):
            if np.sum(tooth_nms_id[:i] == tooth_nms_id[i]) > 0:
                pred_cls_id = np.argwhere(tooth_nms_id == tooth_nms_id[i]).squeeze()[0]
                arranged_pred_cls[i] = arranged_pred_cls[pred_cls_id]
                continue
            if arranged_pred_cls[i] - arranged_pred_cls[i - 1] > 1 and (arranged_pred_cls[i] // 10) == (
                    arranged_pred_cls[i - 1] // 10):
                hole_stack.append(i)
            if arranged_pred_cls[i] - arranged_pred_cls[i - 1] > 1 and (arranged_pred_cls[i] // 10) != (
                    arranged_pred_cls[i - 1] // 10):
                hole_stack = []
                if arranged_pred_cls[i] % 10 > 1:
                    hole_stack.append(i)

            # 不是一颗牙但标签相同
            if arranged_pred_cls[i - 1] == arranged_pred_cls[i]:
                max_of_this_part = (arranged_pred_cls[i] // 10) * 10 + 8

                # if len(hole_stack) > 0:
                #     # Hole
                #     for j in range(hole_stack[-1], i):
                #         arranged_pred_cls[j] -= 1
                #     hole_stack = hole_stack[0:-1]
                # elif np.sum(arranged_pred_cls == max_of_this_part) > 0:
                #     if first_round:
                #         # Move cur side
                #         j = i - 1
                #         expand_other_side = False
                #         while j >= 0:
                #             if tooth_nms_id[j] == tooth_nms_id[j + 1]:
                #                 arranged_pred_cls[j] = arranged_pred_cls[j + 1]
                #                 continue
                #             if arranged_pred_cls[j] // 10 != arranged_pred_cls[i] // 10:
                #                 break
                #             arranged_pred_cls[j] -= 1
                #             if arranged_pred_cls[j] % 10 == 0:
                #                 expand_other_side = True
                #             j -= 1
                #
                #         # 这一排从磨牙向切牙移动，下一次从切牙开始
                #         if expand_other_side:
                #             # 移动到另一侧切牙位置
                #             if (arranged_pred_cls[i] // 10) % 2 == 0:
                #                 arranged_pred_cls[arranged_pred_cls == arranged_pred_cls[j + 1]] -= 9
                #             else:
                #                 arranged_pred_cls[arranged_pred_cls == arranged_pred_cls[j + 1]] += 11
                #
                #             all_pred_cls_sort = np.argsort(arranged_pred_cls)
                #             tooth_nms_id = tooth_nms_id[all_pred_cls_sort]
                #             params = params[all_pred_cls_sort]
                #             arranged_pred_cls = arranged_pred_cls[all_pred_cls_sort]
                #             return do_expand(arranged_pred_cls, tooth_nms_id, params, False)
                #     else:
                #         print('FATAL', arranged_pred_cls)
                # else:
                #     # print(f'arranged_pred_cls[{i}] = {arranged_pred_cls[i]} Move forward => {arranged_pred_cls[i] + 1}')
                #     for j in range(i, len(arranged_pred_cls)):
                #         if arranged_pred_cls[i - 1] % 10 != arranged_pred_cls[j] % 10:
                #             break
                #         arranged_pred_cls[j] += 1

                if not np.sum(arranged_pred_cls == max_of_this_part) > 0:
                    print(f'arranged_pred_cls[{i}] = {arranged_pred_cls[i]} Move forward => {arranged_pred_cls[i] + 1}')
                    for j in range(i, len(arranged_pred_cls)):
                        if arranged_pred_cls[i - 1] % 10 != arranged_pred_cls[j] % 10:
                            break
                        arranged_pred_cls[j] += 1
                elif len(hole_stack) > 0:
                    # Hole
                    for j in range(hole_stack[-1], i):
                        arranged_pred_cls[j] -= 1
                    hole_stack = hole_stack[0:-1]
                else:
                    if first_round:
                        # Move cur side
                        j = i - 1
                        expand_other_side = False
                        while j >= 0:
                            if tooth_nms_id[j] == tooth_nms_id[j + 1]:
                                arranged_pred_cls[j] = arranged_pred_cls[j + 1]
                                continue
                            if arranged_pred_cls[j] // 10 != arranged_pred_cls[i] // 10:
                                break
                            arranged_pred_cls[j] -= 1
                            if arranged_pred_cls[j] % 10 == 0:
                                expand_other_side = True
                            j -= 1

                        # 这一排从磨牙向切牙移动，下一次从切牙开始
                        if expand_other_side:
                            # 移动到另一侧切牙位置
                            if (arranged_pred_cls[i] // 10) % 2 == 0:
                                arranged_pred_cls[arranged_pred_cls == arranged_pred_cls[j + 1]] -= 9
                            else:
                                arranged_pred_cls[arranged_pred_cls == arranged_pred_cls[j + 1]] += 11

                            all_pred_cls_sort = np.argsort(arranged_pred_cls)
                            tooth_nms_id = tooth_nms_id[all_pred_cls_sort]
                            params = params[all_pred_cls_sort]
                            arranged_pred_cls = arranged_pred_cls[all_pred_cls_sort]
                            return do_expand(arranged_pred_cls, tooth_nms_id, params, False)
                    else:
                        print('FATAL', arranged_pred_cls)

        return arranged_pred_cls, tooth_nms_id, params

    # print('Expand', tooth_nms_id, arranged_pred_cls)
    arranged_pred_cls, tooth_nms_id, params = do_expand(arranged_pred_cls, tooth_nms_id, params, True)
    # print('After expanding ', arranged_pred_cls, params)

    # Step 1.6 Fix same-group id error by shrink
    # repeat teeth id == 4, for numpy requires dimensions to be same
    teeth_groups = np.array([
        [1, 2, 2], [3, 3, 3], [4, 4, 5], [6, 7, 8]
    ])
    for i in range(1, len(arranged_pred_cls)):
        # Check the same part
        if (arranged_pred_cls[i] // 10) != (arranged_pred_cls[i - 1] // 10):
            continue
        if arranged_pred_cls[i] == arranged_pred_cls[i - 1]:
            continue
        tg_i = np.argwhere(teeth_groups == arranged_pred_cls[i] % 10)[0]
        tg_i_1 = np.argwhere(teeth_groups == arranged_pred_cls[i - 1] % 10)[0]
        if tg_i[0] == tg_i_1[0]:
            # same group
            if arranged_pred_cls[i] - arranged_pred_cls[i - 1] > 1:
                arranged_pred_cls[i] = arranged_pred_cls[i - 1] + 1
        else:
            # different group, check the first element id
            if tg_i[1] != 0:
                arranged_pred_cls[i] = (arranged_pred_cls[i] // 10) * 10 + teeth_groups[tg_i[0], 0]

    # Step 2. Reorder
    # 处理params非单调的情况
    # print(arranged_pred_cls, params)
    for part in [1, 0]:
        part_indices = ((arranged_pred_cls // 10) % 2 == part)
        part_cls = arranged_pred_cls[part_indices]
        part_param = params[part_indices]
        part_nms_id = tooth_nms_id[part_indices]
        # print(part_param)
        if not is_monotony(part_param, part == 1) or True:
            # 交换不单调的类别标签
            sorted_cls = np.argsort(part_param)
            if part == 0:
                sorted_cls = sorted_cls[::-1]

            # print('not monotony', sorted_cls, part_cls, part_cls[sorted_cls], part_nms_id)

            unique_cls = np.unique(part_cls)

            if len(unique_cls) != len(np.unique(part_nms_id)):
                # print('pass')
                continue

            last_nms_id = -1
            last_cls_id = 0
            for i in sorted_cls:
                param = part_param[i]
                # print(f'{i} param={param}, nms_id={part_nms_id[part_param == param]}, last_cls_id={last_cls_id}')
                cur_nms_id = part_nms_id[part_param == param]
                if last_nms_id == cur_nms_id:
                    continue
                last_nms_id = cur_nms_id
                arranged_pred_cls[tooth_nms_id == cur_nms_id] = unique_cls[last_cls_id]
                last_cls_id += 1

    # TRICK
    # 一边有8号牙，一边只有6号及以下的牙
    part_maximum = [np.max(arranged_pred_cls[(arranged_pred_cls // 10) % 2 == 0]),
                    np.max(arranged_pred_cls[(arranged_pred_cls // 10) % 2 == 1])]

    part_to_move = None
    if part_maximum[0] % 10 == 8 and part_maximum[1] % 10 < 7:
        part_to_move = 0
    if part_maximum[1] % 10 == 8 and part_maximum[0] % 10 < 7:
        part_to_move = 1
    if part_to_move is not None:
        part_idx = (arranged_pred_cls // 10) % 2 == part
        other_part_idx = (arranged_pred_cls // 10) % 2 == abs(part - 1)
        arranged_pred_cls[part_idx] -= 1
        if np.sum(arranged_pred_cls[part_idx] % 10 == 0) > 0:
            if np.sum(arranged_pred_cls[other_part_idx] % 10 == 1) > 0:
                arranged_pred_cls[other_part_idx] += 1
            arranged_pred_cls_part_idx = arranged_pred_cls[part_idx]
            arranged_pred_cls_part_idx[arranged_pred_cls[part_idx] % 10 == 0] = \
                (arranged_pred_cls[other_part_idx][0] // 10) * 10 + 1
            arranged_pred_cls[part_idx] = arranged_pred_cls_part_idx

    final_labels = np.zeros((pred_seg.shape[1],))
    for index, tooth_id in enumerate(tooth_nms_id):
        if np.all(final_labels[temp_labels == tooth_id] > 0):
            # print('repeated', tooth_id, np.unique(final_labels[temp_labels == tooth_id]))
            continue
        # print(f'tooth_id: {index} - {tooth_id}, label: {arranged_pred_cls[index]}')
        final_labels[temp_labels == tooth_id] = arranged_pred_cls[index]
    return final_labels


def rearrange_id_flip(points, pred_seg, arranged_pred_cls, tooth_nms_id, params, times=0):
    if times > 5:
        return None

    def possible_pred_cls(arranged_pred_cls, params, param):
        if param == 1:
            return arranged_pred_cls[params == 2]
        elif param == np.max(params):
            return arranged_pred_cls[params == param - 1]
        p_lt, p_gt = arranged_pred_cls[params == param - 1], arranged_pred_cls[params == param + 1]
        p = arranged_pred_cls[params == param]
        if p_lt == p:
            return p_gt
        elif p_gt == p:
            return p_lt

        if p_lt // 10 != p_gt // 10:
            # 选取个位数较大的
            if p_lt % 10 > p_gt % 10:
                return p_lt
            else:
                return p_gt
        else:
            return p_lt if p_lt != p else p_gt

    sign = -len(arranged_pred_cls) // 3
    # 记录连续相同牙齿的数量
    # same_nms_id_count = 0
    for i in range(1, len(arranged_pred_cls)):
        if tooth_nms_id[i] == tooth_nms_id[i - 1]:
            # same_nms_id_count += 1
            continue
        # Check the same part
        if (arranged_pred_cls[i] // 10) != (arranged_pred_cls[i - 1] // 10):
            sign = -sign
            # same_nms_id_count = 0
            continue

        # i starts from 0, params starts from 1
        param_i = params[i]
        param_i_1 = params[i - 1]
        if param_i - param_i_1 < sign < 0 or param_i - param_i_1 > sign > 0:
            # Error occurred
            # print('Error occurred, id flip', arranged_pred_cls[i], arranged_pred_cls, i, params, param_i, param_i_1,
            #       sign, tooth_nms_id)
            target_pred_cls = possible_pred_cls(arranged_pred_cls, params, param_i)
            if arranged_pred_cls[i] != target_pred_cls:
                arranged_pred_cls[tooth_nms_id == tooth_nms_id[i]] = target_pred_cls
                return rearrange_id_flip(points, pred_seg, arranged_pred_cls, tooth_nms_id, params, times + 1)

    params_argsort = np.argsort(params)
    params = params[params_argsort]
    arranged_pred_cls = arranged_pred_cls[params_argsort]
    tooth_nms_id = tooth_nms_id[params_argsort]
    return arranged_pred_cls, params, tooth_nms_id


def rearrange_labels2(points, pred_seg, pred_cls, id_flip=True):
    def fdi_diff(a, b):
        sign = -1 if (a // 10) % 2 == 0 else 1
        if a // 10 == b // 10:
            return sign * (a - b)
        return sign * (a % 10 - 1 + b % 10)

    # 剔除分类为背景的Patch
    pred_seg = pred_seg[pred_cls > 0, :]
    pred_cls = pred_cls[pred_cls > 0]
    # pred_quad = pred_quad[pred_cls > 0]

    # 若最大置信度<0.6，则认为该Patch为负样本
    # 不进行该步处理，则有可能出现NaN centroid
    good_seg_id = np.squeeze(np.argwhere(np.max(pred_seg, axis=1) >= 0.6))
    pred_seg = pred_seg[good_seg_id, :]
    pred_cls = pred_cls[good_seg_id]

    if len(pred_seg.shape) < 2:
        pred_seg = np.expand_dims(pred_seg, 0)
        pred_cls = np.expand_dims(pred_cls, 0)

    # 确定属于上颌还是下颌
    # 基于多数的牙齿会被正确分类，且牙列呈单调的先验
    upper_cnt, lower_cnt = np.sum(pred_cls < 30), np.sum(pred_cls > 30)

    # 纠正上下颌分类错误
    # FDI标签，上下颌对应位置相差20
    if upper_cnt > lower_cnt > 0:
        pred_cls[pred_cls > 30] -= 20
    elif lower_cnt > upper_cnt > 0:
        pred_cls[pred_cls < 30] += 20

    seg_labels = np.zeros((pred_seg.shape[1],))
    temp_labels, tooth_nms_id = seg_nms(pred_seg, seg_labels)

    nms_selected = []
    # 去重
    for nms_id in np.unique(tooth_nms_id):
        if points[temp_labels == nms_id].shape[0] > 0:
            nms_selected.append(np.argwhere(tooth_nms_id == nms_id)[0].squeeze())
    nms_selected = np.array(nms_selected)

    if len(nms_selected) > 0:
        tooth_nms_id = tooth_nms_id[nms_selected]
        pred_cls = pred_cls[nms_selected]
        pred_seg = pred_seg[nms_selected]

        # 求params
        # 计算牙列质心，按照tooth_nms_id的顺序排列
        tooth_nms_centroids = []
        for tooth_id in tooth_nms_id:
            if points[temp_labels == tooth_id].shape[0] > 0:
                ct = np.mean(points[temp_labels == tooth_id], axis=0)
                tooth_nms_centroids.append(ct)

        tooth_nms_centroids = np.array(tooth_nms_centroids)

        if len(tooth_nms_centroids) > 2:
            # 22.05.25 Note: 投影方法牙弓曲线受到质心数量的影响较大，故直接使用XY平面的坐标进行拟合
            point_on_curve = geometry.parameterize_points_on_xy_plane(tooth_nms_centroids)[0]
            point_on_curve = np.argsort(point_on_curve)
            params = np.zeros((len(point_on_curve),), dtype=np.int32)
            for i in range(len(point_on_curve)):
                params[point_on_curve[i]] = i + 1

            if id_flip:
                params_sort = np.argsort(pred_cls)
                params = params[params_sort]
                tooth_nms_id = tooth_nms_id[params_sort]
                pred_cls = pred_cls[params_sort]
                pred_seg = pred_seg[params_sort, :]
                results_tuple = rearrange_id_flip(points, pred_seg, pred_cls, tooth_nms_id, params)
                if results_tuple is None:
                    return rearrange_labels2(points, pred_seg, pred_cls, False)
                pred_cls, params, tooth_nms_id = results_tuple
            else:
                params_argsort = np.argsort(params)
                params = params[params_argsort]
                pred_cls = pred_cls[params_argsort]
                tooth_nms_id = tooth_nms_id[params_argsort]
                pred_seg = pred_seg[params_argsort]

            def group_hole_filling(group, group_params, group_standards):
                if len(group) == 0:
                    return group
                # group_params 应当连续
                max_cont_start = 0
                max_cont = 0
                cur_cont = 1
                cur_cont_start = 0
                for ii in range(1, len(group_params) + 1):
                    if ii == len(group_params) or abs(group_params[ii] - group_params[ii - 1]) > 1:
                        if ii == len(group_params) or max_cont < cur_cont:
                            max_cont = cur_cont
                            max_cont_start = cur_cont_start
                        cur_cont_start = ii
                        cur_cont = 0
                    cur_cont += 1

                if max_cont > len(group_standards) - max_cont_start:
                    max_cont = len(group_standards) - max_cont_start

                if max_cont == 0:
                    return group

                if np.min(group_standards % 10) >= 6:
                    group[max_cont_start:(max_cont_start + max_cont)] = group_standards[:max_cont]
                else:
                    holes = []
                    for std_grp in group_standards:
                        if np.sum(group[max_cont_start:(max_cont_start + max_cont)] == std_grp) == 0:
                            holes.append(std_grp)

                    repeated = len(group[max_cont_start:(max_cont_start + max_cont)]) \
                               - len(np.unique(group[max_cont_start:(max_cont_start + max_cont)]))
                    group[max_cont_start:(max_cont_start + max_cont)] = np.unique([
                        *np.unique(group[max_cont_start:(max_cont_start + max_cont)]), *holes[:repeated]])
                # print(f'{group} {max_cont_start}:{max_cont_start + max_cont} {repeated} {group_standards[:repeated]}')
                return group

            teeth_groups = [
                [1, 2], [3], [4, 5], [6, 7, 8]
            ]
            parts = np.unique(np.array([np.min(pred_cls), np.max(pred_cls)]) // 10) * 10

            for part in parts:
                # TRICK
                # if np.sum(pred_cls == part + 4) > 0 and np.sum(pred_cls == part + 5) == 0:
                #     pred_cls[pred_cls == part + 4] += 1
                for t_group in teeth_groups:
                    group_index = np.zeros((len(pred_cls),))
                    for grp_id in t_group:
                        group_index[pred_cls == (grp_id + part)] += 1
                    group_index = group_index > 0
                    pred_cls[group_index] = \
                        group_hole_filling(pred_cls[group_index], params[group_index], np.array(t_group) + part)

            # pred_cls should be 27-26...21-11-...16-17
            # rearrange pred_cls by hole-filling
            sorted_cls = np.sort(pred_cls)
            holes = []
            repeated_0 = len(pred_cls[(pred_cls // 10) % 2 == 0]) - len(np.unique(pred_cls[(pred_cls // 10) % 2 == 0]))
            repeated_1 = len(pred_cls[(pred_cls // 10) % 2 == 1]) - len(np.unique(pred_cls[(pred_cls // 10) % 2 == 1]))
            for i in range(1, 9):
                for part in parts + i:
                    if np.sum(sorted_cls == part) == 0:
                        holes.append(part)
            holes = np.sort(holes)

            if repeated_0 + repeated_1 <= len(holes[holes % 10 < 8]):
                holes_not_8 = holes[holes % 10 < 8]
                holes_0 = holes_not_8[(holes_not_8 // 10) % 2 == 0].tolist()
                holes_1 = holes_not_8[(holes_not_8 // 10) % 2 == 1].tolist()

                if repeated_0 >= len(holes_0):
                    holes_selected = holes_0 + holes_1[:repeated_0 - len(holes_0) + repeated_1]
                elif repeated_1 >= len(holes_1):
                    holes_selected = holes_1 + holes_0[:repeated_1 - len(holes_1) + repeated_0]
                else:
                    holes_selected = holes_0[:repeated_0] + holes_1[:repeated_1]
                    if len(holes_selected) == 0:
                        holes_selected = holes[:(repeated_0 + repeated_1)]
            elif repeated_0 + repeated_1 < len(holes):
                holes_0 = holes[(holes // 10) % 2 == 0].tolist()
                holes_1 = holes[(holes // 10) % 2 == 1].tolist()

                if repeated_0 >= len(holes_0):
                    holes_selected = holes_0 + holes_1[:repeated_0 - len(holes_0) + repeated_1]
                elif repeated_1 >= len(holes_1):
                    holes_selected = holes_1 + holes_0[:repeated_1 - len(holes_1) + repeated_0]
                else:
                    holes_selected = holes_0[:repeated_0] + holes_1[:repeated_1]
                    if len(holes_selected) == 0:
                        holes_selected = holes[:(repeated_0 + repeated_1)]
            else:
                holes_selected = holes

            pred_cls = np.unique([*sorted_cls, *holes_selected])

            for i in range(len(pred_cls) - 1):
                for j in range(i + 1, len(pred_cls)):
                    if fdi_diff(pred_cls[i], pred_cls[j]) > 0:
                        tmp = pred_cls[i]
                        pred_cls[i] = pred_cls[j]
                        pred_cls[j] = tmp

    final_labels = np.zeros((pred_seg.shape[1],))
    for index, tooth_id in enumerate(tooth_nms_id):
        if np.all(final_labels[temp_labels == tooth_id] > 0) or index >= len(pred_cls):
            continue
        # print(f'tooth_id: {index} - {tooth_id}, label: {arranged_pred_cls[index]}')
        final_labels[temp_labels == tooth_id] = pred_cls[index]
    return final_labels


def rearrange_labels3_backup(points, pred_seg, pred_cls):
    final_labels = np.zeros((points.shape[0],))
    pred_seg = pred_seg[pred_cls > 0, :]
    pred_cls = pred_cls[pred_cls > 0]
    # pred_quad = pred_quad[pred_cls > 0]

    # 若最大置信度<0.6，则认为该Patch为负样本
    # 不进行该步处理，则有可能出现NaN centroid
    good_seg_id = np.squeeze(np.argwhere(np.max(pred_seg, axis=1) >= 0.6))
    pred_seg = pred_seg[good_seg_id, :]
    pred_cls = pred_cls[good_seg_id]

    for i, seg in enumerate(pred_seg):
        final_labels[seg > 0.5] = pred_cls[i]

    return final_labels
