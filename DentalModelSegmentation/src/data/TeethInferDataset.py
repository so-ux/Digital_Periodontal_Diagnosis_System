"""
单个牙齿上/下颌模型数据集

在训练完成的推理时用，目的是保持与训练、测试代码的统一性
"""
import os.path

import numpy as np
import torch
import torch.utils.data as data
from numpy import ndarray
from scipy import linalg
from sklearn.neighbors import KDTree

from src.pointnet2.pointnet2_utils import furthest_point_sample
from src.utils.input import *
from src.utils.interpolation import point_labels_interpolation
from src.utils.mesh_subdivide import infer_mesh_subdivide
from src.utils.output import write_obj
from src.utils.pc_nn import pc_dist
from src.utils.pc_normalize import pc_normalize
from src.utils.preprocessing import align_model, align_model_on_xy
from src.utils.remove_braces import remove_braces_mesh, model_flat_indices
from src.vis.anatomy_colors import *
import open3d as o3d
import math


class TeethInferDataset(data.Dataset):
    all_data: ndarray

    def __init__(self, input_file):
        super().__init__()
        self.rot_matrix = np.eye(3)
        self.rot_matrix_inv = np.eye(3)
        self.patch_norm_c = None
        self.patch_norm_m = None
        self.down_results = None
        self.input_file = input_file

        v, f, vn = read_mesh(input_file, True)
        self.faces = f

        v = np.concatenate((v, vn), axis=1)

        # v = self.data_augmentation_rotation(v, input_file.__contains__('_U_'))
        self.origin_data = np.copy(v)

        # Align
        v[:, 0:6], self.rotated, self.rot_matrix_pca = align_model(v)
        self.rot_matrix_pca_inv = np.linalg.inv(self.rot_matrix_pca)
        # self.rotated = False
        # self.rot_matrix_pca_inv = np.eye(3)

        # 0. Subdivide mesh
        v, _, vn = infer_mesh_subdivide(v[:, 0:3], f)
        v = np.concatenate((v, vn), axis=1)

        # 1. Remove braces
        self.id_selected, f = remove_braces_mesh(v, f, 10)
        # self.id_selected = np.arange(0, len(v))
        v = v[self.id_selected]

        # 2. Normalize
        v, self.c, self.m = pc_normalize(v)

        # 3. FPS
        data_tensor = torch.Tensor(np.array([v[:, 0:3]])).cuda()

        # For each tooth cropping area, sample 2048 points
        # Assume that there are 16 teeth on each model
        fps_indices = furthest_point_sample(data_tensor, 2048 * 16)[0]
        self.fps_indices_16k = furthest_point_sample(data_tensor[:, fps_indices.to(dtype=torch.int64), :],
                                                     1024 * 16).detach().cpu().numpy()[0]
        fps_indices = fps_indices.detach().cpu().numpy()
        # fps_indices = np.unique(fps_indices)

        self.real_size = fps_indices.shape[0]
        self.fps_indices = fps_indices

        v = v[self.fps_indices]

        flat_indices = np.expand_dims(model_flat_indices(v), 1)
        v = np.concatenate((v, flat_indices), axis=1)

        self.all_data = np.array([v])

        data_tensor = torch.Tensor(np.array([v[:, 0:3]])).cuda()
        self.fps_indices_4k_from_all_data = furthest_point_sample(data_tensor, 256 * 16).detach().cpu().numpy()[0]

        # ----------------------------------------------------------------------------
        # After all_tooth_seg_net, teeth will be cropped by their predicted labels,
        # however at data init stage, the following variables are unknown
        self.patches_16k = None
        self.patches_32k = None
        self.patch_indices_16k = None
        self.patch_indices_32k = None
        self.patch_centroids_16k = None
        self.patch_centroids_32k = None
        self.patch_heatmap_16k = None
        self.patch_heatmap_32k = None

        self.class_results = None

    def data_augmentation_rotation(self, origin_data, haha):
        # Data augmentation
        origin_data_aug = np.copy(origin_data)
        mesh = o3d.geometry.PointCloud()
        mesh.points = o3d.utility.Vector3dVector(origin_data_aug[:, 0:3])
        mesh.normals = o3d.utility.Vector3dVector(origin_data_aug[:, 3:6])
        # angle_x = math.pi * (random.random() - 0.5) * 2
        # angle_y = math.pi * (random.random() - 0.5) * 2
        # angle_z = math.pi * (random.random() - 0.5) * 2
        # print('** Data augmentation, X {}, Y {}, Z {}'.format(
        #     180 * angle_x / np.pi,
        #     180 * angle_y / np.pi,
        #     180 * angle_z / np.pi
        # ))
        # angle_x = -71.54607797769255 * np.pi / 180
        # angle_y = -4.108654208486797 * np.pi / 180
        # angle_z = -158.92727482918684 * np.pi / 180
        angle_x = 0
        angle_y = np.pi if haha else 0
        angle_z = -90 * np.pi / 180

        mesh.rotate(mesh.get_rotation_matrix_from_xyz((angle_x, angle_y, angle_z)))

        origin_data_aug[:, 0:6] = np.concatenate((np.asarray(mesh.points), np.asarray(mesh.normals)), axis=-1)
        return origin_data_aug

    def data_rotation(self, rot_axis=np.array([0, 0, 1]), rot_angle=0., inplace=False):
        rot_matrix = linalg.expm(np.cross(np.eye(3), rot_axis / linalg.norm(rot_axis) * rot_angle))
        _data = np.copy(self.all_data[0, :, 0:6])
        _data[:, 0:3] = np.matmul(rot_matrix, _data[:, 0:3, np.newaxis])[:, :, 0]
        _data[:, 3:6] = np.matmul(rot_matrix, _data[:, 3:6, np.newaxis])[:, :, 0]

        if inplace:
            self.all_data[0, :, 0:6] = _data

            centroid = np.mean(self.origin_data[:, 0:3], axis=0)
            max_dis = np.sqrt(np.max(np.sum((self.origin_data[:, 0:3] - centroid) ** 2, axis=-1)))
            self.origin_data[:, 0:3] = (self.origin_data[:, 0:3] - centroid) / max_dis
            self.origin_data[:, 0:3] = np.matmul(rot_matrix, self.origin_data[:, 0:3, np.newaxis])[:, :, 0]
            self.origin_data[:, 0:3] = self.origin_data[:, 0:3] * max_dis + centroid
            self.origin_data[:, 3:6] = np.matmul(rot_matrix, self.origin_data[:, 3:6, np.newaxis])[:, :, 0]

            self.rot_matrix_pca_inv = np.linalg.inv(np.matmul(rot_matrix, self.rot_matrix_pca))

        return np.concatenate((_data, self.all_data[0, :, 6:]), axis=-1)

    def normalize_after_pred_seg(self, data_tensor, seg):
        positive_points = data_tensor[0, seg > 0.5, 0:3].data.cpu().numpy()
        positive_dists_centroid = np.mean(positive_points, axis=0)
        # 略微扩大一些范围，但不超过1倍，否则模型变小
        positive_dists_max = min(1, 1.2 * np.sqrt(np.max(np.sum(
            (positive_points - positive_dists_centroid) ** 2, axis=-1))))
        kdtree = KDTree(positive_points - positive_dists_centroid)
        indices = kdtree.query_radius([[0, 0, 0]], positive_dists_max, return_distance=False)[0]

        data_tensor = data_tensor[:, seg > 0.5, :][:, indices, :]

        data_tensor[0, :, 0:3] = (data_tensor[0, :, 0:3] - torch.Tensor(positive_dists_centroid).to(device='cuda',
                                                                                                    dtype=torch.float32)) / positive_dists_max
        return data_tensor, positive_dists_centroid, positive_dists_max

    def __getitem__(self, index):
        return self.all_data[index]

    def __len__(self):
        return len(self.all_data)

    def make_patches_centroids(self, pred_centroids, n_points=4096, _16k=False):
        """
        Make patches according to predicted class labels
        :param pred_centroids: [B, N, 3]
        :type pred_centroids:
        :param n_points: Point size of one patch
        :type n_points: int
        :return:
        :rtype:
        """
        points = self.__getitem__(0)[:self.real_size]
        if _16k:
            points = points[self.fps_indices_16k]

        p_points, p_indices, p_centroids = [], [], []
        for centroid in pred_centroids:
            # Find the nearest 4096 points
            sorted_indices = pc_dist(points[:, 0:3], centroid)[:n_points]
            p_indices.append(sorted_indices)
            # Normalize
            patch_points_norm = points[sorted_indices, :]
            patch_points_c = np.mean(patch_points_norm[:, 0:3], axis=0)
            patch_points_m = np.max(np.sqrt(
                np.sum((patch_points_norm[:, 0:3] - patch_points_c) ** 2, axis=1)), axis=0)
            patch_points_norm[:, 0:3] = (patch_points_norm[:, 0:3] - patch_points_c) / patch_points_m
            p_points.append(patch_points_norm)
            p_centroids.append((centroid - patch_points_c) / patch_points_m)

        if _16k:
            self.patches_16k = np.array(p_points)
            self.patch_indices_16k = np.array(p_indices)
            self.patch_centroids_16k = np.array(p_centroids)
            self.patch_centroids_16k = np.expand_dims(self.patch_centroids_16k, 1)
        else:
            self.patches_32k = np.array(p_points)
            self.patch_indices_32k = np.array(p_indices)
            self.patch_centroids_32k = np.array(p_centroids)
            self.patch_centroids_32k = np.expand_dims(self.patch_centroids_32k, 1)

        # self.patches[:, :, 6] /= (np.max(self.patches[:, :, 6], axis=-1, keepdims=True) + 1e-9)

    def return_back_interpolation(self, final_labels):
        """
        Return predicted model back to original size and indices

        The output model's shape/size/indices are required to be identical to the original one's,
        so this step is necessary.
        This method will roll back operations in order: Patch -> FPS -> Normalize -> Add braces.
        """
        if self.patches_32k is None:
            origin_cls = np.zeros((self.origin_data.shape[0],))
            self.class_results = origin_cls
            return

        # return patch back
        # 网格细分的新增顶点均位于列表尾部，因此选择[0, len(self.origin_data)]即为原模型的顶点
        id_selected = self.id_selected[self.id_selected < len(self.origin_data)]
        origin_cls_no_braces = np.zeros((self.origin_data[id_selected].shape[0],))

        down_points = self.all_data[0, :, 0:3]
        down_points = np.matmul(down_points, self.rot_matrix_inv.T)
        # Normalize back
        down_points = down_points * self.m + self.c
        down_points = np.matmul(self.rot_matrix_pca_inv, down_points[:, :, np.newaxis])[:, :, 0]

        # 对每颗牙进行插值
        for tooth_id in np.unique(final_labels):
            pred_seg_on_whole_points = np.zeros((down_points.shape[0],))
            pred_seg_on_whole_points[final_labels == tooth_id] = 1

            interpolation_result = \
                point_labels_interpolation(self.origin_data[id_selected, 0:3], down_points, pred_seg_on_whole_points)[
                    0]  # [N]

            origin_cls_no_braces[interpolation_result >= 0.5] = tooth_id

        # Add brace
        origin_cls = np.zeros((self.origin_data.shape[0],))
        origin_cls[id_selected] = origin_cls_no_braces
        self.class_results = origin_cls

    def save_output(self, obj_file):
        # obj_file = os.path.join(output_dir, 'teeth_ins_seg.obj')
        verts = self.origin_data[:, 0:3]
        faces = self.faces
        colors = np.ones((verts.shape[0], 3)) * 0.5
        # colors_down = np.ones((self.all_data[0].shape[0], 3)) * 0.5
        anatomyColors = AnatomyColors()
        for tooth_id in np.unique(self.class_results):
            tooth_id = int(tooth_id)
            colors[self.class_results == tooth_id] = anatomyColors.get_tooth_district_color(tooth_id, False)
            # colors_down[self.down_results == tooth_id] = anatomyColors.get_tooth_color(tooth_id, True)

        if not os.path.exists(os.path.dirname(obj_file)):
            os.makedirs(os.path.dirname(obj_file), exist_ok=True)
        write_obj(obj_file, np.concatenate((verts, colors), axis=1), faces, None)
        # write_obj(obj_file.replace('.obj', '.down.obj'), np.concatenate((self.all_data[0, :, 0:3] * self.m + self.c, colors_down), axis=1), None, self.all_data[0, :, 3:])

    def save_output_test(self):
        mesh = trimesh.load(self.input_file, process=False)
        verts = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        labels = self.class_results
        return verts, labels# faces,

    def get_data_tensor(self, full_rotation=False, four_k=False):
        if four_k:
            indices = self.fps_indices_4k_from_all_data
        else:
            indices = np.arange(0, self.all_data.shape[1])

        # 绕z轴旋转90deg
        if not full_rotation:
            tensor = torch.Tensor(np.array([self.all_data[0, indices, :]]))
        else:
            if self.rotated:
                tensor = torch.Tensor(np.array([
                    self.all_data[0, indices, :],
                    self.data_rotation(rot_angle=math.pi / 2, inplace=False)[indices, :],
                    self.data_rotation(rot_angle=math.pi, inplace=False)[indices, :],
                    self.data_rotation(rot_angle=-math.pi / 2, inplace=False)[indices, :]
                ]))
            else:
                tensor = torch.Tensor(np.array([self.all_data[0, indices, :]]))
        tensor = tensor.to("cuda", dtype=torch.float32, non_blocking=True)
        return tensor

    def set_rot(self, rot_id, positive_points):
        """
        从n个旋转了90度的模型中挑一个最好的
        输入模型的时候以将up方向转正，因此投影到xy平面来做
        """
        if rot_id == 1:
            self.data_rotation(rot_angle=math.pi / 2, inplace=True)
        elif rot_id == 2:
            self.data_rotation(rot_angle=math.pi, inplace=True)
        elif rot_id == 3:
            self.data_rotation(rot_angle=-math.pi / 2, inplace=True)

        # FPS
        data_tensor = torch.Tensor(np.array([positive_points[:, 0:3]])).cuda()
        fps_count = min(len(positive_points), 2000)
        fps_indices = furthest_point_sample(data_tensor, fps_count).detach().cpu().numpy()[0]

        # Fit ellipse
        rot_angle = align_model_on_xy(positive_points[fps_indices, 0:2]) % np.pi
        rot_angle_threshold = 10 * np.pi / 180
        # 如果rot_angle在n*90deg的10deg以内，则对齐到n*90deg
        if np.abs(rot_angle - np.pi / 2) < rot_angle_threshold:
            rot_angle = np.pi / 2
        elif np.abs(rot_angle - np.pi) < rot_angle_threshold:
            rot_angle = np.pi
        elif np.abs(rot_angle - 3 * np.pi / 2) < rot_angle_threshold or np.abs(
                -np.pi / 2 - rot_angle) < rot_angle_threshold:
            rot_angle = -np.pi / 2
        elif np.abs(rot_angle - 2 * np.pi) < rot_angle_threshold or np.abs(rot_angle) < rot_angle_threshold:
            rot_angle = 0
        print('Rot', rot_angle)

        # 把self.origin_data进行旋转
        self.data_rotation(rot_angle=rot_angle, inplace=True)

    def set_rotate_matrix_inplace(self, rot_matrix):
        self.all_data[0, :, 0:3] = np.matmul(self.all_data[0, :, 0:3], rot_matrix.T)
        self.all_data[0, :, 3:6] = np.matmul(self.all_data[0, :, 3:6], rot_matrix.T)

        self.rot_matrix = rot_matrix
        self.rot_matrix_inv = np.linalg.inv(rot_matrix)

        # centroid = np.mean(self.origin_data[:, 0:3], axis=0)
        # max_dis = np.sqrt(np.max(np.sum((self.origin_data[:, 0:3] - centroid) ** 2, axis=-1)))
        # self.origin_data[:, 0:3] = (self.origin_data[:, 0:3] - centroid) / max_dis
        # self.origin_data[:, 0:3] = np.matmul(self.origin_data[:, 0:3], rot_matrix.T)
        # self.origin_data[:, 0:3] = self.origin_data[:, 0:3] * max_dis + centroid
        # self.origin_data[:, 3:6] = np.matmul(self.origin_data[:, 3:6], rot_matrix.T)

    def get_patches_tensor(self, _16k=False):
        if _16k:
            centroids_pointer = self.patches_16k[:, :, 0:3] - self.patch_centroids_16k
            self.patch_heatmap_16k = np.exp(-2 * np.sum(centroids_pointer ** 2, axis=2))
            self.patch_heatmap_16k = np.expand_dims(self.patch_heatmap_16k, 2)
            patches = np.concatenate((self.patches_16k, self.patch_heatmap_16k), axis=2)
        else:
            centroids_pointer = self.patches_32k[:, :, 0:3] - self.patch_centroids_32k
            self.patch_heatmap_32k = np.exp(-2 * np.sum(centroids_pointer ** 2, axis=2))
            self.patch_heatmap_32k = np.expand_dims(self.patch_heatmap_32k, 2)
            patches = np.concatenate((self.patches_32k, self.patch_heatmap_32k), axis=2)

        # patches = np.concatenate((self.patches, dist_heatmap, centroids_pointer), axis=2)

        tensor = torch.Tensor(patches)
        tensor = tensor.to("cuda", dtype=torch.float32, non_blocking=True)
        return tensor

    def sample_from_new_centroid(self, patch_id, centroid, n_points=4096):
        """
        为Patch设置CFDP采样的新质心，返回裁剪结果

        :param n_points:
        :type n_points:
        :param patch_id:
        :type patch_id:
        :param centroid:
        :type centroid:
        :return:
        :rtype:
        """

        # 还原至原Patch位置
        centroid = centroid * self.patch_norm_m[patch_id] + self.patch_norm_c[patch_id]

        points = self.__getitem__(0)[:self.real_size, 0:6]

        p_points, p_indices, p_centroids = [], [], []
        # Find the nearest 4096 points
        sorted_indices = pc_dist(points[:, 0:3], centroid)[:n_points]
        p_indices.append(sorted_indices)
        # Normalize
        patch_points_norm = points[sorted_indices, :]
        patch_points_c = np.mean(patch_points_norm[:, 0:3], axis=0)
        patch_points_m = np.max(np.sqrt(
            np.sum((patch_points_norm[:, 0:3] - patch_points_c) ** 2, axis=1)), axis=0)
        patch_points_norm[:, 0:3] = (patch_points_norm[:, 0:3] - patch_points_c) / patch_points_m
        p_points.append(patch_points_norm)
        p_centroids.append((centroid - patch_points_c) / patch_points_m)

        return patch_points_norm, np.array(p_centroids)

    def get_cls_patches_tensor(self, pred_seg, down_sample=False, point_size=16384, sample_size=4096):
        """
        获取用于牙齿分类的Tensor

        :param pred_seg: [B, N, 1]
        :type pred_seg: np.ndarray
        :return:
        :rtype:
        """
        points = self.__getitem__(0)[:self.real_size, 0:6]
        if down_sample:
            points = points[self.fps_indices_16k]
        patch_indices = self.patch_indices_32k if not down_sample else self.patch_indices_16k

        if down_sample:
            points_tensor = torch.Tensor(points).cuda()
            points_fps_indices = \
                furthest_point_sample(points_tensor[:, 0:3].unsqueeze(0).contiguous(), point_size).detach().cpu().numpy()[0]
        else:
            points_fps_indices = np.arange(0, len(points))

        all_data = []
        all_resamples = []
        for patch_id in range(pred_seg.shape[0]):
            points_with_seg = np.zeros((points.shape[0], points.shape[1] + 1))
            points_with_seg[:, :points.shape[1]] = points
            points_with_seg[patch_indices[patch_id], points.shape[1]] = np.squeeze(pred_seg[patch_id])

            # if dist_heatmap is not None:
            #     points_with_seg = np.concatenate([points_with_seg, dist_heatmap], -1)

            all_data.append(points_with_seg[points_fps_indices, :])

            resample = np.zeros((patch_indices[patch_id].shape[0], 7))
            resample[:, 0:6] = points[patch_indices[patch_id]]
            resample[:, 0:3], _, _ = pc_normalize(resample[:, 0:3])
            resample[:, -1] = np.squeeze(pred_seg[patch_id])
            all_resamples.append(resample)

        all_resamples = torch.FloatTensor(np.array(all_resamples))

        batch_size = 8
        if down_sample:
            all_resamples_down = torch.zeros((all_resamples.shape[0], sample_size, all_resamples.shape[2]), dtype=torch.float32)
            fps_indices = torch.zeros((all_resamples.shape[0], sample_size), dtype=torch.int64)
            for bi in range(math.ceil(len(all_resamples) / batch_size)):
                indices = \
                    furthest_point_sample(all_resamples[bi * batch_size:(bi + 1) * batch_size].cuda(), sample_size).detach().cpu()
                fps_indices[bi * batch_size:(bi + 1) * batch_size] = indices
            for bi in range(len(fps_indices)):
                all_resamples_down[bi] = all_resamples[bi, fps_indices[bi], :]
            all_resamples = all_resamples_down

        return torch.FloatTensor(np.array(all_data)).cuda(), all_resamples.cuda()

    def remove_curvatures_on_tooth(self, pred_seg):
        """
        移除牙窝和牙齿表面的曲率信息

        :param pred_seg:
        :type pred_seg:
        :return:
        :rtype:
        """
        pred_seg_int = np.array(pred_seg)
        pred_seg_int[pred_seg_int > 0.5] = 1 #crown
        pred_seg_int[pred_seg_int < 1] = 0
        pred_seg_int = np.asarray(pred_seg_int, dtype=np.int32)

        if np.sum(pred_seg_int > 0) < 10:
            return

        nd_data = self.all_data[0, :, :]

        if len(nd_data) > 500:
            # 缩小一圈pred_seg
            kdtree = KDTree(nd_data[:, 0:3])
            indices = kdtree.query(nd_data[pred_seg_int > 0, 0:3], 10, return_distance=False)
            neighbour_zero_count = np.sum(pred_seg_int[indices], axis=1) - 1
            indices = np.zeros(pred_seg.shape, dtype=np.int32)
            indices[pred_seg_int > 0] = neighbour_zero_count

            self.all_data[0, indices > 0, 6] = 0
            # self.all_data[0, :, 6] /= np.max(self.all_data[0, :, 6])
