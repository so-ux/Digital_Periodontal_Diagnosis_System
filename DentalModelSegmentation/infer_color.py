import argparse
import json
import math
import os.path
import sys
import time
import numpy as np
import torch
from loguru import logger
from scipy import linalg
from sklearn.neighbors import KDTree
from tqdm import tqdm
from src.data.TeethInferDataset import TeethInferDataset
from src.models import *
from src.models.pct_models import Pct, PctPatchRefine
from src.utils import postprocessing
from src.utils.cfdp import get_clustered_centroids
from src.utils.interpolation import batch_point_labels_interpolation
from src.vis.vis_teeth_cls import *
from glob import glob
import os
from pymeshlab import Mesh, MeshSet
import trimesh
from torchsummary import summary

# =============== Global variables ===============
model_rotation = Pct({})
model_centroid_prediction = PointnetCentroidPredictionNet()
model_centroid_prediction_10k = PointnetCentroidPredictionNet(out_points=1024)
model_all_tooth_seg = AllToothSegNet(3)
model_refine_pct_16k = PctPatchRefine(2)
model_refine_pct_32k = PctPatchRefine()

model_cls = ToothClsNet()

torch.random.manual_seed(2000228)
torch.cuda.manual_seed(2000228)


# ============= Program init methods =============
def init_logger():
    logger.remove()
    logger.add(sys.stdout, colorize=True, enqueue=True, backtrace=True, diagnose=True)


def make_arg_parser() -> argparse.ArgumentParser:
    """
    Prepare CLI arg parser

    :return: argparse instance
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser()
    # Basic information
    parser.add_argument('--input', '-i', type=str, help='Path of input model', required=True)
    parser.add_argument('--output', '-o', type=str, help='Path of output results', required=False)
    parser.add_argument('--model_path', '-m', type=str, help='Path of trained models', required=False,
                        default='../model')
    return parser


def check_and_prepare_files() -> None:
    """
    Check and prepare needed files
    """
    # if not os.path.exists(ARGS.input) or os.path.isdir(ARGS.input):
    #     logger.error('Invalid input [{}]', ARGS.input)
    #     exit(-1)

    if not os.path.exists(ARGS.output):
        os.makedirs(ARGS.output, exist_ok=True)

    if not os.path.exists(ARGS.model_path):
        logger.error('Model path [{}] not exists', ARGS.model_path)
        exit(-1)


# =========== Data persistence methods ===========
def load_checkpoint(model_path) -> None:
    model_rotation_file = os.path.join(model_path, 'model_rotation')
    model_centroid_prediction_file = os.path.join(model_path, 'model_pred_centroid')
    model_centroid_prediction_10k_file = os.path.join(model_path, 'model_pred_centroid_10k')
    model_all_tooth_seg_file = os.path.join(model_path, 'model_all_tooth_seg')
    model_refine_file = os.path.join(model_path, 'model_pct_refine_16k')
    model_refine_file_32k = os.path.join(model_path, 'model_pct_refine_32k')
    model_cls_file = os.path.join(model_path, 'model_cls')

    model_rotation.load_state_dict(torch.load(model_rotation_file))
    model_centroid_prediction_10k.load_state_dict(torch.load(model_centroid_prediction_10k_file))
    model_centroid_prediction.load_state_dict(torch.load(model_centroid_prediction_file))
    model_all_tooth_seg.load_state_dict(torch.load(model_all_tooth_seg_file))
    model_refine_pct_16k.load_state_dict(torch.load(model_refine_file))
    model_refine_pct_32k.load_state_dict(torch.load(model_refine_file_32k))
    model_cls.load_state_dict(torch.load(model_cls_file))


def label2mesh(mesh, label, labels):
    vert_idxes = np.arange(mesh.vertices.shape[0])
    vert_idxes = vert_idxes[labels == label]
    faces = mesh.faces[np.isin(mesh.faces, vert_idxes).all(axis=1)]
    mesh_split = trimesh.Trimesh(vertices=mesh.vertices, faces=faces)
    return mesh_split


def mesh_split(path):
    mesh = trimesh.load(path)
    edges = trimesh.geometry.faces_to_edges(mesh.faces)
    labels = trimesh.graph.connected_component_labels(edges)

    mesh1 = label2mesh(mesh, 0, labels)
    mesh2 = label2mesh(mesh, 1, labels)

    mesh1.export(os.path.join(os.path.dirname(path), '1.stl'))
    mesh2.export(os.path.join(os.path.dirname(path), '2.stl'))


def rotation_matrix_torch(axis, theta):
    """
    Generalized 3d rotation via Euler-Rodriguez formula, https://www.wikiwand.com/en/Euler%E2%80%93Rodrigues_formula
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    if torch.sqrt(torch.dot(axis, axis)) < 1e-8:
        return torch.eye(3, requires_grad=True).cuda()
    axis = axis / torch.sqrt(torch.dot(axis, axis))

    a = torch.cos(theta / 2.0)
    b, c, d = -axis * torch.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return torch.tensor([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                         [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                         [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def apply_rotation(nd_data, axis_up, axis_forward):
    rot_axis = np.cross(axis_up, np.array([0, 0, 1]))
    cos_theta = np.sum(axis_up * np.array([0, 0, 1])) / np.linalg.norm(axis_up)
    rot_angle = np.arccos(np.clip(cos_theta, -1, 1))
    rot_matrix = linalg.expm(np.cross(np.eye(3), rot_axis / linalg.norm(rot_axis) * rot_angle))
    nd_data[:, 0:3] = np.matmul(rot_matrix, nd_data[:, 0:3, np.newaxis])[:, :, 0]
    nd_data[:, 3:6] = np.matmul(rot_matrix, nd_data[:, 3:6, np.newaxis])[:, :, 0]
    axis_up = np.matmul(rot_matrix, axis_up)
    axis_forward = np.matmul(rot_matrix, axis_forward)

    rot_axis = np.cross(axis_forward, [0, -1, 0])
    cos_theta = np.sum(axis_forward * np.array([0, -1, 0])) / np.linalg.norm(axis_forward)
    rot_angle = np.arccos(np.clip(cos_theta, -1, 1))
    rot_matrix = linalg.expm(np.cross(np.eye(3), rot_axis / linalg.norm(rot_axis) * rot_angle))
    nd_data[:, 0:3] = np.matmul(rot_matrix, nd_data[:, 0:3, np.newaxis])[:, :, 0]
    nd_data[:, 3:6] = np.matmul(rot_matrix, nd_data[:, 3:6, np.newaxis])[:, :, 0]
    axis_up = np.matmul(rot_matrix, axis_forward)
    axis_forward = np.matmul(rot_matrix, axis_forward)

    return nd_data, axis_up, axis_forward


def inference(input, output, model_path='../model', rotation=True, debug_vis=False, write_mesh=False):
    inf_start_time = time.time()
    ##########################################
    # Init
    ##########################################
    load_checkpoint(model_path)
    batch_size = 8

    ##########################################
    # Data init
    ##########################################
    infer_set = TeethInferDataset(input)

    ##########################################
    # Model
    ##########################################
    model_rotation.cuda()
    model_rotation.eval()
    model_all_tooth_seg.cuda()
    model_all_tooth_seg.eval()
    model_centroid_prediction.cuda()
    model_centroid_prediction.eval()
    model_centroid_prediction_10k.cuda()
    model_centroid_prediction_10k.eval()
    model_refine_pct_16k.cuda()
    model_refine_pct_16k.eval()
    model_refine_pct_32k.cuda()
    model_refine_pct_32k.eval()
    model_cls.cuda()
    model_cls.eval()

    with torch.no_grad():
        start_time = time.time()

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # % Stage 0: rotate dental model
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if rotation:
            data_tensor = infer_set.get_data_tensor(False, four_k=True)
            print('data_tensor', data_tensor.shape)
            pred_axis_up, pred_axis_forward = model_rotation(data_tensor[:, :, 0:6].permute(0, 2, 1))
            # print('model_rotation',summary(model_rotation,input_size=(data_tensor[:, :, 0:6].permute(0, 2, 1)).shape))
            print('pred_axis_up', pred_axis_up.shape)
            pred_axis_up, pred_axis_forward = pred_axis_up[0], pred_axis_forward[0]

            standard_ax_up = torch.Tensor([0, 0, 1]).cuda()
            standard_ax_forward = torch.Tensor([0, -1, 0]).cuda()

            if torch.dot(pred_axis_up, standard_ax_up) < 0.99:
                R_ax_up = torch.nn.functional.normalize(torch.cross(pred_axis_up, standard_ax_up), p=2, dim=-1)
                R_angle_up = torch.arccos(
                    torch.dot(pred_axis_up, standard_ax_up) / (torch.norm(pred_axis_up) * torch.norm(standard_ax_up)))
                R_up = rotation_matrix_torch(R_ax_up, R_angle_up).cuda()

                pred_axis_forward = torch.matmul(R_up, pred_axis_forward.unsqueeze(1)).squeeze()
            else:
                R_up = torch.eye(3).cuda()

            pred_axis_forward[-1] = 0

            data_tensor[0, :, 0:3] = torch.matmul(data_tensor[0, :, 0:3], R_up.T)
            data_tensor[0, :, 3:6] = torch.matmul(data_tensor[0, :, 3:6], R_up.T)

            if torch.dot(pred_axis_forward, standard_ax_forward) < 0.99:
                R_ax_forward = torch.nn.functional.normalize(
                    torch.cross(pred_axis_forward, standard_ax_forward), p=2, dim=-1)
                R_angle_forward = torch.arccos(torch.dot(pred_axis_forward, standard_ax_forward) / (
                        torch.norm(pred_axis_forward) * torch.norm(standard_ax_forward)))
                R_forward = rotation_matrix_torch(R_ax_forward, R_angle_forward).cuda()
            else:
                R_forward = torch.eye(3).cuda()

            data_tensor[0, :, 0:3] = torch.matmul(data_tensor[0, :, 0:3], R_forward.T)
            data_tensor[0, :, 3:6] = torch.matmul(data_tensor[0, :, 3:6], R_forward.T)

            if debug_vis:
                np.savetxt(
                    f'/public/bme/home/v-tanmh/DentalModelSegmentation/txt/{os.path.basename(output)}.rot.xyz',
                    data_tensor.data.cpu().numpy()[0, :, 0:6])

            rot_matrix = torch.matmul(R_forward, R_up)

            infer_set.set_rotate_matrix_inplace(rot_matrix.detach().cpu().numpy())

            # logger.info('Stage 0: "rotation" done, spent {}s', time.time() - start_time)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # % Stage 1: all tooth seg
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        data_tensor = infer_set.get_data_tensor(False)
        start_time = time.time()
        pred_seg = torch.argmax(model_all_tooth_seg(data_tensor[:, :, 0:6]), dim=1)
        pred_seg = pred_seg.detach().cpu().numpy()[0]
        print('all tooth seg--pred_seg', pred_seg.shape)
        # print('model_all_tooth_seg',summary(model_all_tooth_seg,input_size=data_tensor[:, :, 0:6].shape))
        if debug_vis:
            np.savetxt(
                f'/public/bme/home/v-tanmh/DentalModelSegmentation/txt/{os.path.basename(output)}.pn.txt',
                vis_teeth_seg(data_tensor[0, :, 0:6].data.cpu().numpy(), pred_seg))

        infer_set.remove_curvatures_on_tooth(pred_seg)
        data_tensor = infer_set.get_data_tensor(False)
        print('data_tensor', data_tensor.shape)
        seg = np.array(pred_seg)
        seg[seg > 0.5] = 1
        seg[seg < 1] = 0
        if np.max(seg) < 0.5:
            seg[seg < 0.5] = 1
        else:

            nd_data = data_tensor[0].data.cpu().numpy()[:, 0:3]
            tree = KDTree(nd_data)
            neighbours = tree.query_radius(nd_data[seg > 0], 0.15, return_distance=False)
            for n in neighbours:
                seg[n] = 1

        # 0714: ta_tensor, positive_dists_centroid, positive_dists_max = infer_set.normalize_after_pred_seg(data_tensor, seg)

        # logger.info('Stage 1: "all tooth seg" done, spent {}s', time.time() - start_time)

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # % Stage 2: centroid prediction
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        start_time = time.time()

        # kp_reg1, kp_score, seed_xyz = model_centroid_prediction(data_tensor, False, False)
        kp_reg, kp_score, seed_xyz = model_centroid_prediction(data_tensor[:, seg > 0.5, :], False, False)
        # print('model_centroid_prediction',summary(model_centroid_prediction,input_size=data_tensor[:, seg > 0.5, :].shape))
        print('kp_reg1', kp_reg.shape)
        print('kp_score1', kp_score.shape)
        print('seed_xyz1', seed_xyz.shape)
        kp_reg2, kp_score2, seed_xyz = model_centroid_prediction_10k(data_tensor[:, seg > 0.5, :], False, False)
        print('kp_reg2', kp_reg.shape)
        print('kp_score2', kp_score.shape)
        print('seed_xyz2', seed_xyz.shape)

        kp_reg = torch.cat([kp_reg, kp_reg2], dim=1)
        print('kp_reg', kp_reg.shape)
        kp_score = torch.cat([kp_score, kp_score2], dim=1)

        kp_reg = kp_reg.detach().cpu().numpy()
        kp_score = kp_score.detach().cpu().numpy()

        kp_reg = kp_reg[0, kp_score[0] < 0.2, :]
        if debug_vis:
            np.savetxt(f'/public/bme/home/v-tanmh/DentalModelSegmentation/txt/{os.path.basename(output)}.ct.xyz',
                       kp_reg)

        kp_reg = get_clustered_centroids(np.asarray(kp_reg, dtype=np.float64))
        print('kp_reg', kp_reg.shape)

        if len(kp_reg) == 0:
            final_labels = np.zeros((pred_seg.shape[0],))
        else:
            if debug_vis:
                np.savetxt(f'/public/bme/home/v-tanmh/DentalModelSegmentation/txt/{os.path.basename(output)}.ct.txt',
                           vis_teeth_centroids(data_tensor[0, :, 0:6].data.cpu().numpy(), kp_reg))

            # logger.info('Stage 2: "centroid prediction" done, spent {}s', time.time() - start_time)

            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            # % Stage 3: patches refine
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            start_time = time.time()

            # Stage 2 preprocess: split patches
            infer_set.make_patches_centroids(kp_reg, _16k=True)
            infer_set.make_patches_centroids(kp_reg, _16k=False)

            # Stage 2: patch inference
            patches_tensor_16k = infer_set.get_patches_tensor(True)
            print('patches_tensor_16k', patches_tensor_16k.shape)
            print('patches_tensor_16k', patches_tensor_16k[0])
            patches_tensor_32k = infer_set.get_patches_tensor(False)
            print('patches_tensor_32k', patches_tensor_32k.shape)
            print('patches_tensor_32k', patches_tensor_32k[0])
            all_pred_seg = np.array([])
            all_pred_seg_16k = np.array([])
            for i in range(math.ceil(len(patches_tensor_32k) / batch_size)):
                if i * batch_size == len(patches_tensor_32k) - 1:
                    pred_seg1 = model_refine_pct_16k(
                        patches_tensor_16k[i * batch_size:(i + 1) * batch_size].repeat(2, 1, 1).transpose(2,
                                                                                                          1))  # [:, 0, :]
                    pred_seg1 = torch.argmax(pred_seg1, dim=1)
                    pred_seg1 = pred_seg1[0:1]

                    pred_seg2 = model_refine_pct_32k(
                        patches_tensor_32k[i * batch_size:(i + 1) * batch_size].repeat(2, 1, 1).transpose(2,
                                                                                                          1))  # [:, 0, :]
                    pred_seg2 = torch.argmax(pred_seg2, dim=1)
                    pred_seg2 = pred_seg2[0:1]
                else:
                    pred_seg1 = model_refine_pct_16k(
                        patches_tensor_16k[i * batch_size:(i + 1) * batch_size].transpose(2, 1))  # [:, 0, :]
                    pred_seg1 = torch.argmax(pred_seg1, dim=1)

                    pred_seg2 = model_refine_pct_32k(
                        patches_tensor_32k[i * batch_size:(i + 1) * batch_size].transpose(2, 1))  # [:, 0, :]
                    pred_seg2 = torch.argmax(pred_seg2, dim=1)

                # pred_seg1 +=> pred_seg2
                pred_seg1 = pred_seg1.detach().cpu().numpy()
                pred_seg2 = pred_seg2.detach().cpu().numpy()

                patch_idx_16k = infer_set.patch_indices_16k[i * batch_size:(i + 1) * batch_size]
                patch_idx_32k = infer_set.patch_indices_32k[i * batch_size:(i + 1) * batch_size]
                points_16k = infer_set.__getitem__(0)[patch_idx_16k][:, :, 0:3]
                points_32k = infer_set.__getitem__(0)[patch_idx_32k][:, :, 0:3]
                feats_16k_32k = batch_point_labels_interpolation(points_32k, points_16k, pred_seg1)
                print('feats_16k_32k', feats_16k_32k.shape)
                feats_16k_32k[feats_16k_32k > 0.8] = 1
                feats_16k_32k[feats_16k_32k < 1] = 0
                pred_seg2 += feats_16k_32k.astype(np.int64)
                print('pred_seg2', pred_seg2.shape)
                pred_seg2 = np.clip(pred_seg2, 0, 1)
                all_pred_seg = np.array([*all_pred_seg, *pred_seg2])
                print('all_pred_seg', all_pred_seg.shape)
                all_pred_seg_16k = np.array([*all_pred_seg_16k, *pred_seg1])

            all_pred_seg_16k[all_pred_seg_16k > 0.5] = 1
            all_pred_seg_16k[all_pred_seg_16k < 1] = 0
            pred_seg = all_pred_seg
            pred_seg[pred_seg > 0.5] = 1
            pred_seg[pred_seg < 1] = 0
            print('pred_seg', pred_seg.shape)

            pred_seg = postprocessing.infer_labels_denoise(infer_set.patches_32k, pred_seg)
            print('pred_seg', pred_seg.shape)
            # logger.info('Stage 3: "patch inference" done, spent {}s', time.time() - start_time)

            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            # % Stage 4: patches classification
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            start_time = time.time()

            patches_tensor_16k, resamples_tensor_16k = infer_set.get_cls_patches_tensor(all_pred_seg_16k, True)
            print('patches_tensor_16k', patches_tensor_16k.shape)
            print('resamples_tensor_16k', resamples_tensor_16k.shape)
            all_pred_cls = np.array([])
            # all_pred_quad = np.array([])
            for i in range(math.ceil(len(patches_tensor_16k) / batch_size)):
                if i * batch_size == len(patches_tensor_16k) - 1:
                    pred_cls = model_cls(patches_tensor_16k[i * batch_size:(i + 1) * batch_size].repeat(2, 1, 1),
                                         resamples_tensor_16k[i * batch_size:(i + 1) * batch_size].repeat(2, 1, 1))
                    pred_cls = pred_cls[0:1]
                else:
                    pred_cls = model_cls(patches_tensor_16k[i * batch_size:(i + 1) * batch_size],
                                         resamples_tensor_16k[i * batch_size:(i + 1) * batch_size])
                pred_cls = pred_cls.detach().cpu().numpy()
                pred_cls = np.argmax(pred_cls, axis=1)

                # 0-32
                pred_cls[pred_cls > 0] = np.ceil(pred_cls[pred_cls > 0] / 8) * 10 + (pred_cls[pred_cls > 0] - 1) % 8 + 1
                all_pred_cls = np.array([*all_pred_cls, *pred_cls])

            if debug_vis:
                #########################################################
                #
                out_dir = f'/public/bme/home/v-tanmh/DentalModelSegmentation/txt/{os.path.basename(output)}_patches'
                os.makedirs(out_dir, exist_ok=True)
                for patch_id in range(pred_seg.shape[0]):
                    nd_out = vis_teeth_heatmap(infer_set.patches_16k[patch_id, :, 0:6],
                                               infer_set.patch_heatmap_16k[patch_id, :, 0])
                    np.savetxt(os.path.join(out_dir, f'heatmap_{patch_id}.txt'), nd_out)
                    nd_out = vis_teeth_seg_cls(infer_set.patches_16k[patch_id, :, 0:6], all_pred_seg_16k[patch_id],
                                               all_pred_cls[patch_id],
                                               infer_set.patch_centroids_16k[patch_id])
                    # nd_out = vis_teeth_seg_cls(patches_tensor[patch_id, :, 0:6], patches_tensor[patch_id, :, 6],
                    # all_pred_cls[patch_id], None)
                    np.savetxt(os.path.join(out_dir, f'{patch_id}.txt'), nd_out)
                    # np.savetxt(os.path.join(out_dir, f'mask_{patch_id}.txt'), infer_set.patches_32k[patch_id, pred_seg[patch_id] > 0.5, :])
                #########################################################

            # logger.info('Stage 4: "classification" done, spent {}s', time.time() - start_time)

            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            # % Stage 5: post processing
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            patches_tensor, _ = infer_set.get_cls_patches_tensor(pred_seg, False)
            try:
                final_labels = postprocessing.rearrange_labels2(patches_tensor[0, :, 0:3].data.cpu().numpy(),
                                                                patches_tensor[:, :, 6].data.cpu().numpy(),
                                                                all_pred_cls)
            except:
                print('rearrange_labels2 error for {}'.format(input))
                return

        infer_set.return_back_interpolation(final_labels)
        # logger.info('Inference spent {}s', time.time() - inf_start_time)

        # Output
        verts, faces, labels = infer_set.save_output_test()
        np.savez(os.path.join(output, os.path.basename(input).split('.')[0] + '.npz'), verts=verts, faces=faces,
                 labels=labels)

        # get stl,txt
        path = os.path.join(output, os.path.basename(input).split('.')[0] + '.npz')  #
        data = np.load(path)  #
        lst = data.files

        verts = data['verts']
        faces = data['faces']
        labels = data['labels']

        if max(labels) < 30:
            half_name = '_UpperJawScan'
        else:
            half_name = '_LowerJawScan'

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        mesh.export(os.path.join(output, output.split('/')[-2] + half_name + '.off'))
        np.savetxt(os.path.join(output, output.split('/')[-2] + half_name + '_label.txt'), labels, fmt='%.00f')

        os.rename(os.path.join(output, os.path.basename(input).split('.')[0] + '.npz'),
                  os.path.join(output, output.split('/')[-2] + half_name + '.npz'))

        # process different ids
        face_exist_idxes=[]
        if write_mesh:
            tids = list(range(11, 19)) + list(range(21, 29)) + list(range(31, 39)) + list(
                range(41, 49))
            for tid in tids:
                vert_idxes = np.arange(verts.shape[0])
                vert_id_idxes = vert_idxes[labels == tid]
                if vert_id_idxes.shape[0] > 0:
                    face_id_idxes = np.isin(faces, vert_id_idxes)  # bool
                    face_id_idxes = np.sum(face_id_idxes, axis=-1)
                    face_id_idxes = np.where(face_id_idxes == 3)[0]

                    face_id = faces[face_id_idxes]

                    idxes = np.arange(verts.shape[0])
                    for idx, val in enumerate(vert_id_idxes):
                        idxes[val] = idx
                    face_id = np.take(idxes, face_id)
                    face_exist_idxes.append(face_id_idxes)
                    mesh = trimesh.Trimesh(vertices=verts[vert_id_idxes], faces=face_id, process=True)
                    mesh.export(os.path.join(output, str(tid) + '.off'))
                    save_ply_with_color(os.path.join(output, str(tid) + '.ply'), verts[vert_id_idxes], face_id,
                                        color=label_to_rgb[int(tid)])
            #

            face_all_idx=list(range(0,faces.shape[0]))
            result = []
            for arr in face_exist_idxes:
                result.extend(arr.tolist())
            face_idx = np.delete(face_all_idx,result,axis=0)
            face_id=faces[face_idx]
            unique_numbers = list(set([item for sublist in face_id for item in sublist]))
            idxes = np.arange(verts.shape[0])
            for idx, val in enumerate(unique_numbers):
                idxes[val] = idx
            face_id = np.take(idxes, face_id)
            mesh = trimesh.Trimesh(vertices=verts[unique_numbers], faces=face_id, process=True)
            mesh.export(os.path.join(output, 'gum.off'))
            save_ply_with_color(os.path.join(output, 'gum.ply'), verts[unique_numbers], face_id,
                                color=(0,0,0))



from plyfile import PlyData, PlyElement
def save_ply_with_color(filename, verts, faces, color=(255, 0, 0)):
    # 定义顶点元素，包括颜色
    vertex = np.array([(v[0], v[1], v[2], color[0], color[1], color[2]) for v in verts],
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    # 定义面元素
    faces = np.array([(f,) for f in faces],
                     dtype=[('vertex_indices', 'i4', (3,))])

    # 创建PlyData对象
    ply_data = PlyData([PlyElement.describe(vertex, 'vertex'),
                        PlyElement.describe(faces, 'face')], text=True)

    # 保存PLY文件
    ply_data.write(filename)
def hex_to_rgb(hex_str):
    # 去除可能的'#'字符
    hex_str = hex_str.replace('#', '')
    # 将十六进制转换为三个十进制数
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))


color_txt=r"E:\PD\code\IOSModelSegmentation\16.txt"
# 创建一个空字典来存储标签和对应的RGB值
label_to_rgb = {}

with open(color_txt, 'r') as file:
    for line in file:
        # 分割每行为编号和十六进制颜色代码
        parts = line.strip().split()
        if len(parts) == 2:  # 确保这行有两部分内容
            label, hex_color = parts
            # 转换十六进制颜色为RGB并存储在字典中
            label_to_rgb[int(label)] = hex_to_rgb(hex_color)
        else:
            print(f'Error parsing line: {line}')


if __name__ == '__main__':

    init_logger()
    parser = make_arg_parser()
    ARGS = parser.parse_args()
    logger.info(ARGS)
    check_and_prepare_files()

    f = open(ARGS.input).read().splitlines()
    for i in f:
        if os.path.exists(i):
            output_path = os.path.join(ARGS.output, i.split('/')[-2], 'oral scan seg')
            # i.split('/')[-1])  # os.path.basename(i).split('_')[0])#
            if not os.path.exists(output_path):
                os.makedirs(output_path, exist_ok=True)
            mesh = trimesh.load(i)
            inference(i, output_path, ARGS.model_path, rotation=True, debug_vis=True, write_mesh=True)

