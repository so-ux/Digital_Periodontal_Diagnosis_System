import argparse
import math
import os.path
import sys
import time
import json

import numpy as np
import torch
from loguru import logger
from sklearn.neighbors import KDTree

from src.data.TeethInferDataset import TeethInferDataset
from src.models import *
from src.metrics.metrics2 import teeth_localization_accuracy, teeth_identification_rate, teeth_segmentation_accuracy, \
    global_ranking_score
from src.utils import postprocessing
from src.utils.cfdp import get_clustered_centroids
from src.utils.tensorboard_utils import TensorboardUtils
from src.vis.vis_teeth_cls import *

# =============== Global variables ===============
torch.random.manual_seed(20000228)
torch.cuda.manual_seed(20000228)

model_centroid_prediction = PointnetCentroidPredictionNet()
model_all_tooth_seg = AllToothSegNet(3)
model_refine = DGCNNRefineNet({
    'k': 32,
    'emb_dims': 1024,
    'dropout': 0.1
})
# model_refine = RefineNet(7)
# model_refine = torch.nn.DataParallel(model_refine)
model_cls = ToothClsNet()
model_cls = torch.nn.DataParallel(model_cls)


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
    parser.add_argument('--data_dir', '-d', type=str, help='Path of input models', required=True)
    parser.add_argument('--output', '-o', type=str, help='Path of output results', required=True)
    parser.add_argument('--model_path', '-m', type=str, help='Path of trained models', required=True)
    parser.add_argument('--stage_1', type=str, help='Choose the first stage algorithm', required=False,
                        choices=['centroid', 'all_tooth'], default='centroid')
    return parser


def check_and_prepare_files() -> None:
    """
    Check and prepare needed files
    """
    if not os.path.exists(os.path.dirname(ARGS.output)):
        os.makedirs(os.path.dirname(ARGS.output), exist_ok=True)

    if not os.path.exists(os.path.dirname(ARGS.model_path)):
        logger.error('Model path [{}] not exists', ARGS.model_path)
        exit(-1)


# =========== Data persistence methods ===========
def load_checkpoint() -> None:
    model_centroid_prediction_file = os.path.join(ARGS.model_path, 'model_pred_centroid')
    model_all_tooth_seg_file = os.path.join(ARGS.model_path, 'model_all_tooth_seg_best')
    model_refine_file = os.path.join(ARGS.model_path, 'model_refine')
    model_cls_file = os.path.join(ARGS.model_path, 'model_cls')

    if ARGS.stage_1 == 'centroid':
        if not os.path.exists(model_centroid_prediction_file):
            logger.error('Cannot find model file [{}]', model_centroid_prediction_file)
        model_centroid_prediction.load_state_dict(torch.load(model_centroid_prediction_file))
        model_all_tooth_seg.load_state_dict(torch.load(model_all_tooth_seg_file))
    else:
        if not os.path.exists(model_all_tooth_seg_file):
            logger.error('Cannot find model file [{}]', model_all_tooth_seg_file)
        model_all_tooth_seg.load_state_dict(torch.load(model_all_tooth_seg_file))
    if not os.path.exists(model_refine_file):
        logger.error('Cannot find model file [{}]', model_refine_file)

    if not os.path.exists(model_cls_file):
        logger.error('Cannot find model file [{}]', model_cls_file)

    model_refine.load_state_dict(torch.load(model_refine_file))
    model_cls.load_state_dict(torch.load(model_cls_file))


# 连接vertices和labels
def cat_vertices_labels(ver, labels):
    labels = np.expand_dims(labels, 1)
    target = np.concatenate((ver, labels), axis=1)
    return target


# 读取labels 的json文件
def read_labels(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        labels = np.array(data["labels"])
    return labels


def main(input_file, label_file, output_file):
    ##########################################
    # Init
    ##########################################
    batch_size = 8
    target_labels = read_labels(label_file)

    ##########################################
    # Data init
    ##########################################
    infer_set = TeethInferDataset(input_file)
    data_tensor = infer_set.get_data_tensor()

    ##########################################
    # Model
    ##########################################
    with torch.no_grad():
        start_time = time.time()
        # model_all_tooth_seg.cuda()

        start_time = time.time()
        # Stage 1: all tooth seg
        pred_seg = model_all_tooth_seg(data_tensor[:, :, 0:6])
        pred_seg = pred_seg.detach().cpu().numpy()[0]

        np.savetxt(
            f'/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/test_small_tooth_output/{os.path.basename(output_file)}.pn.txt',
            vis_teeth_seg(data_tensor[0].data.cpu().numpy()[:, 0:6], pred_seg))

        infer_set.remove_curvatures_on_tooth(pred_seg)
        data_tensor = infer_set.get_data_tensor()

        seg = np.array(pred_seg)
        seg[seg > 0.5] = 1
        seg[seg < 1] = 0

        # 扩大一圈预测
        nd_data = data_tensor[0].data.cpu().numpy()[:, 0:3]
        tree = KDTree(nd_data)
        neighbours = tree.query_radius(nd_data[seg > 0], 0.1, return_distance=False)
        for n in neighbours:
            seg[n] = 1

        logger.info('Stage 1: "all tooth seg" done, spent {}s', time.time() - start_time)

        kp_reg, kp_score, seed_xyz = model_centroid_prediction(data_tensor[:, seg > 0.5, :], False, False)

        kp_reg = kp_reg.detach().cpu().numpy()
        kp_score = kp_score.detach().cpu().numpy()

        kp_reg = kp_reg[0, :, :]

        np.savetxt(f'/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/test_small_tooth_output/{os.path.basename(output_file)}.ct.xyz',
                   kp_reg)
        kp_reg = get_clustered_centroids(np.asarray(kp_reg, dtype=np.float64))

        np.savetxt(f'/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/test_small_tooth_output/{os.path.basename(output_file)}.ct.txt',
                   vis_teeth_centroids(data_tensor[0, :, 0:6].data.cpu().numpy(), kp_reg))

        logger.info('Stage 1: "centroid prediction" done, spent {}s', time.time() - start_time)
        start_time = time.time()

        # Stage 2 preprocess: split patches
        infer_set.make_patches_centroids(kp_reg)

        # Stage 2: patch inference
        patches_tensor = infer_set.get_patches_tensor()

        all_pred_cls = np.array([])
        all_pred_seg = np.array([])
        for i in range(math.ceil(len(patches_tensor) / batch_size)):
            if i * batch_size == len(patches_tensor) - 1:
                # pred_cls, pred_seg = model_refine(patches_tensor[i * batch_size:(i + 1) * batch_size].repeat(2, 1, 1).transpose(2, 1))
                # pred_cls = pred_cls[0:1]
                # pred_seg = pred_seg[0:1]
                pred_seg = model_refine(patches_tensor[i * batch_size:(i + 1) * batch_size].repeat(2, 1, 1).transpose(2, 1))[:, 0, :]
                pred_seg = pred_seg[0:1]
                # pred_seg = model_refine(patches_tensor[i * batch_size:(i + 1) * batch_size].transpose(2, 1))
                # pred_seg = pred_seg.permute(0, 2, 1)
                # pred_seg = torch.argmax(pred_seg[0:1], -1)
            else:
                # pred_cls, pred_seg = model_refine(patches_tensor[i * batch_size:(i + 1) * batch_size].transpose(2, 1))
                pred_seg = model_refine(patches_tensor[i * batch_size:(i + 1) * batch_size].transpose(2, 1))[:, 0, :]
                # pred_seg = model_refine(patches_tensor[i * batch_size:(i + 1) * batch_size].transpose(2, 1))
                # pred_seg = pred_seg.permute(0, 2, 1)
                # pred_seg = torch.argmax(pred_seg, -1)

            # pred_cls = pred_cls.detach().cpu().numpy()
            # pred_cls = np.argmax(pred_cls, axis=1)
            pred_seg = pred_seg.detach().cpu().numpy()

            # all_pred_cls = np.array([*all_pred_cls, *pred_cls])
            all_pred_seg = np.array([*all_pred_seg, *pred_seg])

        pred_seg = all_pred_seg
        # pred_cls = all_pred_cls
        # pred_cls, pred_seg = model_refine(patches_tensor)

        pred_seg = postprocessing.infer_labels_denoise(infer_set.patches, pred_seg)
        logger.info('Stage 2: "patch inference" done, spent {}s', time.time() - start_time)
        start_time = time.time()

        # Stage 3: classification
        model_cls.cuda()
        model_cls.eval()

        patches_tensor, resamples_tensor = infer_set.get_cls_patches_tensor(pred_seg)

        all_pred_cls = np.array([])
        for i in range(math.ceil(len(patches_tensor) / batch_size)):
            if i * batch_size == len(patches_tensor) - 1:
                pred_cls = model_cls(patches_tensor[i * batch_size:(i + 1) * batch_size].repeat(2, 1, 1),
                                     resamples_tensor[i * batch_size:(i + 1) * batch_size].repeat(2, 1, 1))
                pred_cls = pred_cls[0:1]
            else:
                pred_cls = model_cls(patches_tensor[i * batch_size:(i + 1) * batch_size],
                                     resamples_tensor[i * batch_size:(i + 1) * batch_size])
            pred_cls = pred_cls.detach().cpu().numpy()
            pred_cls = np.argmax(pred_cls, axis=1)

            # 0-32还原为0-48
            pred_cls[pred_cls > 0] = np.ceil(pred_cls[pred_cls > 0] / 8) * 10 + (pred_cls[pred_cls > 0] - 1) % 8 + 1
            all_pred_cls = np.array([*all_pred_cls, *pred_cls])

        #########################################################
        # 保存预测的Patch
        out_dir = f'/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/test_small_tooth_output/{os.path.basename(output_file)}_patches'
        os.makedirs(out_dir, exist_ok=True)
        for patch_id in range(pred_seg.shape[0]):
            # nd_out = vis_teeth_heatmap(infer_set.patches[patch_id, :, 0:6], infer_set.patches[patch_id, :, 6])
            nd_out = vis_teeth_seg_cls(infer_set.patches[patch_id, :, 0:6], pred_seg[patch_id], all_pred_cls[patch_id],
                                       infer_set.patch_centroids[patch_id])
            np.savetxt(os.path.join(out_dir, f'{patch_id}.txt'), nd_out)
        #     np.savetxt(os.path.join(out_dir, f'mask_{patch_id}.txt'), infer_set.patches[patch_id, pred_seg[patch_id] > 0.5, :])
        #########################################################

        final_labels = postprocessing.rearrange_labels(patches_tensor[0, :, 0:3].data.cpu().numpy(),
                                                       patches_tensor[:, :, 6].data.cpu().numpy(),
                                                       all_pred_cls, True)

        # seg[final_labels > 0] = 0
        infer_set.return_back_interpolation(final_labels)
        # infer_set.return_back(pred_seg, all_pred_cls)

        # loop(final_labels > 0, tree)

        # Output
        infer_set.save_output(output_file)
        vertices, pred_labels = infer_set.save_output_test()

        pred = cat_vertices_labels(vertices, pred_labels)
        pred_tsa = pred_labels
        pred = np.expand_dims(pred, 0)
        pred = torch.Tensor(pred)
        target = cat_vertices_labels(vertices, target_labels)
        target_tsa = target_labels
        target = np.expand_dims(target, 0)
        target = torch.Tensor(target)

        tla, teeth_count = teeth_localization_accuracy(pred, target)
        tir, teeth_count = teeth_identification_rate(pred, target)
        tsa = teeth_segmentation_accuracy(torch.Tensor(np.array(pred_tsa)), torch.Tensor(np.array(target_tsa)))

        logger.info('[{}] TLA: {}, TIR: {}, TSA: {}, Score: {}', os.path.basename(input_file),
                    math.exp(-tla), tir, tsa, global_ranking_score(tla, tir, tsa))
        return tsa, tir, tla, global_ranking_score(tla, tir, tsa), teeth_count


if __name__ == '__main__':
    import glob

    init_logger()
    parser = make_arg_parser()
    ARGS = parser.parse_args()
    logger.info(ARGS)
    check_and_prepare_files()
    load_checkpoint()

    model_all_tooth_seg.cuda()
    model_centroid_prediction.cuda()
    model_refine.cuda()
    model_cls.cuda()
    model_all_tooth_seg.eval()
    model_centroid_prediction.eval()
    model_refine.eval()
    model_cls.eval()

    writer = TensorboardUtils(os.path.join(ARGS.output, 'run1')).writer
    all_tsa, all_tir, all_tla, all_scores, all_count = [], [], [], [], []
    low_scores = []

    step = 1
    root_dir = f'/run/media/zsj/DATA/Data/miccai/small_tooth/'
    test_file_list = ['1']
    # test_file_list = glob.glob(f'{root_dir}*.obj')
    # test_file_list = glob.glob(f'/home/zsj/Downloads/Z83V9A9D_lower.obj')
    # test_file_list = glob.glob(f'{root_dir}4MC4KRQV_upper.obj')
    # test_file_list = glob.glob(f'{root_dir}Z83V9A9D_lower.obj')
    patient, jaw = 'W82LGNOE', 'lower'

    for file in test_file_list:
        if file.strip() == '':
            continue
        # patient, jaw = os.path.basename(file).replace('.obj', '').split('_')
        input_file = os.path.join(ARGS.data_dir, '3D_scans_per_patient_obj_files', patient, f'{patient}_{jaw}.obj')
        # input_file = file
        label_file = os.path.join(ARGS.data_dir, 'ground-truth_labels_instances', patient, f'{patient}_{jaw}.json')
        output_file = os.path.join(ARGS.output, f'{patient}_{jaw}.obj')
        tsa, tir, tla, score, teeth_count = main(input_file, label_file, output_file)
        all_tsa.append(tsa)
        all_tir.append(tir)
        all_tla.append(tla)
        all_scores.append(score)
        all_count.append(teeth_count)

        # if score < 0.9:
        #     low_scores.append({
        #         'file': input_file,
        #         'tsa': tsa,
        #         'tir': tir,
        #         'tla': tla,
        #         'score': score
        #     })
        #     writer.add_text('Bad items', f'File: {input_file}\nTSA: {tsa}; TIR: {tir}; TLA: {tla}; Score: {score}', step)

        writer.add_scalar('Single/TSA', tsa, step)
        writer.add_scalar('Single/TIR', tir, step)
        writer.add_scalar('Single/TLA', tla, step)
        writer.add_scalar('Single/Score', score, step)

        total_tsa = np.sum(all_tsa) / step
        total_tir = np.sum(np.array(all_tir) * np.array(all_count)) / np.sum(all_count)
        total_tla = np.exp(-np.sum(np.array(all_tla) * np.array(all_count)) / np.sum(all_count))

        writer.add_scalar('Total/TSA', total_tsa, step)
        writer.add_scalar('Total/TIR', total_tir, step)
        writer.add_scalar('Total/TLA', total_tla, step)
        writer.add_scalar('Total/Score', (total_tsa + total_tir + total_tla) / 3, step)

        step += 1

        writer.flush()

    print(low_scores)
