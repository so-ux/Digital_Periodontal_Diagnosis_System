import argparse
import glob
import os.path
import sys
import traceback

script_dir = os.path.dirname((os.path.abspath(__file__)))
sys.path.append(script_dir)

import numpy as np
import torch
from loguru import logger
from src.metrics.metrics2 import teeth_localization_accuracy, teeth_identification_rate, \
    teeth_segmentation_accuracy, \
    global_ranking_score
from src.utils.tensorboard_utils import TensorboardUtils
from src.vis.vis_teeth_cls import *
import json
from src.inference import inference

# =============== Global variables ===============
torch.random.manual_seed(20000228)
torch.cuda.manual_seed(20000228)


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
    parser.add_argument('--exp_name', '-n', type=str, help='Experiment name', required=False, default='default')
    return parser


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


def main(input_file, label_file, output_file, model_path):
    logger.info('Testing {}', os.path.basename(input_file))
    target_labels = read_labels(label_file)

    # Output
    vertices, pred_labels = inference(input_file, output_file, model_path)

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

    error = classification_error(tir, pred_labels, target_labels)
    logger.info('[{}] TLA: {}, TIR: {}, TSA: {}, Score: {}, Error: {}', os.path.basename(input_file),
                np.exp(-tla), tir, tsa, global_ranking_score(tla, tir, tsa), error)
    return tsa, tir, tla, global_ranking_score(tla, tir, tsa), teeth_count, error


def classification_error(tir, pred, gt):
    u_pred = np.unique(pred)
    u_gt = np.unique(gt)

    error = []
    if tir < 1:
        for gt_id in u_gt:
            if np.sum(u_pred == gt_id) == 0:
                error.append(gt_id % 10)

    if len(error) == 0 and tir > 0.999:
        return None

    if len(error) > 3:
        return f"Continuous {error}"

    msg = 'Missing'
    for eid in error:
        msg = f'{msg} {int(eid)}'
    return msg


if __name__ == '__main__':
    init_logger()
    parser = make_arg_parser()
    ARGS = parser.parse_args()
    logger.info(ARGS)

    writer = TensorboardUtils(os.path.join(ARGS.output, ARGS.exp_name)).writer
    all_tsa, all_tir, all_tla, all_scores = [], [], [], []
    all_count = []
    low_scores = []

    step = 1
    test_file_list = os.listdir(ARGS.data_dir)
    test_files = [os.path.join(ARGS.data_dir, line) for line in test_file_list]

    for i, file in enumerate(test_files):
        ls=glob.glob(file+'*.obj')
        input_file = os.path.join(file, ls[0])
        label_file = os.path.join(file, ls[0].replace('.obj','.json'))
        output_file = os.path.join(file, ls[0].replace('.obj','_pred.obj'))
        try:
            tsa, tir, tla, score, teeth_count, error = main(input_file, label_file, output_file, ARGS.model_path)
        except Exception as e:
            traceback.print_exc()
            logger.error('Error occurred: {}', e)
            writer.add_text('error', file, step)
            tsa, tir, tla, score, teeth_count, error = 0, 0, 5, 0, 1, ''
        all_tsa.append(tsa)
        all_tir.append(tir)
        all_tla.append(tla)
        all_scores.append(score)
        all_count.append(teeth_count)

        if error is not None:
            writer.add_text('ClsError', f'{file} {error}', i)

        writer.add_scalar('Single/TSA', tsa, step)
        writer.add_scalar('Single/TIR', tir, step)
        writer.add_scalar('Single/TLA', np.exp(-tla), step)
        writer.add_scalar('Single/Score', score, step)

        total_tsa = np.sum(all_tsa) / step
        total_tir = np.sum(np.array(all_tir) * np.array(all_count)) / np.sum(all_count)
        total_tla = np.exp(-np.sum(np.array(all_tla) * np.array(all_count)) / np.sum(all_count))

        writer.add_scalar('Total/TSA', total_tsa, step)
        writer.add_scalar('Total/TIR', total_tir, step)
        writer.add_scalar('Total/TLA', total_tla, step)
        writer.add_scalar('Total/Score', (total_tsa + total_tir + total_tla) / 3, step)
        writer.flush()

        step += 1

    print(low_scores)
