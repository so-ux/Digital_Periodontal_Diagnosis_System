import os.path

import numpy as np
import random
import json
from src.utils.input import read_obj
from src.utils.output import write_obj
from src.vis.anatomy_colors import AnatomyColors

colors = AnatomyColors()


def load_gt_json(json_path):
    f = open(json_path, 'r')
    return json.load(f)['labels']


def load_model(model_path):
    v, f = read_obj(model_path)
    return v, f


def get_color(index):
    if index == 0:
        return [0.7, 0.7, 0.7]
    # group = int(index // 10)
    # alpha = (index % 10 - 1) / 10
    # if colors[group] is None:
    #     colors[group] = [random.random(), random.random(), random.random()]
    # base_c = colors[group]
    # color = [
    #     base_c[0] * (1 - alpha) + alpha,
    #     base_c[1] * (1 - alpha) + alpha,
    #     base_c[2] * (1 - alpha) + alpha
    # ]
    # return color
    return colors.get_color(index, True)


def colorize(v, l):
    nd_color = []
    for i in range(len(v)):
        nd_color.append(get_color(l[i]))
    return np.concatenate((v, np.array(nd_color)), axis=1)


pids = ['6L4N5MA2', '019CP4YZ']
model_dir = '/run/media/zsj/DATA/Data/miccai/3D_scans_per_patient_obj_files_b2/'
gt_dir = '/run/media/zsj/DATA/Data/miccai/ground-truth_labels_instances_b2/'
out_dir = '/home/zsj/Projects/data/miccai/results/gt/'

for pid in pids:
    for ori in ['lower', 'upper']:
        v, f = load_model(os.path.join(model_dir, pid, '{}_{}.obj'.format(pid, ori)))
        lbl = load_gt_json(os.path.join(gt_dir, pid, '{}_{}.json'.format(pid, ori)))
        print(len(v), len(lbl))
        write_obj(os.path.join(out_dir, '{}_{}.obj'.format(pid, ori)), colorize(v, lbl), f, None)
