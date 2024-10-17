import numpy as np
from src.vis.anatomy_colors import AnatomyColors
import os
import json

anatomy = AnatomyColors()


def export(obj, labels):
    out = obj.replace('.obj', '.json')
    f = open(out, 'w')
    json.dump({
        'id_patient': os.path.basename(obj).split('_')[0],
        'jaw': os.path.basename(obj).split('.')[0].split('_')[1],
        'labels': labels
    }, f)

    f.close()


def n(color):
    return '%03d%03d%03d' % (color[0], color[1], color[2])


def extract_labels(colors):
    colors = np.round(np.array(colors) * 255).astype(np.int32)
    labels = np.zeros((colors.shape[0], ), dtype=np.int32)
    for i in range(colors.shape[0]):
        nm = n(colors[i])
        if nm == '166203132':
            nm = '166205130'
        labels[i] = color_table[nm]
    return labels


if __name__ == '__main__':
    color_table = {}
    color_table['178178178'] = 0
    for i in range(1, 5):
        for j in range(1, 9):
            color = anatomy.get_tooth_district_color(i * 10 + j)
            color_table[n(color)] = i * 10 + j
    input_dir = '/home/zsj/Projects/data/miccai/results/gt_relabel/'

    for obj in os.listdir(input_dir):
        if not obj.endswith('obj'):
            continue
        print(obj)
        file = open(os.path.join(input_dir, obj), 'r')
        verts = []
        colors = []
        while True:
            line = file.readline()
            if not line:
                break
            strs = line.split(' ')
            if strs[0] == 'v':
                verts.append([float(strs[1]), float(strs[2]), float(strs[3])])
                colors.append([float(strs[4]), float(strs[5]), float(strs[6])])
        file.close()
        labels = extract_labels(colors)
        export(os.path.join(input_dir, obj), labels.tolist())

