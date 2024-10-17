import os.path
import glob

import numpy as np
from sklearn.neighbors import KDTree
import json

from src.utils.input import read_obj
from src.utils.output import write_obj

if __name__ == '__main__':
    input_file = '/home/zsj/Projects/data/口扫+CBCT成套数据/lv_test/out/43074710_shell_occlusion_l.obj'
    label_file = '/home/zsj/Projects/data/口扫+CBCT成套数据/lv_test/out/43074710_shell_occlusion_l.json'

    for input_file in glob.glob('/home/zsj/Projects/data/口扫+CBCT成套数据/lv_test/out/*.obj'):
        label_file = input_file.replace('.obj', '.json')
        with open(label_file, 'r') as fp:
            dic = json.load(fp)

        labels = np.array(dic['labels'], dtype=np.int32)
        verts, faces, normals = read_obj(input_file, True)
        kdtree = KDTree(verts)

        for tid in np.unique(labels):
            if tid == 0:
                continue
            crop_v_mask = labels == tid
            crop_v_id = np.argwhere(crop_v_mask).squeeze()
            crop_v_id_rev = {}
            for i, v in enumerate(crop_v_id):
                crop_v_id_rev[v] = i
            crop_f = []
            for f in faces:
                if np.sum(crop_v_mask[f]) == 3:
                    crop_f.append([crop_v_id_rev[f[0]], crop_v_id_rev[f[1]], crop_v_id_rev[f[2]]])

            out_dir = os.path.join(f'/home/zsj/Projects/data/口扫+CBCT成套数据/lv_test/out/单牙/{os.path.basename(input_file)}')
            os.makedirs(out_dir, exist_ok=True)
            write_obj(os.path.join(out_dir, f'{tid}.obj'), verts[crop_v_mask], crop_f, None)

