import json

import numpy as np
import glob
import os

import tqdm
import trimesh

from src.utils.input import read_obj
from src.vis.anatomy_colors import AnatomyColors


def color2str(color):
    return f'{color[0]} {color[1]} {color[2]}'


if __name__ == '__main__':
    directory = '/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/data_rot/gt/'
    anatomy = AnatomyColors()

    for obj in tqdm.tqdm(glob.glob(f'{directory}*.obj')):
        verts = []
        faces = []
        with open(obj, 'r') as fp:
            for line in fp.readlines():
                if line.startswith('v '):
                    verts.append([float(s) for s in line.strip().split(' ')[1:] if s != ''])
                if line.startswith('f '):
                    faces.append([int(s.split('//')[0]) - 1 for s in line.strip().split(' ')[1:] if s != ''])
        verts = np.array(verts)
        faces = np.array(faces, dtype=np.int32)

        mesh = trimesh.load(obj)
        mesh_vertices = np.asarray(mesh.vertices)

        vertex_not_in_faces = np.zeros(verts.shape[0], dtype=np.int32)
        for face in faces:
            vertex_not_in_faces[face] = 1

        verts = verts[vertex_not_in_faces > 0]

        # print(verts.shape)
        # print(verts.shape, np.asarray(mesh.vertices).shape)
        # exit(0)

        if len(verts) != len(mesh_vertices):
            print('ERROR', obj)
            continue

        colors = verts[:, 3:6]
        unique_colors = np.unique(colors, axis=0)
        labels = np.zeros((verts.shape[0], ), dtype=np.int32)

        c2l = {}

        try:
            for pi in range(len(verts)):
                if not c2l.__contains__(color2str(colors[pi])):
                    for i in range(1, 5):
                        for j in range(1, 9):
                            lc = anatomy.get_tooth_district_color(i * 10 + j, True)
                            if abs(colors[pi, 0] - lc[0]) < 1e-2 and abs(colors[pi, 1] - lc[1]) < 1e-2 and abs(colors[pi, 2] - lc[2]) < 1e-2:
                                c2l[color2str(colors[pi])] = i * 10 + j
                    if abs(colors[pi, 0] - 0.7) < 1e-2 and abs(colors[pi, 1] - 0.7) < 1e-2 and abs(colors[pi, 2] - 0.7) < 1e-2:
                        c2l[color2str(colors[pi])] = 0
                    if abs(colors[pi, 0] - 0.50196078) < 1e-2 and abs(colors[pi, 1] - 0.50196078) < 1e-2 and abs(colors[pi, 2] - 0.50196078) < 1e-2:
                        c2l[color2str(colors[pi])] = 0
                labels[pi] = c2l[color2str(colors[pi])]
        except:
            print('ERROR! Skip', obj)
            continue
        with open(f'{directory}{os.path.basename(obj).replace(".obj", ".json")}', 'w') as fp:
            json.dump({
                "id": obj,
                "jaw": obj,
                "labels": labels.tolist(),
                "instances": []
            }, fp)
            fp.close()
