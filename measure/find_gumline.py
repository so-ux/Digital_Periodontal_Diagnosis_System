import glob
import os
import networkx as nx
import numpy as np
import trimesh
from sklearn.neighbors import KDTree
import measure.utils as utils
from scipy.interpolate import splprep, splev

def find_gumline(cfg, d, row):
    centers = []
    a = 0
    ids = [os.path.basename(e).split('.off')[0] for e in
           glob.glob(os.path.join(d, row, cfg.path.aligned_dental, '*.off'))]

    gum_lines = [[] for _ in range(len(ids))]
    broken_vertice_indices_dict = {}
    for id in ids:
        mesh = trimesh.load(os.path.join(cfg.path.aligned_dental, id + '.off'))  # trimesh.load(file)#

        broken_face_indices = trimesh.repair.broken_faces(mesh)
        broken_faces = mesh.faces[broken_face_indices]
        edges = trimesh.geometry.faces_to_edges(broken_faces)
        edges = np.sort(edges, axis=1)
        edges_str = ['_'.join([str(a), str(b)]) for a, b in edges.tolist()]

        adj_edges = mesh.face_adjacency_edges
        adj_edges = np.sort(adj_edges, axis=1)
        adj_edges_str = ['_'.join([str(a), str(b)]) for a, b in adj_edges.tolist()]

        broken_edges = [e for e in edges_str if e not in adj_edges_str]
        broken_edges = np.array([list(map(int, e.split('_'))) for e in broken_edges])
        G = nx.Graph()
        G.add_edges_from(broken_edges)

        # Find all cycles in the graph
        cycles = nx.cycle_basis(G)
        # Find the longest cycle
        broken_vertice_indices = np.array(max(cycles, key=len))
        assert len(broken_vertice_indices) > 10
        broken_vertices = mesh.vertices[broken_vertice_indices]

        gum_line = trimesh.Trimesh(vertices=broken_vertices)
        gum_line.export(os.path.join(cfg.path.gum_line, id + '.off'))  # os.path.basename(file)

        broken_vertice_indices_dict[id] = broken_vertice_indices
        centers.append(np.mean(broken_vertices, 0))
        gum_lines[a].append(broken_vertices)
        a += 1
    print('Finished: line')

    tooth_tree = KDTree(centers)
    tooth_dis, tooth_idx = tooth_tree.query(centers, k=3, return_distance=True)
    refined_gum_line_dict = {}
    for i in range(len(ids)):  # 0-10 #0 1 10
        fg = gum_lines[len(ids) - i - 1]  # 10 9
        bg = gum_lines[tooth_idx[-i - 1][-1]] + gum_lines[tooth_idx[-i - 1][-2]]
        tree = KDTree(fg[0])
        ind1 = tree.query_radius(bg[0], r=1.4)  # 1.5
        ind2 = tree.query_radius(bg[1], r=1.4)  # 1.5
        ind = np.concatenate((ind1, ind2), axis=0)

        remove_indices = []
        b = [e for e in ind if e.size > 0]  # e != []
        for e in b:
            for f in e:
                remove_indices.append(f)
        d1 = np.unique(remove_indices)
        keep_indices = np.setdiff1d(np.arange(len(fg[0])), d1)
        refined_gum_line_vertices = np.array(fg)[0][keep_indices]
        refined_gum_line = trimesh.Trimesh(vertices=refined_gum_line_vertices)
        refined_gum_line.export(os.path.join(cfg.path.gum_line_refined, '%d.off' % int(ids[len(ids) - i - 1])))
        refined_gum_line_dict[int(ids[len(ids) - i - 1])] = refined_gum_line_vertices
    print('Finished: refine')

    for id in ids:
        mesh = trimesh.load(os.path.join(cfg.path.aligned_dental, id + '.off'))
        mesh_v = mesh.vertices

        tree = KDTree(mesh_v)
        distances, indices = tree.query(refined_gum_line_dict[int(id)])
        indices_gum = indices[distances <= np.e ** (-6)].squeeze()

        gum_indices = broken_vertice_indices_dict[id][[i in indices_gum for i in broken_vertice_indices_dict[id]]]
        gum_indices = np.append(gum_indices, gum_indices[0])
        x = mesh_v[gum_indices]

        tck, u = splprep([x[:, 0], x[:, 1], x[:, 2]], s=1, per=True)
        u_fine = np.linspace(0, 1, 360)
        x_fine, y_fine, z_fine = splev(u_fine, tck)

        new_points = np.array([x_fine, y_fine, z_fine]).T
        res = trimesh.Trimesh(vertices=new_points)
        res.export(os.path.join(cfg.path.gum_line_refined_complemented, id + '.off'))
    print('Finished: complement')

if __name__ == '__main__':
    dirs = glob.glob(os.path.join(utils.get_dataset(), '*'))

    for row in ['lower', 'upper']:  #
        for d in dirs:
            cfg = utils.get_path(os.path.join(d, row))
            print(d)
            find_gumline(cfg, d, row)
