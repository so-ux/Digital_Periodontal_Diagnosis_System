import torch
from sklearn.cluster import DBSCAN
import numpy as np

from pointnet2_utils import furthest_point_sample
from sklearn.neighbors import KDTree


def remove_braces(verts, min_samples=10):
    db = DBSCAN(eps=1, min_samples=min_samples).fit(verts[:, 0:3])
    indices, counts = np.unique(db.labels_, return_counts=True)
    which = np.argmax(counts)
    selected = np.argwhere(db.labels_ == indices[which]).squeeze()
    return selected


def remove_braces_mesh(verts, faces, min_samples=10):
    db = DBSCAN(eps=1, min_samples=min_samples).fit(verts[:, 0:3])
    indices, counts = np.unique(db.labels_, return_counts=True)
    if np.max(counts) < 4000:
        return np.arange(0, len(verts)), faces
    which = np.argwhere(counts > 1000)

    selected = np.zeros((len(verts),), dtype=np.int32)
    for i in which:
        selected[db.labels_ == indices[i]] += 1

    removed = selected == 0
    selected = selected > 0

    # which = np.argmax(counts)
    # selected = np.argwhere(db.labels_ == indices[which]).squeeze()
    # removed = np.argwhere(db.labels_ != indices[which]).squeeze()

    new_v_id = np.zeros((len(verts), ))
    new_v_id[selected] = np.arange(0, np.sum(selected))
    # Remove all faces with id_removed
    face_select_id = np.sum(removed[faces], axis=1) == 0
    r_f = new_v_id[faces[face_select_id]]
    return np.argwhere(selected).squeeze(), np.asarray(r_f, dtype=np.int32)


def model_flat_indices(verts):
    tree = KDTree(verts[:, 0:3])
    neighbours = tree.query(verts[:, 0:3], 20, return_distance=False)
    norms = verts[neighbours][:, :, 3:]
    norms /= np.sqrt(np.sum(norms ** 2, axis=-1, keepdims=True))
    mean_norm = np.mean(norms, axis=1, keepdims=True)
    flats = np.einsum('ijk,ikn->ijn', norms, mean_norm.transpose([0, 2, 1])).squeeze()
    flats = np.mean(np.arccos(np.clip(flats, -1, 1)), axis=-1)
    return flats


def flat_index(verts_with_norm):
    norms = verts_with_norm[:, 3:]
    norms /= np.sqrt(np.sum(norms ** 2, axis=1, keepdims=True))
    mean_norm = np.mean(norms, axis=0, keepdims=True)

    # 此处加入np.clip来约束点积结果严格在[-1, 1]，否则会出现点积结果为1.0000000000000002，取arccos为nan
    return np.mean(np.arccos(np.clip(np.dot(norms, mean_norm.T), -1, 1)))


# Debug test method
if __name__ == '__main__':
    from input import read_obj, read_off
    from pc_normalize import pc_normalize

    # v, f, n = read_off('/run/media/zsj/4BC83987399C4779/Gordon/model_1/HBF_10227_L_S.off', True)
    v, f, n = read_obj(
        '/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/3D_scans_per_patient_obj_files/ZKJEPFDD/ZKJEPFDD_lower.obj',
        True)
    v = np.concatenate((v, n), axis=1)

    selected, f = remove_braces_mesh(v, f)
    v = v[selected]

    v, c, m = pc_normalize(v)

    # mesh = o3d.geometry.TriangleMesh()
    # mesh.vertices = o3d.utility.Vector3dVector(v[:, 0:3])
    # mesh.triangles = o3d.utility.Vector3iVector(f)
    # mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh])

    # 3. FPS
    data_tensor = torch.Tensor(np.array([v[:, 0:3]])).cuda()

    # For each tooth cropping area, sample 2048 points
    # Assume that there are 16 teeth on each model
    fps_indices = furthest_point_sample(data_tensor, 2048 * 16).detach().cpu().numpy()[0]
    fps_indices = np.unique(fps_indices)

    v = v[fps_indices]

    # tree = KDTree(v[:, 0:3])
    # flat_indices = []
    # for i, vert in enumerate(tqdm.tqdm(v)):
    #     neighbours = tree.query([vert[0:3]], 10, return_distance=False)[0]
    #     # print(neighbours)
    #     flat_indices.append(flat_index(v[neighbours]))
    #
    # flat_indices = np.array(flat_indices)
    flat_indices = model_flat_indices(v)
    print(np.min(flat_indices), np.max(flat_indices))
    # flat_indices = (flat_indices - np.min(flat_indices)) / (np.max(flat_indices) - np.min(flat_indices))
    flat_indices[flat_indices < 1] = 0

    colors = np.ones((32768, 4))
    colors[:, 0] = flat_indices
    colors[:, 1] = 0 * flat_indices
    colors[:, 2] = 1 - flat_indices

    nd_out = np.concatenate((v, colors), axis=1)
    np.savetxt('/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/test_vis/000.txt', nd_out)

    # np.savetxt('../test.xyz', v)
