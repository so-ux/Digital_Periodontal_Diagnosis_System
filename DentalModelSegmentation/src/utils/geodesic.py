import numpy as np
from src.utils.output import write_obj


def gt_mesh_geodesic_field(vert, faces, labels):
    # Init queue
    q = np.empty((faces.shape[0], 2))
    q_head = 0
    q_tail = 0
    vis = np.ones((faces.shape[0], )) == 0

    adj_mat = np.empty((vert.shape[0], 0), dtype=list).tolist()
    for i, face in enumerate(faces):
        tooth_point_cnt = 0
        for face_i in face:
            adj_mat[face_i].append(i)
            if labels[face_i] > 0:
                tooth_point_cnt += 1
        if tooth_point_cnt == 3:
            # q.put([i, 0])
            q[q_tail, 0] = i
            q[q_tail, 1] = 0
            q_tail += 1
            vis[i] = True

    dist = np.ones((vert.shape[0], )) * 1000000
    # Mark all tooth points as distance = 0
    dist[labels > 0] = 0
    while q_head < q_tail:
        face_index, face_dis = q[q_head, :]
        face_index = int(face_index)

        q_head += 1
        for vert_i in faces[face_index]:
            dist[vert_i] = min(dist[vert_i], face_dis + 1)
            # Add this point's adj faces to queue
            for adj_face_i in adj_mat[vert_i]:
                if not vis[adj_face_i]:
                    q[q_tail, 0] = adj_face_i
                    q[q_tail, 1] = dist[vert_i]
                    q_tail += 1
                    vis[adj_face_i] = True
    return dist


def mesh_geodesic_field(vert, faces):
    # Init queue
    q = np.empty((faces.shape[0], 2))
    q_head = 0
    q_tail = 1
    vis = np.ones((faces.shape[0], )) == 0

    adj_mat = np.empty((vert.shape[0], 0), dtype=list).tolist()
    for i, face in enumerate(faces):
        for face_i in face:
            adj_mat[face_i].append(i)

    zero_face_index = 0
    vis[zero_face_index] = True

    dist = np.ones((vert.shape[0], )) * 1000000
    # Mark faces[0] points as distance = 0
    dist[faces[zero_face_index]] = 0
    q[0, 0] = zero_face_index
    q[0, 1] = 0
    while q_head < q_tail:
        face_index, face_dis = q[q_head, :]
        face_index = int(face_index)

        q_head += 1
        for vert_i in faces[face_index]:
            dist[vert_i] = min(dist[vert_i], face_dis + 1)
            # Add this point's adj faces to queue
            for adj_face_i in adj_mat[vert_i]:
                if not vis[adj_face_i]:
                    q[q_tail, 0] = adj_face_i
                    q[q_tail, 1] = dist[vert_i]
                    q_tail += 1
                    vis[adj_face_i] = True
    return dist


if __name__ == '__main__':
    import h5py
    from src.utils.input import read_obj

    fn = '/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/3D_scans_per_patient_obj_files/XBDVGZ7V/XBDVGZ7V_lower.obj'
    f = h5py.File('/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/h5/XBDVGZ7V_lower.h5')
    _, faces = read_obj(fn)
    dist = f['gd'][:]
    dist /= np.max(dist)
    dist = np.exp(-(dist ** 2) / (2 * 0.09))
    print(np.sum(dist > 0.95))

    selected = f['id_selected'][:]
    verts = f['data'][:][selected]

    color = np.ones((verts.shape[0], 3)) * 255
    color[:, 0] *= dist
    color[:, 1] *= 0
    color[:, 2] *= (1 - dist)
    print(np.sum(dist > 1000000 - 1))
    write_obj('/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/00000.obj', np.concatenate((verts[:, 0:3], color), axis=1), faces, None)

# if __name__ == '__main__':
#     import h5py
#     from src.utils.input import read_obj
#     fn = '/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/3D_scans_per_patient_obj_files/XBDVGZ7V/XBDVGZ7V_lower.obj'
#     h5 = h5py.File('/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/h5/XBDVGZ7V_lower.h5')
#
#     verts, faces = read_obj(fn)
#
#     dist = mesh_geodesic_field(verts, faces)
#     dist /= np.max(dist)
#
#     selected = h5['id_selected'][:]
#     fps_indices = h5['fps_indices'][:]
#     verts = verts[selected][fps_indices]
#     dist = dist[selected][fps_indices]
#
#     color = np.ones((verts.shape[0], 3)) * 255
#     color[:, 2] *= dist
#     color[:, 1] *= 0
#     color[:, 0] *= (1 - dist)
#     print(np.sum(dist > 1000000 - 1))
#     write_obj('/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/00000.obj', np.concatenate((verts[:, 0:3], color), axis=1), None, None)