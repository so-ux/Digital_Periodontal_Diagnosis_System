import numpy as np
import open3d
import trimesh


def read_off(filename, normal=False):
    file = open(filename, 'r')
    header = file.readline().strip()
    if header not in ['OFF', 'COFF']:
        print('Not a valid OFF header')
    n_verts, n_faces, _ = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')[0:3]] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:4] for i_face in range(n_faces)]
    verts = np.vstack(verts)
    faces = np.vstack(faces)

    if normal:
        mesh = open3d.geometry.TriangleMesh()
        mesh.vertices = open3d.utility.Vector3dVector(verts)
        mesh.triangles = open3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()
        normals = np.asarray(mesh.vertex_normals)
        return verts, faces, normals
    return verts, faces


def read_obj(filename, normal=False):
    file = open(filename, 'r')

    verts = []
    faces = []
    normals = []

    for line in file.readlines():
        if not line:
            break
        strs = line.strip().split(' ')
        if strs[0] == 'v':
            verts.append([float(strs[1]), float(strs[2]), float(strs[3])])
        elif strs[0] == 'n' or strs[0] == 'vn':
            normals.append([float(strs[1]), float(strs[2]), float(strs[3])])
        elif strs[0] == 'f':
            face = []
            for fi in strs[1:]:
                # OBJ face is 1-based
                face.append(int(fi.split('//')[0]) - 1)
            faces.append(face)

    verts = np.array(verts)
    faces = np.array(faces, dtype=np.int32)
    normals = np.array(normals)

    vertex_not_in_faces = np.zeros(verts.shape[0], dtype=np.int32)
    for face in faces:
        vertex_not_in_faces[face] = 1
    verts = verts[vertex_not_in_faces > 0]

    if normal:
        # mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        mesh = trimesh.load(filename, process=False)
        normals = np.asarray(mesh.vertex_normals)
        # return verts, faces, normals
        return np.asarray(mesh.vertices), np.asarray(mesh.faces), normals
    else:
        return verts, faces


def read_mesh(filename, normal=False):
    mesh = trimesh.load(filename, process=False)
    if normal:
        return np.asarray(mesh.vertices), np.asarray(mesh.faces), np.asarray(mesh.vertex_normals)
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


def read_txt(filename, feat=False):
    file_nd = np.loadtxt(filename)
    if feat:
        return file_nd[:, 0:3], file_nd[3:]
    else:
        return file_nd[:, 0:3]


if __name__ == '__main__':
    read_obj('/home/zsj/Projects/data/miccai/3D_scans_per_patient_obj_files/0EJBIPTC/0EJBIPTC_lower.obj')
