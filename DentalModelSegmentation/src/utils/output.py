import numpy as np


def write_off(filename, nd_verts, nd_faces):
    file = open(filename, 'w')
    file.write('OFF\n')
    file.write('{} {} 0\n'.format(len(nd_verts), 0 if nd_faces is None else len(nd_faces)))

    for vert in nd_verts:
        for i in range(len(vert)):
            file.write('{}'.format(vert[i]))
            if i == len(vert) - 1:
                file.write('\n')
            else:
                file.write(' ')

    if nd_faces is not None:
        for face in nd_faces:
            file.write('{}'.format(len(face)))
            for fi in face:
                file.write(' {}'.format(fi))
            file.write('\n')


def write_obj(filename, nd_verts, nd_faces, nd_normals):
    file = open(filename, 'w')
    for vert in nd_verts:
        file.write('v')
        for i in range(len(vert)):
            file.write(' {}'.format(vert[i]))
        file.write('\n')

    if nd_faces is not None:
        for face in nd_faces:
            file.write('f')
            for fi in face:
                file.write(' {}'.format(int(fi) + 1))
            file.write('\n')

    if nd_normals is not None:
        for norm in nd_normals:
            file.write('vn {} {} {}\n'.format(norm[0], norm[1], norm[2]))


def write_txt(filename, nd_verts):
    np.savetxt(filename, nd_verts)