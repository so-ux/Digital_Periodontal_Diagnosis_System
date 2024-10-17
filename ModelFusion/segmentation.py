import os
import glob
import argparse
import trimesh
import pymeshlab
import numpy as np
import SimpleITK as sitk
from skimage import measure
import ModelFusion.utils as utils

def mesh_generate_cbct(image, mask, output):
    """
    Generate meshes for each tooth from CBCT annotation.

    Args:
        image (.nii.gz): CBCT image file path
        mask (.nii.gz): CBCT mask file path
        output (directory): output path

    Returns:
        labels (list): tooth labels of cbct

    """
    image = sitk.ReadImage(image)
    mask = sitk.ReadImage(mask)
    arr = sitk.GetArrayFromImage(mask).T
    spacing = np.round(image.GetSpacing(), 5)[0]

    # exclude the background label (0)
    labels = list(np.unique(arr)[1:].astype(int))

    for label in labels:
        # create binary voxel
        patch = np.where(arr == label, 1, 0)

        # get vertices and faces by marching cubes
        # but the created normals are reversed
        # verts, faces, normals, values = measure.marching_cubes(patch,spacing=(spacing[0],spacing[1],spacing[2]))
        verts, faces, normals, values = measure.marching_cubes(patch, 0)

        ms = pymeshlab.MeshSet()
        mesh = pymeshlab.Mesh(vertex_matrix=verts, face_matrix=faces)
        ms.add_mesh(mesh)

        # correct normals
        ms.meshing_invert_face_orientation()

        # remove isolated pieces
        # p = pymeshlab.Percentage(5)
        # ms.meshing_remove_connected_component_by_diameter(mincomponentdiag=p)

        # add smoothing
        # ms.apply_coord_laplacian_smoothing(stepsmoothnum=10, cotangentweight=False)
        # set spacing
        ms.compute_matrix_from_scaling_or_normalization(axisx=spacing, axisy=spacing, axisz=spacing)
        output_path = os.path.join(output, '%s.off' % label)
        if label < 30:
            output_path = output_path.replace('lower', 'upper')
        else:
            output_path = output_path.replace('upper', 'lower')

        ms.save_current_mesh(output_path, save_vertex_color=False, save_face_color=False)
        print('Save: %s' % output_path)

    return labels


def mesh_generate_dental(mesh, label, output):
    """
    Generate meshes for each tooth from dental model annotation.

    Args:
        mesh (.off): 3D dental model mesh
        label (.txt): contains the label for each vertex in a single row
        output (directory): output path

    Returns:
        ids (list): tooth labels of dental model

    """
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mesh)
    mesh = ms.current_mesh()
    vertices = mesh.vertex_matrix()
    faces = mesh.face_matrix()
    vertex_normals = mesh.vertex_normal_matrix()
    face_normals = mesh.face_normal_matrix()

    f = open(label)
    labels = f.read().splitlines()
    labels = np.array(labels, dtype='int')
    assert len(labels) == len(vertices)

    ids = sorted(list(set(labels)))[1:]
    for i in ids:
        out_file = os.path.join(output, '%d.off' % i)

        indV = np.where(labels == i)[0]
        newV = vertices[indV]
        newVN = vertex_normals[indV]

        indF = np.isin(faces, indV)
        indF = np.sum(indF, axis=-1)
        indF = np.where(indF == 3)[0]
        newF = faces[indF]
        newFN = face_normals[indF]

        # pick vertices and update their indices in faces
        fg = np.zeros(len(vertices) + 1)
        for ind, val in enumerate(indV):
            fg[val] = ind
        newF = np.take(fg, newF)

        new_mesh = trimesh.Trimesh(vertices=newV, faces=newF,vertex_normals=newVN, face_normals=newFN,process=True)

        ms = pymeshlab.MeshSet()
        refined_mesh = pymeshlab.Mesh(vertex_matrix=new_mesh.vertices,
                                      face_matrix=new_mesh.faces,
                                      v_normals_matrix=new_mesh.vertex_normals,
                                      f_normals_matrix=new_mesh.face_normals)
        ms.add_mesh(refined_mesh)
        # ms.apply_coord_laplacian_smoothing()

        # need rotate?
        # ms.compute_matrix_from_rotation(rotaxis=0, rotcenter=0, angle=90)
        # ms.compute_matrix_from_rotation(rotaxis=2, rotcenter=0, angle=345)
        # -------------------------------------------------------------------

        # remove isolated pieces
        # p = pymeshlab.Percentage(5) # you can change based on your data
        # ms.meshing_remove_connected_component_by_diameter(mincomponentdiag=p)
        ms.save_current_mesh(out_file, save_face_color=False, save_vertex_color=False)

        print('Save: %s' % out_file)

    return ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CBCT image segmentation')
    parser.add_argument('--input', '-i', type=str, default='input')
    args = parser.parse_args()
    path = args.input
    dirs = glob.glob(os.path.join(path, '*'))

    for d in dirs:
        print(d)
        # check whether orientation is RAI
        image_path=os.path.join(d,os.path.basename(d)+'_image.nii.gz')
        image = sitk.ReadImage(image_path)
        image_spacing =image.GetSpacing()
        mask_path = os.path.join(d, os.path.basename(d) + '_label.nii.gz')
        if not os.path.exists(mask_path):
            print('Do not have CBCT label!')
        else:
            mask = sitk.ReadImage(mask_path)
            mask_spacing = mask.GetSpacing()
            mask.SetSpacing(image_spacing)
            sitk.WriteImage(mask, os.path.join(d, os.path.basename(d) + '_label.nii.gz'))
            if image.GetDirection()!=(1,0,0,0,1,0,0,0,1):
                print('%s image orientation is not RAI' % os.path.basename(d))
                image.SetDirection((1,0,0,0,1,0,0,0,1))
                image.SetSpacing(image_spacing)
                sitk.WriteImage(image,os.path.join(d,os.path.basename(d)+'_image.nii.gz'))
            if mask.GetDirection()!=(1,0,0,0,1,0,0,0,1):
                print('%s mask orientation is not RAI' % os.path.basename(d))
                mask.SetDirection((1,0,0,0,1,0,0,0,1))
                sitk.WriteImage(mask,os.path.join(d,os.path.basename(d)+'_label.nii.gz'))
            else:
                pass

            dental_ids = []
            cfg = utils.get_path(os.path.join(d, 'lower'))
            dental_ids += mesh_generate_dental(mesh=cfg.oral_scan_seg_file.lower_dental_mesh,
                                               label=cfg.oral_scan_seg_file.lower_dental_label,
                                               output=cfg.path.dental)

            cfg = utils.get_path(os.path.join(d, 'upper'))
            dental_ids += mesh_generate_dental(mesh=cfg.oral_scan_seg_file.upper_dental_mesh,
                                               label=cfg.oral_scan_seg_file.upper_dental_label,
                                                output=cfg.path.dental)

            # if the CBCT meshes have been generated with other softwares (e.g., ITK-SNAP)
            # please comment the following line to skip this step, but remember to scale the generated meshes with spacing
            cbct_ids = mesh_generate_cbct(image=cfg.root_file.cbct_image,
                                          mask=cfg.root_file.cbct_mask,
                                          output=cfg.path.cbct)

