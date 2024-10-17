import os
import glob
import json
import shutil
import trimesh
import warnings
import pymeshlab
import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA
import ModelFusion.utils as utils


def mesh_merge(mesh_list, output_merged, output_centers):
    """
    Merge the tooth meshes to get the patient-level mesh.

    Args:
        mesh_list (directory): the path of meshes to merge
        output_merged (.off): the merged mesh
        output_centers (.json): center points of each mesh

    Returns:

    """
    ms = pymeshlab.MeshSet()
    mesh_list = glob.glob(os.path.join(mesh_list, '*.off'))
    centers = {}

    for mesh in mesh_list:
        ms.load_new_mesh(mesh)
        vertices = ms.current_mesh().vertex_matrix()
        id = os.path.basename(mesh).split('.')[0]
        centers[id] = list((np.min(vertices, axis=0) + np.max(vertices, axis=0)) / 2)

    ms.generate_by_merging_visible_meshes()

    ms.save_current_mesh(output_merged, save_vertex_color=False, save_face_color=False)
    print('Save: %s' % output_merged)

    with open(output_centers, 'w') as f:
        json.dump(centers, f, indent=4)
        print('Save: %s' % output_centers)


def align_pca(src, dst, src_centers, dst_centers, output_prealigned, output_transformation):
    """ 
    Registration
    Stage 1: coarse registration on patient-level

    Args:
        src (.off): merged patient-level 3D dental model path
        dst (.off): merged patient-level CBCT model path
        src_centers (.json): center points of each 3D dental model
        dst_centers (.json): center points of each CBCT model
        output_prealigned (.off): output path of the prealigned mesh
        output_transformation (.json): output path of transformation

    Returns:
        transformation (np.array)

    """
    source = trimesh.load(src)

    target = trimesh.load(dst)

    source_pca = PCA(n_components=3)
    source_pca.fit(source.vertices)
    target_pca = PCA(n_components=3)
    target_pca.fit(target.vertices)

    # get axis of src and dst, each column for an axis
    s = source_pca.components_.T.copy()
    t = target_pca.components_.T.copy()

    with open(src_centers, 'r') as f:
        src_points = dict(sorted(json.load(f).items()))

    with open(dst_centers, 'r') as f:
        dst_points = dict(sorted(json.load(f).items()))

    assert list(sorted(src_points.keys())) == list(sorted(dst_points.keys())), 'id not paired'

    src_coords = np.array(list(src_points.values()))
    dst_coords = np.array(list(dst_points.values()))

    # get the directions by SVD
    # 零均值化
    mu_src = np.mean(src_coords, axis=0)
    mu_dst = np.mean(dst_coords, axis=0)
    ss = src_coords - mu_src
    dd = dst_coords - mu_dst
    # 协方差矩阵
    H = ss.T @ dd
    # S对角线上是奇异值σ
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if R[2, 2] < 0:
        R[:, 2] *= -1

    # fit the direction of axis
    rotation = t @ s.T
    for col in range(3):
        idx = np.argmax(np.abs(R[:, col])), col
        if R[idx] * rotation[idx] < 0:
            rotation[:, col] *= -1

    new_coords = source.vertices.copy()
    new_coords = new_coords @ rotation.T

    # find center points
    cpS = np.mean(new_coords, axis=0)
    cpT = np.mean(target.vertices, axis=0)

    # translation
    # remind that the dental model only contains tooth crowns, which causes a gap in height
    axis = target_pca.components_.T[:, -1]
    sH = new_coords @ axis.T
    sLen = sH.max() - sH.min()
    tH = target.vertices @ axis.T
    tLen = tH.max() - tH.min()

    translation = cpT - cpS - axis.T * (tLen - sLen) / 2
    new_coords += translation

    vertex_normals = source.vertex_normals @ rotation.T
    face_normals = source.face_normals @ rotation.T

    mesh = trimesh.Trimesh(vertices=new_coords, faces=source.faces,
                           face_normals=face_normals,
                           vertex_normals=vertex_normals)
    mesh.export(output_prealigned)

    # save the transformation
    transformation = np.eye(4)
    transformation[:3, :3] = rotation
    transformation[:3, -1] = translation
    transformation = transformation

    with open(output_transformation, 'w') as f:
        json.dump(transformation.tolist(), f, indent=4)
        print('Save: %s' % output_transformation)

    return transformation


def rigid_transform_patient(src, dst, output_aligned, output_transformation):
    """
    Registration
    Stage 2: rigid ICP on patient-level

    Args:
        src (.off): merged 3D dental model path
        dst (.off): merged CBCT model path
        output_aligned (.off): output mesh path
        output_transformation (.json): output path of transformation

    Returns:
        transformation (np.array)

    """

    source = o3d.io.read_triangle_mesh(src)
    target = o3d.io.read_triangle_mesh(dst)

    sourceV = np.asarray(source.vertices)
    targetV = np.asarray(target.vertices)

    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(sourceV)
    source_pcd.estimate_normals()
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(targetV)
    target_pcd.estimate_normals()

    reg_p2l = o3d.pipelines.registration.registration_icp(
        source=source_pcd,
        target=target_pcd,
        max_correspondence_distance=0.5,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
    )

    print(''.center(50, '='))
    print(reg_p2l)
    print(f'fitness: {reg_p2l.fitness} \t inlier_mse: {reg_p2l.inlier_rmse}')
    print(''.center(50, '='))

    if reg_p2l.fitness < 0.8:
        warnings.warn('fitness < 0.8')

    transformation = reg_p2l.transformation
    source.transform(transformation)

    o3d.io.write_triangle_mesh(os.path.join(output_aligned), source,write_vertex_normals=True)
    print('Save: %s' % output_aligned)

    with open(output_transformation, 'w') as f:
        json.dump(transformation.tolist(), f, indent=4)
        print('Save: %s' % output_transformation)

    return transformation


def transform(vertices, transformation):
    """
    Apply the transformation for vertices.

    Args:
        vertices (np.array): vertices
        transformation (np.array): transformation

    Returns:
        transformed vertices (np.array)

    """
    pts = np.concatenate([vertices, np.ones((vertices.shape[0], 1))], axis=-1)
    res = (pts @ transformation.T)[:, :3]

    return res


def rigid_transform_tooth(src_path, dst_path, stage1, stage2, output_merged,
                          output_aligned, output_transformation_tooth, output_transformation_all):
    #,aligned_dental_ok):
    """
    Registration
    Stage 3: rigid ICP on tooth-level

    Args:
        src_path (directory): path of tooth-level dental models
        dst_path (directory): path of tooth-level CBCT models
        stage1 (.json): stage1 transformation
        stage2 (.json): stage2 transformation
        output_merged (.off): the merged aligned mesh
        output_aligned (directory): path of aligned dental models
        output_transformation_tooth (.json): stage3 transformation
        output_transformation_all (.json): combined transformation from stage1 to stage3
        aligned_dental_ok (directory): path of aligned successed dental models

    Returns:
        transformation (np.array)

    """

    tooth_info = {}
    all_info = {}
    src_files = glob.glob(os.path.join(src_path, '*.off'))
    dst_files = glob.glob(os.path.join(dst_path, '*.off'))

    assert list(map(lambda x: os.path.basename(x).split('.')[0], src_files)) == list(
        map(lambda x: os.path.basename(x).split('.')[0], dst_files))

    with open(stage1, 'r') as f:
        stage1_transformation = np.array(json.load(f))
    with open(stage2, 'r') as f:
        stage2_transformation = np.array(json.load(f))

    for file in src_files:
        id = os.path.basename(file).split('.')[0]

        src_mesh = trimesh.load(file)
        src_vertices = src_mesh.vertices

        dst_mesh = trimesh.load(os.path.join(dst_path, os.path.basename(file)))

        # apply patient-level transformation
        patient_transformation = stage2_transformation @ stage1_transformation
        new_src_vertices = transform(src_vertices, patient_transformation)

        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(new_src_vertices)
        source_pcd.estimate_normals()
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(dst_mesh.vertices)
        target_pcd.estimate_normals()

        reg_p2l = o3d.pipelines.registration.registration_icp(
            source=source_pcd,
            target=target_pcd,
            max_correspondence_distance=0.5,
            init=np.eye(4),
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
        )

        print('id: {:s} \t fitness: {:20s} inlier_rmse: {:20s}'.format(id, str(reg_p2l.fitness),
                                                                       str(reg_p2l.inlier_rmse)))
        if reg_p2l.fitness < 0.8:
            tooth_transformation = np.eye(4)
            print(f'---- WARNING ---- id: {id} \t fitness < 0.8, skip tooth-level transformation')
        else:
            tooth_transformation = reg_p2l.transformation
            new_src_vertices = transform(new_src_vertices, tooth_transformation)

        tooth_info[id] = tooth_transformation.tolist()
        all_info[id] = (tooth_transformation @ patient_transformation).tolist()

        new_mesh = trimesh.Trimesh(vertices=new_src_vertices, faces=src_mesh.faces,
                                   face_normals=src_mesh.face_normals,
                                   vertex_normals=src_mesh.vertex_normals)

        output_aligned_path = os.path.join(output_aligned, os.path.basename(file))
        new_mesh.export(output_aligned_path)

    with open(output_transformation_tooth, 'w') as f:
        json.dump(tooth_info, f, indent=4)
        print('Save: %s' % output_transformation_tooth)

    with open(output_transformation_all, 'w') as f:
        json.dump(all_info, f, indent=4)
        print('Save: %s' % output_transformation_all)

    ms = pymeshlab.MeshSet()
    mesh_list = glob.glob(os.path.join(output_aligned, '*.off'))

    for mesh in mesh_list:
        ms.load_new_mesh(mesh)

    ms.generate_by_merging_visible_meshes()
    ms.save_current_mesh(output_merged, save_vertex_color=False, save_face_color=False)
    print('Save: %s' % output_merged)
    print('End of registration'.center(100, '='))

def compare_num_move(less_list,more_list):
    file_path1 = less_list
    file_path2 = more_list
    f1 = []
    f2 = []
    for filename in os.listdir(file_path1):
        f1.append(filename)
    for filename in os.listdir(file_path2):
        f2.append(filename)
    output_path=os.path.join(os.path.dirname(more_list),os.path.basename(more_list)+'_more')
    output_path1 = os.path.join(os.path.dirname(less_list), os.path.basename(less_list) + '_more')
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    if not os.path.exists(output_path1):
        os.makedirs(output_path1, exist_ok=True)
    for filename2 in os.listdir(file_path2):
        if filename2 not in f1:
            shutil.move(os.path.join(file_path2, filename2), os.path.join(output_path, filename2))
    for filename1 in os.listdir(file_path1):
        if filename1 not in f2:
            shutil.move(os.path.join(file_path1, filename1), os.path.join(output_path1, filename1))

def delete(source_folder,destination_folder):
    source_files = os.listdir(source_folder)
    for file_name in source_files:
        source_file_path = os.path.join(source_folder, file_name)
        destination_file_path = os.path.join(destination_folder, file_name)
        if os.path.exists(destination_file_path):
            os.remove(destination_file_path)
            shutil.move(source_file_path, destination_file_path)

    print("File move completed！")


if __name__ == '__main__':
    dirs = glob.glob(os.path.join(r'E:\PD\test', '*'))

    for row in ['lower','upper']:
        for d in dirs:
            cfg = utils.get_path(os.path.join(d, row))
            print(d)
            out_more = os.path.join(d, row, '00_cbct_more')
            if os.path.exists(out_more):
                delete(cfg.path.cbct, out_more)
            if len(glob.glob(os.path.join(cfg.path.cbct, '*'))) <= len(glob.glob(os.path.join(cfg.path.dental, '*'))):
                compare_num_move(less_list=cfg.path.cbct, more_list=cfg.path.dental)
            elif len(glob.glob(os.path.join(cfg.path.cbct, '*'))) > len(glob.glob(os.path.join(cfg.path.dental, '*'))):
                compare_num_move(less_list=cfg.path.dental, more_list=cfg.path.cbct)
            else:
                print('same num!')

            mesh_merge(mesh_list=cfg.path.cbct, output_merged=cfg.file.cbct_merged,
                       output_centers=cfg.file.cbct_centers)
            mesh_merge(mesh_list=cfg.path.dental, output_merged=cfg.file.dental_merged,
                       output_centers=cfg.file.dental_centers)

            align_pca(src=cfg.file.dental_merged, dst=cfg.file.cbct_merged,
                      src_centers=cfg.file.dental_centers, dst_centers=cfg.file.cbct_centers,
                      output_prealigned=cfg.file.dental_prealigned, output_transformation=cfg.file.stage1)
            rigid_transform_patient(src=cfg.file.dental_prealigned, dst=cfg.file.cbct_merged,
                                    output_aligned=cfg.file.dental_patient_aligned,
                                    output_transformation=cfg.file.stage2)
            rigid_transform_tooth(src_path=cfg.path.dental, dst_path=cfg.path.cbct,
                                  stage1=cfg.file.stage1, stage2=cfg.file.stage2,
                                  output_merged=cfg.file.dental_tooth_aligned,
                                  output_aligned=cfg.path.aligned_dental,
                                  output_transformation_tooth=cfg.file.stage3,
                                  output_transformation_all=cfg.file.transformation)
