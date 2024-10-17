from measure.find_gumline import find_gumline
from measure.tooth_axis import find_tooth_axis
import datetime
import numpy as np
import pymeshlab
from numpy import tile
from skimage import measure
from skimage import morphology
import SimpleITK as sitk
from collections import OrderedDict
import glob
import os
import measure.utils as utils
from measure.utils import Logger
import trimesh
from sklearn.neighbors import KDTree
from measure.nii_bone_line_v2 import postprocess_with_tooth
from numpy import multiply


def dst(p, q, rs):
    x = p-q
    dst=np.linalg.norm(
        np.outer(np.dot(rs-q, x)/np.dot(x, x), x)+q-rs,
        axis=1)
    min_dst=np.min(dst)
    min_key=np.argmin(dst)
    return min_dst,min_key

def calVectorFrom2Points(Point_1, Point_2):
    d = np.linalg.norm(Point_2 - Point_1)
    return np.array((Point_2 - Point_1) / d)

def calDistanceFrom2Points(P_1, P_2):
    P_1 = np.array(P_1)
    P_2 = np.array(P_2)

    diff = P_1 - P_2
    Dis = np.linalg.norm(diff, axis=0)

    return Dis


def tooth_bone_linetrimesh(tooth_path, bone_path, out):

    starttime11 = datetime.datetime.now()
    cbct_tooth = trimesh.load(tooth_path, force='mesh')
    bone_whole = trimesh.load(bone_path, force='mesh')

    interab_mesh = trimesh.boolean.intersection(list([cbct_tooth, bone_whole]), engine='blender')
    diffba_mesh = trimesh.boolean.difference(list([bone_whole, cbct_tooth]), engine='blender')
    diffab_mesh = trimesh.boolean.difference(list([cbct_tooth, bone_whole]), engine='blender')

    interab = interab_mesh.vertices
    diffba = diffba_mesh.vertices
    diffab = diffab_mesh.vertices

    tree_diffba = KDTree(diffba)
    diffba_distances, diffba_indices = tree_diffba.query(interab)
    diffba_indices = diffba_indices[diffba_distances == 0].squeeze()
    diffba_out = trimesh.Trimesh(vertices=diffba[diffba_indices])
    # diffba_out.export(r'D:\9th-3D-202301\new\92\R000092\no smooth\diffba.off')

    tree_diffab = KDTree(diffab)
    diffab_distances, diffab_indices = tree_diffab.query(interab)
    diffab_indices = diffab_indices[diffab_distances == 0].squeeze()
    diffab_out = trimesh.Trimesh(vertices=diffab[diffab_indices])
    # diffab_out.export(r'D:\9th-3D-202301\new\92\R000092\no smooth\diffab.off')

    tree_res = KDTree(diffba_out.vertices)
    res_distances, res_indices = tree_res.query(diffab_out.vertices)
    res_indices = res_indices[res_distances == 0].squeeze()
    res_out = trimesh.Trimesh(vertices=diffba_out.vertices[res_indices])
    res_out.export(out)

    starttime21 = datetime.datetime.now()
    print('The tooth_bone_line time is %s:'%(starttime21 - starttime11))

    # line_v2 = np.array(diffba_out.vertices[res_indices])
    return diffba_out.vertices[res_indices]

# use pymeshlab  maybe obtain difference results
def tooth_bone_line(tooth_path, bone_path, out):

    starttime11 = datetime.datetime.now()
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(tooth_path)
    ms.load_new_mesh(bone_path)
    ms.generate_boolean_intersection(first_mesh=0, second_mesh=1)
    interab = ms.current_mesh().vertex_matrix()
    ms.generate_boolean_difference(first_mesh=1, second_mesh=0)
    diffba = ms.current_mesh().vertex_matrix()
    ms.generate_boolean_difference(first_mesh=0, second_mesh=1)
    diffab = ms.current_mesh().vertex_matrix()

    tree_diffba = KDTree(diffba)
    diffba_distances, diffba_indices = tree_diffba.query(interab)
    diffba_indices = diffba_indices[diffba_distances == 0].squeeze()
    diffba_out = trimesh.Trimesh(vertices=diffba[diffba_indices])
    # diffba_out.export(r'D:\9th-3D-202301\new\92\R000092\no smooth\diffba.off')

    tree_diffab = KDTree(diffab)
    diffab_distances, diffab_indices = tree_diffab.query(interab)
    diffab_indices = diffab_indices[diffab_distances == 0].squeeze()
    diffab_out = trimesh.Trimesh(vertices=diffab[diffab_indices])
    # diffab_out.export(r'D:\9th-3D-202301\new\92\R000092\no smooth\diffab.off')

    tree_res = KDTree(diffba_out.vertices)
    res_distances, res_indices = tree_res.query(diffab_out.vertices)
    res_indices = res_indices[res_distances == 0].squeeze()
    res_out = trimesh.Trimesh(vertices=diffba_out.vertices[res_indices])
    res_out.export(out)

    starttime21 = datetime.datetime.now()
    print('The tooth_bone_line time is %s:'%(starttime21 - starttime11))

    # line_v2 = np.array(diffba_out.vertices[res_indices])
    return diffba_out.vertices[res_indices]


def nii2mesh(nii_array,spacing,output_path):
    verts, faces, normals, values = measure.marching_cubes(nii_array, 0)
    ms = pymeshlab.MeshSet()
    mesh = pymeshlab.Mesh(vertex_matrix=verts, face_matrix=faces)
    ms.add_mesh(mesh)

    ms.meshing_invert_face_orientation()

    p = pymeshlab.Percentage(50)  # 50
    ms.meshing_remove_connected_component_by_diameter(mincomponentdiag=p)

    # ms.apply_coord_laplacian_smoothing(stepsmoothnum=10, cotangentweight=False)#10

    ms.compute_matrix_from_scaling_or_normalization(axisx=spacing[0], axisy=spacing[0], axisz=spacing[0])

    ms.save_current_mesh(output_path, save_vertex_color=False, save_face_color=False)

def nii(nii_path,row,mesh_path):
    bone = sitk.ReadImage(nii_path)
    arr = sitk.GetArrayFromImage(bone).T
    spacing = np.round(bone.GetSpacing(), 5)

    if row == 'upper':
        half_bone = (arr == 1)
    else:
        half_bone = (arr == 2)
    half_bone = morphology.remove_small_objects(half_bone.astype(bool), 200000)
    nii2mesh(half_bone,spacing,output_path=mesh_path)

def bone_line(bone_tree1,bone_faces1,bone_verts1,gum_line_verts1, eigen_vectors1,bone_line_outpath1):
    triangle_index = bone_tree1.intersects_first(gum_line_verts1, eigen_vectors1)
    faces_planes = bone_faces1[triangle_index]
    positions = bone_verts1[faces_planes]
    plane_origins = positions[:, 0, :]
    plane_normals, triangle_ok = trimesh.triangles.normals(positions)
    bone_point, valid, distance = trimesh.intersections.planes_lines(plane_origins[triangle_ok], plane_normals, gum_line_verts1[triangle_ok],
                                                                      eigen_vectors1[triangle_ok], return_distance=True,
                                                                      return_denom=False)
    # save
    bone_line = trimesh.Trimesh(vertices=bone_point)
    bone_line.export(bone_line_outpath1)
    return bone_point,triangle_ok


def curve_fitting(lab_img,label_tooth,order=4):
    '''
        Curve fitting to the label image.
        @params:
            lab_img [np.ndarray]: label image
            order [int]: order of the polynomial
        @returns:
            curve [np.ndarray]: fitted parameters
    '''

    upper_centers = {}
    lower_centers = {}

    row_upper_mask = ((20 < label_tooth) & (label_tooth < 29)) | ((10 < label_tooth) & (label_tooth < 19))
    row_lower_mask = ((30 < label_tooth) & (label_tooth < 39)) | ((40 < label_tooth) & (label_tooth < 49))

    label_indices = np.indices(lab_img.shape)
    label_indices = label_indices.transpose(1, 2, 3, 0)  # Transpose to (x, y, z, indices)

    for lt in label_tooth[row_upper_mask]:
        upper_label_mask = lab_img == lt
        upper_label_center = np.mean(label_indices[upper_label_mask], axis=0)
        upper_centers[lt] = upper_label_center

    for lt in label_tooth[row_lower_mask]:
        lower_label_mask = lab_img == lt
        lower_label_center = np.mean(label_indices[lower_label_mask], axis=0)
        lower_centers[lt] = lower_label_center

    upper_center_array = np.array(list(upper_centers.values()))
    lower_center_array = np.array(list(lower_centers.values()))

    # Use numpy's polyfit function directly
    upper_curve = np.polyfit(upper_center_array[:, 0], upper_center_array[:, 1], order)
    upper_func = np.poly1d(upper_curve)

    lower_curve = np.polyfit(lower_center_array[:, 0], lower_center_array[:, 1], order)
    lower_func = np.poly1d(lower_curve)

    return upper_func, upper_centers,lower_func,lower_centers

def closest_point(points, target):
    points = np.array(points)
    target = np.array(target)

    distances = np.linalg.norm(points - target, axis=1)
    min_dist_idx = np.argmin(distances)

    min_dist = distances[min_dist_idx]
    closest = points[min_dist_idx]

    return closest


def find_closest_tangent(func,center,id):
    x = [v[0] for k, v in center.items() if k == int(id)][0]
    y = [v[1] for k, v in center.items() if k == int(id)][0]

    points_x=np.array(np.linspace(x-5,x+5,num=20))
    points_y=func(points_x)
    points=np.stack((points_x,points_y), axis=-1)

    closest=closest_point(points, np.stack((x,y), axis=-1))
    slope = func.deriv()(closest[0])
    return slope


def point_project_array(points, para):
    '''
    :param points:
    :param para: para[0, 0, 1, 0]
    :return:
    '''
    para = np.array(para)
    d = para[0] ** 2 + para[1] ** 2 + para[2] ** 2
    t = -(np.matmul(points[:, :3], para[:3].T) + para[3]) / d
    points = np.matmul(t[:, np.newaxis], para[np.newaxis, :3]) + points[:, :3]
    return points

def angle(axis,points,center_point):
    sectors = [[] for _ in range(6)]
    for i in range(len(points)):
        b=points[i]-center_point
        cos_ = axis.dot(b) / (np.linalg.norm(axis) * np.linalg.norm(b))
        angle_hu=np.arccos(cos_)
        angle_d = angle_hu * 180 / np.pi
        if axis[0] * b[1] - axis[1] * b[0] < 0:
            angle_d = 360 - angle_d
        sector = int(angle_d / 60)
        sectors[sector].append(i)
    return sectors


def GBD(cfg, d, row,sector_flag):
    starttime1 = datetime.datetime.now()
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # find gingival contour
    find_gumline(cfg, d, row)
    starttime2 = datetime.datetime.now()
    print('Time of finding gingival contour:', starttime2 - starttime1)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # find tooth’s longitudinal axis
    for file in os.listdir(cfg.path.cbct):
        mesh = trimesh.load(os.path.join(cfg.path.cbct, file), force='mesh', process=False)
        find_tooth_axis(mesh, cfg.path.tooth_axis, file)
    print('Finished: axis')
    starttime3 = datetime.datetime.now()
    print('Time of finding tooth’s longitudinal axis:', starttime3 - starttime1)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # measure distance
    mask = sitk.ReadImage(os.path.join(cfg.root_file.cbct_mask))
    spacing = mask.GetSpacing()
    arr = sitk.GetArrayFromImage(mask).T
    label_tooth = np.unique(arr)[1:]

    if sector_flag:
        upper_func, upper_centers, lower_func, lower_centers = curve_fitting(arr, label_tooth, order=4)

    ids = [os.path.basename(e).split('.off')[0] for e in
           glob.glob(os.path.join(d, row, cfg.path.gum_line_refined_complemented, '*.off'))]

    # postprocess of bone------------------------------------------------------------------------------------------
    mesh_output_path = os.path.join(d, row, os.path.basename(d) + '_bone_processed.off')
    cbct_bone_processed = os.path.join(d, os.path.basename(d) + '_bone_processed.nii.gz')
    if not os.path.exists(cbct_bone_processed):
        print('strat bone postprocessing and converting into mesh!')
        postprocess_with_tooth(arr, spacing, label_tooth, cfg.root_file.cbct_bone, cbct_bone_processed)
    else:
        print('The bone has been postprocessed!')

    if not os.path.exists(mesh_output_path):
        nii(cbct_bone_processed, row, mesh_output_path)
    else:
        print('The bone has been converted into mesh!')

    bone = trimesh.load(mesh_output_path, force='mesh')
    #
    bone_verts = np.asarray(bone.vertices)
    bone_faces = np.asarray(bone.faces)
    bone_tree = trimesh.ray.ray_triangle.RayMeshIntersector(bone)

    starttime4 = datetime.datetime.now()
    print('The time for preprocessing stage:', starttime4 - starttime3)

    for id in ids:
        print(id)
        starttime5 = datetime.datetime.now()
        # tooth_bone_line------------------------------------------------------------------
        print('start finding tooth_bone_line!')
        tooth = os.path.join(cfg.path.cbct, id + '.off')
        out = os.path.join(cfg.path.tooth_bone_line, id + '.off')
        line_v = tooth_bone_line(tooth, mesh_output_path, out)

        gum_line = trimesh.load(os.path.join(cfg.path.gum_line_refined_complemented, id + '.off'))
        gum_line_verts = np.asarray(gum_line.vertices)
        tooth_axis = trimesh.load(os.path.join(cfg.path.tooth_axis, id + '.off'))
        tooth_axis_verts = np.asarray(tooth_axis.vertices)

        # tooth eigen_vectors--------------------------------------------------------
        point1 = tooth_axis_verts[0]
        point2 = tooth_axis_verts[1]
        # RAI
        if row == 'lower':
            if sector_flag:
                func = lower_func
                centers = lower_centers
            else:
                pass
            if point1[2] > point2[2]:
                eigen_vector = calVectorFrom2Points(point1, point2)
            else:
                eigen_vector = calVectorFrom2Points(point2, point1)
        else:
            if sector_flag:
                func = upper_func
                centers = upper_centers
            else:
                pass
            if point1[2] > point2[2]:
                eigen_vector = calVectorFrom2Points(point2, point1)
            else:
                eigen_vector = calVectorFrom2Points(point1, point2)

        eigen_vectors = tile(eigen_vector, (gum_line_verts.shape[0], 1))

        # bone_line--------------------------------------------------------------------
        print('start finding bone_line!')
        starttime6 = datetime.datetime.now()
        bone_line_outpath = os.path.join(cfg.path.bone_line, id + '.off')
        bone_points, triangle_ok = bone_line(bone_tree, bone_faces, bone_verts, gum_line_verts, eigen_vectors,
                                             bone_line_outpath)

        starttime7 = datetime.datetime.now()
        print(starttime7 - starttime6)

        # bone_line_refined-------------------------------------------------------------
        print('start refining bone_line!')
        line_key = []
        dsts = []

        gum_key = []
        distance = Logger(os.path.join(cfg.path.csv), id)

        line_v_new = np.zeros_like(gum_line_verts)
        for i in range(gum_line_verts.shape[0]):
            min_dst, min_key = dst(gum_line_verts[i], bone_points[i], line_v)
            line_key.append(min_key)
            dst1 = calDistanceFrom2Points(gum_line_verts[i], line_v[min_key])
            line_v_new[i, :] = line_v[min_key]
            dsts.append(dst1)

        mean = np.mean(dsts)
        dsts_new = dsts.copy()
        line_key_new = line_key.copy()
        gum_line_verts_copy = gum_line_verts.copy()
        a = []
        for j in range(len(dsts)):
            if dsts[j] > mean + int(5):
                a.append(j)
                dsts_new.remove(dsts[j])
                line_key_new.remove(line_key[j])

            else:
                index_results = OrderedDict(
                    {'id': j, 'gum_line_verts': str(gum_line_verts[j]), 'line_v': str(line_v[line_key[j]]),
                     'distance_infer': dsts[j]})
                distance.update(os.path.basename(d), index_results)

        line = line_v[np.array(line_key_new)]
        line_trimesh = trimesh.Trimesh(vertices=line)
        line_trimesh.export(os.path.join(cfg.path.bone_line_final, id + '.off'))

        starttime8 = datetime.datetime.now()
        print(starttime8 - starttime7)

        if sector_flag:
            part_gum = Logger(os.path.join(cfg.path.csv), id + '_part_gum')
            gum_line_verts_new = np.delete(gum_line_verts_copy, a, axis=0)
            slope = find_closest_tangent(func, centers, id)

            mesh = trimesh.load(os.path.join(cfg.path.cbct, id + '.off'))
            center_point = np.mean(np.asarray(mesh.vertices), axis=0)

            intercept = -(eigen_vector[0] * center_point[0] + eigen_vector[1] * center_point[1] +
                          eigen_vector[2] * center_point[2])

            project_pane = [eigen_vector[0], eigen_vector[1], eigen_vector[2], intercept]
            gum_line_project = point_project_array(gum_line_verts_new, project_pane)

            tangent_direction = np.array([1, slope, 0])
            tangent_direction /= np.linalg.norm(tangent_direction)
            tangent_line = [center_point + multiply(tangent_direction, j) for j in range(-17, 5, 1)]
            tangent_p2 = point_project_array(np.array(tangent_line), project_pane)

            if 10 < int(id) < 19 or 40 < int(id) < 49:
                part = ['DB', 'B', 'MB', 'ML', 'L', 'DL']
            elif 20 < int(id) < 29 or 30 < int(id) < 39:
                part = ['MB', 'B', 'DB', 'DL', 'L', 'ML']
            start_axis = tangent_p2[0] - center_point
            sectors = angle(axis=start_axis, points=gum_line_project, center_point=center_point)

            gum_6 = []
            bone_6 = []
            for i in range(len(sectors)):
                gum_line_sector = trimesh.Trimesh(vertices=np.array(gum_line_verts_new[sectors[i]]))
                # gum_line_sector.export(os.path.join(r'D:\9th-3D-202301\TEST\part_gum',os.path.basename(d)+'_'+id+'_'+str(i)+'.off'))
                part_distance = [dsts_new[sectors[i][j]] for j in range(len(sectors[i]))]
                max_part_dst = np.max(part_distance)
                max_part_dst_key = dsts_new.index(max_part_dst)
                gum_6.append(gum_line_verts_new[max_part_dst_key])
                bone_6.append(line_v_new[max_part_dst_key])
                index_results1 = OrderedDict(
                    {'part': part[i], 'key': max_part_dst_key,
                     'gum_line_verts': str(gum_line_verts_new[max_part_dst_key]),
                     'line_v': str(line_v_new[max_part_dst_key]), 'distance_infer': max_part_dst}
                )
                part_gum.update(os.path.basename(d), index_results1)

            # save 6 points
            gum_6_line_trimesh = trimesh.Trimesh(vertices=np.asarray(gum_6))
            gum_6_line_trimesh.export(os.path.join(cfg.path.gum_6, id + '.off'))

            bone_6_line_trimesh = trimesh.Trimesh(vertices=np.asarray(bone_6))
            bone_6_line_trimesh.export(os.path.join(cfg.path.bone_6, id + '.off'))

        starttime9 = datetime.datetime.now()
        print('Time to process a tooth:', starttime9 - starttime4)
    starttime10 = datetime.datetime.now()
    print('Time to process %s of %s is %s:' % (row, d, starttime10 - starttime3))


if __name__ == '__main__':
    dirs = glob.glob(os.path.join(r'E:\PD\test', '*'))

    # Do we need to calculate the distance between 6 points? 0--No, 1--Yes
    sector_flag = 0

    for row in ['lower', 'upper']:
        for d in dirs:
            cfg = utils.get_path(os.path.join(d, row))
            print(d)
            GBD(cfg, d, row,sector_flag)