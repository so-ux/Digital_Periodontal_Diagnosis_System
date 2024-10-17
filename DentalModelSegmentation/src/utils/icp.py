import open3d as o3d
import numpy as np
import pickle

#读取电脑中的 ply 点云文件
from src.utils.input import read_obj

# source_data, _, normals = read_obj("/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/3D_scans_per_patient_obj_files/0EAKT1CU/0EAKT1CU_lower.obj", True)
# source_data = np.concatenate((source_data, normals), axis=-1)
from src.utils.pc_normalize import pc_normalize

source_data = np.loadtxt("/run/media/zsj/DATA/Data/miccai/3D_scans_per_patient_obj_files/01EGTCG0/01EGTCG0_lower.xyz")
target_data = np.loadtxt("/run/media/zsj/DATA/Data/Gordon/model_1/HBF_10227_L_S.xyz")

source_data[:, 0:3], _, _ = pc_normalize(source_data[:, 0:3])
target_data[:, 0:3], _, _ = pc_normalize(target_data[:, 0:3])

source = o3d.geometry.PointCloud()
source.points = o3d.utility.Vector3dVector(source_data[:, 0:3])
source.normals = o3d.utility.Vector3dVector(source_data[:, 3:6])
target = o3d.geometry.PointCloud()
target.points = o3d.utility.Vector3dVector(target_data[:, 0:3])
target.normals = o3d.utility.Vector3dVector(target_data[:, 3:6])


#为两个点云上上不同的颜色
source.paint_uniform_color([1, 0.706, 0])    #source 为黄色
target.paint_uniform_color([0, 0.651, 0.929])#target 为蓝色

threshold = 1.0  #移动范围的阀值
trans_init = np.asarray([[1,0,0,0],   # 4x4 identity matrix，这是一个转换矩阵，
                         [0,1,0,0],   # 象征着没有任何位移，没有任何旋转，我们输入
                         [0,0,1,0],   # 这个矩阵为初始变换
                         [0,0,0,1]])

# src_feats = o3d.pipelines.registration.Feature()
# feats_data = np.loadtxt('/home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/fpfh.feats')
# src_feats.data = feats_data
src_feats = o3d.pipelines.registration.compute_fpfh_feature(source, o3d.geometry.KDTreeSearchParamKNN(knn=32))
print(src_feats.data)
# with open('./fpfh_feats.pkl', 'wb') as fp:
#     pickle.dump({
#         "pc": source_data[:, 0:6],
#         "feats": np.asarray(src_feats.data)
#     }, fp)
#     fp.close()
print('src_feats complete')
target_feats = o3d.pipelines.registration.compute_fpfh_feature(target, o3d.geometry.KDTreeSearchParamKNN(knn=32))

print('src_feats complete')

reg_p2p = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
    source, target, src_feats, target_feats
)

#将我们的矩阵依照输出的变换矩阵进行变换
print(reg_p2p)
source.transform(reg_p2p.transformation)
target.translate([0, 0, 0])

#创建一个 o3d.visualizer class
vis = o3d.visualization.Visualizer()
vis.create_window()

#将两个点云放入visualizer
vis.add_geometry(source)
vis.add_geometry(target)

#让visualizer渲染点云
# vis.update_geometry()
vis.poll_events()
vis.update_renderer()

vis.run()

