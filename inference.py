import glob
import os
import pandas as pd
import os.path as op
import SimpleITK as sitk
from CBCTSegmentation.Teeth_bone_segmentation import infer
import argparse
from measure.utils import get_path
from ModelFusion.registration import delete, compare_num_move, mesh_merge, align_pca, rigid_transform_patient, \
    rigid_transform_tooth
from ModelFusion.segmentation import mesh_generate_dental, mesh_generate_cbct
from measure.GBD_module import GBD


def makedirs(folder):
    if not op.exists(folder):
        os.makedirs(folder)


def check_RAI(image, mask, image_spacing):
    if image.GetDirection() != (1, 0, 0, 0, 1, 0, 0, 0, 1):
        print('%s image orientation is not RAI' % patient)
        image.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
        image.SetSpacing(image_spacing)
        sitk.WriteImage(image, os.path.join(patient_path, patient + '_image.nii.gz'))
    if mask.GetDirection() != (1, 0, 0, 0, 1, 0, 0, 0, 1):
        print('%s mask orientation is not RAI' % patient)
        mask.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
        mask.SetSpacing(image_spacing)
        sitk.WriteImage(mask, os.path.join(patient_path, patient + '_label.nii.gz'))
    else:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CBCT image segmentation')
    parser.add_argument('--input', '-i', type=str, help='Path of patients', required=True)
    parser.add_argument('--model_path', '-m', type=str, help='Path of trained models', required=False,
                        default='../model')
    parser.add_argument('--b_r', type=str, default='5000',
                        help='Remove non-bone parts smaller than this value in size.')
    parser.add_argument('--six_sector', '-six', help='Weather calculate six site distance',
                        default=False)
    args = parser.parse_args()
    path = args.input
    model_path = args.model_path

    dict_from_csv = pd.read_csv(r"CBCTSegmentation/mapping.CSV", header=None, index_col=0).squeeze('columns').to_dict()
    mapped = {v: k for k, v in dict_from_csv.items()}
    # ---------------------------------------------------------------------------------------------------
    for patient in os.listdir(path):
        print('**********process the patient:', patient)
        patient_path = op.join(path, patient)
        output_path = op.join(patient_path, 'oral scan seg')
        makedirs(output_path)

        # Step1: IOS model segmentation
        # for name in ['_LowerJawScan','_UpperJawScan']:
        #     ios_path=op.join(path,patient,patient+name+'.off')
        #     mesh = trimesh.load()
        #     inference(ios_path, output_path, args.model_path, rotation=True, debug_vis=False)

        # Step2: CBCT image segmentation
        # infer(path, patient, args.model_path, mapped)

        # Step3: Multimodal data fusion
        # check whether orientation is RAI

        image_path = op.join(patient_path, patient + '_image.nii.gz')
        image = sitk.ReadImage(image_path)
        image_spacing = image.GetSpacing()
        mask_path = op.join(patient_path, patient + '_label.nii.gz')
        mask = sitk.ReadImage(mask_path)
        if not op.exists(mask_path):
            print('Do not have CBCT label!')
        else:
            check_RAI(image, mask, image_spacing)

        # 1. segmentation
        dental_ids = []
        cfg = get_path(op.join(patient_path, 'lower'))
        dental_ids += mesh_generate_dental(mesh=cfg.oral_scan_seg_file.lower_dental_mesh,
                                           label=cfg.oral_scan_seg_file.lower_dental_label,
                                           output=cfg.path.dental)
        cfg = get_path(op.join(patient_path, 'upper'))
        dental_ids += mesh_generate_dental(mesh=cfg.oral_scan_seg_file.upper_dental_mesh,
                                           label=cfg.oral_scan_seg_file.upper_dental_label,
                                           output=cfg.path.dental)
        # if the CBCT meshes have been generated with other softwares (e.g., ITK-SNAP)
        # please comment the following line to skip this step, but remember to scale the generated meshes with spacing
        cbct_ids = mesh_generate_cbct(image=cfg.root_file.cbct_image,
                                      mask=cfg.root_file.cbct_mask,
                                      output=cfg.path.cbct)

        for row in ['lower', 'upper']:
            cfg = get_path(op.join(patient_path, row))
            out_more = op.join(patient_path, row, '00_cbct_more')
            if op.exists(out_more):
                delete(cfg.path.cbct, out_more)
            if len(glob.glob(os.path.join(cfg.path.cbct, '*'))) <= len(glob.glob(os.path.join(cfg.path.dental, '*'))):
                compare_num_move(less_list=cfg.path.cbct, more_list=cfg.path.dental)
            elif len(glob.glob(os.path.join(cfg.path.cbct, '*'))) > len(glob.glob(os.path.join(cfg.path.dental, '*'))):
                compare_num_move(less_list=cfg.path.dental, more_list=cfg.path.cbct)
            else:
                print('same num!')

            # 2. registration
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

            # Step4: Gingiva-bone distance measurement
            GBD(cfg, patient_path, row, args.six_sector)
