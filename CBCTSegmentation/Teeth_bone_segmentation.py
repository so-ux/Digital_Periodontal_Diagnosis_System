import SimpleITK as sitk
from skimage.segmentation import watershed
from skimage import morphology
from scipy import ndimage
import argparse
import numpy as np
from nnunet.inference.predict import predict_cases_fastest, predict_cases
import datetime
import os
import shutil
import torch
import pandas as pd
import os.path as op

def load_model(model_path):
    # nnunet model for predicting _2_label_map
    model_2_label = op.join(model_path,'Task610_Teeth/nnUNetTrainerV2__nnUNetPlansv2.1/')
    # nnunet model for predicting _3_label_map
    model_3_label = op.join(model_path,'707_3_label_model/nnUNetTrainerV2__nnUNetPlansv2.1/')

    # nnunet model for predicting ID_map
    model_ID = op.join(model_path,'id_model_cropped/nnUNetTrainerNoMirroring__nnUNetPlansv2.1/')

    # nnunet model for predicting bone_map using "886-bone"
    model_bone = op.join(model_path,'Task886_Bone/nnUNetTrainerV2__nnUNetPlansv2.1/')

    return model_2_label, model_3_label, model_ID, model_bone


def label_rescale(image_label, w_ori, h_ori, z_ori, flag):
    w_ori, h_ori, z_ori = int(w_ori), int(h_ori), int(z_ori)
    # resize label map (int)
    if flag == 'trilinear':
        teeth_ids = np.unique(image_label)
        image_label_ori = torch.zeros((w_ori, h_ori, z_ori)).cuda(0)
        image_label = torch.from_numpy(image_label).cuda(0)
        for label_id in range(len(teeth_ids)):
            image_label_bn = (image_label == teeth_ids[label_id]).float()
            image_label_bn = image_label_bn[None, None, :, :, :]
            image_label_bn = torch.nn.functional.interpolate(image_label_bn, size=(w_ori, h_ori, z_ori),
                                                             mode='trilinear')
            image_label_bn = image_label_bn[0, 0, :, :, :]
            image_label_ori[image_label_bn > 0.5] = teeth_ids[label_id]
        image_label = image_label_ori.cpu().data.numpy()

    if flag == 'nearest':
        image_label = torch.from_numpy(image_label).cuda(0)
        image_label = image_label[None, None, :, :, :].float()
        image_label = torch.nn.functional.interpolate(image_label, size=(w_ori, h_ori, z_ori), mode='nearest')
        image_label = image_label[0, 0, :, :, :].cpu().data.numpy()
    return image_label


def img_crop(image_bbox):
    image_bbox = morphology.remove_small_objects(image_bbox.astype(bool), 2500, connectivity=3).astype(int)
    if image_bbox.sum() > 0:
        # if None:
        x_min = np.nonzero(image_bbox)[0].min() - 32
        x_max = np.nonzero(image_bbox)[0].max() + 32

        y_min = np.nonzero(image_bbox)[1].min() - 16
        y_max = np.nonzero(image_bbox)[1].max() + 16

        z_min = np.nonzero(image_bbox)[2].min() - 16
        z_max = np.nonzero(image_bbox)[2].max() + 16

        if x_min < 0:
            x_min = 0
        if y_min < 0:
            y_min = 0
        if z_min < 0:
            z_min = 0
        if x_max > image_bbox.shape[0]:
            x_max = image_bbox.shape[0]
        if y_max > image_bbox.shape[1]:
            y_max = image_bbox.shape[1]
        if z_max > image_bbox.shape[2]:
            z_max = image_bbox.shape[2]
    if image_bbox.sum() == 0:
        x_min, x_max, y_min, y_max, z_min, z_max = -1, image_bbox.shape[0], 0, image_bbox.shape[1], 0, image_bbox.shape[
            2]
    return x_min, x_max, y_min, y_max, z_min, z_max


def makedirs(folder):
    if not op.exists(folder):
        os.makedirs(folder)


def temporary_folder():
    # Set up some temporary storage folders for files
    # 2_label
    input_2 = 'temp/nnunet_input_2'
    makedirs(input_2)
    output_2 = 'temp/nnunet_infer_2'
    makedirs(output_2)
    # 3_label
    input_3 = 'temp/nnunet_input_3'
    makedirs(input_3)
    output_3 = 'temp/nnunet_infer_3'
    makedirs(output_3)
    # id
    output_id = 'temp/nnunet_infer_id'
    makedirs(output_id)
    return input_2,output_2,input_3,output_3,output_id


def infer(path, file, model_path, mapped):
    model_2_label, model_3_label, model_ID, model_bone = load_model(model_path)

    input_2, output_2, input_3, output_3, output_id=temporary_folder()
    # Set up output folder
    output_path = op.join(path, file)

    # Load the input data
    img_path = op.join(path, file,file+'_image.nii.gz')
    data_nii = sitk.ReadImage(img_path)
    origin = data_nii.GetOrigin()
    spacing = data_nii.GetSpacing()
    direction = data_nii.GetDirection()
    image_array = (sitk.GetArrayFromImage(data_nii).T).astype(np.float32)
    w, h, d = image_array.shape

    starttime1 = datetime.datetime.now()
    # ------------------------------------------------------------------------------------------------
    # Use nnUNet to predict _2_label_map
    # change name to 'Teeth_   _0000.nii.gz'
    new_name = 'Teeth_' + file + '_0000.nii.gz'

    shutil.copy(img_path, input_2)
    os.rename(op.join(input_2, file+'_image.nii.gz'), op.join(input_2, new_name))

    predict_cases(model_2_label, [[op.join(input_2, new_name)]],
                  [op.join(output_2, new_name.replace('_0000', ''))], folds=4,
                  num_threads_preprocessing=1, num_threads_nifti_save=2, segs_from_prev_stage=None, do_tta=False,
                  mixed_precision=True, all_in_gpu=True, step_size=0.5, checkpoint_name='model_final_checkpoint',
                  remove=0)

    _2_label_map = (sitk.GetArrayFromImage(
        sitk.ReadImage(op.join(output_2, new_name.replace('_0000', '')))).T).astype(np.float32)
    del model_2_label
    starttime2 = datetime.datetime.now()
    print('The time of predicting _2_label_map stage:', starttime2 - starttime1)

    # ------------------------------------------------------------------------------------------------
    # Use nnUNet to predict _3_label_map
    # rescale the input spacing to 0.2 mm
    image1 = label_rescale(image_array, w * (spacing[0] / 0.2), h * (spacing[1] / 0.2), d * (spacing[2] / 0.2),
                           'nearest')

    label1 = label_rescale(_2_label_map.astype(np.int64), w * (spacing[0] / 0.2), h * (spacing[1] / 0.2),
                           d * (spacing[2] / 0.2), 'trilinear')

    # crop image
    x_min, x_max, y_min, y_max, z_min, z_max = img_crop(label1)
    image_array = image1[x_min:x_max, y_min:y_max, z_min:z_max]
    label_array = label1[x_min:x_max, y_min:y_max, z_min:z_max]

    image_array1 = sitk.GetImageFromArray(image_array.astype(np.float32).T)
    sitk.WriteImage(image_array1, op.join(input_3, new_name))

    predict_cases(model_3_label, [[op.join(input_3, new_name)]],
                  [op.join(output_3, new_name.replace('_0000', ''))], folds=4,
                  num_threads_preprocessing=1, num_threads_nifti_save=2, segs_from_prev_stage=None, do_tta=False,
                  mixed_precision=True, all_in_gpu=True, step_size=0.5, checkpoint_name='model_final_checkpoint',
                  remove=0)
    del model_3_label
    _3_label_map = sitk.ReadImage(op.join(output_3, new_name.replace('_0000', '')))
    _3_label_map = sitk.GetArrayFromImage(_3_label_map).T
    starttime3 = datetime.datetime.now()
    print('The time of predicting _3_label_map stage:', starttime3 - starttime2)

    # 1->0,2->1
    _3_label_map = np.where(_3_label_map == 1, 0, _3_label_map)
    _3_label_map = np.where(_3_label_map == 2, 1, _3_label_map)

    seeds_map = _3_label_map
    seeds_map = seeds_map * label_array
    seeds_map = morphology.remove_small_objects(seeds_map.astype(bool), 50, connectivity=1)
    markers = ndimage.label(seeds_map)[0]

    whole_label = np.zeros((int(w * (spacing[0] / 0.2)), int(h * (spacing[1] / 0.2)), int(d * (spacing[2] / 0.2))))
    whole_label[x_min:x_max, y_min:y_max, z_min:z_max] = markers
    whole_label = label_rescale(whole_label, w, h, d, 'trilinear')

    # Compute the distance map of the binary map
    distance = ndimage.distance_transform_edt(_2_label_map)

    # Compute the watershed transform
    tooth_array = watershed(-distance, markers=whole_label, mask=_2_label_map)

    # Save the output
    # tooth_array1 = sitk.GetImageFromArray(tooth_array.astype(np.int32).T)
    starttime4 = datetime.datetime.now()
    print('The time of rescale+watershed transform stage:', starttime4 - starttime3)

    # =============================================================================================================================================================================================
    # Use nnUNet to predict id
    predict_cases_fastest(model_ID, [[op.join(input_3, new_name)]],
                          [op.join(output_id, new_name.replace('_0000', ''))], folds=4,
                          num_threads_preprocessing=1, do_tta=False, mixed_precision=True,
                          all_in_gpu=True,
                          step_size=1, checkpoint_name='model_best')
    id_mask = sitk.ReadImage(op.join(output_id, new_name.replace('_0000', '')))
    id_mask = sitk.GetArrayFromImage(id_mask).T
    del model_ID
    starttime5 = datetime.datetime.now()
    print('The time of predicting id_map stage:', starttime5 - starttime4)
    # ------------------------------------------------------------------------------------------------
    # mapping id
    labels = np.unique(tooth_array[tooth_array != 0])
    true_id_tooth = np.zeros_like(tooth_array)
    exist_id = []
    for label in labels:
        single = np.where(tooth_array == label, 1, 0) * id_mask
        single = single[single != 0]
        ids, counts = np.unique(single, return_counts=True)
        a = 0
        if len(ids) > 1:
            id_map = ids[np.argsort(counts)[::-1]]
            true_id = mapped[id_map[0]]
            if true_id in exist_id:
                a += 1
                true_id = mapped[id_map[-(a + 1)]]

        elif len(ids) == 1:
            true_id = mapped[ids[0]]
        else:
            true_id = 0
        # 0919
        exist_id.append(true_id)
        true_id_tooth[tooth_array == label] = true_id

    # Save the output
    true_id_tooth1 = sitk.GetImageFromArray(true_id_tooth.astype(np.int32).T)
    true_id_tooth1.SetOrigin(origin)
    true_id_tooth1.SetSpacing(spacing)
    true_id_tooth1.SetDirection(direction)
    sitk.WriteImage(true_id_tooth1, op.join(output_path, file+'_label.nii.gz'))

    starttime6 = datetime.datetime.now()
    print('The time of postprocess stage:', starttime6 - starttime5)

    # Use nnUNet to predict _bone_map-----------------------------------------------------------------------------
    predict_cases(model_bone, [[op.join(input_2, new_name)]],
                  [op.join(output_path, new_name.replace('_0000', ''))], folds=5,
                  num_threads_preprocessing=1, segs_from_prev_stage=None, do_tta=True, mixed_precision=True,
                  all_in_gpu=True, step_size=0.5, checkpoint_name='model_final_checkpoint', remove=int(args.b_r))
    os.rename(op.join(output_path, new_name.replace('_0000', '')), op.join(output_path, file+'_bone.nii.gz'))
    del model_bone
    starttime7 = datetime.datetime.now()
    print('Predicting the time of a patient:', starttime7 - starttime1)

    # delete temp files
    os.remove(op.join(input_2, new_name))
    os.remove(op.join(input_3, new_name))
    os.remove(op.join(output_2, new_name.replace('_0000', '')))
    os.remove(op.join(output_3, new_name.replace('_0000', '')))
    os.remove(op.join(output_id, new_name.replace('_0000', '')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CBCT image segmentation')
    parser.add_argument('--input', '-i', type=str, default='input')
    parser.add_argument('--model_path', '-m', type=str, help='Path of trained models', required=False,
                        default='../model')
    parser.add_argument('--b_r', type=str, default='5000',
                        help='Remove non-bone parts smaller than this value in size.')
    args = parser.parse_args()
    path = args.input
    model_path=args.model_path

    dict_from_csv = pd.read_csv(r"CBCTSegmentation/mapping.CSV", header=None, index_col=0, squeeze=True).to_dict()
    mapped = {v: k for k, v in dict_from_csv.items()}

    for file in os.listdir(path):
        print('**********process the patient:', file)
        infer(path, file, model_path,mapped)
