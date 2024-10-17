from scipy.ndimage import binary_erosion
import os
import nibabel as nib
import numpy as np
import torch
import SimpleITK as sitk
import datetime
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def read_data(data_patch):
    src_data_file = data_patch+'_image.nii.gz'
    src_data_vol = nib.load(src_data_file)
    image = src_data_vol.get_data()
    w, h, d = image.shape
    spacing = src_data_vol.header['pixdim'][1:4]
    image = label_rescale(image, w*(spacing[0]/0.2), h*(spacing[0]/0.2), d*(spacing[0]/0.2), 'nearest')
    nib.save(nib.Nifti1Image(image.astype(np.float32), np.eye(4)),
             os.path.join(os.path.dirname(image_list[data_id]),
                          os.path.basename(image_list[data_id]).split('.')[0] + '_image_2.nii.gz'))

    return image, w, h, d
#
#
def label_rescale(image_label, w_ori, h_ori, z_ori, flag):
    w_ori, h_ori, z_ori = int(w_ori), int(h_ori), int(z_ori)
    # resize label map (int)
    if flag == 'trilinear':
        teeth_ids = np.unique(image_label)
        image_label_ori = torch.zeros((w_ori, h_ori, z_ori)).cuda(0)
        image_label = torch.from_numpy(image_label).cuda(0)
        for label_id in range(len(teeth_ids)):
            image_label_bn = (image_label == teeth_ids[label_id]).float()
            #image_label_bn = torch.from_numpy(image_label_bn)#.astype(float)
            image_label_bn = image_label_bn[None, None, :, :, :]
            image_label_bn = torch.nn.functional.interpolate(image_label_bn, size=(w_ori, h_ori, z_ori),
                                                             mode='trilinear')
            image_label_bn = image_label_bn[0, 0, :, :, :]
            image_label_ori[image_label_bn > 0.5] = teeth_ids[label_id]
        image_label = image_label_ori.cpu().data.numpy()
    # image
    if flag == 'nearest':
        image_label = torch.from_numpy(image_label).cuda(0)
        image_label = image_label[None, None, :, :, :].float()
        image_label = torch.nn.functional.interpolate(image_label, size=(w_ori, h_ori, z_ori), mode='nearest')
        image_label = image_label[0, 0, :, :, :].cpu().data.numpy()


    return image_label
if __name__ == '__main__':

    input_image = r'G:\Nanjing_Data\imagesTr'
    input_mask=r'G:\Nanjing_Data\CBCT_Nanjing_Teeth_bi'
    out_image=r'G:\Nanjing_Data\imagesTr_crop'
    out_mask=r'G:\Nanjing_Data\CBCT_Nanjing_Teeth_bi_crop'
    for file in os.listdir(input_image):
        print(file)
        starttime1 = datetime.datetime.now()
        image = os.path.join(input_image, file)
        img1 = nib.load(image)
        image_array = img1.get_fdata()
        spacing = img1.header['pixdim'][1:4]
        w, h, d = image_array.shape

        label = os.path.join(input_mask, file.replace('Bone_','Teeth_').replace('_0000.nii.gz', '.nii.gz'))
        mask1 = nib.load(label)
        mask_array = mask1.get_fdata()

        # spacing-->0.2mm
        image1 = label_rescale(image_array, w * (spacing[0] / 0.2), h * (spacing[1] / 0.2), d * (spacing[2] / 0.2), 'nearest')
        label1=label_rescale(mask_array.astype(np.int64), w * (spacing[0] / 0.2), h * (spacing[1] / 0.2), d * (spacing[2] / 0.2), 'trilinear')
        # nib.save(nib.Nifti1Image(image.astype(np.float32), np.eye(4)),
        #          os.path.join(os.path.dirname(image_list[data_id]),
        #                       os.path.basename(image_list[data_id]).split('.')[0] + '_image_2.nii.gz'))
        img1 = sitk.GetImageFromArray(image1.T.astype(np.float32))
        sitk.WriteImage(img1, r'/hpc/data/home/bme/v-tanmh/nnUNetFrame/DATASET/nnUNet_raw_data_base/nnUNet_raw_data/Task601_Teeth/imagesTr/' + file)
        lab1 = sitk.GetImageFromArray(label1.T.astype(np.int32))
        sitk.WriteImage(lab1, r'/hpc/data/home/bme/v-tanmh/nnUNetFrame/DATASET/nnUNet_raw_data_base/nnUNet_raw_data/Task601_Teeth/labels/' +'Teeth_'+ name+'.nii.gz')
        print('label_rescale finshed')
        mask_array1 = label1
        labels = np.unique(mask_array1)[1:]
        boundary_all=np.zeros_like(mask_array1)
        for label in labels:
            mask_array2 = np.where(mask_array1 == label, 1, 0)
            # dilated_label = binary_dilation(mask_array).astype(np.float32)
            eroded_label = binary_erosion(mask_array2,iterations=3)#.astype(np.float32)
            boundary = mask_array2 - eroded_label
            boundary_all =np.where(boundary==1,boundary,boundary_all)
            #img1 = sitk.GetImageFromArray(boundary.astype(np.int32).T)
            #sitk.WriteImage(img1, r'/hpc/data/home/bme/v-tanmh/nnUNetFrame/DATASET/nnUNet_raw_data_base/nnUNet_raw_data/Task600_Teeth/labelsTr_2/' + str(label) + '.nii.gz')
        # img2 = sitk.GetImageFromArray(boundary_all.T)
        # sitk.WriteImage(img2, r'D:\boundary\2\1000813648_20180116_boundary_all.nii.gz')

    # boundary_all=r'D:\boundary\X2315779_boundary_all.nii.gz'
    # boundary_all = sitk.ReadImage(boundary_all)
    # boundary_all = sitk.GetArrayFromImage(boundary_all).T

        mask_array3 = np.where(mask_array1!=0,2,0)
        label2=np.where(boundary_all==1,1,mask_array3)
        lab2 = sitk.GetImageFromArray(label2.T.astype(np.int32))
        sitk.WriteImage(lab2,r'/hpc/data/home/bme/v-tanmh/nnUNetFrame/DATASET/nnUNet_raw_data_base/nnUNet_raw_data/Task601_Teeth/labelsTr/' +'Teeth_'+ name+'.nii.gz')
    # # dilated_label = binary_dilation(mask_array).astype(np.float32)
    # eroded_label = binary_erosion(mask_array).astype(np.float32)
    # boundary=mask_array - eroded_label
    # img1 = sitk.GetImageFromArray(boundary.T)
    # sitk.WriteImage(img1,r'D:\boundary\Teeth_1000813648-20180116_boundary1.nii.gz')