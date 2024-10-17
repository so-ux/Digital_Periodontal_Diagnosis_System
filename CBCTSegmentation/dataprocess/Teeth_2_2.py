import os
import datetime
import numpy as np
import nibabel as nib
from skimage import morphology


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
if __name__ == '__main__':

    input_image = r'G:\Nanjing_Data\imagesTr'
    input_mask=r'G:\Nanjing_Data\CBCT_Nanjing_Teeth_bi'
    out_image=r'G:\Nanjing_Data\imagesTr_crop'
    out_mask=r'G:\Nanjing_Data\CBCT_Nanjing_Teeth_bi_crop'
    for file in os.listdir(input_image):
        if not os.path.exists(os.path.join(out_image, file.replace('Bone','Teeth'))):
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
            # crop image
            x_min, x_max, y_min, y_max, z_min, z_max = img_crop(mask_array)
            image = image_array[x_min:x_max, y_min:y_max, z_min:z_max]
            label = mask_array[x_min:x_max, y_min:y_max, z_min:z_max]

            nib.save(nib.Nifti1Image(image.astype(np.float32), img1.affine),
                     os.path.join(out_image, file.replace('Bone','Teeth')))
            nib.save(nib.Nifti1Image(label.astype(np.float32), img1.affine),
                     os.path.join(out_mask, file.replace('Bone','Teeth').replace('_0000.nii.gz', '.nii.gz')))

            starttime2 = datetime.datetime.now()
            print('The time of this stage:', starttime2 - starttime1)
