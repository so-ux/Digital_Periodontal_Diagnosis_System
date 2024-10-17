import os
import datetime
import numpy as np
import nibabel as nib
from skimage import morphology
from collections import OrderedDict
import pandas as pd
class Logger():
    def __init__(self, save_path,save_name):
        self.save_path = save_path
        self.log = None
        self.save_name = save_name

    def update(self, epoch,train_log):
        item = OrderedDict({'name':epoch})
        item.update(train_log)
        print("\033[0;33mTrain:\033[0m", train_log)
        self.update_csv(item)

    def update_csv(self,item):
        tmp = pd.DataFrame(item, index=[0])
        if self.log is not None:
            self.log = self.log.append(tmp, ignore_index=True)
        else:
            self.log = tmp
        self.log.to_csv('%s/%s.csv' % (self.save_path, self.save_name), index=False)

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

    input_image = r'G:\Nanjing_Data\CBCT_Nanjing_Bone'
    input_mask=r'G:\Nanjing_Data\CBCT_Nanjing_Teeth'
    # out_image=r''
    out_mask=r'G:\Nanjing_Data\CBCT_Nanjing_Teeth_bi'
    for file in os.listdir(input_image):
        if not os.path.exists(os.path.join(out_mask, file.replace('Bone','Teeth'))):
            print(file)
            starttime1 = datetime.datetime.now()
            image = os.path.join(input_image, file)
            img1 = nib.load(image)
            image_array = img1.get_fdata()
            spacing = img1.header['pixdim'][1:4]
            w, h, d = image_array.shape

            label = os.path.join(input_mask, 'Teeth_'+file.replace('_','-').replace('.nii.gz', '-0000_0000.nii.gz'))
            if os.path.exists(label):
                mask1 = nib.load(label)
                mask_array = mask1.get_fdata()

                # label-->1&change affine of mask
                bi=np.where(mask_array!=0,1,0)
                nib.save(nib.Nifti1Image(bi.astype(np.float32), img1.affine),
                         os.path.join(out_mask, file.replace('Bone','Teeth')))


            starttime2 = datetime.datetime.now()
            print('The time of this stage:', starttime2 - starttime1)
