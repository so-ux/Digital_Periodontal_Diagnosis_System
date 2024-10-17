import datetime
import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_dilation, binary_erosion
import SimpleITK as sitk
from skimage import morphology

def postprocess_with_tooth(mask_array,spacing,labels,bone_path,save_path):
    time1 = datetime.datetime.now()
    up_tooth=np.zeros_like(mask_array, dtype=bool)
    lower_tooth=np.zeros_like(mask_array, dtype=bool)

    bone1 = sitk.ReadImage(bone_path)
    bone_array = (sitk.GetArrayFromImage(bone1).T).astype(np.int32)
    up_bone_array = np.where(bone_array==1,1,0).astype(bool)
    lower_bone_array = np.where(bone_array==2,1,0).astype(bool)

    for label in labels:
        if label < 29:
            up_tooth |= (mask_array == label)
        else:
            lower_tooth |= (mask_array == label)
    up_whole = up_tooth | up_bone_array
    dilated_label = binary_dilation(up_whole,iterations=1).astype(np.float32)
    fill_holes_label = ndimage.binary_fill_holes(dilated_label).astype(int)
    eroded_label = binary_erosion(fill_holes_label,iterations=2).astype(np.float32)

    lower_whole = lower_tooth | lower_bone_array
    dilated_label1 = binary_dilation(lower_whole,iterations=1).astype(np.float32)
    fill_holes_label1 = ndimage.binary_fill_holes(dilated_label1).astype(int)
    eroded_label1 = binary_erosion(fill_holes_label1,iterations=2).astype(np.float32)

    whole=np.where(eroded_label1==1,2,eroded_label)

    #-tooth
    whole1 = whole * (mask_array == 0)

    #remove small
    up = (whole1 == 1).astype(dtype=bool)
    up = morphology.remove_small_objects(up, 10000)
    lower = (whole1 == 2).astype(dtype=bool)
    lower = morphology.remove_small_objects(lower, 20000)

    dilated = binary_dilation(up,iterations=1).astype(np.float32)

    dilated1 = binary_dilation(lower,iterations=1).astype(np.float32)

    whole1=np.where(dilated1==1,2,dilated)

    whole1=np.where(mask_array!=0,0,whole1)

    whole_array1 = sitk.GetImageFromArray(whole1.T)
    whole_array1.SetSpacing(spacing)
    sitk.WriteImage(whole_array1, save_path)
    time3 = datetime.datetime.now()
    print('The time of postprocessing stage:',time3-time1)





