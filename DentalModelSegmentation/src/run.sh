#!/bin/bash

for patient in `ls /run/media/zsj/DATA/Data/miccai/3D_scans_per_patient_obj_files/`; do
    upper="/run/media/zsj/DATA/Data/miccai/3D_scans_per_patient_obj_files/${patient}/${patient}_upper.obj"
    lower="/run/media/zsj/DATA/Data/miccai/3D_scans_per_patient_obj_files/${patient}/${patient}_lower.obj"
    /home/zsj/anaconda3/envs/pointnet2/bin/python inference.py --input ${upper} -n experiment_centroids_all --n_epochs 1615 --output /home/zsj/PycharmProjects/miccai-3d-teeth-seg/temp/validate/test_1615/${patient}_upper/
    /home/zsj/anaconda3/envs/pointnet2/bin/python inference.py --input ${lower} -n experiment_centroids_all --n_epochs 1615 --output /home/zsj/PycharmProjects/miccai-3d-teeth-seg/temp/validate/test_1615/${patient}_lower/
done
