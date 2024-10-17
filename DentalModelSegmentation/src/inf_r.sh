#!/bin/bash
export PYTHONPATH=/home/zsj/PycharmProjects/miccai-3d-teeth-seg

fn=$1
patient="${fn%%_*}"
jaw="${fn##*_}"
echo $patient
echo $jaw

/home/zsj/anaconda3/envs/pointnet2/bin/python /home/zsj/PycharmProjects/miccai-3d-teeth-seg/src/inference_rotation.py -m /home/zsj/PycharmProjects/miccai-3d-teeth-seg/model -i /home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/3D_scans_per_patient_obj_files/${patient}/${patient}_${jaw}.obj -o /home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/test_vis/${patient}_${jaw}.obj
