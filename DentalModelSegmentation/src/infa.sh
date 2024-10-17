#!/bin/bash
export PYTHONPATH=/home/zsj/PycharmProjects/miccai-3d-teeth-seg

out=$2

if [ -z $out ]; then
    out=/home/zsj/vis.obj
fi

/home/zsj/anaconda3/envs/pointnet2/bin/python /home/zsj/PycharmProjects/miccai-3d-teeth-seg/src/inference.py -m /home/zsj/PycharmProjects/miccai-3d-teeth-seg/model -i "$1" -o "$out"

