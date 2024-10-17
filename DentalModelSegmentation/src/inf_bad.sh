#!/bin/bash

for file in `ls /home/zsj/PycharmProjects/miccai-3d-teeth-seg/data/test_vis/ | grep -E "\.obj$"`; do
    ./inf.sh "${file%%.obj*}"
done
