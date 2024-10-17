#!/bin/bash
#SBATCH -p bme_gpu4
#BATCH -J main
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -o out%j.out
#SBATCH -e error%j.out
#SBATCH -t 24:00:00
#SBATCH --mem 128G

module load compiler/gnu/8.3.0
nvidia-smi
echo "${SLURM_JOB_NODELIST}"
echo start on $(date)
python /public/bme/home/v-tanmh/Periodontal_Diagnosis/DentalModelSegmentation/test.py -d /public/bme/home/v-tanmh/dental_file/test/input/lower/ -o /public/bme/home/v-tanmh/dental_file/test/input/lower/ -m /public/bme/home/v-tanmh/Periodontal_Diagnosis/DentalModelSegmentation/model/ -n test_infer
echo end on $(date)
