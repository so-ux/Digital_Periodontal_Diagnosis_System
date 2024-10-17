#!/bin/bash
#SBATCH -p bme_gpu4
#BATCH -J main
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -o out%j.out
#SBATCH -e error%j.out
#SBATCH -t 24:00:00

echo "${SLURM_JOB_NODELIST}"
echo start on $(date)
python /public/bme/home/v-tanmh/DentalModelSegmentation/infer.py -i /public/bme/home/v-tanmh/file.txt -o /public/bme/home/v-tanmh/dental_file/out/ -m /public/bme/home/v-tanmh/DentalModelSegmentation/model/
echo end on $(date)
