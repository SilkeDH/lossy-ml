#!/bin/sh

#SBATCH --nodes=1
#SBATCH -C LSDF
#SBATCH --time=15:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_8

module load devel/jupyter_ml
date
python3 lossycomp/testing_compressor.py
date
