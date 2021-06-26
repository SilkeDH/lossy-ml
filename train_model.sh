#!/bin/sh

#SBATCH --nodes=1
#SBATCH -C LSDF
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu_4

module load devel/jupyter_ml
date
python3 lossycomp/train_model.py -o FINAL/OPTIM/19 -c 4 -f 21 -k 5 -res 1
date
