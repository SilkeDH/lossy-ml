#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:4
#SBATCH --account=haf
#SBATCH --partition=batch

source /p/home/jusers/donayreholtz1/hdfml/hyp_opt/bin/activate

module load Stages/2020
module load GCCcore/.9.3.0
module load TensorFlow/2.3.1-Python-3.8.5
module load dask/2.22.0-Python-3.8.5
pip3 install xarray

python3 lossy-ml/lossycomp/train_model.py -o /FINAL_2/model_soil_res_hyp_400k_2 -m True
