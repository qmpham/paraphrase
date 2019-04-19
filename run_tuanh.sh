#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=20
#SBATCH --nodelist=n102
#SBATCH --output=sparse_src_masking_30.log.out
#SBATCH --error=sparse_src_masking_30.log.err

source ~/anaconda3/bin/activate py3.6
#export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64/

which python
python -c 'import tensorflow; print("tensorflow OK"); import opennmt; print("opennmt OK")'
python -u runner.py --config_file configs/config_tuanh.yml 1> log.out 2> log.err &
