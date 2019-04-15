#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=20
#SBATCH --nodelist=n102
#SBATCH --output=sparse_src_masking_30.log.out
#SBATCH --error=sparse_src_masking_30.log.err

source ~/miniconda3/bin/activate nmt_env_standard
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64/

which python
python -c 'import tensorflow; print("tensorflow OK"); import opennmt; print("opennmt OK")'
python -u runner.py --config_file configs/sparse_src_masking_30.yml
