#!/bin/bash
#SBATCH --job-name=model1        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=2        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G        # memory per cpu-core (4G is default)
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:1             # number of gpus per node

source ~/.bashrc
conda activate simple_thesis_model
nvidia-smi
nvcc --version
conda list pytorch

python train_CNN3D.py