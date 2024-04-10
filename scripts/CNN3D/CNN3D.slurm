#!/bin/bash
#SBATCH --job-name=test         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=2G         # memory per cpu-core (4G is default)
#SBATCH --gpus=1

module purge
module load anaconda3/2023.9
conda activate pytools-env

python scripts/CNN3D/train_CNN3D.py