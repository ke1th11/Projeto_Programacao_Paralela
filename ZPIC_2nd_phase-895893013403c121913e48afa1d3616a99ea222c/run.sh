#!/bin/bash
#SBATCH --job-name=zpic_cpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:10:00
#SBATCH --partition=normal-a100-40
#SBATCH --account=f202500010hpcvlabuminhog

# Carregar módulos
ml CUDA

# Correr o programa
srun ./zpic
