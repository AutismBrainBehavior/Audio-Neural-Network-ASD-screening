#!/bin/sh

#SBATCH --partition=shared-gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00

module load GCCcore/8.2.0 Singularity/3.4.0-Go-1.12

OPENPOSE_SIMG=audio.simg

CMD="python train_net.py"

srun singularity exec --nv $OPENPOSE_SIMG $CMD
