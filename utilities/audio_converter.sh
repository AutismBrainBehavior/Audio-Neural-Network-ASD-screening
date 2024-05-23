#!/bin/sh

#SBATCH --partition=shared-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00

module load GCCcore/8.2.0
module load FFmpeg/4.1.3

CMD="python vac.py"

srun $CMD
