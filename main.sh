#!/bin/bash
#SBATCH --job-name=Connoisseur
#SBATCH --output=Connoisseur-output.txt
#SBATCH --error=Connoisseur-errors.txt
#SBATCH --pty -p sched_mit_sloan_batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --time=48:00:00
module load python/3.6.3
module load sloan/python/modules/python-3.6/tensorflow/1.9.0/cpu
python3.6 main.py