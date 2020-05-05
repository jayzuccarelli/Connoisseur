#!/bin/bash
#SBATCH --job-name=connoisseur
#SBATCH --output=output_pretrained.txt
#SBATCH --error=errors_pretrained.txt
#SBATCH -p sched_mit_sloan_gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=12G
#SBATCH --time=1-00:00:00
module load python/3.6.3
module load sloan/python/modules/python-3.6/tensorflow/1.9.0/gpu
python3.6 main_pretrained.py