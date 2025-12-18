#!/bin/bash 

#SBATCH --job-name=gpu_notebook

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=10:00:00
#SBATCH --qos=co_long_gpu
#SBATCH --output=logslurm/gpu_notebook-%A_%a.out
#SBATCH --error=logslurm/gpu_notebook-%A_%a.err

pixi run jupyter notebook --no-browser --ip=0.0.0.0 --port=8888