#!/bin/bash 

#SBATCH --job-name=extract

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=0
#SBATCH --time=7-0:00:00
#SBATCH --qos=co_long_std
#SBATCH --output=logslurm/extract-%A_%a.out
#SBATCH --error=logslurm/extract-%A_%a.err

pixi run python src/spectral_diff/extract_ms_dataset.py