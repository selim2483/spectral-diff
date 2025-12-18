#!/bin/bash 

#SBATCH --job-name=cloudy_video_lr1e-5
#SBATCH --ntasks-per-node=48
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --output=logslurm/cloudy_video_lr1e-5-%A_%a.out
#SBATCH --error=logslurm/cloudy_video_lr1e-5-%A_%a.err

nvidia-smi
pixi run python src/spectral_diff/video.py fit --trainer config/trainer.yaml --model.model config/unet.yaml --model.scheduler config/ddpm.yaml --data config/cloudy_conditional.yaml --model.initial_lr 1e-5 --trainer.val_check_interval 1000