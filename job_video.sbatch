#!/bin/bash 

#SBATCH --job-name=cloudy_video_lr1e-4_T100

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=7-0:00:00
#SBATCH --qos=co_long_gpu
#SBATCH --output=logslurm/cloudy_video_lr1e-4_T100_video-%A_%a.out
#SBATCH --error=logslurm/cloudy_video_lr1e-4_T100-%A_%a.err
#SBATCH --constraint=rtx6000

nvidia-smi
pixi run python src/spectral_diff/video.py fit --trainer config/trainer.yaml --model.model config/unet.yaml --model.scheduler config/ddpm.yaml --data config/cloudy_conditional.yaml --model.initial_lr 1e-4 --trainer.val_check_interval 1000 --model.scheduler.num_train_timesteps 100