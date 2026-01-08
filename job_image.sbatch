#!/bin/bash 

#SBATCH --job-name=cloudy_image_resize256_1e-4

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=7-0:00:00
#SBATCH --qos=co_long_gpu
#SBATCH --output=logslurm/cloudy_image_resize_1e-4-%A_%a.out
#SBATCH --error=logslurm/cloudy_image_resize_1e-4-%A_%a.err
#SBATCH --constraint=rtx6000

nvidia-smi
pixi run python src/spectral_diff/image.py fit\
 --trainer config/trainer_image.yaml\
 --model.model config/unet_image.yaml\
 --model.scheduler config/ddpm.yaml\
 --data config/cloudy_image.yaml\
 --model.initial_lr 1e-4\
 --trainer.val_check_interval 5000\
 --ckpt_path /scratchm/sollivie/spectral-diff/runs/cloudy_image_resize256_1e-4/model-step=50000.ckpt