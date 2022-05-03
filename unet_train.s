#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=47:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=unet_train
#SBATCH --mail-type=END
#SBATCH --mail-user=etg259@nyu.edu
#SBATCH --output=logs/unet_train_%j.out

singularity exec --nv /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif nvidia-smi

singularity exec --nv --overlay ../Wave-U-Net-Pytorch/overlay-10GB-400K.ext3:ro /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif /bin/bash -c 'source /ext3/env.sh; python train.py --cuda --data_dir ../Wave-U-Net-Pytorch/hdf/ --num_in_chan 1 --num_out_chan 1 --patience 20'
