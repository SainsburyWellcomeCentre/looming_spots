#!/bin/bash
#
#SBATCH -p gpu # partition (queue)
#SBATCH -N 1   # number of nodes
#SBATCH --mem 120G # memory pool for all cores
#SBATCH --gres=gpu:1
#SBATCH -t 0-5:0 # time (D-HH:MM)
#SBATCH -o dlc_tracking.out
#SBATCH -e dlc_tracking.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s.lenzi@ucl.ac.uk

mouse_id="1097613"
config_path='/nfs/winstor/margrie/slenzi/Ben_grainger_materials/config.yaml'

echo "Loading CUDA"
module load /ceph/apps/ubuntu-20/modulefiles/cuda/11.2

echo "Loading conda environment"
module load /ceph/apps/ubuntu-20/modulefiles/miniconda/4.9.2

conda activate dlc_tf

export DLClight=True

echo "Running dlc_tracking"
python ~/code/python/looming_spots/looming_spots/tracking_dlc/track_mouse.py $mouse_id