#!/bin/bash -l

#SBATCH --job-name="eta_540"
#SBATCH --partition=medium
#SBATCH --nodes=6
#SBATCH --time=2-00:00:00
#SBATCH --mem=40GB
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL


#SBATCH --output="job/test-%jo"
#SBATCH --error="job/test-%je"

source /deac/csc/chenGrp/software/tensorflow/bin/activate
python /home/xue20/workspace/ATN_Beta/auto.py --num 1 --data eta1 --k 6 --kmeans 1 --main_epoch 1000 --alpha 0.00001 --beta 1 --h_dim 8
