#!/bin/bash -l

#SBATCH --job-name="eta2_2"
#SBATCH --partition=medium
#SBATCH --nodes=6
#SBATCH --time=2-00:00:00
#SBATCH --mem=20GB
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL


#SBATCH --output="job/eta2_2-%jo"
#SBATCH --error="job/eta2_2-%je"

source /deac/csc/chenGrp/software/tensorflow/bin/activate

python /deac/generalGrp/chenGrp/xue20/ATN_Beta/auto.py --num 20 --data eta2 --k 6 --kmeans 1 --main_epoch 1000 --alpha 0.00001 --beta 0.1 --h_dim 8
python /deac/generalGrp/chenGrp/xue20/ATN_Beta/auto.py --num 20 --data eta2 --k 6 --kmeans 1 --main_epoch 1000 --alpha 0.00001 --beta 0.1 --h_dim 12
python /deac/generalGrp/chenGrp/xue20/ATN_Beta/auto.py --num 20 --data eta2 --k 6 --kmeans 1 --main_epoch 1000 --alpha 0.00001 --beta 0.1 --h_dim 16