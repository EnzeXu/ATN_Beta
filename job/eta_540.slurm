#!/bin/bash -l

#SBATCH --job-name="eta_540"
#SBATCH --partition=medium
#SBATCH --nodes=4
#SBATCH --time=2-00:00:00
#SBATCH --mem=20GB
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL


#SBATCH --output="job/eta_540-%jo"
#SBATCH --error="job/eta_540-%je"

source /deac/csc/chenGrp/software/tensorflow/bin/activate
python /home/xue20/workspace/ATN_Beta/auto.py --num 20 --data eta1 --k 6 --kmeans 1 --main_epoch 1000 --alpha 0.00001 --beta 1 --h_dim 8
python /home/xue20/workspace/ATN_Beta/auto.py --num 20 --data eta1 --k 6 --kmeans 1 --main_epoch 1000 --alpha 0.00001 --beta 1 --h_dim 12
python /home/xue20/workspace/ATN_Beta/auto.py --num 20 --data eta1 --k 6 --kmeans 1 --main_epoch 1000 --alpha 0.00001 --beta 1 --h_dim 16
python /home/xue20/workspace/ATN_Beta/auto.py --num 20 --data eta1 --k 6 --kmeans 1 --main_epoch 1000 --alpha 0.00001 --beta 0.1 --h_dim 8
python /home/xue20/workspace/ATN_Beta/auto.py --num 20 --data eta1 --k 6 --kmeans 1 --main_epoch 1000 --alpha 0.00001 --beta 0.1 --h_dim 12
python /home/xue20/workspace/ATN_Beta/auto.py --num 20 --data eta1 --k 6 --kmeans 1 --main_epoch 1000 --alpha 0.00001 --beta 0.1 --h_dim 16
python /home/xue20/workspace/ATN_Beta/auto.py --num 20 --data eta1 --k 6 --kmeans 1 --main_epoch 1000 --alpha 0.00001 --beta 0.01 --h_dim 8
python /home/xue20/workspace/ATN_Beta/auto.py --num 20 --data eta1 --k 6 --kmeans 1 --main_epoch 1000 --alpha 0.00001 --beta 0.01 --h_dim 12
python /home/xue20/workspace/ATN_Beta/auto.py --num 20 --data eta1 --k 6 --kmeans 1 --main_epoch 1000 --alpha 0.00001 --beta 0.01 --h_dim 16

python /home/xue20/workspace/ATN_Beta/auto.py --num 20 --data eta1 --k 6 --kmeans 1 --main_epoch 1000 --alpha 0.001 --beta 1 --h_dim 8
python /home/xue20/workspace/ATN_Beta/auto.py --num 20 --data eta1 --k 6 --kmeans 1 --main_epoch 1000 --alpha 0.001 --beta 1 --h_dim 12
python /home/xue20/workspace/ATN_Beta/auto.py --num 20 --data eta1 --k 6 --kmeans 1 --main_epoch 1000 --alpha 0.001 --beta 1 --h_dim 16
python /home/xue20/workspace/ATN_Beta/auto.py --num 20 --data eta1 --k 6 --kmeans 1 --main_epoch 1000 --alpha 0.001 --beta 0.1 --h_dim 8
python /home/xue20/workspace/ATN_Beta/auto.py --num 20 --data eta1 --k 6 --kmeans 1 --main_epoch 1000 --alpha 0.001 --beta 0.1 --h_dim 12
python /home/xue20/workspace/ATN_Beta/auto.py --num 20 --data eta1 --k 6 --kmeans 1 --main_epoch 1000 --alpha 0.001 --beta 0.1 --h_dim 16
python /home/xue20/workspace/ATN_Beta/auto.py --num 20 --data eta1 --k 6 --kmeans 1 --main_epoch 1000 --alpha 0.001 --beta 0.01 --h_dim 8
python /home/xue20/workspace/ATN_Beta/auto.py --num 20 --data eta1 --k 6 --kmeans 1 --main_epoch 1000 --alpha 0.001 --beta 0.01 --h_dim 12
python /home/xue20/workspace/ATN_Beta/auto.py --num 20 --data eta1 --k 6 --kmeans 1 --main_epoch 1000 --alpha 0.001 --beta 0.01 --h_dim 16

python /home/xue20/workspace/ATN_Beta/auto.py --num 20 --data eta1 --k 6 --kmeans 1 --main_epoch 1000 --alpha 0.1 --beta 1 --h_dim 8
python /home/xue20/workspace/ATN_Beta/auto.py --num 20 --data eta1 --k 6 --kmeans 1 --main_epoch 1000 --alpha 0.1 --beta 1 --h_dim 12
python /home/xue20/workspace/ATN_Beta/auto.py --num 20 --data eta1 --k 6 --kmeans 1 --main_epoch 1000 --alpha 0.1 --beta 1 --h_dim 16
python /home/xue20/workspace/ATN_Beta/auto.py --num 20 --data eta1 --k 6 --kmeans 1 --main_epoch 1000 --alpha 0.1 --beta 0.1 --h_dim 8
python /home/xue20/workspace/ATN_Beta/auto.py --num 20 --data eta1 --k 6 --kmeans 1 --main_epoch 1000 --alpha 0.1 --beta 0.1 --h_dim 12
python /home/xue20/workspace/ATN_Beta/auto.py --num 20 --data eta1 --k 6 --kmeans 1 --main_epoch 1000 --alpha 0.1 --beta 0.1 --h_dim 16
python /home/xue20/workspace/ATN_Beta/auto.py --num 20 --data eta1 --k 6 --kmeans 1 --main_epoch 1000 --alpha 0.1 --beta 0.01 --h_dim 8
python /home/xue20/workspace/ATN_Beta/auto.py --num 20 --data eta1 --k 6 --kmeans 1 --main_epoch 1000 --alpha 0.1 --beta 0.01 --h_dim 12
python /home/xue20/workspace/ATN_Beta/auto.py --num 20 --data eta1 --k 6 --kmeans 1 --main_epoch 1000 --alpha 0.1 --beta 0.01 --h_dim 16
