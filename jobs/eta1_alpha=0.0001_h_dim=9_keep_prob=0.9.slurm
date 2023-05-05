#!/bin/bash
#SBATCH --job-name="eta1_alpha=0.0001_h_dim=9_keep_prob=0.9"
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=50GB
#SBATCH --ntasks-per-node=8
#SBATCH --account=chenGrp
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="jobs_oe/eta1_alpha=0.0001_h_dim=9_keep_prob=0.9-%j.o"
#SBATCH --error="jobs_oe/eta1_alpha=0.0001_h_dim=9_keep_prob=0.9-%j.e"

echo $(pwd) > "jobs/pwd.txt"
source /deac/csc/chenGrp/software/tensorflow/bin/activate
python run.py --num 1 --alpha 0.0001 --h_dim 9 --keep_prob 0.9 --data eta1 --k 6 --beta 1.0 --kmeans 1 --main_epoch 1000
