#!/bin/bash -l

#SBATCH --job-name="test"
#SBATCH --partition=medium
#SBATCH --nodes=4
#SBATCH --time=0-05:00:00
#SBATCH --mem=30GB
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL


#SBATCH --output="job/test-%jo"
#SBATCH --error="job/test-%je"

source /deac/csc/chenGrp/software/tensorflow/bin/activate
python /deac/generalGrp/chenGrp/xue20/ATN_Beta/tools.py