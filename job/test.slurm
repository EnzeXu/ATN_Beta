#!/bin/bash #on the first line, meaning that the script should always be run with bash, rather than another shell.

#SBATCH --job-name="test"
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --nodes=2
#SBATCH --time=0-05:00:00
#SBATCH --mem=10GB
#SBATCH --mail-user=xue20@wfu.edu
#SBATCH --mail-type=BEGIN,END,FAIL


#SBATCH --output="job/test-%jo"
#SBATCH --error="job/test-%je"

source /deac/csc/chenGrp/software/tensorflow/bin/activate
python /deac/generalGrp/chenGrp/xue20/ATN_Beta/tools.py