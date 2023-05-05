#!/bin/bash

for alpha in "0.0001" "1e-5" "1e-6"
do
    for h_dim in "7" "8" "9"
    do
        for keep_prob in "0.7" "0.8" "0.9"
        do
            sbatch jobs/eta1_alpha\=${alpha}_h_dim\=${h_dim}_keep_prob\=${keep_prob}.slurm
        done
    done
done