#!/bin/bash

# Purge all loaded modules
module purge

eval "$(conda shell.bash hook)"
conda activate species

# Run the Python script with the input arguments
python synthetic_experiment_gt_knn.py $1 $2 $3 $4 $5 $6
