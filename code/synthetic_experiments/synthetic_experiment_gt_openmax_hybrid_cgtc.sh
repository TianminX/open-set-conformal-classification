#!/bin/bash

# Purge all loaded modules
module purge

eval "$(conda shell.bash hook)"
conda activate species

# Run the Python script with the input arguments
python synthetic_experiment_gt_openmax_hybrid_cgtc.py "$@"
