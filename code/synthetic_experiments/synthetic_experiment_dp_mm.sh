#!/bin/bash

# Purge all loaded modules
module purge

eval "$(conda shell.bash hook)"
conda activate species

# export OPENBLAS_NUM_THREADS=1

# Run the Python script with the input arguments
python synthetic_experiment_dp_mm.py "$@"
