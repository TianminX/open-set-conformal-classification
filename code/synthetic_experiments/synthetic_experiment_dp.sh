#!/bin/bash

# Purge all loaded modules
module purge

eval "$(conda shell.bash hook)"
conda activate species

# export OPENBLAS_NUM_THREADS=1

# Run the Python script with the input arguments
python synthetic_experiment_dp.py $1 $2 $3 $4 $5 $6 $7 $8