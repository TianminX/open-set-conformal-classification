#!/bin/bash
#
# STALE -- superseded by synthetic_experiment_dp_mm_plugin.sh
# Implements an earlier missing-mass adjustment (mu_hat = M1/n, no cap on
# alpha_unseen) that does NOT match the paper. Kept only to reproduce the
# 'CGTC (orig)' / 'CGTC (adj-a)' baselines. Do not use for new runs.
# Purge all loaded modules
module purge

eval "$(conda shell.bash hook)"
conda activate species

# export OPENBLAS_NUM_THREADS=1

# Run the Python script with the input arguments
python synthetic_experiment_dp_mm_stale.py "$@"
