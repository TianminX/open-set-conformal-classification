#!/bin/bash

module purge

eval "$(conda shell.bash hook)"
conda activate species

# Pin BLAS/OpenMP thread pools to the allocated CPUs to avoid oversubscription
# on shared nodes (the KNN black box uses n_jobs=-1).
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Run the Python script with the input arguments:
#   theta n_ref n_test calib_num alpha_total batch family mode
python3 synthetic_experiment_dp_openset_bench.py "$@"
