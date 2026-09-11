#!/bin/bash

module purge

eval "$(conda shell.bash hook)"
conda activate species

# Pin BLAS/OpenMP thread pools to the allocated CPUs to avoid oversubscription
# on shared nodes (the KNN black box uses n_jobs=-1; PROSER uses torch threads).
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Run the Python script with the input arguments:
#   n_ref n_test calib_num alpha_total n_label_total k_top k_bot batch family mode
python3 real_experiment_celeb_openset_bench.py "$@"
