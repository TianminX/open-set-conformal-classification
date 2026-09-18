#!/bin/bash

# Submit SLURM jobs for the open-set benchmarks on the Dirichlet-process
# synthetic data (synthetic_experiment_dp_openset_bench.py): the theta sweep
# at n_ref = 2000 of submit_synthetic_experiment_dp_mm_plugin.sh (same theta
# grid, n_test, calibration size, alpha, batches and seeds), so that every
# benchmark row pairs with a CGTC+ row of results/dp_tuned_mixed_labels_mm_plugin.
#
#   naive : Naive baseline (feature-blind Good-Turing constant)
#           -> results/dp_bench_naive/
#   occ   : OCC (LOF) raw + recalibrated
#           -> results/dp_bench_occ_<mode>/
#   knn1  : KNN-dist (k=1) raw + recalibrated
#           -> results/dp_bench_knn1_<mode>/
#
# Default modes reproduce the CelebA headline benchmarks: occ in mode scale
# (the multiplicative Recal-OCC (LOF)) and knn1 in mode isotonic
# (Iso-KNN-dist (k=1)). MODES="scale isotonic" runs both modes for both
# families. Jobs whose output CSV already exists are skipped.
#
# Usage: bash submit_synthetic_experiment_dp_openset_bench.sh [family ...]
#   (default: naive occ knn1)

if [[ $# -gt 0 ]]; then
  FAMILY_LIST=("$@")
else
  FAMILY_LIST=(naive occ knn1)
fi

# Recalibration modes per family (env MODES overrides both)
modes_for_family () {
  if [[ -n "${MODES:-}" ]]; then
    read -r -a MODE_LIST <<< "$MODES"
    return
  fi
  case "$1" in
    naive) MODE_LIST=(none) ;;
    occ)   MODE_LIST=(scale) ;;
    knn1)  MODE_LIST=(isotonic) ;;
    *) echo "Unknown family: $1 (use naive, occ, knn1)"; exit 1 ;;
  esac
}

# Theta grid of the CGTC+ runs (results/dp_tuned_mixed_labels_mm_plugin at n_ref 2000)
THETA_LIST=(12 25 50 100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500)

# List of different n_ref values
N_REF_LIST=(2000)

# List of n_test values
N_TEST_LIST=(1000)

# Calibration size: n_ref / CALIB_DENOM (10% of n_ref)
CALIB_DENOM=10

# List of alpha_total values (single miscoverage budget; no alpha split)
ALPHA_TOTAL_LIST=(0.10)

# List of batch numbers (5 repetitions each, as in the CGTC+ runs)
BATCH_LIST=$(seq 1 10)

# SLURM parameters (each job: 5 repetitions of one or two KNN fits on
# n_ref = 2000 points; a couple of minutes)
MEMO=4G
TIME=00-00:30:00
CPUS=2

ORDP="sbatch --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task="$CPUS" --time="$TIME

for FAMILY in "${FAMILY_LIST[@]}"; do
  modes_for_family "$FAMILY"

  for MODE in "${MODE_LIST[@]}"; do
    if [[ "$FAMILY" == "naive" ]]; then
      TAG="dp_bench_naive"
    else
      TAG="dp_bench_${FAMILY}_${MODE}"
    fi

    # Ensure the results and logs directories exist
    mkdir -p "results/${TAG}/"
    mkdir -p "logs/${TAG}/"

    for BATCH in $BATCH_LIST; do
      echo "Processing ${TAG}, Batch $BATCH..."

      for THETA in "${THETA_LIST[@]}"; do
        for N_REF in "${N_REF_LIST[@]}"; do
          for N_TEST in "${N_TEST_LIST[@]}"; do
            CALIB_NUM=$(( N_REF / CALIB_DENOM ))

            for ALPHA_TOTAL in "${ALPHA_TOTAL_LIST[@]}"; do
              ALPHA_TOTAL_FMT=$(printf "%.3f" "$ALPHA_TOTAL")

              JOBN="db_${FAMILY}_${MODE}_th${THETA}_n${N_REF}_t${N_TEST}_c${CALIB_NUM}_aT${ALPHA_TOTAL_FMT}_b${BATCH}"
              OUTF="logs/${TAG}/${JOBN}.out"
              ERRF="logs/${TAG}/${JOBN}.err"

              # Must match the Python output path
              OUT_FILE="results/${TAG}/${TAG}_theta${THETA}_nref${N_REF}_ntest${N_TEST}_cs${CALIB_NUM}_atotal${ALPHA_TOTAL_FMT}_batch${BATCH}.csv"

              if [[ ! -f $OUT_FILE ]]; then
                SCRIPT="synthetic_experiment_dp_openset_bench.sh $THETA $N_REF $N_TEST $CALIB_NUM $ALPHA_TOTAL_FMT $BATCH $FAMILY $MODE"
                ORD=$ORDP" -J $JOBN -o $OUTF -e $ERRF $SCRIPT"
                echo "Submitting job: $JOBN (family=$FAMILY, mode=$MODE, theta=$THETA, batch=$BATCH)"
                $ORD
              else
                echo "Skipping job: $JOBN (output file already exists)"
              fi
            done
          done
        done
      done

      echo "Completed all parameter combinations for ${TAG}, Batch $BATCH"
      echo "----------------------------------------"
    done
  done
done

echo "Job submission complete!"
