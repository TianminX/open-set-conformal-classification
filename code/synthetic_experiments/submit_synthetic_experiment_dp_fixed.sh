#!/bin/bash

# Submit script for synthetic_experiment_dp.py with FIXED alpha allocation
# Usage: bash submit_synthetic_experiment_dp_fixed.sh

# List of different theta values to experiment with
THETA_LIST=(12 25 50 100 200 300 400 500 600 700 800 900 1000)

# List of different n_ref values
N_REF_LIST=(2000)

# List of n_test values
N_TEST_LIST=(1000)

# Calibration proportion (0.1 means 10% of n_ref)
CALIB_PROPORTION=0.1

# List of alpha_total values (total significance budget)
ALPHA_TOTAL_LIST=(0.10)

# Lambda weight (not used for tuning in fixed mode, but still required as argument)
LAMBDA_WEIGHT_LIST=(0.50)

# List of batch numbers
BATCH_LIST=$(seq 1 10)

# Fixed alpha allocations: alpha_class, alpha_unseen, alpha_seen
# These must sum to <= alpha_total
ALPHA_CLASS=0.09
ALPHA_UNSEEN=0.01
ALPHA_SEEN=0.0

# Tuning method flag: -1 for fixed
TUNING_METHOD=-1

# SLURM parameters
MEMO=16G                             # Memory required
TIME=00-10:00:00                     # Time required

# SBATCH command template
ORDP="sbatch --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task=1 --time="$TIME

# Ensure the results and logs directories exist
mkdir -p "results/dp_tuned_mixed_labels/"
mkdir -p "logs/dp_fixed/"

# Loop through all combinations
for BATCH in $BATCH_LIST; do
  echo "Processing Batch $BATCH..."

  for THETA in "${THETA_LIST[@]}"; do
    for N_REF in "${N_REF_LIST[@]}"; do
      for N_TEST in "${N_TEST_LIST[@]}"; do
        CALIB_NUM=$(echo "$N_REF * $CALIB_PROPORTION" | bc | cut -d. -f1)
        for ALPHA_TOTAL in "${ALPHA_TOTAL_LIST[@]}"; do
          for LAMBDA_WEIGHT in "${LAMBDA_WEIGHT_LIST[@]}"; do

            # Format values consistently
            ALPHA_TOTAL_FMT=$(printf "%.3f" "$ALPHA_TOTAL")
            LAMBDA_WEIGHT_FMT=$(printf "%.2f" "$LAMBDA_WEIGHT")

            # Create a unique job name
            JOBN="dp_fixed_theta${THETA}_n${N_REF}_t${N_TEST}_c${CALIB_NUM}_aT${ALPHA_TOTAL_FMT}_ac${ALPHA_CLASS}_au${ALPHA_UNSEEN}_as${ALPHA_SEEN}_b${BATCH}"

            # Define output and error log files
            OUTF="logs/dp_fixed/${JOBN}.out"
            ERRF="logs/dp_fixed/${JOBN}.err"

            # Check for existing output
            OUT_FILE_FMT="results/dp_tuned_mixed_labels/dp_theta%s_nref${N_REF}_ntest${N_TEST}_cs${CALIB_NUM}_atotal${ALPHA_TOTAL_FMT}_lambda${LAMBDA_WEIGHT_FMT}_tune${TUNING_METHOD}_batch${BATCH}.csv"

            OUT_FILE_INT=$(printf "$OUT_FILE_FMT" "$THETA")
            OUT_FILE_DOT=$(printf "$OUT_FILE_FMT" "${THETA}.0")

            if [[ -f "$OUT_FILE_INT" || -f "$OUT_FILE_DOT" ]]; then
              echo "Skipping job: $JOBN (output exists: $( [[ -f $OUT_FILE_INT ]] && echo "$OUT_FILE_INT" || echo "$OUT_FILE_DOT" ))"
            else
              SCRIPT="synthetic_experiment_dp.sh $THETA $N_REF $N_TEST $CALIB_NUM $ALPHA_TOTAL_FMT $LAMBDA_WEIGHT_FMT $BATCH $TUNING_METHOD $ALPHA_CLASS $ALPHA_UNSEEN $ALPHA_SEEN"
              ORD=$ORDP" -J $JOBN -o $OUTF -e $ERRF $SCRIPT"
              echo "Submitting job: $JOBN"
              $ORD
            fi

          done
        done
      done
    done
  done
  echo "Completed all parameter combinations for Batch $BATCH"
  echo "----------------------------------------"
done

echo "Job submission complete!"
