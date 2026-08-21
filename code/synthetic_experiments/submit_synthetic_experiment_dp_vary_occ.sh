#!/bin/bash

# OCC sensitivity rerun (response letter, Referee 1 Q5): same parameters as the
# main synthetic run (submit_synthetic_experiment_dp.sh, CV-selected beta),
# except the one-class classifier is iforest or ocsvm instead of lof.
# Results go to results/dp_tuned_mixed_labels/vary_occ/ so they are kept
# separate from the lof data pooled by dp_original_appendix_plots.R.

# One-class classifiers to run (main results use lof)
OCC_LIST=(iforest ocsvm)

# List of different theta values to experiment with
THETA_LIST=(12 25 50 100 200 300 400 500 600 700 800 900 1000)

# List of different n_ref values
N_REF_LIST=(2000)

# List of n_test values
N_TEST_LIST=(1000)

# Calibration proportion (e.g., 0.1 means 10% of n_ref)
CALIB_PROPORTION_LIST=(0.1)

# List of alpha_total values (total significance budget)
ALPHA_TOTAL_LIST=(0.10)

# List of lambda_weight values (weight parameter for loss function, between 0 and 1)
LAMBDA_WEIGHT_LIST=(0.50)

# List of batch numbers
BATCH_LIST=$(seq 1 10)

# Tuning modes matching the main lof run: 0 = tuned alphas (random splitting),
# -1 = fixed alphas (0.09 / 0.01 / 0.00, same as the tune-1 lof batches).
# The response-letter vary-OCC figures only use tune0; drop -1 here to halve
# the job count if the fixed-alpha runs are not needed.
TUNING_METHOD_LIST=(0 -1)

# Fixed alpha allocation used when TUNING_METHOD is -1
ALPHA_CLASS_FIXED=0.09
ALPHA_UNSEEN_FIXED=0.01
ALPHA_SEEN_FIXED=0.00


# SLURM parameters
MEMO=16G                             # Memory required
TIME=00-10:00:00                     # Time required

# SBATCH command template
ORDP="sbatch --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task=1 --time="$TIME

# Ensure the results and logs directories exist
# NOTE: Python script writes to results/dp_tuned_mixed_labels/vary_occ/
mkdir -p "results/dp_tuned_mixed_labels/vary_occ/"
mkdir -p "logs/dp_vary_occ/"

# Loop through all combinations
for BATCH in $BATCH_LIST; do
  echo "Processing Batch $BATCH..."

  for OCC in "${OCC_LIST[@]}"; do
    for THETA in "${THETA_LIST[@]}"; do
      for N_REF in "${N_REF_LIST[@]}"; do
        for N_TEST in "${N_TEST_LIST[@]}"; do
          for CALIB_PROPORTION in "${CALIB_PROPORTION_LIST[@]}"; do
            CALIB_NUM=$(echo "$N_REF * $CALIB_PROPORTION" | bc | cut -d. -f1)
            for ALPHA_TOTAL in "${ALPHA_TOTAL_LIST[@]}"; do
              for LAMBDA_WEIGHT in "${LAMBDA_WEIGHT_LIST[@]}"; do
                for TUNING_METHOD in "${TUNING_METHOD_LIST[@]}"; do

                  # Format values consistently
                  ALPHA_TOTAL_FMT=$(printf "%.3f" "$ALPHA_TOTAL")
                  LAMBDA_WEIGHT_FMT=$(printf "%.2f" "$LAMBDA_WEIGHT")

                  # Create a unique job name
                  JOBN="dp_${OCC}_theta${THETA}_n${N_REF}_t${N_TEST}_c${CALIB_NUM}_aT${ALPHA_TOTAL_FMT}_l${LAMBDA_WEIGHT_FMT}_tm${TUNING_METHOD}_b${BATCH}"

                  # Define output and error log files
                  OUTF="logs/dp_vary_occ/${JOBN}.out"
                  ERRF="logs/dp_vary_occ/${JOBN}.err"

                  # Check for existing output (matches Python's output path)
                  OUT_FILE_FMT="results/dp_tuned_mixed_labels/vary_occ/dp_occ${OCC}_betacv_theta%s_nref${N_REF}_ntest${N_TEST}_cs${CALIB_NUM}_atotal${ALPHA_TOTAL_FMT}_lambda${LAMBDA_WEIGHT_FMT}_tune${TUNING_METHOD}_batch${BATCH}.csv"

                  # Two possible outputs (integer vs integer.0)
                  OUT_FILE_INT=$(printf "$OUT_FILE_FMT" "$THETA")
                  OUT_FILE_DOT=$(printf "$OUT_FILE_FMT" "${THETA}.0")

                  if [[ -f "$OUT_FILE_INT" || -f "$OUT_FILE_DOT" ]]; then
                    echo "Skipping job: $JOBN (output exists: $( [[ -f $OUT_FILE_INT ]] && echo "$OUT_FILE_INT" || echo "$OUT_FILE_DOT" ))"
                  else
                    if [[ "$TUNING_METHOD" == "-1" ]]; then
                      SCRIPT="synthetic_experiment_dp_vary_occ.sh $THETA $N_REF $N_TEST $CALIB_NUM $ALPHA_TOTAL_FMT $LAMBDA_WEIGHT_FMT $BATCH $TUNING_METHOD $ALPHA_CLASS_FIXED $ALPHA_UNSEEN_FIXED $ALPHA_SEEN_FIXED $OCC"
                    else
                      SCRIPT="synthetic_experiment_dp_vary_occ.sh $THETA $N_REF $N_TEST $CALIB_NUM $ALPHA_TOTAL_FMT $LAMBDA_WEIGHT_FMT $BATCH $TUNING_METHOD $OCC"
                    fi
                    ORD=$ORDP" -J $JOBN -o $OUTF -e $ERRF $SCRIPT"
                    echo "Submitting job: $JOBN"
                    $ORD
                  fi
                done
              done
            done
          done
        done
      done
    done
  done
  echo "Completed all parameter combinations for Batch $BATCH"
  echo "----------------------------------------"
done

echo "Job submission complete!"
