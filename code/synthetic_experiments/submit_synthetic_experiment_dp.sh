#!/bin/bash

# List of different theta values to experiment with
THETA_LIST=(1000) #(500 1000) #12 25 50 100 200 300 400 500 600 700 800 900 1000

# List of different n_ref values
N_REF_LIST=(2000)

# List of n_test values
N_TEST_LIST=(1000)

# Calibration proportion (e.g., 0.1 means 10% of n_ref)
CALIB_PROPORTION_LIST=(0.05 0.1 0.2 0.5) # (0.1)

# List of alpha_total values (total significance budget)
ALPHA_TOTAL_LIST=(0.10)

# List of lambda_weight values (weight parameter for loss function, between 0 and 1)
LAMBDA_WEIGHT_LIST=(0.50)

# List of batch numbers
BATCH_LIST=$(seq 1 10) # (seq 1 10)

# List of tuning methods (0 for 'random', 1 for 'bernoulli')
TUNING_METHOD_LIST=(0)

# List of one-class classifiers
OCC_NAME_LIST=(lof) #iforest ocsvm lof 

# Whether to use CV for beta (True or False)
BETA_CV_LIST=(False) #True False 


# SLURM parameters
MEMO=16G                             # Memory required
TIME=00-10:00:00                     # Time required

# SBATCH command template
ORDP="sbatch --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task=1 --time="$TIME

# Results directory for this run
# NOTE: must match the output_file path in synthetic_experiment_dp.py
RESULTS_DIR="results/dp_tuned_mixed_labels/rebuttal/2026-08-24"

# Ensure the results and logs directories exist
mkdir -p "$RESULTS_DIR/"
mkdir -p "logs/dp/"

# Loop through all combinations
for BATCH in $BATCH_LIST; do
  echo "Processing Batch $BATCH..."

  for THETA in "${THETA_LIST[@]}"; do
    for N_REF in "${N_REF_LIST[@]}"; do
      for N_TEST in "${N_TEST_LIST[@]}"; do
        for CALIB_PROPORTION in "${CALIB_PROPORTION_LIST[@]}"; do
          CALIB_NUM=$(echo "$N_REF * $CALIB_PROPORTION" | bc | cut -d. -f1)
          for ALPHA_TOTAL in "${ALPHA_TOTAL_LIST[@]}"; do
            for LAMBDA_WEIGHT in "${LAMBDA_WEIGHT_LIST[@]}"; do
              for TUNING_METHOD in "${TUNING_METHOD_LIST[@]}"; do
                for OCC_NAME in "${OCC_NAME_LIST[@]}"; do
                  for BETA_CV in "${BETA_CV_LIST[@]}"; do

                    # Format values consistently
                    ALPHA_TOTAL_FMT=$(printf "%.3f" "$ALPHA_TOTAL")
                    LAMBDA_WEIGHT_FMT=$(printf "%.2f" "$LAMBDA_WEIGHT")
                    BETA_LABEL=$( [[ "$BETA_CV" == "True" ]] && echo "betacv" || echo "beta1.6" )

                    # Create a unique job name
                    JOBN="dp_occ${OCC_NAME}_beta${BETA_LABEL}_theta${THETA}_n${N_REF}_t${N_TEST}_c${CALIB_NUM}_aT${ALPHA_TOTAL_FMT}_l${LAMBDA_WEIGHT_FMT}_tm${TUNING_METHOD}_b${BATCH}"

                    # Define output and error log files
                    OUTF="logs/dp/${JOBN}.out"
                    ERRF="logs/dp/${JOBN}.err"

                    # Check for existing output
                    OUT_FILE="${RESULTS_DIR}/dp_occ${OCC_NAME}_beta${BETA_LABEL}_theta${THETA}_nref${N_REF}_ntest${N_TEST}_cs${CALIB_NUM}_atotal${ALPHA_TOTAL_FMT}_lambda${LAMBDA_WEIGHT_FMT}_tune${TUNING_METHOD}_batch${BATCH}.csv"

                    if [[ -f "$OUT_FILE" ]]; then
                      echo "Skipping job: $JOBN (output exists: $OUT_FILE)"
                    else
                      SCRIPT="synthetic_experiment_dp.sh $THETA $N_REF $N_TEST $CALIB_NUM $ALPHA_TOTAL_FMT $LAMBDA_WEIGHT_FMT $BATCH $TUNING_METHOD $OCC_NAME $BETA_CV"
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
  done
  echo "Completed all parameter combinations for Batch $BATCH"
  echo "----------------------------------------"
done

echo "Job submission complete!"
