#!/bin/bash

# List of different theta values to experiment with (formatted to 2 decimal places)
THETA_LIST=(1000) # (12 25 50 100 200 300 400 500 600 700 800 900 1000)
#(25 50 100 200 400 800 1600 3200 6400 12800 25600 51200 102400)

# List of different n_ref values
N_REF_LIST=(1000 3000 4000 5000 6000) # (2000)

# List of n_test values
N_TEST_LIST=(1000)

# Calibration proportion (0.1 means 10% of n_ref)
CALIB_PROPORTION=0.1

# List of alpha_total values (total significance budget)
ALPHA_TOTAL_LIST=(0.10) # 0.20

# List of lambda_weight values (weight parameter for loss function, between 0 and 1)
LAMBDA_WEIGHT_LIST=(0.50)

# List of batch numbers
BATCH_LIST=$(seq 1 10)

# List of tuning methods (0 for 'random', 1 for 'bernoulli')
TUNING_METHOD_LIST=(0)


# SLURM parameters
MEMO=16G                             # Memory required (e.g., 8 GB)
TIME=00-10:00:00                    # Time required (e.g., 30 minutes)

# SBATCH command template
ORDP="sbatch --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task=1 --time="$TIME

# Ensure the results and logs directories exist
mkdir -p "results/dp/"
mkdir -p "logs/dp/"

# Loop through all combinations of theta, n_ref, batch_num, calib_num, alpha_total, and lambda_weight
# Loop with BATCH as the outermost loop
for BATCH in $BATCH_LIST; do
  echo "Processing Batch $BATCH..."
    
  for THETA in "${THETA_LIST[@]}"; do
    for N_REF in "${N_REF_LIST[@]}"; do
      for N_TEST in "${N_TEST_LIST[@]}"; do
        CALIB_NUM=$(echo "$N_REF * $CALIB_PROPORTION" | bc | cut -d. -f1)
        for ALPHA_TOTAL in "${ALPHA_TOTAL_LIST[@]}"; do
          for LAMBDA_WEIGHT in "${LAMBDA_WEIGHT_LIST[@]}"; do
            for TUNING_METHOD in "${TUNING_METHOD_LIST[@]}"; do

              # Format values consistently
              # THETA_FMT=$(printf "%.2f" "$THETA")
              ALPHA_TOTAL_FMT=$(printf "%.3f" "$ALPHA_TOTAL")
              LAMBDA_WEIGHT_FMT=$(printf "%.2f" "$LAMBDA_WEIGHT")

              # Create a unique job name
              JOBN="dp_theta${THETA}_n${N_REF}_t${N_TEST}_c${CALIB_NUM}_aT${ALPHA_TOTAL_FMT}_l${LAMBDA_WEIGHT_FMT}_tm${TUNING_METHOD}_b${BATCH}"

              # Define output and error log files
              OUTF="logs/dp/${JOBN}.out"
              ERRF="logs/dp/${JOBN}.err"

              # Define the base pattern once
              OUT_FILE_FMT="results/dp/dp_theta%s_nref${N_REF}_ntest${N_TEST}_cs${CALIB_NUM}_atotal${ALPHA_TOTAL_FMT}_lambda${LAMBDA_WEIGHT_FMT}_tune${TUNING_METHOD}_batch${BATCH}.csv"

              # Two possible outputs (integer vs integer.0)
              OUT_FILE_INT=$(printf "$OUT_FILE_FMT" "$THETA")
              OUT_FILE_DOT=$(printf "$OUT_FILE_FMT" "${THETA}.0")

              if [[ -f "$OUT_FILE_INT" || -f "$OUT_FILE_DOT" ]]; then
                echo "Skipping job: $JOBN (output exists: $( [[ -f $OUT_FILE_INT ]] && echo "$OUT_FILE_INT" || echo "$OUT_FILE_DOT" ))"
              else
                SCRIPT="synthetic_experiment_dp.sh $THETA $N_REF $N_TEST $CALIB_NUM $ALPHA_TOTAL_FMT $LAMBDA_WEIGHT_FMT $BATCH $TUNING_METHOD"
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
  echo "Completed all parameter combinations for Batch $BATCH"
  echo "----------------------------------------"
done

echo "Job submission complete!"
