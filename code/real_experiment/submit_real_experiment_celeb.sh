#!/bin/bash

# List of different n_ref values
N_REF_LIST=(2000 3000 4000 5000 6000)

# List of n_test values
N_TEST_LIST=(1000)

# Calibration proportion (0.1 means 10% of n_ref)
CALIB_PROPORTION=0.1

# List of alpha_total values (total significance budget)
ALPHA_TOTAL_LIST=(0.20)

# List of lambda_weight values (weight parameter for loss function, between 0 and 1)
LAMBDA_WEIGHT_LIST=(0.50)

# List of n_label_total values (for uniform sampling, 0 to skip)
N_LABEL_TOTAL_LIST=(2000) # (500 1000 3000 4000) # 

# List of k_top values (top k celebrities to keep, 0 to skip)
K_TOP_LIST=(0)

# List of k_bot values (bottom k celebrities to keep, 0 to skip)
K_BOT_LIST=(0)

# List of tuning methods (0 for 'random', 1 for 'bernoulli')
TUNING_METHOD_LIST=(1)

# Note: Set n_label_total to 0 and k_top/k_bot to non-zero for top-bottom sampling
# Set all three to 0 to use full dataset

# List of batch numbers
BATCH_LIST=$(seq 1 20)

# SLURM parameters
MEMO=16G                            # Memory required (increased for real data)
TIME=00-25:00:00                    # Time required (increased for real data)

# SBATCH command template
ORDP="sbatch --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task=1 --time="$TIME

# Ensure the results and logs directories exist
mkdir -p "results/celeb/"
mkdir -p "logs/celeb/"

# Loop with BATCH as the outermost loop
for BATCH in $BATCH_LIST; do
  echo "Processing Batch $BATCH..."

  for N_REF in "${N_REF_LIST[@]}"; do
    for N_TEST in "${N_TEST_LIST[@]}"; do
      # Calculate CALIB_NUM as 10% of N_REF
      CALIB_NUM=$(echo "$N_REF * $CALIB_PROPORTION" | bc | cut -d. -f1)

      for ALPHA_TOTAL in "${ALPHA_TOTAL_LIST[@]}"; do
        for LAMBDA_WEIGHT in "${LAMBDA_WEIGHT_LIST[@]}"; do
          for N_LABEL_TOTAL in "${N_LABEL_TOTAL_LIST[@]}"; do
            for K_TOP in "${K_TOP_LIST[@]}"; do
              for K_BOT in "${K_BOT_LIST[@]}"; do
                for TUNING_METHOD in "${TUNING_METHOD_LIST[@]}"; do

                  # Format values consistently
                  ALPHA_TOTAL_FMT=$(printf "%.3f" "$ALPHA_TOTAL")
                  LAMBDA_WEIGHT_FMT=$(printf "%.2f" "$LAMBDA_WEIGHT")

                  # Create a unique job name (shortened to avoid SLURM limitations)
                  JOBN="cel_n${N_REF}_t${N_TEST}_c${CALIB_NUM}_aT${ALPHA_TOTAL_FMT}_l${LAMBDA_WEIGHT_FMT}_nl${N_LABEL_TOTAL}_k${K_TOP}_${K_BOT}_tm${TUNING_METHOD}_b${BATCH}"

                  # Define output and error log files
                  OUTF="logs/celeb/${JOBN}.out"
                  ERRF="logs/celeb/${JOBN}.err"

                  # Define output CSV file name based on parameters (including tuning method)
                  OUT_FILE="results/celeb/celeb_nref${N_REF}_ntest${N_TEST}_cs${CALIB_NUM}_atotal${ALPHA_TOTAL_FMT}_lambda${LAMBDA_WEIGHT_FMT}_nlabel${N_LABEL_TOTAL}_ktop${K_TOP}_kbot${K_BOT}_tune${TUNING_METHOD}_batch_${BATCH}.csv"

                  if [[ ! -f $OUT_FILE ]]; then
                    # If the file doesn't exist, submit the job
                    SCRIPT="real_experiment_celeb.sh $N_REF $N_TEST $CALIB_NUM $ALPHA_TOTAL_FMT $LAMBDA_WEIGHT_FMT $N_LABEL_TOTAL $K_TOP $K_BOT $BATCH $TUNING_METHOD"
                    ORD=$ORDP" -J $JOBN -o $OUTF -e $ERRF $SCRIPT"

                    # Display which tuning method is being used
                    if [ "$TUNING_METHOD" -eq 0 ]; then
                      METHOD_NAME="random"
                    else
                      METHOD_NAME="bernoulli"
                    fi

                    echo "Submitting job: $JOBN (batch=$BATCH, n_ref=$N_REF, calib_num=$CALIB_NUM, n_label_total=$N_LABEL_TOTAL, tuning=$METHOD_NAME)"
                    $ORD
                  else
                    echo "Skipping job: $JOBN (output file already exists)"
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
