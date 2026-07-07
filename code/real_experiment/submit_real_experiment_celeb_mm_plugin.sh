#!/bin/bash

# Submit SLURM jobs for the CV-tuned plug-in missing-mass allocation on CelebA.
# Real-data counterpart of submit_synthetic_experiment_dp_mm_plugin.sh.
# One job per parameter combination; jobs whose output CSV already exists are skipped.

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
N_LABEL_TOTAL_LIST=(2000) # (500 1000 3000 4000)

# List of k_top values (top k celebrities to keep, 0 to skip)
K_TOP_LIST=(0)

# List of k_bot values (bottom k celebrities to keep, 0 to skip)
K_BOT_LIST=(0)

# Splitting method used inside the plug-in CV (0 = random, 1 = bernoulli)
SPLITTING_METHOD_LIST=(0 1)

# Cap-grid size G for the plug-in CV (number of subdivisions of alpha_remaining)
GRID_SIZE_LIST=(20)

# Beta label embedded in the Python output filename.
# Must match the Python script: beta_cv=False, default_beta=1.6  ->  "beta1.6".
BETA_LABEL="beta1.6"

# Note: Set n_label_total to 0 and k_top/k_bot to non-zero for top-bottom sampling.
# Set all three to 0 to use the full dataset.

# List of batch numbers
BATCH_LIST=$(seq 1 20)

# SLURM parameters
MEMO=16G                            # Memory required (increased for real data)
TIME=00-35:00:00                    # Time required (increased for real data)
CPUS=4                              # CPUs per task (KNN black box uses n_jobs=-1)

# SBATCH command template
ORDP="sbatch --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task="$CPUS" --time="$TIME

# Ensure the results and logs directories exist
mkdir -p "results/celeb_mm_plugin/"
mkdir -p "logs/celeb_mm_plugin/"

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
                for SPLITTING_METHOD in "${SPLITTING_METHOD_LIST[@]}"; do
                  for GRID_SIZE in "${GRID_SIZE_LIST[@]}"; do

                    # Format values consistently
                    ALPHA_TOTAL_FMT=$(printf "%.3f" "$ALPHA_TOTAL")
                    LAMBDA_WEIGHT_FMT=$(printf "%.2f" "$LAMBDA_WEIGHT")

                    # Create a unique job name (shortened to avoid SLURM limitations)
                    JOBN="celmmpl_n${N_REF}_t${N_TEST}_c${CALIB_NUM}_aT${ALPHA_TOTAL_FMT}_l${LAMBDA_WEIGHT_FMT}_nl${N_LABEL_TOTAL}_k${K_TOP}_${K_BOT}_sm${SPLITTING_METHOD}_G${GRID_SIZE}_b${BATCH}"

                    # Define output and error log files
                    OUTF="logs/celeb_mm_plugin/${JOBN}.out"
                    ERRF="logs/celeb_mm_plugin/${JOBN}.err"

                    # Define output CSV file name (must match the Python output path)
                    OUT_FILE="results/celeb_mm_plugin/celeb_${BETA_LABEL}_nref${N_REF}_ntest${N_TEST}_cs${CALIB_NUM}_atotal${ALPHA_TOTAL_FMT}_lambda${LAMBDA_WEIGHT_FMT}_nlabel${N_LABEL_TOTAL}_ktop${K_TOP}_kbot${K_BOT}_split${SPLITTING_METHOD}_G${GRID_SIZE}_batch_${BATCH}.csv"

                    if [[ ! -f $OUT_FILE ]]; then
                      # If the file doesn't exist, submit the job
                      SCRIPT="real_experiment_celeb_mm_plugin.sh $N_REF $N_TEST $CALIB_NUM $ALPHA_TOTAL_FMT $LAMBDA_WEIGHT_FMT $N_LABEL_TOTAL $K_TOP $K_BOT $BATCH $SPLITTING_METHOD $GRID_SIZE"
                      ORD=$ORDP" -J $JOBN -o $OUTF -e $ERRF $SCRIPT"

                      # Display which splitting method is being used
                      if [ "$SPLITTING_METHOD" -eq 0 ]; then
                        METHOD_NAME="random"
                      else
                        METHOD_NAME="bernoulli"
                      fi

                      echo "Submitting job: $JOBN (batch=$BATCH, n_ref=$N_REF, calib_num=$CALIB_NUM, n_label_total=$N_LABEL_TOTAL, split=$METHOD_NAME, G=$GRID_SIZE)"
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
  done

  echo "Completed all parameter combinations for Batch $BATCH"
  echo "----------------------------------------"
done

echo "Job submission complete!"
