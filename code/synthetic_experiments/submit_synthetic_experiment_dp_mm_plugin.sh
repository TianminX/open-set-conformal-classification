#!/bin/bash

# Submit SLURM jobs for the CV-tuned plug-in missing-mass allocation experiment.
# One job per parameter combination; jobs whose output CSV already exists are skipped.

# Fixed theta for the asymptotic regime (we vary n_ref instead)
THETA_LIST=(500)

# Reference sizes, log-spaced from 1000 to 100000 (asymptotic regime)
N_REF_LIST=(1000 2000 5000 10000 20000 50000 100000)

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

# Splitting method used inside the CV (0 = random, 1 = bernoulli)
SPLITTING_METHOD_LIST=(0)

# Cap-grid size G for the plug-in CV (number of subdivisions of alpha_remaining)
GRID_SIZE_LIST=(20)


# SLURM parameters
MEMO=16G                             # Memory required
TIME=00-12:00:00                     # Time required
CPUS=4                               # CPUs per task (KNN black box uses n_jobs=-1)

# SBATCH command template
ORDP="sbatch --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task="$CPUS" --time="$TIME

# Ensure the results and logs directories exist
mkdir -p "results/dp_tuned_mixed_labels_mm_plugin/"
mkdir -p "logs/dp_mm_plugin/"

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
              for SPLITTING_METHOD in "${SPLITTING_METHOD_LIST[@]}"; do
                for GRID_SIZE in "${GRID_SIZE_LIST[@]}"; do

                # Format values consistently
                ALPHA_TOTAL_FMT=$(printf "%.3f" "$ALPHA_TOTAL")
                LAMBDA_WEIGHT_FMT=$(printf "%.2f" "$LAMBDA_WEIGHT")

                # Create a unique job name
                JOBN="dpmmplugin_theta${THETA}_n${N_REF}_t${N_TEST}_c${CALIB_NUM}_aT${ALPHA_TOTAL_FMT}_l${LAMBDA_WEIGHT_FMT}_sm${SPLITTING_METHOD}_G${GRID_SIZE}_b${BATCH}"

                # Define output and error log files
                OUTF="logs/dp_mm_plugin/${JOBN}.out"
                ERRF="logs/dp_mm_plugin/${JOBN}.err"

                # Check for existing output (matches Python's output path)
                OUT_FILE_FMT="results/dp_tuned_mixed_labels_mm_plugin/dp_occ%s_betacv_theta%s_nref${N_REF}_ntest${N_TEST}_cs${CALIB_NUM}_atotal${ALPHA_TOTAL_FMT}_lambda${LAMBDA_WEIGHT_FMT}_split${SPLITTING_METHOD}_G${GRID_SIZE}_batch${BATCH}.csv"

                # Two possible outputs (integer vs integer.0)
                OUT_FILE_INT=$(printf "$OUT_FILE_FMT" "lof" "$THETA")
                OUT_FILE_DOT=$(printf "$OUT_FILE_FMT" "lof" "${THETA}.0")

                if [[ -f "$OUT_FILE_INT" || -f "$OUT_FILE_DOT" ]]; then
                  echo "Skipping job: $JOBN (output exists: $( [[ -f $OUT_FILE_INT ]] && echo "$OUT_FILE_INT" || echo "$OUT_FILE_DOT" ))"
                else
                  SCRIPT="synthetic_experiment_dp_mm_plugin.sh $THETA $N_REF $N_TEST $CALIB_NUM $ALPHA_TOTAL_FMT $LAMBDA_WEIGHT_FMT $BATCH $SPLITTING_METHOD $GRID_SIZE"
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
