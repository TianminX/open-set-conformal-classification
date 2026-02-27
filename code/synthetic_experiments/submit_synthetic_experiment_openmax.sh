#!/bin/bash

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

# List of batch numbers
BATCH_LIST=$(seq 1 10)

# SLURM parameters
MEMO=16G                             # Memory required
TIME=00-02:00:00                     # Time required

# SBATCH command template
ORDP="sbatch --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task=1 --time="$TIME

# Ensure the results and logs directories exist
mkdir -p "results/dp_openmax/"
mkdir -p "logs/dp_openmax/"

# Loop through all combinations
for BATCH in $BATCH_LIST; do
  echo "Processing Batch $BATCH..."

  for THETA in "${THETA_LIST[@]}"; do
    for N_REF in "${N_REF_LIST[@]}"; do
      for N_TEST in "${N_TEST_LIST[@]}"; do
        CALIB_NUM=$(echo "$N_REF * $CALIB_PROPORTION" | bc | cut -d. -f1)
        for ALPHA_TOTAL in "${ALPHA_TOTAL_LIST[@]}"; do

          # Format values consistently
          ALPHA_TOTAL_FMT=$(printf "%.3f" "$ALPHA_TOTAL")

          # Create a unique job name
          JOBN="dp_openmax_theta${THETA}_n${N_REF}_t${N_TEST}_c${CALIB_NUM}_aT${ALPHA_TOTAL_FMT}_b${BATCH}"

          # Define output and error log files
          OUTF="logs/dp_openmax/${JOBN}.out"
          ERRF="logs/dp_openmax/${JOBN}.err"

          # Define the base pattern once
          OUT_FILE_FMT="results/dp_openmax/dp_theta%s_nref${N_REF}_ntest${N_TEST}_cs${CALIB_NUM}_atotal${ALPHA_TOTAL_FMT}_batch${BATCH}.csv"

          # Two possible outputs (integer vs integer.0)
          OUT_FILE_INT=$(printf "$OUT_FILE_FMT" "$THETA")
          OUT_FILE_DOT=$(printf "$OUT_FILE_FMT" "${THETA}.0")

          if [[ -f "$OUT_FILE_INT" || -f "$OUT_FILE_DOT" ]]; then
            echo "Skipping job: $JOBN (output exists: $( [[ -f $OUT_FILE_INT ]] && echo "$OUT_FILE_INT" || echo "$OUT_FILE_DOT" ))"
          else
            SCRIPT="synthetic_experiment_openmax.sh $THETA $N_REF $N_TEST $CALIB_NUM $ALPHA_TOTAL_FMT $BATCH"
            ORD=$ORDP" -J $JOBN -o $OUTF -e $ERRF $SCRIPT"
            echo "Submitting job: $JOBN"
            $ORD
          fi

        done
      done
    done
  done
  echo "Completed all parameter combinations for Batch $BATCH"
  echo "----------------------------------------"
done

echo "Job submission complete!"
