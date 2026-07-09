#!/bin/bash

# Submit SLURM jobs for the GT-recalibrated OpenMax classifiers on CelebA
# (unknown column rescaled so its average matches the Good-Turing estimate
# M1/n of the training split; keeps OpenMax's x-dependent unknown ranking).
# Same n_ref grid, subsampling scheme, and batch seeds as
# submit_real_experiment_celeb_openmax.sh so the reference/test splits match
# the raw OpenMax, GT-KNN, and plug-in runs exactly.
# One job per parameter combination; jobs whose output CSV already exists are skipped.

# List of different n_ref values
N_REF_LIST=(2000 3000 4000 5000 6000)

# List of n_test values
N_TEST_LIST=(1000)

# Calibration proportion (0.1 means 10% of n_ref)
CALIB_PROPORTION=0.1

# List of alpha_total values (single miscoverage budget; no alpha split)
ALPHA_TOTAL_LIST=(0.20)

# List of n_label_total values (for uniform sampling, 0 to skip)
N_LABEL_TOTAL_LIST=(2000)

# List of k_top values (top k celebrities to keep, 0 to skip)
K_TOP_LIST=(0)

# List of k_bot values (bottom k celebrities to keep, 0 to skip)
K_BOT_LIST=(0)

# List of batch numbers
BATCH_LIST=$(seq 1 20)

# SLURM parameters
MEMO=8G                            # Memory required
TIME=00-01:00:00                    # Time required (each base is fit twice: scoring copy + full model)
CPUS=2                              # CPUs per task (KNN black box uses n_jobs=-1)

# SBATCH command template
ORDP="sbatch --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task="$CPUS" --time="$TIME

# Ensure the results and logs directories exist
mkdir -p "results/celeb_openmax_recal/"
mkdir -p "logs/celeb_openmax_recal/"

# Loop with BATCH as the outermost loop
for BATCH in $BATCH_LIST; do
  echo "Processing Batch $BATCH..."

  for N_REF in "${N_REF_LIST[@]}"; do
    for N_TEST in "${N_TEST_LIST[@]}"; do
      # Calculate CALIB_NUM as 10% of N_REF
      CALIB_NUM=$(echo "$N_REF * $CALIB_PROPORTION" | bc | cut -d. -f1)

      for ALPHA_TOTAL in "${ALPHA_TOTAL_LIST[@]}"; do
        for N_LABEL_TOTAL in "${N_LABEL_TOTAL_LIST[@]}"; do
          for K_TOP in "${K_TOP_LIST[@]}"; do
            for K_BOT in "${K_BOT_LIST[@]}"; do

              # Format values consistently
              ALPHA_TOTAL_FMT=$(printf "%.3f" "$ALPHA_TOTAL")

              # Create a unique job name (shortened to avoid SLURM limitations)
              JOBN="celomrcl_n${N_REF}_t${N_TEST}_c${CALIB_NUM}_aT${ALPHA_TOTAL_FMT}_nl${N_LABEL_TOTAL}_k${K_TOP}_${K_BOT}_b${BATCH}"

              # Define output and error log files
              OUTF="logs/celeb_openmax_recal/${JOBN}.out"
              ERRF="logs/celeb_openmax_recal/${JOBN}.err"

              # Define output CSV file name (must match the Python output path)
              OUT_FILE="results/celeb_openmax_recal/celeb_openmax_recal_nref${N_REF}_ntest${N_TEST}_cs${CALIB_NUM}_atotal${ALPHA_TOTAL_FMT}_nlabel${N_LABEL_TOTAL}_ktop${K_TOP}_kbot${K_BOT}_batch_${BATCH}.csv"

              if [[ ! -f $OUT_FILE ]]; then
                # If the file doesn't exist, submit the job
                SCRIPT="real_experiment_celeb_openmax_recal.sh $N_REF $N_TEST $CALIB_NUM $ALPHA_TOTAL_FMT $N_LABEL_TOTAL $K_TOP $K_BOT $BATCH"
                ORD=$ORDP" -J $JOBN -o $OUTF -e $ERRF $SCRIPT"

                echo "Submitting job: $JOBN (batch=$BATCH, n_ref=$N_REF, calib_num=$CALIB_NUM, n_label_total=$N_LABEL_TOTAL)"
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

  echo "Completed all parameter combinations for Batch $BATCH"
  echo "----------------------------------------"
done

echo "Job submission complete!"
