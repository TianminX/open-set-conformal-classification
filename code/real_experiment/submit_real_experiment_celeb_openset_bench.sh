#!/bin/bash

# Submit SLURM jobs for the unified open-set benchmark runner on CelebA
# (real_experiment_celeb_openset_bench.py): every recalibrated benchmark
# under ONE uniform recalibration procedure, plus the new raw scores.
#
#   families : knn (Recal KNN-dist k=10, Recal KNN-MSP)
#              knn1 (KNN-dist k=1 raw + Recal KNN-dist k=1)
#              proser (Recal PROSER)
#              openmax (Recal OpenMax-MLP, Recal OpenMax-KNN)
#              occ (raw + Recal OCC for lof / iforest / ocsvm / ocsvm20)
#   modes    : center (uniform centering + Good-Turing anchoring; the
#              recalibration used for the paper's Recal rows once these
#              results land), isotonic (conditional recalibration, robustness
#              check). Add "scale" to reproduce the original multiplicative
#              recalibration inside the same runner.
#
# Same n_ref grid, subsampling scheme, and batch seeds as
# submit_real_experiment_celeb_openmax.sh so the reference/test splits match
# the other runs exactly. One job per parameter combination; jobs whose
# output CSV already exists are skipped.
#
# Usage: bash submit_real_experiment_celeb_openset_bench.sh [family ...]
#   (default: all five families; e.g. "bash submit_... knn1 occ" for two)

if [[ $# -gt 0 ]]; then
  FAMILY_LIST=("$@")
else
  FAMILY_LIST=(knn knn1 proser openmax occ)
fi

# Recalibration modes to run
MODE_LIST=(center isotonic)

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
BATCH_LIST=$(seq 1 50)

# SLURM resources per family (each recalibrated base is fit twice: scoring
# copy + full model; occ fits four one-class classifiers twice each).
resources_for_family () {
  case "$1" in
    knn|knn1) MEMO=4G;  TIME=00-00:40:00; CPUS=2 ;;
    occ)      MEMO=4G;  TIME=00-01:00:00; CPUS=2 ;;
    openmax)  MEMO=8G;  TIME=00-01:30:00; CPUS=2 ;;
    proser)   MEMO=8G;  TIME=00-04:00:00; CPUS=4 ;;
    *) echo "Unknown family: $1"; exit 1 ;;
  esac
}

for FAMILY in "${FAMILY_LIST[@]}"; do
  resources_for_family "$FAMILY"
  ORDP="sbatch --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task="$CPUS" --time="$TIME

  for MODE in "${MODE_LIST[@]}"; do
    TAG="celeb_bench_${FAMILY}_${MODE}"

    # Ensure the results and logs directories exist
    mkdir -p "results/${TAG}/"
    mkdir -p "logs/${TAG}/"

    # Loop with BATCH as the outermost loop
    for BATCH in $BATCH_LIST; do
      echo "Processing ${TAG}, Batch $BATCH..."

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
                  JOBN="cb_${FAMILY}_${MODE}_n${N_REF}_t${N_TEST}_c${CALIB_NUM}_aT${ALPHA_TOTAL_FMT}_nl${N_LABEL_TOTAL}_k${K_TOP}_${K_BOT}_b${BATCH}"

                  # Define output and error log files
                  OUTF="logs/${TAG}/${JOBN}.out"
                  ERRF="logs/${TAG}/${JOBN}.err"

                  # Define output CSV file name (must match the Python output path)
                  OUT_FILE="results/${TAG}/${TAG}_nref${N_REF}_ntest${N_TEST}_cs${CALIB_NUM}_atotal${ALPHA_TOTAL_FMT}_nlabel${N_LABEL_TOTAL}_ktop${K_TOP}_kbot${K_BOT}_batch_${BATCH}.csv"

                  if [[ ! -f $OUT_FILE ]]; then
                    # If the file doesn't exist, submit the job
                    SCRIPT="real_experiment_celeb_openset_bench.sh $N_REF $N_TEST $CALIB_NUM $ALPHA_TOTAL_FMT $N_LABEL_TOTAL $K_TOP $K_BOT $BATCH $FAMILY $MODE"
                    ORD=$ORDP" -J $JOBN -o $OUTF -e $ERRF $SCRIPT"

                    echo "Submitting job: $JOBN (family=$FAMILY, mode=$MODE, batch=$BATCH, n_ref=$N_REF, calib_num=$CALIB_NUM)"
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

      echo "Completed all parameter combinations for ${TAG}, Batch $BATCH"
      echo "----------------------------------------"
    done
  done
done

echo "Job submission complete!"
