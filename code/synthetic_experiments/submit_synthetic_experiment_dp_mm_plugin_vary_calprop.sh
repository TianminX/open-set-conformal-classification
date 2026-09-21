#!/bin/bash

# Submit SLURM jobs for the calibration-size sweep of the CV-tuned plug-in
# missing-mass method (CGTC+), the CGTC+ counterpart of the original-CGTC sweep
# in results_hpc/dp_tuned_mixed_labels/vary_calprop (Appendix on the
# sensitivity to the calibration sample size and the coverage boxplot).
#
# Design: theta = 1000 and n_ref = 2000 fixed, calibration size in
# {100, 200, 400, 1000} (5% to 50% of n_ref), 10 batches x 5 replicates
# (num_exp = 5 inside synthetic_experiment_dp_mm_plugin.py), random CV split
# (splitting_method_flag = 0, the variant used by the manuscript figures).
#
# Results are written to results/dp_tuned_mixed_labels_mm_plugin_vary_calprop/
# through the DP_MM_RESULTS_SUFFIX environment variable read by
# synthetic_experiment_dp_mm_plugin.py, so the main theta-sweep folder
# (results/dp_tuned_mixed_labels_mm_plugin/) is left untouched.  Copy the
# folder to results_hpc/ locally, then run from code/synthetic_experiments/:
#   VARY_CALPROP_VARIANT=mm Rscript dp_vary_calprop_plots.R
#   VARY_CALPROP_VARIANT=mm Rscript dp_coverage_boxplot_varyCalib.R
#
# Jobs whose output CSV already exists are skipped.

THETA_LIST=(1000)
N_REF_LIST=(2000)

# Test-set size.  1000 matches every other synthetic figure; the binomial
# noise of the coverage estimate is then sqrt(0.09/1000) ~ 0.95 percentage
# points, comparable to the calibration-induced spread at 1000 calibration
# points.  Use 4000 to halve that floor if the boxplot is the main target.
N_TEST_LIST=(1000)

# Calibration sizes (absolute), 5% / 10% / 20% / 50% of n_ref = 2000
CALIB_NUM_LIST=(100 200 400 1000)

ALPHA_TOTAL_LIST=(0.10)
LAMBDA_WEIGHT_LIST=(0.50)
BATCH_LIST=$(seq 1 10)

# Splitting method used inside the CV (0 = random, 1 = bernoulli); the
# manuscript figures use 0.
SPLITTING_METHOD_LIST=(0)

# Cap-grid size G for the plug-in CV
GRID_SIZE_LIST=(20)

# Results-folder suffix (read by the Python runner)
SUFFIX="_vary_calprop"

# SLURM parameters (same as submit_synthetic_experiment_dp_mm_plugin.sh)
MEMO=16G
TIME=00-24:00:00
CPUS=4

ORDP="sbatch --mem=${MEMO} --nodes=1 --ntasks=1 --cpus-per-task=${CPUS} --time=${TIME} --export=ALL,DP_MM_RESULTS_SUFFIX=${SUFFIX}"

RESULTS_DIR="results/dp_tuned_mixed_labels_mm_plugin${SUFFIX}"
LOG_DIR="logs/dp_mm_plugin${SUFFIX}"
mkdir -p "${RESULTS_DIR}/" "${LOG_DIR}/"

for BATCH in $BATCH_LIST; do
  echo "Processing Batch $BATCH..."
  for THETA in "${THETA_LIST[@]}"; do
    for N_REF in "${N_REF_LIST[@]}"; do
      for N_TEST in "${N_TEST_LIST[@]}"; do
        for CALIB_NUM in "${CALIB_NUM_LIST[@]}"; do
          for ALPHA_TOTAL in "${ALPHA_TOTAL_LIST[@]}"; do
            for LAMBDA_WEIGHT in "${LAMBDA_WEIGHT_LIST[@]}"; do
              for SPLITTING_METHOD in "${SPLITTING_METHOD_LIST[@]}"; do
                for GRID_SIZE in "${GRID_SIZE_LIST[@]}"; do

                  ALPHA_TOTAL_FMT=$(printf "%.3f" "$ALPHA_TOTAL")
                  LAMBDA_WEIGHT_FMT=$(printf "%.2f" "$LAMBDA_WEIGHT")

                  JOBN="dpmmvc_theta${THETA}_n${N_REF}_t${N_TEST}_c${CALIB_NUM}_aT${ALPHA_TOTAL_FMT}_l${LAMBDA_WEIGHT_FMT}_sm${SPLITTING_METHOD}_G${GRID_SIZE}_b${BATCH}"
                  OUTF="${LOG_DIR}/${JOBN}.out"
                  ERRF="${LOG_DIR}/${JOBN}.err"

                  # Must match the output path built by synthetic_experiment_dp_mm_plugin.py
                  OUT_FILE_FMT="${RESULTS_DIR}/dp_occ%s_betacv_theta%s_nref${N_REF}_ntest${N_TEST}_cs${CALIB_NUM}_atotal${ALPHA_TOTAL_FMT}_lambda${LAMBDA_WEIGHT_FMT}_split${SPLITTING_METHOD}_G${GRID_SIZE}_batch${BATCH}.csv"
                  OUT_FILE_INT=$(printf "$OUT_FILE_FMT" "lof" "$THETA")
                  OUT_FILE_DOT=$(printf "$OUT_FILE_FMT" "lof" "${THETA}.0")

                  if [[ -f "$OUT_FILE_INT" || -f "$OUT_FILE_DOT" ]]; then
                    echo "Skipping job: $JOBN (output exists)"
                  else
                    SCRIPT="synthetic_experiment_dp_mm_plugin.sh $THETA $N_REF $N_TEST $CALIB_NUM $ALPHA_TOTAL_FMT $LAMBDA_WEIGHT_FMT $BATCH $SPLITTING_METHOD $GRID_SIZE"
                    ORD="$ORDP -J $JOBN -o $OUTF -e $ERRF $SCRIPT"
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
