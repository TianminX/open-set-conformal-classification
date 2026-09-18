#!/bin/bash

# Submit SLURM jobs for the large-n_ref extension of the CelebA open-set
# comparison: the four methods of the headline figure (setting "main" of
# real_celeb_compare_mm_plugin_vs_openmax.R) at reference sample sizes beyond
# the paper's 2000-6000 grid, to show the methods approaching each other as
# the missing mass vanishes and the problem becomes (almost) closed-set.
#
#   cgtc  : CGTC+                 real_experiment_celeb_mm_plugin.sh
#                                 (betacv, lambda 0.50, random CV split, G = 20)
#   naive : Naive baseline        real_experiment_celeb_gt_knn.sh
#   occ   : Recal-OCC (LOF)       real_experiment_celeb_occ_recal.sh
#                                 (the legacy multiplicative recalibration)
#   knn1  : Iso-KNN-dist (k=1)    real_experiment_celeb_openset_bench.sh knn1 isotonic
#
# Same subsampling scheme (2,000 uniformly sampled identities, 39,827 images),
# n_test, calibration proportion, alpha and batch seeds as the existing runs,
# so the reference/test splits match those of the paper's grid exactly.
#
# Results go to the sibling folders results/<folder>_large/ (the Python
# runners read CELEB_RESULTS_SUFFIX), so the figure scripts that read
# results/<folder>/ only are unaffected; real_celeb_compare_mm_plugin_vs_openmax.R
# reads a folder together with its _large sibling and plots the extended grid
# in its setting "main_large". Jobs whose output CSV exists are skipped.
#
# Usage: bash submit_real_experiment_celeb_large_nref.sh [method ...]
#   (default: all four methods; e.g. "bash submit_... naive occ knn1" submits
#   the cheap benchmarks only)
# Environment overrides:
#   N_REFS="8000 10000 ..."  n_ref grid (default below; needs n_ref + n_test <= 39827)
#   BATCHES_CGTC=20          CGTC+ batches (the figures use batches 1-20)
#   BATCHES_BENCH=50         benchmark batches (as in the existing runs)
#   SUFFIX=_large            results-folder suffix

if [[ $# -gt 0 ]]; then
  METHOD_LIST=("$@")
else
  METHOD_LIST=(cgtc naive occ knn1)
fi

# n_ref grid beyond the paper's 2000-6000 (unseen fraction of the test
# identities: ~0.03 at 8000, ~0.014 at 10000, ~0.009 at 12000, ~0.005 at
# 20000, ~0.003 at 30000). A job is skipped when its output exists in the
# _large folder or in the original results/<folder>/, so the paper's grid
# may be listed without rerunning or duplicating it.
read -r -a N_REF_LIST <<< "${N_REFS:-8000 10000 12000 15000 20000 25000 30000}"

# Batches: CGTC+ uses batches 1-20 in the figures, the benchmarks 1-50.
BATCHES_CGTC=${BATCHES_CGTC:-20}
BATCHES_BENCH=${BATCHES_BENCH:-50}

# Results-folder suffix (see the Python runners)
SUFFIX=${SUFFIX:-_large}

# Fixed settings, identical to the other CelebA submit scripts
N_TEST=1000
CALIB_DENOM=10                 # calib_num = n_ref / 10
ALPHA_TOTAL_FMT=$(printf "%.3f" 0.20)
LAMBDA_WEIGHT_FMT=$(printf "%.2f" 0.50)   # CGTC+ only
N_LABEL_TOTAL=2000
K_TOP=0
K_BOT=0
SPLITTING_METHOD=0             # CGTC+ plug-in CV split: random
GRID_SIZE=20                   # CGTC+ cap grid
N_IMAGES=39827                 # images among the 2,000 sampled identities

# SLURM resources per method. CGTC+: the plug-in CV recomputes the
# frequency-based p-values on every grid point, roughly 4 to 5 hours per job
# at n_ref = 30000 (about half an hour at 6000); the benchmarks take minutes.
resources_for_method () {
  case "$1" in
    cgtc)  MEMO=16G; TIME=00-35:00:00; CPUS=4 ;;
    naive) MEMO=4G;  TIME=00-00:30:00; CPUS=2 ;;
    occ)   MEMO=4G;  TIME=00-00:30:00; CPUS=2 ;;
    knn1)  MEMO=4G;  TIME=00-01:00:00; CPUS=2 ;;
    *) echo "Unknown method: $1 (use cgtc, naive, occ, knn1)"; exit 1 ;;
  esac
}

for METHOD in "${METHOD_LIST[@]}"; do
  resources_for_method "$METHOD"
  ORDP="sbatch --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task="$CPUS" --time="$TIME" --export=ALL,CELEB_RESULTS_SUFFIX="$SUFFIX

  case "$METHOD" in
    cgtc)  TAG_BASE="celeb_mm_plugin";           N_BATCH=$BATCHES_CGTC ;;
    naive) TAG_BASE="celeb_gt_knn";              N_BATCH=$BATCHES_BENCH ;;
    occ)   TAG_BASE="celeb_occ";                 N_BATCH=$BATCHES_BENCH ;;
    knn1)  TAG_BASE="celeb_bench_knn1_isotonic"; N_BATCH=$BATCHES_BENCH ;;
  esac
  TAG="${TAG_BASE}${SUFFIX}"

  # Ensure the results and logs directories exist
  mkdir -p "results/${TAG}/"
  mkdir -p "logs/${TAG}/"

  for BATCH in $(seq 1 "$N_BATCH"); do
    echo "Processing ${TAG}, Batch $BATCH..."

    for N_REF in "${N_REF_LIST[@]}"; do
      if (( N_REF + N_TEST > N_IMAGES )); then
        echo "Skipping n_ref=$N_REF: n_ref + n_test exceeds the $N_IMAGES available images"
        continue
      fi
      CALIB_NUM=$(( N_REF / CALIB_DENOM ))

      case "$METHOD" in
        cgtc)
          JOBN="cl_cgtc_n${N_REF}_t${N_TEST}_c${CALIB_NUM}_aT${ALPHA_TOTAL_FMT}_l${LAMBDA_WEIGHT_FMT}_nl${N_LABEL_TOTAL}_k${K_TOP}_${K_BOT}_sm${SPLITTING_METHOD}_G${GRID_SIZE}_b${BATCH}"
          OUT_NAME="celeb_betacv_nref${N_REF}_ntest${N_TEST}_cs${CALIB_NUM}_atotal${ALPHA_TOTAL_FMT}_lambda${LAMBDA_WEIGHT_FMT}_nlabel${N_LABEL_TOTAL}_ktop${K_TOP}_kbot${K_BOT}_split${SPLITTING_METHOD}_G${GRID_SIZE}_batch_${BATCH}.csv"
          SCRIPT="real_experiment_celeb_mm_plugin.sh $N_REF $N_TEST $CALIB_NUM $ALPHA_TOTAL_FMT $LAMBDA_WEIGHT_FMT $N_LABEL_TOTAL $K_TOP $K_BOT $BATCH $SPLITTING_METHOD $GRID_SIZE"
          ;;
        naive)
          JOBN="cl_naive_n${N_REF}_t${N_TEST}_c${CALIB_NUM}_aT${ALPHA_TOTAL_FMT}_nl${N_LABEL_TOTAL}_k${K_TOP}_${K_BOT}_b${BATCH}"
          OUT_NAME="celeb_gt_knn_nref${N_REF}_ntest${N_TEST}_cs${CALIB_NUM}_atotal${ALPHA_TOTAL_FMT}_nlabel${N_LABEL_TOTAL}_ktop${K_TOP}_kbot${K_BOT}_batch_${BATCH}.csv"
          SCRIPT="real_experiment_celeb_gt_knn.sh $N_REF $N_TEST $CALIB_NUM $ALPHA_TOTAL_FMT $N_LABEL_TOTAL $K_TOP $K_BOT $BATCH"
          ;;
        occ)
          JOBN="cl_occ_n${N_REF}_t${N_TEST}_c${CALIB_NUM}_aT${ALPHA_TOTAL_FMT}_nl${N_LABEL_TOTAL}_k${K_TOP}_${K_BOT}_b${BATCH}"
          OUT_NAME="celeb_occ_nref${N_REF}_ntest${N_TEST}_cs${CALIB_NUM}_atotal${ALPHA_TOTAL_FMT}_nlabel${N_LABEL_TOTAL}_ktop${K_TOP}_kbot${K_BOT}_batch_${BATCH}.csv"
          SCRIPT="real_experiment_celeb_occ_recal.sh $N_REF $N_TEST $CALIB_NUM $ALPHA_TOTAL_FMT $N_LABEL_TOTAL $K_TOP $K_BOT $BATCH"
          ;;
        knn1)
          JOBN="cl_knn1_n${N_REF}_t${N_TEST}_c${CALIB_NUM}_aT${ALPHA_TOTAL_FMT}_nl${N_LABEL_TOTAL}_k${K_TOP}_${K_BOT}_b${BATCH}"
          OUT_NAME="celeb_bench_knn1_isotonic_nref${N_REF}_ntest${N_TEST}_cs${CALIB_NUM}_atotal${ALPHA_TOTAL_FMT}_nlabel${N_LABEL_TOTAL}_ktop${K_TOP}_kbot${K_BOT}_batch_${BATCH}.csv"
          SCRIPT="real_experiment_celeb_openset_bench.sh $N_REF $N_TEST $CALIB_NUM $ALPHA_TOTAL_FMT $N_LABEL_TOTAL $K_TOP $K_BOT $BATCH knn1 isotonic"
          ;;
      esac

      OUT_FILE="results/${TAG}/${OUT_NAME}"
      OUT_FILE_BASE="results/${TAG_BASE}/${OUT_NAME}"
      OUTF="logs/${TAG}/${JOBN}.out"
      ERRF="logs/${TAG}/${JOBN}.err"

      if [[ -f $OUT_FILE_BASE ]]; then
        echo "Skipping job: $JOBN (output file already exists in results/${TAG_BASE}/)"
      elif [[ ! -f $OUT_FILE ]]; then
        ORD=$ORDP" -J $JOBN -o $OUTF -e $ERRF $SCRIPT"
        echo "Submitting job: $JOBN (method=$METHOD, batch=$BATCH, n_ref=$N_REF, calib_num=$CALIB_NUM)"
        $ORD
      else
        echo "Skipping job: $JOBN (output file already exists)"
      fi
    done

    echo "Completed all n_ref values for ${TAG}, Batch $BATCH"
    echo "----------------------------------------"
  done
done

echo "Job submission complete!"
