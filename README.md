# Conformal Inference for Open-Set and Imbalanced Classification

This repository contains the reference implementation of Conformal Good–Turing Classification (CGTC) and of its missing-mass-adjusted version (CGTC+), a conformal prediction framework that (1) handles open-set labels via a principled "joker" option for unseen classes and (2) remains efficient under severe class imbalance via selective sample splitting with proper re-weighting.

Accompanying paper: *Conformal Inference for Open-Set and Imbalanced Classification*.

**Abstract.** This paper studies conformal prediction for highly imbalanced and open-set classification, where the data are exchangeable but there are many possible classes and not all may be observed. We show that existing conformal methods implicitly assume that all labels are observed, so they may undercover when new labels arise at test time; moreover, they make inefficient use of imbalanced data, sometimes producing uninformative prediction sets. To address the first limitation, we introduce conformal p-values for testing whether a test point has a new label, establish their optimality, and connect them to classical work by Good and Turing. Building on these, we develop a conformal prediction method that guarantees marginal coverage under exchangeability alone, together with an asymptotically valid adjustment that reduces conservativeness. To address the second limitation, we propose a selective sample splitting scheme that prioritizes rare labels for training, preserving finite-sample validity through reweighting. Experiments with simulated and real data show that our method performs well even with infinitely many possible labels.

## Repository layout

```
code/
  CGTC_Demo.ipynb          Notebook walkthrough of the original CGTC method
  CGTC_Plus_Demo.ipynb     Notebook walkthrough of CGTC+ (plug-in missing-mass allocation)
  cgtc/                    Core CGTC implementation: conformal prediction sets,
                           Good–Turing p-values, selective splitting, alpha allocation
  synthetic_experiments/   Dirichlet process simulations, open-set benchmarks,
                           figure scripts
  real_experiment/         CelebA experiments: data preparation, CGTC and CGTC+,
                           open-set benchmarks, figure and table scripts
  third_party/             Set-valued classification utilities and the FaceNet architecture
dependencies.txt           Tested package versions
```

## Installation

Python 3.12 is recommended, with the following packages:

```
pip install tqdm numpy pillow mtcnn keras tensorflow notebook pandas matplotlib seaborn scikit-learn torch
```

See `dependencies.txt` for the exact versions used for the paper.

## Reproducing the experiments

Each experiment script runs one configuration and writes a CSV file under `results/`. The `synthetic_experiment_*.py` and `real_experiment_celeb*.py` scripts take their settings as positional command-line arguments and print the order of these arguments when called without any. Run each script from its own folder. The argument `batch_num` sets the random seed, so the repetitions in the paper are consecutive batches. Each row of a CSV file reports the coverage, the prediction set size, and the joker frequency of one method (columns `method` and `pvalue_method`) in one repetition. The figures and tables of the paper average these rows.

### Synthetic data

The scripts are in `code/synthetic_experiments/`. Unless stated otherwise, the paper uses `n_test = 1000`, `alpha_total = 0.10`, `lambda_weight = 0.50`, `calib_num = n_ref / 10`, and batches 1 to 10. Each batch runs five repetitions.

| Script | Experiment | Settings in the paper |
| --- | --- | --- |
| `synthetic_experiment_dp_mm_plugin.py` | CGTC+ | `splitting_method_flag = 0` and `grid_size = 20` throughout. Main results: `theta` in {12, 25, 50, 100, 200, ..., 1500} with `n_ref = 2000`. Reference sample size: `n_ref` in {1000, 2000, 5000, 10000, 20000} with `theta = 100`. Calibration sample size: `calib_num` in {100, 200, 400, 1000} with `theta = 1000` and `n_ref = 2000`. |
| `synthetic_experiment_dp.py` | CGTC | `tuning_method_flag = 0` and `occ_name = lof` throughout. Main results: `theta` in {12, 25, 50, 100, 200, ..., 1000} with `n_ref = 2000` and `beta_cv = True`. Calibration sample size: `calib_num` in {100, 200, 400, 1000} with `theta = 1000`, `n_ref = 2000`, and `beta_cv = False`. |
| `synthetic_experiment_dp_vary_occ.py` | CGTC with other one-class classifiers | Main grid of CGTC with `occ_name` in {`iforest`, `ocsvm`}. |
| `synthetic_experiment_dp_openset_bench.py` | Open-set benchmarks | Main grid of CGTC+ with (`family`, `mode`) in {(`naive`, `none`), (`occ`, `scale`), (`knn1`, `isotonic`)}. |
| `beta_sensitivity_experiment.py` | Exponent of the power-law weights | `python beta_sensitivity_experiment.py 100 2026 dp` |
| `ess_selective_splitting.py` | Effective sample size of selective splitting | Default arguments. |

### CelebA data

The scripts are in `code/real_experiment/`. Download the CelebA dataset (the aligned images in `img_align_celeba/` and the file `identity_CelebA.txt`) into `data/celebrity/` at the root of the repository, and place the pretrained FaceNet weights at `code/third_party/keras-facenet/model/facenet_keras.h5`. Then, from `code/real_experiment/`, run `python data_prep/celeb_preprocess.py <batch> 500` for batches 1 to 406, which crops the faces and computes their FaceNet embeddings, followed by `python data_prep/celeb_combine.py`, which collects the embeddings in the file `combined_data.npz` read by the experiment scripts.

Unless stated otherwise, the paper uses `n_ref` in {2000, 3000, 4000, 5000, 6000}, `n_test = 1000`, `calib_num = n_ref / 10`, `alpha_total = 0.20`, and 2,000 identities sampled uniformly at random (`n_label_total = 2000`, `k_top = 0`, `k_bot = 0`). Each batch runs one repetition.

| Script | Experiment | Settings in the paper |
| --- | --- | --- |
| `real_experiment_celeb_mm_plugin.py` | CGTC+ | Batches 1 to 20 and `grid_size = 20` throughout. Main results: `lambda_weight` in {0.30, 0.50, 0.70} with `splitting_method_flag = 0`. Comparison with CGTC: `lambda_weight = 0.50` with `splitting_method_flag = 1`. Larger reference samples: `n_ref` in {8000, 10000, 12000, 15000, 20000, 25000, 30000} with `lambda_weight = 0.50` and `splitting_method_flag = 0`. |
| `real_experiment_celeb.py` | CGTC | Batches 1 to 20 and `lambda_weight = 0.50` throughout. Main results: `tuning_method` in {0, 1}. Number of identities: `n_label_total` in {500, 1000, 2000, 3000} with `n_ref = 2000` and `tuning_method = 1`. |
| `real_experiment_celeb_gt_knn.py` | Naive benchmark | Batches 1 to 50, also with the larger reference samples. |
| `real_experiment_celeb_openmax.py`, `real_experiment_celeb_openmax_osdn.py`, `real_experiment_celeb_openmax_recal.py` | OpenMax: simplified, faithful, and recalibrated | Batches 1 to 50. |
| `real_experiment_celeb_knn_scores_raw.py`, `real_experiment_celeb_knn_scores.py` | Nearest-neighbor scores: raw and recalibrated | Batches 1 to 50. |
| `real_experiment_celeb_proser.py`, `real_experiment_celeb_proser_recal.py` | PROSER: raw and recalibrated | Batches 1 to 50. |
| `real_experiment_celeb_occ_recal.py` | One-class classifier score: raw and recalibrated | Batches 1 to 50, also with the larger reference samples. |
| `real_experiment_celeb_openset_bench.py` | Isotonic recalibration of all scores | Batches 1 to 50, `family` in {`knn`, `knn1`, `proser`, `openmax`, `occ`} with `mode = isotonic`. The family `knn1` is also run with the larger reference samples. |

### Cluster scripts

The experiments were run on a SLURM cluster, with one job per configuration. Two pairs of scripts are included as examples, for CGTC+ on synthetic data and on CelebA:

- `synthetic_experiments/submit_synthetic_experiment_dp_mm_plugin.sh` with `synthetic_experiment_dp_mm_plugin.sh`
- `real_experiment/submit_real_experiment_celeb_mm_plugin.sh` with `real_experiment_celeb_mm_plugin.sh`

Each `submit_*.sh` script loops over a grid of settings (by default the main results), skips the configurations whose output file already exists, and submits the others through the second script, which activates the conda environment (named `species` here) and calls the Python script. The other experiments were launched in the same way.

### Figures and tables

The scripts below turn the experiment CSV files into the figures and tables of the paper. Each script names the folder it reads near its top, usually a subfolder of `results_hpc/` next to the script, where the outputs of the cluster runs were collected. Run each script from its own folder. The R scripts need the packages `tidyverse`, `data.table`, `scales`, `ggh4x`, and `ggrepel`.

| Script | Output |
| --- | --- |
| `synthetic_experiments/dp_mm_plugin_paper_plots.R` | CGTC+ on synthetic data: performance, coverage by label frequency, comparison of p-values, allocation of the significance level, reference sample size |
| `synthetic_experiments/dp_original_appendix_plots.R` | CGTC on synthetic data |
| `synthetic_experiments/dp_vary_occ_plots.R` | Sensitivity to the one-class classifier |
| `synthetic_experiments/dp_vary_calprop_plots.R`, `dp_coverage_boxplot_varyCalib.R` | Sensitivity to the calibration sample size (set `VARY_CALPROP_VARIANT=mm` for CGTC+) |
| `synthetic_experiments/dp_freq0_freq1_plot.R`, `dp_bin_mass_plot.R` | Composition and probability mass of the label-frequency bins |
| `synthetic_experiments/beta_sensitivity_dp_plot.R` | Sensitivity to the exponent of the power-law weights |
| `synthetic_experiments/dp_compare_mm_plugin_vs_openset_bench.R` | Comparison with open-set classifiers on synthetic data |
| `real_experiment/celeb_mm_plugin_paper_plots.R` | CGTC+ on CelebA: performance, coverage by label frequency, comparison of p-values, allocation of the significance level |
| `real_experiment/celeb_original_appendix_plots.R`, `celeb_nlab_plots.R` | CGTC on CelebA, as a function of the reference sample size and of the number of identities |
| `real_experiment/real_celeb_compare_mm_plugin_vs_original.R` | Comparison of CGTC+ and CGTC on CelebA |
| `real_experiment/real_celeb_mm_plugin_lambda_compare.R` | Sensitivity to the preference parameter of the allocation loss |
| `real_experiment/real_celeb_compare_mm_plugin_vs_openmax.R`, `make_openset_benchmark_table.py`, `make_openset_diagnostics_table.py` | Comparison with open-set classifiers on CelebA (figures and tables) |

The CelebA data and the experiment outputs are not included in this repository.
