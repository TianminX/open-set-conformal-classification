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
  synthetic_experiments/   Dirichlet process simulations, benchmarks, and figure scripts
  real_experiment/         CelebA experiments: data preparation, CGTC pipeline,
                           open-set benchmarks, and figure/table scripts
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

**Synthetic experiments.** Each `synthetic_experiment_*.py` script runs one configuration and writes a CSV of results under `results/`. The `submit_*.sh` wrappers record the full parameter grids used in the paper and are written for a Slurm cluster, but the Python scripts can also be invoked directly with the same command-line arguments.

**Real-data experiment.** Download the CelebA dataset, run the scripts in `real_experiment/data_prep/` to crop faces and compute FaceNet embeddings, then run the `real_experiment_celeb*.py` scripts. The benchmark scripts share the same interface and evaluation as the CGTC pipeline.

**Figures and tables.** The scripts below aggregate the experiment CSVs (collected under `results_hpc/` in each experiment folder) into the figures and tables of the paper. Run each one from its own folder.

| Script | Output |
| --- | --- |
| `synthetic_experiments/dp_mm_plugin_paper_plots.R` | CGTC+ on synthetic data: performance, coverage by label frequency, comparison of p-values, allocation of the significance level, reference sample size |
| `synthetic_experiments/dp_original_appendix_plots.R` | CGTC on synthetic data |
| `synthetic_experiments/dp_vary_occ_plots.R` | Sensitivity to the one-class classifier |
| `synthetic_experiments/dp_vary_calprop_plots.R`, `dp_coverage_boxplot_varyCalib.R` | Sensitivity to the calibration sample size (set `VARY_CALPROP_VARIANT=mm` for CGTC+) |
| `synthetic_experiments/dp_freq0_freq1_plot.R`, `dp_bin_mass_plot.R` | Composition and probability mass of the label-frequency bins |
| `synthetic_experiments/beta_sensitivity_experiment.py`, `beta_sensitivity_dp_plot.R` | Sensitivity to the exponent of the power-law weights |
| `synthetic_experiments/ess_selective_splitting.py` | Effective sample size of the selective-splitting weights |
| `synthetic_experiments/dp_compare_mm_plugin_vs_openset_bench.R` | Comparison with open-set classifiers on synthetic data |
| `real_experiment/celeb_mm_plugin_paper_plots.R` | CGTC+ on CelebA: performance, coverage by label frequency, comparison of p-values, allocation of the significance level |
| `real_experiment/celeb_original_appendix_plots.R`, `celeb_nlab_plots.R` | CGTC on CelebA, as a function of the reference sample size and of the number of identities |
| `real_experiment/real_celeb_compare_mm_plugin_vs_original.R` | Comparison of CGTC+ and CGTC on CelebA |
| `real_experiment/real_celeb_mm_plugin_lambda_compare.R` | Sensitivity to the preference parameter of the allocation loss |
| `real_experiment/real_celeb_compare_mm_plugin_vs_openmax.R`, `make_openset_benchmark_table.py`, `make_openset_diagnostics_table.py` | Comparison with open-set classifiers on CelebA (figures and tables) |

The CelebA data and raw experiment outputs are not included in this repository.
