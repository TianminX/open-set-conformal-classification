# Conformal Inference for Open-Set and Imbalanced Classification

This repository contains the reference implementation of Conformal Good–Turing Classification (CGTC), a conformal prediction framework that (1) handles open-set labels via a principled "joker" option for unseen classes and (2) remains efficient under severe class imbalance via selective sample splitting with proper re-weighting.

Accompanying paper: *Conformal Inference for Open-Set and Imbalanced Classification*.

**Abstract.** This paper presents a conformal prediction method for classification in highly imbalanced and open-set settings, where there are many possible classes and not all may be represented in the available data. Existing methods require a finite, known label space and typically involve random sample splitting, which implicitly assumes the availability of a sufficient number of observations from each class. Consequently, they have two limitations: (i) they may become invalid when encountering previously unseen labels at test time, and (ii) they tend to become inefficient under extreme class imbalance. To obtain informative conformal prediction sets with valid coverage in the presence of unseen labels, we compute and integrate into our predictions a new type of conformal p-values inspired by the classical Good-Turing estimator, which can be used to rigorously test whether a new data point belongs to a previously unseen class. To make more efficient use of imbalanced data, we develop a selective sample splitting algorithm that partitions training and calibration data based on label frequency. Despite breaking exchangeability, this approach allows maintaining finite-sample coverage through proper re-weighting. With both simulated and real data, we demonstrate that our method leads to prediction sets with valid coverage even in very challenging open-set scenarios with infinite numbers of possible labels, and produces more informative predictions under extreme class imbalance.

## Repository layout

```
code/
  CGTC_Demo.ipynb                       Notebook walkthrough of the method
  cgtc/                                 Core CGTC implementation
    conformal_methods.py                Conformal prediction sets: CGTC and standard split
                                        conformal, with random or Bernoulli (selective) splitting
    testing.py                          GT / RGT / XGT conformal p-values and the
                                        unseen-label hypothesis test
    split.py                            Standard and Bernoulli selective sample splitting
    alpha_tune_function.py              Data-driven alpha allocation (cross-validation)
    alpha_tune_plugin.py                Plug-in missing-mass alpha allocation
    distributions*.py                   Data-generating utilities for the synthetic experiments
    utils.py                            Shared helpers
  synthetic_experiments/
    synthetic_experiment_dp.py          Dirichlet process simulations (main experiment)
    synthetic_experiment_dp_mm_plugin.py  Missing-mass plug-in allocation variant
    synthetic_experiment_*.py           Benchmarks and ablations (OpenMax, EVM, hybrids)
    submit_*.sh                         Batch wrappers with the parameter grids of the paper
    dp_mm_plugin_paper_plots.R          Main figures for the synthetic experiments
    beta_sensitivity_dp_plot.R          Beta sensitivity figures
    dp_vary_occ_plots.R                 One-class-classifier ablation figures
  real_experiment/
    data_prep/                          CelebA preprocessing: MTCNN crops + FaceNet embeddings
    real_experiment_celeb.py            CGTC pipeline on the CelebA embeddings
    real_experiment_celeb_mm_plugin.py  Missing-mass plug-in allocation variant
    real_experiment_celeb_*.py          Open-set benchmarks (OpenMax, PROSER, KNN scores, OCC)
    celeb_mm_plugin_paper_plots.R       Main figures for the real-data experiments
    real_celeb_compare_*.R              Benchmark comparison figures
    make_openset_benchmark_table.py     Benchmark comparison table
  third_party/
    arc/                                Set-valued classification utilities
    keras-facenet/                      FaceNet architecture used for the embeddings
dependencies.txt                        Tested package versions
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

**Figures and tables.** The R scripts and `make_openset_benchmark_table.py` aggregate the experiment CSVs (collected under `results_hpc/`) into the figures and tables of the paper.

The CelebA data and raw experiment outputs are not included in this repository.
