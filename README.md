# Conformal Inference for Open-Set and Imbalanced Classification

This repository contains the reference implementation of Conformal Good–Turing Classification (CGTC), a conformal prediction framework that (1) handles open-set labels via a principled "joker" option for unseen classes and (2) remains efficient under severe class imbalance via selective sample splitting with proper re-weighting.

Accompanying paper: *Conformal Inference for Open-Set and Imbalanced Classification*.

**Abstract.** This paper presents a conformal prediction method for classification in highly imbalanced and open-set settings, where there are many possible classes and not all may be represented in the available data. Existing methods require a finite, known label space and typically involve random sample splitting, which implicitly assumes the availability of a sufficient number of observations from each class. Consequently, they have two limitations: (i) they may become invalid when encountering previously unseen labels at test time, and (ii) they tend to become inefficient under extreme class imbalance. To obtain informative conformal prediction sets with valid coverage in the presence of unseen labels, we compute and integrate into our predictions a new type of conformal p-values inspired by the classical Good-Turing estimator, which can be used to rigorously test whether a new data point belongs to a previously unseen class. To make more efficient use of imbalanced data, we develop a selective sample splitting algorithm that partitions training and calibration data based on label frequency. Despite breaking exchangeability, this approach allows maintaining finite-sample coverage through proper re-weighting. With both simulated and real data, we demonstrate that our method leads to prediction sets with valid coverage even in very challenging open-set scenarios with infinite numbers of possible labels, and produces more informative predictions under extreme class imbalance.

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

**Figures and tables.** The R scripts and `make_openset_benchmark_table.py` aggregate the experiment CSVs (collected under `results_hpc/`) into the figures and tables of the paper.

The CelebA data and raw experiment outputs are not included in this repository.
