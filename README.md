# Conformal Inference for Open-Set and Imbalanced Classification

This repository contains the reference implementation of Conformal Good–Turing Classification (CGTC) and of its missing-mass-adjusted version (CGTC+), a conformal prediction framework that (1) handles open-set labels via a principled "joker" option for unseen classes and (2) remains efficient under severe class imbalance via selective sample splitting with proper re-weighting.

Accompanying paper: *Conformal Inference for Open-Set and Imbalanced Classification*.

**Abstract.** This paper studies conformal prediction for highly imbalanced and open-set classification, where the data are exchangeable but there are many possible classes and not all may be observed. We show that existing conformal methods implicitly assume that all labels are observed, so they may undercover when new labels arise at test time; moreover, they make inefficient use of imbalanced data, sometimes producing uninformative prediction sets. To address the first limitation, we introduce conformal p-values for testing whether a test point has a new label, study their optimality, and connect them to classical work by Good and Turing. Building on these, we develop a conformal prediction method that guarantees marginal coverage under exchangeability alone, together with an asymptotically valid adjustment that reduces conservativeness. To address the second limitation, we propose a selective sample splitting scheme that prioritizes rare labels for training, preserving finite-sample validity through reweighting. Experiments with simulated and real data show that our method performs well even with infinitely many possible labels.

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

Each `synthetic_experiment_*.py` and `real_experiment_celeb*.py` script runs one configuration and writes a CSV file under `results/`. Run it from its own folder. Called without arguments, it prints the arguments it expects. In the file names, `dp` (Dirichlet process) refers to the synthetic data, `celeb` to CelebA, `mm_plugin` to CGTC+, and `original` to CGTC.

**Synthetic data.** In `code/synthetic_experiments/`, `synthetic_experiment_dp_mm_plugin.py` runs CGTC+ and `synthetic_experiment_dp.py` runs CGTC. The other Python scripts run the additional analyses and the open-set benchmarks.

**CelebA data.** Download the CelebA images (`img_align_celeba/`) and the file `identity_CelebA.txt` into `data/celebrity/`, and place the pretrained FaceNet weights at `code/third_party/keras-facenet/model/facenet_keras.h5`. From `code/real_experiment/`, run `python data_prep/celeb_preprocess.py <batch> 500` for batches 1 to 406 to crop the faces and compute their FaceNet embeddings, then `python data_prep/celeb_combine.py` to collect them. In the same folder, `real_experiment_celeb_mm_plugin.py` runs CGTC+, `real_experiment_celeb.py` runs CGTC, and the other `real_experiment_celeb_*.py` scripts run the open-set benchmarks.

**Cluster.** The experiments were run on a SLURM cluster, with one job per configuration. The `submit_*.sh` scripts, with the job scripts they call, are included as examples for CGTC+. They show the settings of its main results.

**Figures and tables.** The R scripts and the two `make_openset_*_table.py` scripts turn the CSV files into the figures and tables of the paper. Each one reads the folder named near its top, usually under `results_hpc/`, where the outputs of the cluster runs were collected.

The CelebA data and the experiment outputs are not included in this repository.
