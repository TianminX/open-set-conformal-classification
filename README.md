# Conformal Inference for Open-Set and Imbalanced Classification

This repository contains the reference implementation for Conformal Good–Turing Classification (CGTC) — a conformal prediction framework that (1) handles open-set labels via a principled “joker” option for unseen classes and (2) remains efficient under severe class imbalance via selective sample splitting with proper re-weighting.

Accompanying paper: *Conformal Inference for Open-Set and Imbalanced Classification*. \url{}

Paper abstract: This paper presents a conformal prediction method for classification in highly imbalanced and open-set settings, where there are many possible classes and not all may be represented in the available data. Existing methods require a finite, known label space and typically involve random sample splitting, which implicitly assumes the availability of a sufficient number of observations from each class. Consequently, they have two limitations: (i) they may become invalid when encountering previously unseen labels at test time, and (ii) they tend to become inefficient under extreme class imbalance. To obtain informative conformal prediction sets with valid coverage in the presence of unseen labels, we compute and integrate into our predictions a new type of conformal p-values inspired by the classical Good-Turing estimator, which can be used to rigorously test whether a new data point belongs to a previously unseen class. To make more efficient use of imbalanced data, we develop a selective sample splitting algorithm that partitions training and calibration data based on label frequency. Despite breaking exchangeability, this approach allows maintaining finite-sample coverage through proper re-weighting. With both simulated and real data, we demonstrate that our method leads to prediction sets with valid coverage even in very challenging open-set scenarios with infinite numbers of possible labels, and produces more informative predictions under extreme class imbalance.


## Contents
open-set-conformal-classification/
- `code/`
    - `cgtc/`                           # Core CGTC implementation
        - `conformal_methods.py`         # Build conformal prediction sets using: 1) CGTC with random splitting, 2) CGTC with Bernoulli splitting, 3) Standard split conformal with random splitting, and 4) Standard split conformal with Bernoulli splitting
        - `testing.py`                   # Run hypothesis testing and evaluate GT / RGT / XGT p-values
        - `split.py`                     # Standard & Bernoulli selective split conformal
        - `alpha_tune_function.py`       # Data-driven $\alpha$ allocation (CV)
        - `distributions*.py`            # Utilities used in synthetic runs
        - `utils.py`                     # Helpers
    - `synthetic_experiments/`
        - `synthetic_experiment_dp.py`   # Dirichlet–Process simulations
        - `submit_*.sh`                  # Example batch scripts
    - `real_experiment/`
        - `real_experiment_celeb.py`     # CelebA pipeline (expects preprocessed data)
        - `data_prep/`
            - `celeb_preprocess.py`       # MTCNN crop + FaceNet embeddings
            - `celeb_combine.py`          # Combine per-batch embeddings → preprocessed data
        - `*.sh`                      # Example wrappers
    - `third_party/`
        - `arc/`                         # Set-valued classification utilities
- `dependencies.txt`                   # Pin of tested versions
- `paper/`                             # LaTeX sources for the manuscript


## Install
We recommend python=3.12 with the following dependencies install before running CGTC:

pip install tqdm numpy pillow mtcnn keras tensorflow notebook pandas matplotlib seaborn scikit-learn torch


