"""
Empirical coverage check for the CV-tuned plug-in missing-mass allocation.

On an exchangeable label stream (Dirichlet process -> the sequence is
exchangeable), we tune the plug-in hyperparameters (alpha_unseen cap,
alpha_seen) by cross-validation, deploy the derived allocation
  mu_hat       = (M1 + 1)/(n + 1)
  alpha_unseen = min(cap, mu_hat)
  alpha_class  = min((alpha_total - alpha_seen - alpha_unseen)/(1 - mu_hat), 1)
and verify that the empirical MARGINAL miscoverage of the conformal
Good-Turing classifier stays at or below the target alpha_total (up to Monte
Carlo tolerance). Marginal coverage is 'Coverage (?)' from
evaluate_prediction_sets (the joker counts as covering a genuinely novel label).

This is the tune-and-deploy pipeline, so the CV is run inside every
replication; it is therefore heavy. Keep R / K / G modest, or raise them when
running a serious check. Run from code/synthetic_experiments/ :
    python verify_plugin_allocation_coverage.py
"""

import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import LocalOutlierFactor

os.environ["LOKY_MAX_CPU_COUNT"] = "20"

sys.path.insert(0, os.path.abspath('../third_party'))
import arc
from arc import black_boxes

sys.path.insert(0, '../cgtc/')
from conformal_methods import (
    get_preliminary_sets_naive, finalize_prediction_sets, evaluate_prediction_sets,
)
from alpha_tune_plugin import tune_plugin_allocation_cv, compute_plugin_allocation
from distributions_y import DirichletProcess
from distributions_x import ShiftedNormal


# ============================================================
# Parameters
# ============================================================
theta = 50            # DP concentration -> P(novel) ~ theta/(theta+n)
n_ref = 300           # reference (train + calibration) size
n_test = 300          # test points per replication
calib_num = 150       # expected calibration size
alpha_total = 0.10    # target miscoverage level
lambda_weight = 0.5   # CV loss preference
grid_size = 5         # small cap-grid for speed (G)
n_splits = 3          # small K for speed
R = 200               # Monte Carlo replications
seed = 2024
pvalue_method = 'XGT'
beta = 1.6            # fixed power weight for the seen-label test
num_features = 3
sigma = 1e-4

calib_size = calib_num / n_ref

# Models (same as the experiment driver)
classifier = black_boxes.OpenSetKNN(
    calibrate=False, n_neighbors=5, weights='distance', algorithm='auto',
    leaf_size=30, p=2, metric='minkowski', metric_params=None,
    n_jobs=-1, clip_proba_factor=1e-20,
)
occ = LocalOutlierFactor(n_neighbors=1, novelty=True)


def split_data(X, Y, n_ref, n_test, random_state):
    """Shuffle-split into reference and test (matches the driver)."""
    np.random.seed(random_state)
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    ref, test = idx[:n_ref], idx[n_ref:n_ref + n_test]
    return X[ref], Y[ref], X[test], Y[test]


# ============================================================
# Monte Carlo: tune -> deploy -> evaluate marginal coverage
# ============================================================
label_dist = DirichletProcess(theta=theta)
feature_dist = ShiftedNormal(num_features=num_features, sigma=sigma)

coverage = np.empty(R)
mu_hat_arr = np.empty(R)
a_unseen_arr = np.empty(R)
a_seen_arr = np.empty(R)
a_class_arr = np.empty(R)
cap_arr = np.empty(R)

print(f"Running {R} MC replications (theta={theta}, n_ref={n_ref}, "
      f"alpha={alpha_total}, K={n_splits}, G={grid_size}, pval={pvalue_method})...")

for r in tqdm(range(R)):
    rs = seed + r
    Y = label_dist.sample(n_ref + n_test, random_state=rs)   # exchangeable
    X = feature_dist.sample(Y, random_state=rs)
    X_ref, Y_ref, X_test, Y_test = split_data(X, Y, n_ref, n_test, rs)

    # --- Tune (CV) on the reference sample
    best_cap, best_alpha_seen, _ = tune_plugin_allocation_cv(
        X_ref, Y_ref,
        alpha_total=alpha_total,
        lambda_weight=lambda_weight,
        n_splits=n_splits,
        grid_size=grid_size,
        classifier=classifier,
        occ=occ,
        calibration_probability=None,
        calib_size=calib_size,
        pvalue_method=pvalue_method,
        random_state=rs,
        beta=beta,
        splitting_method='random',
        verbose=False,
    )

    # --- Deploy the realized allocation on the full reference sample
    a_unseen, a_seen, a_class, mu = compute_plugin_allocation(
        Y_ref, alpha_total, best_cap, best_alpha_seen
    )

    # --- Build the conformal Good-Turing prediction sets
    prelim = get_preliminary_sets_naive(
        X_ref, Y_ref, X_test,
        alpha_prime=a_class, black_box=classifier,
        calib_size=calib_size, random_state=rs,
    )
    final = finalize_prediction_sets(
        prelim, X_ref, Y_ref, X_test,
        pvalue_method, a_unseen, a_seen,
        occ=occ, random_state=rs, beta=beta,
    )
    res = evaluate_prediction_sets(final, Y_test, Y_ref, verbose=False)

    coverage[r] = res['Coverage (?)'].iloc[0]
    mu_hat_arr[r] = mu
    a_unseen_arr[r] = a_unseen
    a_seen_arr[r] = a_seen
    a_class_arr[r] = a_class
    cap_arr[r] = best_cap

    # Per-replication allocation sanity (the plug-in rule must hold exactly)
    assert a_unseen <= best_cap + 1e-12, f"alpha_unseen {a_unseen} exceeds cap {best_cap}"
    assert a_unseen <= mu + 1e-12, f"alpha_unseen {a_unseen} exceeds mu_hat {mu}"
    assert abs(a_seen - best_alpha_seen) < 1e-12, "alpha_seen != tuned eps_n"
    assert -1e-12 <= a_class <= 1.0 + 1e-12, f"alpha_class {a_class} out of [0,1]"


# ============================================================
# Aggregate + checks
# ============================================================
mean_cov = coverage.mean()
marg_miscov = 1.0 - mean_cov
se = np.sqrt(alpha_total * (1.0 - alpha_total) / R)   # conservative (R reps as units)
emp_se = coverage.std(ddof=1) / np.sqrt(R)
tol = 3.0 * se

print("\n=== Deployed allocation (averaged over replications) ===")
print(f"  mu_hat        = {mu_hat_arr.mean():.4f} +/- {mu_hat_arr.std():.4f}")
print(f"  alpha_unseen  = {a_unseen_arr.mean():.4f} +/- {a_unseen_arr.std():.4f}")
print(f"  alpha_seen    = {a_seen_arr.mean():.4f} +/- {a_seen_arr.std():.4f}")
print(f"  alpha_class   = {a_class_arr.mean():.4f} +/- {a_class_arr.std():.4f}")
print(f"  cap (tuned)   = {cap_arr.mean():.4f} +/- {cap_arr.std():.4f}")

print("\n=== Marginal coverage ===")
print(f"  mean coverage (?)   = {mean_cov:.4f}")
print(f"  marginal miscoverage = {marg_miscov:.4f}")
print(f"  target alpha         = {alpha_total:.4f}")
print(f"  se (binomial, R)     = {se:.4f}   empirical se = {emp_se:.4f}")
print(f"  tolerance (3*se)     = {tol:.4f}")

print(f"\n[Check] marginal miscoverage <= alpha + 3*se")
print(f"  {marg_miscov:.4f} <= {alpha_total + tol:.4f}  ->  "
      f"{'PASS' if marg_miscov <= alpha_total + tol else 'FAIL'}")

assert marg_miscov <= alpha_total + tol, (
    f"Marginal miscoverage {marg_miscov:.4f} exceeds alpha+3se "
    f"{alpha_total + tol:.4f} (alpha={alpha_total}, R={R})."
)
print("\nAll checks passed.")
