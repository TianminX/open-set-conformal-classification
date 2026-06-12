"""
Plug-in missing-mass budget allocation (paper revision).

This module implements the updated, CV-tuned allocation method:
"Data-Driven Allocation with Missing-Mass Adjustments".

The two pieces are:
  - compute_plugin_allocation: given (alpha, cap, alpha_seen) and a reference
    label sample, derive (alpha_unseen, alpha_seen, alpha_class) from the
    plug-in missing-mass estimate mu_hat = psi0^GT = (M1 + 1)/(n + 1).
  - tune_plugin_allocation_cv: cross-validate the rule hyperparameters
    (alpha_unseen cap, alpha_seen) by minimizing the same loss used by the
    original tuner, deriving the realized allocation per fold from the fold's
    own mu_hat.

It is intentionally kept separate from alpha_tune_function.py (the original
tuner, left untouched as the baseline). The Good-Turing scalar psi0^GT is
reused from testing.py rather than recomputed.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm

from conformal_methods import (
    get_preliminary_sets_Bernoulli,
    get_preliminary_sets_naive,
    finalize_prediction_sets,
)
from testing import psi0_gt


def _derive_allocation(mu_hat, alpha_total, alpha_unseen_cap, alpha_seen):
    """Derive (alpha_unseen, alpha_seen, alpha_class) from a missing-mass
    estimate mu_hat, per the plug-in rule:

        alpha_remaining = alpha_total - alpha_seen
        alpha_unseen    = min(alpha_unseen_cap, mu_hat)
        alpha_class     = min( (alpha_remaining - alpha_unseen) / (1 - mu_hat), 1 )

    The mu_hat -> 1 limit is guarded (alpha_class -> 1, no division by zero).
    """
    alpha_unseen = min(alpha_unseen_cap, mu_hat)
    alpha_remaining = alpha_total - alpha_seen
    denom = 1.0 - mu_hat
    if denom <= 0.0:                       # mu_hat -> 1 degenerate guard
        alpha_class = 1.0
    else:
        alpha_class = min((alpha_remaining - alpha_unseen) / denom, 1.0)
    alpha_class = max(0.0, alpha_class)    # defensive; non-negative budget
    return alpha_unseen, alpha_seen, alpha_class


def compute_plugin_allocation(Y_ref, alpha_total, alpha_unseen_cap, alpha_seen):
    """Plug-in missing-mass allocation deployed on a reference label sample.

    mu_hat = psi0^GT = (1 + M1)/(1 + n) is the plug-in missing-mass estimate
    (M1 = number of singleton labels in Y_ref, n = len(Y_ref)), reused from
    testing.psi0_gt.

    Returns (alpha_unseen, alpha_seen, alpha_class, mu_hat).
    """
    mu_hat = psi0_gt(Y_ref)
    alpha_unseen, alpha_seen, alpha_class = _derive_allocation(
        mu_hat, alpha_total, alpha_unseen_cap, alpha_seen
    )
    return alpha_unseen, alpha_seen, alpha_class, mu_hat


def tune_plugin_allocation_cv(
        X_ref, Y_ref,
        alpha_total=0.1,
        lambda_weight=0.5,
        n_splits=10,
        grid_size=20,
        classifier=None,
        occ=None,
        calibration_probability=None,
        calib_size=None,
        pvalue_method='XGT',
        random_state=2024,
        beta=1.6,
        splitting_method='bernoulli',
        verbose=True,
):
    """Cross-validate the plug-in allocation hyperparameters (alpha_unseen cap,
    alpha_seen), implementing Algorithm "Data-Driven Allocation with
    Missing-Mass Adjustments".

    Search:
        alpha_seen in {0, 0.01, 0.02, 0.05, 0.1} intersect [0, alpha_total)
        for each alpha_seen:
            alpha_remaining = alpha_total - alpha_seen
            cap in {0, alpha_remaining/G, ..., alpha_remaining}   (G = grid_size)
    For each (cap, alpha_seen) and each fold k, the realized allocation is
    derived from the fold's plug-in mu_hat_k:
        alpha_unseen_hat = min(cap, mu_hat_k)
        alpha_class_hat  = min((alpha_remaining - alpha_unseen_hat)/(1 - mu_hat_k), 1)
    The loss is the same as the original tuner:
        L = lambda * normalized_size + (1 - lambda) * joker_waste.

    Returns (best_cap, best_alpha_seen, results_df). The realized allocation is
    NOT returned; deploy it on the full reference sample via
    compute_plugin_allocation(Y_ref, alpha_total, best_cap, best_alpha_seen).
    """
    if splitting_method not in ['bernoulli', 'random']:
        raise ValueError("splitting_method must be either 'bernoulli' or 'random'")

    # alpha_seen candidate set (Algorithm): {0, .01, .02, .05, .1} cap [0, alpha)
    alpha_seen_candidates = [a for a in (0.0, 0.01, 0.02, 0.05, 0.1) if a < alpha_total]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    if verbose:
        tqdm.write(f"\nPlug-in allocation CV (lambda={lambda_weight}, "
                   f"K={n_splits}, G={grid_size})")
        tqdm.write(f"alpha_seen candidates: {alpha_seen_candidates}")

    # ----- Precompute, per fold: the train/val split, seen labels, the fold's
    # plug-in mu_hat, and baseline sets at alpha_total (normalized-size denom).
    fold_cache = {}
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_ref)):
        X_train, X_val = X_ref[train_idx], X_ref[val_idx]
        Y_train, Y_val = Y_ref[train_idx], Y_ref[val_idx]
        seen_labels = np.unique(Y_train)
        mu_hat_fold = psi0_gt(Y_train)

        if splitting_method == 'bernoulli':
            baseline_sets = get_preliminary_sets_Bernoulli(
                X_train, Y_train, X_val,
                alpha_prime=alpha_total,
                black_box=classifier,
                calibration_probability=calibration_probability,
                random_state=random_state + fold_idx,
            )
        else:  # random
            baseline_sets = get_preliminary_sets_naive(
                X_train, Y_train, X_val,
                alpha_prime=alpha_total,
                black_box=classifier,
                calib_size=calib_size,
                random_state=random_state + fold_idx,
            )
        baseline_sizes = [len([y for y in s if y != '?']) for s in baseline_sets]

        fold_cache[fold_idx] = {
            'X_train': X_train, 'Y_train': Y_train,
            'X_val': X_val, 'Y_val': Y_val,
            'seen_labels': seen_labels,
            'mu_hat': mu_hat_fold,
            'baseline_sizes': baseline_sizes,
        }

    if verbose:
        mus = [fold_cache[fi]['mu_hat'] for fi in range(n_splits)]
        tqdm.write(f"Per-fold plug-in mu_hat in [{min(mus):.4f}, {max(mus):.4f}]")

    # ----- Cache preliminary sets by (fold, rounded alpha_class). Since mu_hat
    # is fixed per fold, the derived alpha_class is what determines the set;
    # different (cap, alpha_seen) sharing an alpha_class reuse the same sets.
    prelim_cache = {}

    def _get_prelim_sets(fold_idx, alpha_class):
        key = (fold_idx, round(float(alpha_class), 6))
        if key in prelim_cache:
            return prelim_cache[key]
        fd = fold_cache[fold_idx]
        if splitting_method == 'bernoulli':
            prelim = get_preliminary_sets_Bernoulli(
                fd['X_train'], fd['Y_train'], fd['X_val'],
                alpha_prime=alpha_class,
                black_box=classifier,
                calibration_probability=calibration_probability,
                random_state=random_state + fold_idx,
            )
        else:  # random
            prelim = get_preliminary_sets_naive(
                fd['X_train'], fd['Y_train'], fd['X_val'],
                alpha_prime=alpha_class,
                black_box=classifier,
                calib_size=calib_size,
                random_state=random_state + fold_idx,
            )
        prelim_cache[key] = prelim
        return prelim

    # ----- Search over (alpha_seen, cap).
    results_list = []
    for alpha_seen in alpha_seen_candidates:
        alpha_remaining = alpha_total - alpha_seen
        cap_grid = [j * alpha_remaining / grid_size for j in range(grid_size + 1)]

        for cap in tqdm(cap_grid, desc=f"cap (alpha_seen={alpha_seen:.2f})",
                        disable=not verbose):
            fold_losses, fold_norm_sizes, fold_jokers, fold_covs = [], [], [], []

            for fold_idx in range(n_splits):
                fd = fold_cache[fold_idx]
                mu_k = fd['mu_hat']
                alpha_unseen, _, alpha_class = _derive_allocation(
                    mu_k, alpha_total, cap, alpha_seen
                )

                try:
                    prelim_sets = _get_prelim_sets(fold_idx, alpha_class)
                    final_sets = finalize_prediction_sets(
                        prelim_sets,
                        fd['X_train'], fd['Y_train'], fd['X_val'],
                        pvalue_method,
                        alpha_unseen, alpha_seen,
                        occ=occ,
                        random_state=random_state + fold_idx,
                        beta=beta,
                    )

                    normalized_sizes, joker_wastes, correct = [], [], []
                    for i, (fset, y_true) in enumerate(zip(final_sets, fd['Y_val'])):
                        final_size = len([y for y in fset if y != '?'])
                        baseline_size = fd['baseline_sizes'][i]
                        normalized_sizes.append(
                            final_size / baseline_size if baseline_size > 0 else 1.0
                        )
                        joker_wastes.append(
                            1 if ('?' in fset and y_true in fd['seen_labels']) else 0
                        )
                        correct.append(
                            1 if (y_true in fset or
                                  ('?' in fset and y_true not in fd['seen_labels'])) else 0
                        )

                    avg_norm = np.mean(normalized_sizes)
                    avg_joker = np.mean(joker_wastes)
                    loss = lambda_weight * avg_norm + (1 - lambda_weight) * avg_joker

                    fold_losses.append(loss)
                    fold_norm_sizes.append(avg_norm)
                    fold_jokers.append(avg_joker)
                    fold_covs.append(np.mean(correct))

                except Exception as e:
                    if verbose:
                        tqdm.write(f"Error in fold {fold_idx} "
                                   f"(cap={cap:.3f}, alpha_seen={alpha_seen:.3f}): {e}")
                    fold_losses.append(np.nan)
                    fold_norm_sizes.append(np.nan)
                    fold_jokers.append(np.nan)
                    fold_covs.append(np.nan)

            results_list.append({
                'alpha_unseen_cap': cap,
                'alpha_seen': alpha_seen,
                'avg_loss': np.nanmean(fold_losses),
                'std_loss': np.nanstd(fold_losses),
                'avg_normalized_size': np.nanmean(fold_norm_sizes),
                'avg_joker_waste': np.nanmean(fold_jokers),
                'avg_coverage': np.nanmean(fold_covs),
                'std_coverage': np.nanstd(fold_covs),
                'lambda': lambda_weight,
            })

    results_df = pd.DataFrame(results_list)
    if len(results_df) == 0:
        raise ValueError("No valid (cap, alpha_seen) combinations found!")

    best_idx = results_df['avg_loss'].idxmin()
    best = results_df.loc[best_idx]
    best_cap = float(best['alpha_unseen_cap'])
    best_alpha_seen = float(best['alpha_seen'])

    if verbose:
        tqdm.write("\nBest plug-in hyperparameters (minimum loss):")
        tqdm.write(f"  alpha_unseen_cap = {best_cap:.4f}")
        tqdm.write(f"  alpha_seen       = {best_alpha_seen:.4f}")
        tqdm.write(f"  Average loss     = {best['avg_loss']:.4f}")
        tqdm.write(f"  Components: NormSize={best['avg_normalized_size']:.4f}, "
                   f"JokerWaste={best['avg_joker_waste']:.4f}")
        tqdm.write(f"  Average coverage = {best['avg_coverage']:.4f}")

    return best_cap, best_alpha_seen, results_df
