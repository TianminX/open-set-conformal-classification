import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import KFold
import math

import seaborn as sns
from tqdm import tqdm

from functools import partial

from sklearn.base import clone

import sys
import os

sys.path.insert(0, os.path.abspath('../third_party'))
import arc
from arc import methods

sys.path.insert(0, '/')
from conformal_methods import get_preliminary_sets_Bernoulli, get_preliminary_sets_naive, finalize_prediction_sets, evaluate_prediction_sets
from distributions_x import ShiftedNormal
from distributions_y import DirichletProcess
from utils import calibration_probability_rate, calibration_probability_level, tune_calibration_params
from testing import select_beta_cv

from sklearn.ensemble import RandomForestClassifier

import pdb

def tune_alpha_allocation_bernoulli_coverage(
        X_ref, Y_ref,
        alpha_total=0.1,
        alpha_old_fixed=0.0,
        n_splits=5,
        alpha_class_range=None,
        classifier=None,
        occ=None,
        calibration_probability=None,
        pvalue_method='RGT',
        random_state=2024,
        beta=1.6,
        verbose=True
):
    """
    Tune alpha allocation for Bernoulli method to minimize average coverage
    while maintaining total alpha constraint.
    """

    # Define alpha_class values to try
    if alpha_class_range is None:
        # Try values from 0 to (alpha_total - alpha_old_fixed) in steps
        max_alpha_class = alpha_total - alpha_old_fixed
        alpha_class_range = np.linspace(0, max_alpha_class, 21)

    # Initialize results storage
    results_list = []

    # Create CV splitter
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    if verbose:
        tqdm.write(f"\nTuning alpha allocation with alpha_total={alpha_total}, alpha_old={alpha_old_fixed}")
        tqdm.write(f"Testing {len(alpha_class_range)} different alpha_class values")

    # Try each alpha_class value
    for alpha_class in tqdm(alpha_class_range, desc="Alpha tuning"):
        # Calculate corresponding alpha_new
        alpha_new = alpha_total - alpha_class - alpha_old_fixed

        # Skip if alpha_new is negative
        if alpha_new < 0:
            continue

        # Store coverage for each fold
        fold_coverages = []
        fold_sizes = []

        # Cross-validation
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_ref)):
            X_train, X_val = X_ref[train_idx], X_ref[val_idx]
            Y_train, Y_val = Y_ref[train_idx], Y_ref[val_idx]

            # Get unique labels from training set
            seen_labels = np.unique(Y_train)

            try:
                # Get preliminary sets using Bernoulli
                prelim_sets = get_preliminary_sets_Bernoulli(
                    X_train, Y_train, X_val,
                    alpha_prime=alpha_class,
                    black_box=classifier,
                    calibration_probability=calibration_probability,
                    random_state=random_state + fold_idx
                )

                # Finalize prediction sets
                final_sets = finalize_prediction_sets(
                    prelim_sets,
                    X_train, Y_train, X_val,
                    pvalue_method,
                    alpha_new, alpha_old_fixed,
                    occ=occ,
                    random_state=random_state + fold_idx,
                    beta=beta
                )

                # Evaluate
                eval_results = evaluate_prediction_sets(
                    final_sets, Y_val, seen_labels, verbose=False
                )

                fold_coverages.append(eval_results['Coverage (?)'].values[0])
                fold_sizes.append(eval_results['Size (?)'].values[0])

            except Exception as e:
                if verbose:
                    tqdm.write(f"Error in fold {fold_idx}: {e}")
                fold_coverages.append(np.nan)
                fold_sizes.append(np.nan)

        # Average across folds
        avg_coverage = np.nanmean(fold_coverages)
        avg_size = np.nanmean(fold_sizes)
        std_coverage = np.nanstd(fold_coverages)
        std_size = np.nanstd(fold_sizes)

        # Store results
        results_list.append({
            'alpha_class': alpha_class,
            'alpha_new': alpha_new,
            'alpha_old': alpha_old_fixed,
            'avg_coverage': avg_coverage,
            'std_coverage': std_coverage,
            'avg_size': avg_size,
            'std_size': std_size,
            'coverage_penalty': max(0, alpha_total - avg_coverage),  # Penalty for under-coverage
            'objective': avg_coverage  # Minimize coverage
        })

        # if verbose and len(results_list) % 5 == 0:
        if verbose:
            tqdm.write(f"α_class={alpha_class:.3f}, α_new={alpha_new:.3f}: "
                  f"Coverage={avg_coverage:.3f}±{std_coverage:.3f}")

    # Convert to DataFrame
    results_df = pd.DataFrame(results_list)

    # # Find best allocation (minimum coverage while meeting constraint)
    # # First filter to ensure coverage >= (1 - alpha_total)
    # target_coverage = 1 - alpha_total
    # valid_results = results_df[results_df['avg_coverage'] >= target_coverage - 0.01]  # Small tolerance
    #
    # if len(valid_results) == 0:
    #     if verbose:
    #         tqdm.write(f"Warning: No allocation achieves target coverage {target_coverage:.3f}")
    #         tqdm.write("Selecting allocation with highest coverage")
    #     best_idx = results_df['avg_coverage'].idxmax()
    # else:
    #     # Among valid results, choose the one with smallest coverage (tightest sets)
    #     best_idx = valid_results['avg_coverage'].idxmin()

    best_idx = results_df['avg_coverage'].idxmin()
    best_result = results_df.loc[best_idx]
    best_alpha_class = best_result['alpha_class']
    best_alpha_new = best_result['alpha_new']

    if verbose:
        tqdm.write("\nBest allocation found:")
        tqdm.write(f"  α_class = {best_alpha_class:.3f}")
        tqdm.write(f"  α_new = {best_alpha_new:.3f}")
        tqdm.write(f"  α_old = {alpha_old_fixed:.3f}")
        tqdm.write(f"  Average coverage = {best_result['avg_coverage']:.3f}")
        tqdm.write(f"  Average size = {best_result['avg_size']:.3f}")

    return best_alpha_class, best_alpha_new, results_df

def tune_alpha_allocation_loss(
        X_ref, Y_ref,
        alpha_total=0.1,
        alpha_old_fixed=0.0,
        lambda_weight=0.5,
        n_splits=5,
        alpha_class_range=None,
        classifier=None,
        occ=None,
        calibration_probability=None,
        calib_size=None,
        pvalue_method='RGT',
        random_state=2024,
        beta=1.6,
        splitting_method='bernoulli',
        verbose=True
):
    """
    Tune alpha allocation by minimizing a loss function that balances
    prediction set efficiency and joker waste.

    The loss function is:
    L = λ * (normalized_set_size) + (1-λ) * (joker_waste)

    where:
    - normalized_set_size = |C^0_alpha_class| / |C^0_alpha_total|
    - joker_waste = I{joker in C^1 and Y in seen_labels}

    Parameters:
    -----------
    X_ref, Y_ref : array-like
        Reference data for training and calibration
    alpha_total : float
        Total alpha budget (default: 0.1)
    alpha_old_fixed : float
        Fixed value for alpha_old (default: 0.0)
    lambda_weight : float
        Weight parameter λ ∈ [0,1] balancing size vs joker waste
    n_splits : int
        Number of CV folds
    alpha_class_range : array-like
        Range of alpha_class values to try
    classifier : object
        Classification model
    occ : object
        One-class classifier for GT p-values
    calibration_probability : function
        Function for Bernoulli sampling
    calib_size : float
        Calibration size for non-Bernoulli methods
    pvalue_method : str
        P-value method to use ('GT', 'XGT', or 'RGT')
    random_state : int
        Random seed
    beta : float
        Beta parameter for RGT p-values
    verbose : bool
        Whether to print progress

    Returns:
    --------
    best_alpha_class : float
        Optimal alpha_class value
    best_alpha_new : float
        Optimal alpha_new value
    results_df : DataFrame
        Results for all alpha allocations tested
    """

    # Define alpha_class values to try
    if alpha_class_range is None:
        max_alpha_class = alpha_total - alpha_old_fixed
        alpha_class_range = np.linspace(0, max_alpha_class, 21)

    # Initialize results storage
    results_list = []

    # Create CV splitter
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    if verbose:
        tqdm.write(f"\nTuning alpha allocation with loss function (λ={lambda_weight})")
        tqdm.write(f"Testing {len(alpha_class_range)} different alpha_class values")

    # Try each alpha_class value
    for alpha_class in tqdm(alpha_class_range, desc="Alpha tuning (loss)"):
        # Calculate corresponding alpha_new
        alpha_new = alpha_total - alpha_class - alpha_old_fixed

        # Skip if alpha_new is negative
        if alpha_new < 0:
            continue

        # Store metrics for each fold
        fold_losses = []
        fold_normalized_sizes = []
        fold_joker_wastes = []
        fold_coverages = []

        # Cross-validation
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_ref)):
            X_train, X_val = X_ref[train_idx], X_ref[val_idx]
            Y_train, Y_val = Y_ref[train_idx], Y_ref[val_idx]

            # Get unique labels from training set
            seen_labels = np.unique(Y_train)

            try:
                # Get preliminary sets using selected splitting method
                if splitting_method == 'bernoulli':
                    prelim_sets = get_preliminary_sets_Bernoulli(
                        X_train, Y_train, X_val,
                        alpha_prime=alpha_class,
                        black_box=classifier,
                        calibration_probability=calibration_probability,
                        random_state=random_state + fold_idx
                    )

                    # Also get baseline sets with all alpha allocated to classification
                    baseline_sets = get_preliminary_sets_Bernoulli(
                        X_train, Y_train, X_val,
                        alpha_prime=alpha_total,  # Use full alpha_total as baseline
                        black_box=classifier,
                        calibration_probability=calibration_probability,
                        random_state=random_state + fold_idx
                    )

                elif splitting_method == 'random':
                    prelim_sets = get_preliminary_sets_naive(
                        X_train, Y_train, X_val,
                        alpha_prime=alpha_class,
                        black_box=classifier,
                        calib_size=calib_size,
                        random_state=random_state + fold_idx
                    )

                    # Also get baseline sets with all alpha allocated to classification
                    baseline_sets = get_preliminary_sets_naive(
                        X_train, Y_train, X_val,
                        alpha_prime=alpha_total,  # Use full alpha_total as baseline
                        black_box=classifier,
                        calib_size=calib_size,
                        random_state=random_state + fold_idx
                    )

                # Finalize prediction sets
                final_sets = finalize_prediction_sets(
                    prelim_sets,
                    X_train, Y_train, X_val,
                    pvalue_method,
                    alpha_new, alpha_old_fixed,
                    occ=occ,
                    random_state=random_state + fold_idx,
                    beta=beta
                )

                # Calculate loss components
                normalized_sizes = []
                joker_wastes = []
                correct_predictions = []

                for i, (final_set, prelim_set, baseline_set, y_true) in enumerate(
                        zip(final_sets, prelim_sets, baseline_sets, Y_val)
                ):
                    # Normalized size (excluding joker) - CORRECTED
                    final_size = len([y for y in final_set if y != '?'])  # Use final_set, not prelim_set
                    baseline_size = len([y for y in baseline_set if y != '?'])

                    if baseline_size > 0:
                        normalized_size = final_size / baseline_size
                    else:
                        normalized_size = 1.0  # Handle edge case

                    normalized_sizes.append(normalized_size)

                    # Joker waste: joker included but true label is seen
                    joker_waste = 1 if ('?' in final_set and y_true in seen_labels) else 0
                    joker_wastes.append(joker_waste)

                    # Coverage
                    correct = 1 if (y_true in final_set or
                                    ('?' in final_set and y_true not in seen_labels)) else 0
                    correct_predictions.append(correct)

                # Calculate average metrics
                avg_normalized_size = np.mean(normalized_sizes)
                avg_joker_waste = np.mean(joker_wastes)
                avg_coverage = np.mean(correct_predictions)

                # Calculate loss
                loss = (lambda_weight * avg_normalized_size +
                        (1 - lambda_weight) * avg_joker_waste)

                fold_losses.append(loss)
                fold_normalized_sizes.append(avg_normalized_size)
                fold_joker_wastes.append(avg_joker_waste)
                fold_coverages.append(avg_coverage)

            except Exception as e:
                if verbose:
                    tqdm.write(f"Error in fold {fold_idx}: {e}")
                fold_losses.append(np.nan)
                fold_normalized_sizes.append(np.nan)
                fold_joker_wastes.append(np.nan)
                fold_coverages.append(np.nan)

        # Average across folds
        avg_loss = np.nanmean(fold_losses)
        avg_normalized_size = np.nanmean(fold_normalized_sizes)
        avg_joker_waste = np.nanmean(fold_joker_wastes)
        avg_coverage = np.nanmean(fold_coverages)

        std_loss = np.nanstd(fold_losses)
        std_coverage = np.nanstd(fold_coverages)

        # Store results
        results_list.append({
            'alpha_class': alpha_class,
            'alpha_new': alpha_new,
            'alpha_old': alpha_old_fixed,
            'avg_loss': avg_loss,
            'std_loss': std_loss,
            'avg_normalized_size': avg_normalized_size,
            'avg_joker_waste': avg_joker_waste,
            'avg_coverage': avg_coverage,
            'std_coverage': std_coverage,
            'lambda': lambda_weight
        })

        if verbose:
            tqdm.write(f"α_class={alpha_class:.3f}, α_new={alpha_new:.3f}: "
                       f"Loss={avg_loss:.3f}±{std_loss:.3f}, "
                       f"NormSize={avg_normalized_size:.3f}, "
                       f"JokerWaste={avg_joker_waste:.3f}")

    # Convert to DataFrame
    results_df = pd.DataFrame(results_list)

    # Find best allocation - minimum loss
    best_idx = results_df['avg_loss'].idxmin()
    best_result = results_df.loc[best_idx]
    best_alpha_class = best_result['alpha_class']
    best_alpha_new = best_result['alpha_new']

    if verbose:
        tqdm.write("\nBest allocation found (minimum loss):")
        tqdm.write(f"  α_class = {best_alpha_class:.3f}")
        tqdm.write(f"  α_new = {best_alpha_new:.3f}")
        tqdm.write(f"  α_old = {alpha_old_fixed:.3f}")
        tqdm.write(f"  Average loss = {best_result['avg_loss']:.3f}")
        tqdm.write(f"  Components: NormSize={best_result['avg_normalized_size']:.3f}, "
                   f"JokerWaste={best_result['avg_joker_waste']:.3f}")
        tqdm.write(f"  Average coverage = {best_result['avg_coverage']:.3f}")

    return best_alpha_class, best_alpha_new, results_df


def tune_alpha_allocation_loss_all(
        X_ref, Y_ref,
        alpha_total=0.1,
        lambda_weight=0.5,
        n_splits=5,
        alpha_grid_size=21,
        classifier=None,
        occ=None,
        calibration_probability=None,
        calib_size=None,
        pvalue_method='XGT',
        random_state=2024,
        beta=1.6,
        splitting_method='random',
        verbose=True
):
    """
    Tune all three alpha allocations (alpha_class, alpha_new, alpha_old) by minimizing
    a loss function that balances prediction set efficiency and joker waste.

    The loss function is:
    L = λ * (normalized_set_size) + (1-λ) * (joker_waste)

    Parameters:
    -----------
    X_ref, Y_ref : array-like
        Reference data for training and calibration
    alpha_total : float
        Total alpha budget (default: 0.1)
    lambda_weight : float
        Weight parameter λ ∈ [0,1] balancing size vs joker waste
    n_splits : int
        Number of CV folds
    alpha_grid_size : int
        Number of points for alpha_class grid (alpha_new is computed)
    classifier : object
        Classification model
    occ : object
        One-class classifier for XGT p-values
    calibration_probability : function
        Function for Bernoulli sampling
    calib_size : float
        Calibration size for non-Bernoulli methods
    pvalue_method : str
        P-value method to use ('GT', 'XGT', or 'RGT')
    random_state : int
        Random seed
    beta : float
        Beta parameter for RGT p-values
    verbose : bool
        Whether to print progress

    Returns:
    --------
    best_alpha_class : float
        Optimal alpha_class value
    best_alpha_new : float
        Optimal alpha_new value
    best_alpha_old : float
        Optimal alpha_old value
    results_df : DataFrame
        Results for all alpha allocations tested
    """

    # Validate splitting_method parameter
    if splitting_method not in ['bernoulli', 'random']:
        raise ValueError("splitting_method must be either 'bernoulli' or 'random'")

    # Define candidate values for alpha_old
    alpha_old_candidates = [0, 0.01, 0.02, 0.03, 0.04, 0.05]
    # Filter out values that exceed alpha_total
    alpha_old_candidates = [a for a in alpha_old_candidates if a < alpha_total]

    # For each alpha_old, we'll search over alpha_class values
    # alpha_new will be computed as: alpha_total - alpha_class - alpha_old

    # Initialize results storage
    results_list = []

    # Create CV splitter
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Count total combinations
    total_combinations = len(alpha_old_candidates) * alpha_grid_size

    if verbose:
        tqdm.write(f"\nTuning all alpha allocations with loss function (λ={lambda_weight})")
        tqdm.write(f"Testing {len(alpha_old_candidates)} alpha_old values: {alpha_old_candidates}")
        tqdm.write(f"For each alpha_old, testing {alpha_grid_size} alpha_class values")
        tqdm.write(f"Total combinations: {total_combinations}")

    # Try each alpha_old candidate
    for alpha_old in alpha_old_candidates:
        # Define range for alpha_class given this alpha_old
        max_alpha_class = alpha_total - alpha_old
        alpha_class_values = np.linspace(0.00, max_alpha_class, alpha_grid_size)

        if verbose:
            tqdm.write(f"\nTesting alpha_old = {alpha_old:.3f}")

        # Try each alpha_class value
        for alpha_class in tqdm(alpha_class_values, desc=f"Alpha_class (alpha_old={alpha_old:.2f})"):
            # Calculate corresponding alpha_new
            alpha_new = alpha_total - alpha_class - alpha_old

            # Skip if alpha_new is negative (shouldn't happen with our setup, but just in case)
            if alpha_new < 0:
                continue

            # Store metrics for each fold
            fold_losses = []
            fold_normalized_sizes = []
            fold_joker_wastes = []
            fold_coverages = []

            # Cross-validation
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_ref)):
                X_train, X_val = X_ref[train_idx], X_ref[val_idx]
                Y_train, Y_val = Y_ref[train_idx], Y_ref[val_idx]

                # Get unique labels from training set
                seen_labels = np.unique(Y_train)

                try:
                    # Get preliminary sets using selected splitting method
                    if splitting_method == 'bernoulli':
                        prelim_sets = get_preliminary_sets_Bernoulli(
                            X_train, Y_train, X_val,
                            alpha_prime=alpha_class,
                            black_box=classifier,
                            calibration_probability=calibration_probability,
                            random_state=random_state + fold_idx
                        )

                        # Also get baseline sets with all alpha allocated to classification
                        baseline_sets = get_preliminary_sets_Bernoulli(
                            X_train, Y_train, X_val,
                            alpha_prime=alpha_total,  # Use full alpha_total as baseline
                            black_box=classifier,
                            calibration_probability=calibration_probability,
                            random_state=random_state + fold_idx
                        )

                    elif splitting_method == 'random':
                        prelim_sets = get_preliminary_sets_naive(
                            X_train, Y_train, X_val,
                            alpha_prime=alpha_class,
                            black_box=classifier,
                            calib_size=calib_size,
                            random_state=random_state + fold_idx
                        )

                        # Also get baseline sets with all alpha allocated to classification
                        baseline_sets = get_preliminary_sets_naive(
                            X_train, Y_train, X_val,
                            alpha_prime=alpha_total,
                            black_box=classifier,
                            calib_size=calib_size,
                            random_state=random_state + fold_idx
                        )

                    # Finalize prediction sets
                    final_sets = finalize_prediction_sets(
                        prelim_sets,
                        X_train, Y_train, X_val,
                        pvalue_method,
                        alpha_new, alpha_old,
                        occ=occ,
                        random_state=random_state + fold_idx,
                        beta=beta
                    )

                    # Calculate loss components
                    normalized_sizes = []
                    joker_wastes = []
                    correct_predictions = []

                    for i, (final_set, prelim_set, baseline_set, y_true) in enumerate(
                            zip(final_sets, prelim_sets, baseline_sets, Y_val)
                    ):
                        # Normalized size (excluding joker) - CORRECTED
                        final_size = len([y for y in final_set if y != '?'])  # Use final_set, not prelim_set
                        baseline_size = len([y for y in baseline_set if y != '?'])

                        if baseline_size > 0:
                            normalized_size = final_size / baseline_size
                        else:
                            normalized_size = 1.0  # Handle edge case

                        normalized_sizes.append(normalized_size)

                        # Joker waste: joker included but true label is seen
                        joker_waste = 1 if ('?' in final_set and y_true in seen_labels) else 0
                        joker_wastes.append(joker_waste)

                        # Coverage
                        correct = 1 if (y_true in final_set or
                                        ('?' in final_set and y_true not in seen_labels)) else 0
                        correct_predictions.append(correct)

                    # Calculate average metrics
                    avg_normalized_size = np.mean(normalized_sizes)
                    avg_joker_waste = np.mean(joker_wastes)
                    avg_coverage = np.mean(correct_predictions)

                    # Calculate loss
                    loss = (lambda_weight * avg_normalized_size +
                            (1 - lambda_weight) * avg_joker_waste)

                    fold_losses.append(loss)
                    fold_normalized_sizes.append(avg_normalized_size)
                    fold_joker_wastes.append(avg_joker_waste)
                    fold_coverages.append(avg_coverage)

                except Exception as e:
                    if verbose:
                        tqdm.write(f"Error in fold {fold_idx}: {e}")
                    fold_losses.append(np.nan)
                    fold_normalized_sizes.append(np.nan)
                    fold_joker_wastes.append(np.nan)
                    fold_coverages.append(np.nan)

            # Average across folds
            avg_loss = np.nanmean(fold_losses)
            avg_normalized_size = np.nanmean(fold_normalized_sizes)
            avg_joker_waste = np.nanmean(fold_joker_wastes)
            avg_coverage = np.nanmean(fold_coverages)

            std_loss = np.nanstd(fold_losses)
            std_coverage = np.nanstd(fold_coverages)

            # Store results
            results_list.append({
                'alpha_class': alpha_class,
                'alpha_new': alpha_new,
                'alpha_old': alpha_old,
                'avg_loss': avg_loss,
                'std_loss': std_loss,
                'avg_normalized_size': avg_normalized_size,
                'avg_joker_waste': avg_joker_waste,
                'avg_coverage': avg_coverage,
                'std_coverage': std_coverage,
                'lambda': lambda_weight
            })

    # Convert to DataFrame
    results_df = pd.DataFrame(results_list)

    if len(results_df) == 0:
        raise ValueError("No valid alpha combinations found!")

    # Find best allocation - minimum loss
    best_idx = results_df['avg_loss'].idxmin()
    best_result = results_df.loc[best_idx]
    best_alpha_class = best_result['alpha_class']
    best_alpha_new = best_result['alpha_new']
    best_alpha_old = best_result['alpha_old']

    if verbose:
        tqdm.write("\nBest allocation found (minimum loss):")
        tqdm.write(f"  α_class = {best_alpha_class:.3f}")
        tqdm.write(f"  α_new = {best_alpha_new:.3f}")
        tqdm.write(f"  α_old = {best_alpha_old:.3f}")
        tqdm.write(f"  Sum = {best_alpha_class + best_alpha_new + best_alpha_old:.3f} (should equal {alpha_total:.3f})")
        tqdm.write(f"  Average loss = {best_result['avg_loss']:.3f}")
        tqdm.write(f"  Components: NormSize={best_result['avg_normalized_size']:.3f}, "
                   f"JokerWaste={best_result['avg_joker_waste']:.3f}")
        tqdm.write(f"  Average coverage = {best_result['avg_coverage']:.3f}")

    return best_alpha_class, best_alpha_new, best_alpha_old, results_df



def tune_alpha_allocation_loss_all_optimized(
        X_ref, Y_ref,
        alpha_total=0.1,
        lambda_weight=0.5,
        n_splits=10,
        alpha_step=0.005,      # NEW: fixed step size for α_class grid
        classifier=None,
        occ=None,
        calibration_probability=None,
        calib_size=None,
        pvalue_method='XGT',
        random_state=2024,
        beta=1.6,
        splitting_method='bernoulli',
        verbose=True
):
    """
    Optimized version that caches baseline sets and preliminary sets.

    Changes:
    - α_class grid now starts at 0.01 and advances with a fixed step of 0.005 (ignores alpha_grid_size).
    - Logging updated accordingly.

    Key optimizations (unchanged):
    1. Cache baseline sets (computed once per fold with alpha_total)
    2. Cache preliminary sets for each (fold, alpha_class) combination
    3. Reuse cached values when evaluating different alpha_old/alpha_new combinations
    """

    def _alpha_grid(max_alpha_class: float, start=0.01, step=None, ndigits=3):
        """Return [start, start+step, ...] ≤ max_alpha_class, rounded to ndigits."""
        if step is None:
            step = alpha_step
        if max_alpha_class < start:
            return np.array([], dtype=float)
        # Guard against floating drift in k computation
        k = int(math.floor((max_alpha_class - start) / step + 1e-12))
        grid = start + step * np.arange(k + 1, dtype=float)
        return np.round(grid, ndigits)

    # Validate splitting_method parameter
    if splitting_method not in ['bernoulli', 'random']:
        raise ValueError("splitting_method must be either 'bernoulli' or 'random'")

    # Define candidate values for alpha_old
    alpha_old_candidates = [0, 0.01, 0.02, 0.05, 0.1, 0.15]
    alpha_old_candidates = [a for a in alpha_old_candidates if a < alpha_total]

    # Initialize results storage
    results_list = []

    # Create CV splitter
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    if verbose:
        tqdm.write(f"\nOptimized tuning with loss function (λ={lambda_weight})")
        tqdm.write(f"Testing {len(alpha_old_candidates)} alpha_old values: {alpha_old_candidates}")
        tqdm.write("For each alpha_old, α_class grid = {0.01, 0.015, 0.020, ...} "
                   "with fixed step 0.005 up to α_total - α_old.")

    # Step 1: Precompute baseline sets for each fold
    baseline_cache = {}  # baseline_cache[fold_idx] = (baseline_sets, baseline_sizes, seen_labels, X_train, Y_train, X_val, Y_val)

    if verbose:
        tqdm.write("Precomputing baseline sets for each fold...")

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_ref)):
        X_train, X_val = X_ref[train_idx], X_ref[val_idx]
        Y_train, Y_val = Y_ref[train_idx], Y_ref[val_idx]

        # Get unique labels from training set
        seen_labels = np.unique(Y_train)

        # Compute baseline sets with full alpha_total
        if splitting_method == 'bernoulli':
            baseline_sets = get_preliminary_sets_Bernoulli(
                X_train, Y_train, X_val,
                alpha_prime=alpha_total,
                black_box=classifier,
                calibration_probability=calibration_probability,
                random_state=random_state + fold_idx
            )
        else:  # random
            baseline_sets = get_preliminary_sets_naive(
                X_train, Y_train, X_val,
                alpha_prime=alpha_total,
                black_box=classifier,
                calib_size=calib_size,
                random_state=random_state + fold_idx
            )

        # Calculate baseline sizes
        baseline_sizes = [len([y for y in s if y != '?']) for s in baseline_sets]

        # Store in cache
        baseline_cache[fold_idx] = {
            'baseline_sets': baseline_sets,
            'baseline_sizes': baseline_sizes,
            'seen_labels': seen_labels,
            'X_train': X_train,
            'Y_train': Y_train,
            'X_val': X_val,
            'Y_val': Y_val
        }

    # Step 2: Collect all unique alpha_class values we'll test (fixed grid: start=0.01, step=0.005)
    all_alpha_class_values = set()
    per_old_counts = {}
    for alpha_old in alpha_old_candidates:
        max_alpha_class = alpha_total - alpha_old
        alpha_class_values = _alpha_grid(max_alpha_class)
        per_old_counts[alpha_old] = len(alpha_class_values)
        all_alpha_class_values.update(alpha_class_values)

    all_alpha_class_values = sorted(list(all_alpha_class_values))

    # Step 3: Precompute preliminary sets for each (fold, alpha_class) combination
    prelim_cache = {}  # prelim_cache[alpha_class][fold_idx] = preliminary_sets

    if verbose:
        total_unique = len(all_alpha_class_values)
        tqdm.write(f"Precomputing preliminary sets for {total_unique} unique α_class values...")

    for alpha_class in tqdm(all_alpha_class_values, desc="Caching preliminary sets", disable=not verbose):
        prelim_cache[alpha_class] = {}

        for fold_idx in range(n_splits):
            fold_data = baseline_cache[fold_idx]

            # Compute preliminary sets for this alpha_class
            if splitting_method == 'bernoulli':
                prelim_sets = get_preliminary_sets_Bernoulli(
                    fold_data['X_train'], fold_data['Y_train'], fold_data['X_val'],
                    alpha_prime=float(alpha_class),
                    black_box=classifier,
                    calibration_probability=calibration_probability,
                    random_state=random_state + fold_idx
                )
            else:  # random
                prelim_sets = get_preliminary_sets_naive(
                    fold_data['X_train'], fold_data['Y_train'], fold_data['X_val'],
                    alpha_prime=float(alpha_class),
                    black_box=classifier,
                    calib_size=calib_size,
                    random_state=random_state + fold_idx
                )

            prelim_cache[alpha_class][fold_idx] = prelim_sets

    # Step 4: Evaluate all alpha combinations using cached data
    total_combinations = sum(per_old_counts.values())

    if verbose:
        tqdm.write(f"\nEvaluating {total_combinations} alpha combinations using cached data...")

    for alpha_old in alpha_old_candidates:
        max_alpha_class = alpha_total - alpha_old
        alpha_class_values = _alpha_grid(max_alpha_class)

        # If no feasible α_class (e.g., max_alpha_class < 0.01), skip this alpha_old
        if alpha_class_values.size == 0:
            if verbose:
                tqdm.write(f"Skipping alpha_old={alpha_old:.3f}: max α_class {max_alpha_class:.3f} < 0.01.")
            continue

        for alpha_class in tqdm(alpha_class_values, desc=f"α_class (α_old={alpha_old:.2f})", disable=not verbose):
            # Calculate corresponding alpha_new
            alpha_class = float(alpha_class)  # ensure plain float key
            alpha_new = alpha_total - alpha_class - alpha_old

            if alpha_new < 0:
                continue

            # Store metrics for each fold
            fold_losses = []
            fold_normalized_sizes = []
            fold_joker_wastes = []
            fold_coverages = []

            # Evaluate on each fold using cached data
            for fold_idx in range(n_splits):
                fold_data = baseline_cache[fold_idx]

                try:
                    # Retrieve cached preliminary sets
                    prelim_sets = prelim_cache[alpha_class][fold_idx]

                    # Finalize prediction sets (depends on alpha_new/alpha_old)
                    final_sets = finalize_prediction_sets(
                        prelim_sets,
                        fold_data['X_train'], fold_data['Y_train'], fold_data['X_val'],
                        pvalue_method,
                        alpha_new, alpha_old,
                        occ=occ,
                        random_state=random_state + fold_idx,
                        beta=beta
                    )

                    # Calculate loss components
                    normalized_sizes = []
                    joker_wastes = []
                    correct_predictions = []

                    for i, (final_set, y_true) in enumerate(zip(final_sets, fold_data['Y_val'])):
                        # Normalized size (excluding joker)
                        final_size = len([y for y in final_set if y != '?'])
                        baseline_size = fold_data['baseline_sizes'][i]

                        if baseline_size > 0:
                            normalized_size = final_size / baseline_size
                        else:
                            normalized_size = 1.0

                        normalized_sizes.append(normalized_size)

                        # Joker waste: joker included but true label is seen
                        joker_waste = 1 if ('?' in final_set and y_true in fold_data['seen_labels']) else 0
                        joker_wastes.append(joker_waste)

                        # Coverage
                        correct = 1 if (y_true in final_set or
                                        ('?' in final_set and y_true not in fold_data['seen_labels'])) else 0
                        correct_predictions.append(correct)

                    # Average metrics
                    avg_normalized_size = np.mean(normalized_sizes)
                    avg_joker_waste = np.mean(joker_wastes)
                    avg_coverage = np.mean(correct_predictions)

                    # Loss
                    loss = (lambda_weight * avg_normalized_size +
                            (1 - lambda_weight) * avg_joker_waste)

                    fold_losses.append(loss)
                    fold_normalized_sizes.append(avg_normalized_size)
                    fold_joker_wastes.append(avg_joker_waste)
                    fold_coverages.append(avg_coverage)

                except Exception as e:
                    if verbose:
                        tqdm.write(f"Error in fold {fold_idx}: {e}")
                    fold_losses.append(np.nan)
                    fold_normalized_sizes.append(np.nan)
                    fold_joker_wastes.append(np.nan)
                    fold_coverages.append(np.nan)

            # Average across folds
            avg_loss = np.nanmean(fold_losses)
            avg_normalized_size = np.nanmean(fold_normalized_sizes)
            avg_joker_waste = np.nanmean(fold_joker_wastes)
            avg_coverage = np.nanmean(fold_coverages)

            std_loss = np.nanstd(fold_losses)
            std_coverage = np.nanstd(fold_coverages)

            # Store results
            results_list.append({
                'alpha_class': alpha_class,
                'alpha_new': alpha_new,
                'alpha_old': alpha_old,
                'avg_loss': avg_loss,
                'std_loss': std_loss,
                'avg_normalized_size': avg_normalized_size,
                'avg_joker_waste': avg_joker_waste,
                'avg_coverage': avg_coverage,
                'std_coverage': std_coverage,
                'lambda': lambda_weight
            })

    # Convert to DataFrame
    results_df = pd.DataFrame(results_list)

    if len(results_df) == 0:
        raise ValueError("No valid alpha combinations found!")

    # Find best allocation
    best_idx = results_df['avg_loss'].idxmin()
    best_result = results_df.loc[best_idx]
    best_alpha_class = best_result['alpha_class']
    best_alpha_new = best_result['alpha_new']
    best_alpha_old = best_result['alpha_old']

    if verbose:
        tqdm.write("\nBest allocation found (minimum loss):")
        tqdm.write(f"  α_class = {best_alpha_class:.3f}")
        tqdm.write(f"  α_new = {best_alpha_new:.3f}")
        tqdm.write(f"  α_old = {best_alpha_old:.3f}")
        tqdm.write(f"  Sum = {best_alpha_class + best_alpha_new + best_alpha_old:.3f} (should equal {alpha_total:.3f})")
        tqdm.write(f"  Average loss = {best_result['avg_loss']:.3f}")
        tqdm.write(f"  Components: NormSize={best_result['avg_normalized_size']:.3f}, "
                   f"JokerWaste={best_result['avg_joker_waste']:.3f}")
        tqdm.write(f"  Average coverage = {best_result['avg_coverage']:.3f}")

    return best_alpha_class, best_alpha_new, best_alpha_old, results_df



def tune_alpha_allocation_loss_all_fast(
        X_ref, Y_ref,
        alpha_total=0.1,
        lambda_weight=0.5,
        n_splits=5,
        alpha_grid_size=21,
        classifier=None,
        occ=None,
        calibration_probability=None,
        calib_size=None,
        pvalue_method='XGT',
        random_state=2024,
        beta=1.6,
        splitting_method='bernoulli',
        verbose=True,
        n_jobs_folds=1  # set >1 to parallelize across folds for the finalize step
):
    """
    Same search as tune_alpha_allocation_loss_all but much faster by caching:
      - CV splits (once)
      - baseline sets per fold (once)
      - preliminary sets per (fold, alpha_class) (once)
    Everything else (finalization, metrics, loss) is unchanged.
    """

    if splitting_method not in ['bernoulli', 'random']:
        raise ValueError("splitting_method must be either 'bernoulli' or 'random'")

    # ----- grid -----
    alpha_old_candidates = [a for a in [0, 0.01, 0.02, 0.03, 0.04, 0.05] if a < alpha_total]

    # We'll generate alpha_class grids separately for each alpha_old below,
    # but note: preliminary sets depend only on alpha_class, not on alpha_old,
    # so we will cache by alpha_class value and reuse across alpha_old values.
    # We'll normalize the float as a rounded key to avoid fp drift.
    def key_ac(a): return float(np.round(a, 10))

    # ----- CV splits reused -----
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(kf.split(X_ref))
    if verbose:
        tqdm.write(f"\nTuning all alpha allocations with loss (λ={lambda_weight}). "
                   f"Splitting method: {splitting_method}. "
                   f"Total alpha: {alpha_total}")

    # ----- fold-level caches -----
    # Each fold has fixed train/val; baseline depends only on (fold, alpha_total)
    baseline_cache = {}                 # fold_idx -> list of baseline sets (len = |val|)
    baseline_sizes_cache = {}           # fold_idx -> np.array of baseline set sizes (excluding '?')
    seen_labels_cache = {}              # fold_idx -> np.array of seen labels (set-like use)
    fold_xy_cache = {}                  # fold_idx -> (X_train, Y_train, X_val, Y_val)

    # Preliminary sets depend on (fold, alpha_class)
    prelim_cache = defaultdict(dict)    # prelim_cache[fold_idx][alpha_class_key] -> list of prelim sets

    # Prepare per-fold data and baseline once
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_train, X_val = X_ref[train_idx], X_ref[val_idx]
        Y_train, Y_val = Y_ref[train_idx], Y_ref[val_idx]
        fold_xy_cache[fold_idx] = (X_train, Y_train, X_val, Y_val)

        seen_labels_cache[fold_idx] = np.unique(Y_train)

        # Baseline sets: allocate all alpha to classification (no joker), once per fold
        if splitting_method == 'bernoulli':
            baseline_sets = get_preliminary_sets_Bernoulli(
                X_train, Y_train, X_val,
                alpha_prime=alpha_total,
                black_box=classifier,
                calibration_probability=calibration_probability,
                random_state=random_state + fold_idx
            )
        else:
            baseline_sets = get_preliminary_sets_naive(
                X_train, Y_train, X_val,
                alpha_prime=alpha_total,
                black_box=classifier,
                calib_size=calib_size,
                random_state=random_state + fold_idx
            )

        baseline_cache[fold_idx] = baseline_sets

        # Precompute baseline sizes (exclude joker)
        b_sizes = np.fromiter((len([y for y in s if y != '?']) for s in baseline_sets), dtype=float)
        # Handle empty-baseline edge case by treating size as 1.0 (same as your original)
        b_sizes[b_sizes == 0] = 1.0
        baseline_sizes_cache[fold_idx] = b_sizes

    results = []

    # We flip the loop order:
    # 1) Loop over alpha_class values once (and cache preliminary sets for all folds)
    # 2) For each alpha_old, only finalize/score (cheap compared to re-training)
    #
    # To keep the tested grid identical to your original, we must generate,
    # for each alpha_old, the *same* linspace it would have produced. We do that,
    # but we only compute the union of all alpha_class values once.
    all_alpha_class_values = set()
    per_old_alpha_class = {}  # alpha_old -> np.array of alpha_class values for that alpha_old

    for alpha_old in alpha_old_candidates:
        max_ac = alpha_total - alpha_old
        ac_values = np.linspace(0.0, max_ac, alpha_grid_size)
        per_old_alpha_class[alpha_old] = ac_values
        for ac in ac_values:
            all_alpha_class_values.add(key_ac(ac))

    all_alpha_class_values = sorted(all_alpha_class_values)

    # Cache preliminary sets for every alpha_class in the union (once)
    if verbose:
        tqdm.write(f"\nCaching preliminary sets for {len(all_alpha_class_values)} alpha_class values (union over all alpha_old).")

    for ac_key in tqdm(all_alpha_class_values, desc="Cache prelim"):
        ac = float(ac_key)
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            # Skip if present
            if ac_key in prelim_cache[fold_idx]:
                continue
            X_train, Y_train, X_val, _ = fold_xy_cache[fold_idx]
            if splitting_method == 'bernoulli':
                prelim = get_preliminary_sets_Bernoulli(
                    X_train, Y_train, X_val,
                    alpha_prime=ac,
                    black_box=classifier,
                    calibration_probability=calibration_probability,
                    random_state=random_state + fold_idx
                )
            else:
                prelim = get_preliminary_sets_naive(
                    X_train, Y_train, X_val,
                    alpha_prime=ac,
                    black_box=classifier,
                    calib_size=calib_size,
                    random_state=random_state + fold_idx
                )
            prelim_cache[fold_idx][ac_key] = prelim

    # Helper: evaluate one (alpha_old, alpha_class)
    def eval_combo(alpha_old, alpha_class):
        alpha_new = alpha_total - alpha_class - alpha_old
        if alpha_new < 0:
            return None

        fold_losses = []
        fold_norm_sizes = []
        fold_jw = []
        fold_cov = []

        # Optionally parallelize across folds (safe: seeds are per-fold)
        fold_iter = enumerate(splits)

        # Simple threaded parallel without external deps
        if n_jobs_folds == 1:
            fold_metrics = []
            for fold_idx, _ in fold_iter:
                fold_metrics.append(_eval_fold(fold_idx, alpha_new, alpha_old, alpha_class))
        else:
            import concurrent.futures as cf
            with cf.ThreadPoolExecutor(max_workers=n_jobs_folds) as ex:
                futures = [ex.submit(_eval_fold, fold_idx, alpha_new, alpha_old, alpha_class)
                           for fold_idx, _ in enumerate(splits)]
                fold_metrics = [f.result() for f in futures]

        for loss, ns, jw, cov in fold_metrics:
            fold_losses.append(loss)
            fold_norm_sizes.append(ns)
            fold_jw.append(jw)
            fold_cov.append(cov)

        # Aggregate (use nanmean/nanstd as in your code)
        avg_loss = np.nanmean(fold_losses)
        avg_ns = np.nanmean(fold_norm_sizes)
        avg_jw = np.nanmean(fold_jw)
        avg_cov = np.nanmean(fold_cov)
        std_loss = np.nanstd(fold_losses)
        std_cov = np.nanstd(fold_cov)

        return {
            'alpha_class': alpha_class,
            'alpha_new': alpha_new,
            'alpha_old': alpha_old,
            'avg_loss': avg_loss,
            'std_loss': std_loss,
            'avg_normalized_size': avg_ns,
            'avg_joker_waste': avg_jw,
            'avg_coverage': avg_cov,
            'std_coverage': std_cov,
            'lambda': lambda_weight
        }

    # Per-fold worker (uses only caches + finalize)
    def _eval_fold(fold_idx, alpha_new, alpha_old, alpha_class):
        X_train, Y_train, X_val, Y_val = fold_xy_cache[fold_idx]
        prelim_sets = prelim_cache[fold_idx][key_ac(alpha_class)]
        baseline_sets = baseline_cache[fold_idx]
        b_sizes = baseline_sizes_cache[fold_idx]
        seen_labels = set(seen_labels_cache[fold_idx])

        # Finalize depends on alpha_new/alpha_old
        final_sets = finalize_prediction_sets(
            prelim_sets,
            X_train, Y_train, X_val,
            pvalue_method,
            alpha_new, alpha_old,
            occ=occ,
            random_state=random_state + fold_idx,
            beta=beta
        )

        # Vectorized metrics
        # sizes (exclude '?')
        f_sizes = np.fromiter((len([y for y in s if y != '?']) for s in final_sets), dtype=float)

        # Normalized size
        ns = np.mean(f_sizes / b_sizes)

        # Joker waste: '?' in final_set and true label is seen
        has_joker = np.fromiter(('?' in s for s in final_sets), dtype=int)
        y_seen = np.fromiter((y in seen_labels for y in Y_val), dtype=int)
        jw = np.mean(has_joker * y_seen)

        # Coverage: y_true in final_set OR ('?' in final_set AND y_true unseen)
        contains_true = np.fromiter(((y in s) for y, s in zip(Y_val, final_sets)), dtype=int)
        y_unseen = 1 - y_seen
        cov = np.mean(np.maximum(contains_true, has_joker * y_unseen))

        # Loss
        loss = lambda_weight * ns + (1 - lambda_weight) * jw
        return loss, ns, jw, cov

    # Now score all combos; we reuse cached prelim/baseline
    if verbose:
        tqdm.write("\nEvaluating grid with cached prelim/baseline...")

    for alpha_old in alpha_old_candidates:
        for alpha_class in per_old_alpha_class[alpha_old]:
            out = eval_combo(alpha_old, alpha_class)
            if out is not None:
                results.append(out)

    results_df = pd.DataFrame(results)
    if results_df.empty:
        raise ValueError("No valid alpha combinations found!")

    # Pick best (same criterion as original)
    best_idx = results_df['avg_loss'].idxmin()
    best = results_df.loc[best_idx]
    best_alpha_class = float(best['alpha_class'])
    best_alpha_new = float(best['alpha_new'])
    best_alpha_old = float(best['alpha_old'])

    if verbose:
        tqdm.write("\nBest allocation (minimum loss):")
        tqdm.write(f"  α_class = {best_alpha_class:.3f}")
        tqdm.write(f"  α_new   = {best_alpha_new:.3f}")
        tqdm.write(f"  α_old   = {best_alpha_old:.3f}")
        tqdm.write(f"  Sum     = {best_alpha_class + best_alpha_new + best_alpha_old:.3f} "
                   f"(target {alpha_total:.3f})")
        tqdm.write(f"  Avg loss = {best['avg_loss']:.3f}")
        tqdm.write(f"    components: NormSize={best['avg_normalized_size']:.3f}, "
                   f"JokerWaste={best['avg_joker_waste']:.3f}")
        tqdm.write(f"  Avg coverage = {best['avg_coverage']:.3f}")

    return best_alpha_class, best_alpha_new, best_alpha_old, results_df



