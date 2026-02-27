import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.stats.mstats import mquantiles
from arc.classification import ProbabilityAccumulator as ProbAccum

sys.path.insert(0, '../cgtc/')
from distributions_x import ShiftedNormal
from distributions_y import ZipfDist
from testing import compute_GT_pvalues_testing_new, compute_XGT_pvalues_testing_new, compute_RGT_pvalues_testing_new, compute_RGT_pvalues_testing_old
from split import SplitConformal, SplitConformalFull, SelectiveSplitConformal, BernoulliSplitConformal

sys.path.insert(0, os.path.abspath('../third_party'))
from arc import models, methods, black_boxes, others, coverage

sys.path.insert(0, os.path.abspath('../third_party/arc'))

import pdb



#Refactor the methods to save computation
def get_preliminary_sets_naive(
    X_train_calib, Y_train_calib, X_test,
    alpha_prime, black_box, calib_size,
    random_state=None
):
    """
    1) Runs standard SplitConformal with alpha=alpha_prime on the known label space.
    2) Decodes the resulting sets back to original labels.
    3) Returns a list of 'preliminary' sets (in original label space).
    """
    # if random_state is not None:
    #     np.random.seed(random_state)

    # Encode
    label_encoder = LabelEncoder()
    Y_train_calib_encoded = label_encoder.fit_transform(Y_train_calib)

    # For selective, we need an integer calibration size
    calib_num = int(calib_size * len(Y_train_calib_encoded))

    method_sc = SplitConformal(
        X_train_calib,
        Y_train_calib_encoded,
        black_box,
        alpha_prime,
        calib_num
    )
    encoded_preliminary_sets  = method_sc.predict(X_test)

    # Decode each set to the original label space
    decoded_preliminary_sets = []
    for enc_set in encoded_preliminary_sets:
        decoded_set = label_encoder.inverse_transform(enc_set)
        decoded_preliminary_sets.append(list(decoded_set))

    return decoded_preliminary_sets



def get_preliminary_sets_naive_full(
    X_train_calib, Y_train_calib, X_test,
    alpha_prime, black_box, calib_size,
    random_state=None
):
    """
    1) Runs full SplitConformalFull with alpha=alpha_prime on the known label space.
    2) Decodes the resulting sets back to original labels.
    3) Returns a list of 'preliminary' sets (in original label space).
    """
    # if random_state is not None:
    #     np.random.seed(random_state)

    # Encode
    label_encoder = LabelEncoder()
    Y_train_calib_encoded = label_encoder.fit_transform(Y_train_calib)

    # For selective, we need an integer calibration size
    calib_num = int(calib_size * len(Y_train_calib_encoded))

    method_sc = SplitConformalFull(
        X_train_calib,
        Y_train_calib_encoded,
        black_box,
        alpha_prime,
        calib_num
    )
    encoded_preliminary_sets  = method_sc.predict(X_test)

    # Decode each set to the original label space
    decoded_preliminary_sets = []
    for enc_set in encoded_preliminary_sets:
        decoded_set = label_encoder.inverse_transform(enc_set)
        decoded_preliminary_sets.append(list(decoded_set))

    return decoded_preliminary_sets


def get_preliminary_sets_Bernoulli(
    X_train_calib, Y_train_calib, X_test,
    alpha_prime, black_box, calibration_probability,
    random_state=None
):
    # Encode
    label_encoder = LabelEncoder()
    Y_train_calib_encoded = label_encoder.fit_transform(Y_train_calib)

    # Note: Unlike other methods, calibration is determined by the provided frequency function.
    method_bs = BernoulliSplitConformal(
        X_train_calib,
        Y_train_calib_encoded,
        black_box,
        alpha_prime,
        calibration_probability,
        random_state=random_state
    )
    encoded_preliminary_sets = method_bs.predict(X_test)

    # Decode each set to the original label space.
    decoded_preliminary_sets = []
    for enc_set in encoded_preliminary_sets:
        decoded_set = label_encoder.inverse_transform(enc_set)
        decoded_preliminary_sets.append(list(decoded_set))

    return decoded_preliminary_sets


#Refactor the methods to save computation
def get_preliminary_sets_benchmark(
    X_train_calib, Y_train_calib, X_test,
    alpha, black_box, calib_size,
    random_state=None
):
    """
    1) Runs standard SplitConformal with alpha=alpha_prime on the known label space.
    2) Decodes the resulting sets back to original labels.
    3) Returns a list of 'preliminary' sets (in original label space).
    """
    # if random_state is not None:
    #     np.random.seed(random_state)

    # Encode
    label_encoder = LabelEncoder()
    Y_train_calib_encoded = label_encoder.fit_transform(Y_train_calib)

    # For selective, we need an integer calibration size
    calib_num = int(calib_size * len(Y_train_calib_encoded))

    method_sc = SplitConformal(
        X_train_calib,
        Y_train_calib_encoded,
        black_box,
        alpha,
        calib_num
    )
    encoded_preliminary_sets  = method_sc.predict(X_test)

    # Decode each set to the original label space
    decoded_preliminary_sets = []
    for enc_set in encoded_preliminary_sets:
        decoded_set = label_encoder.inverse_transform(enc_set)
        decoded_preliminary_sets.append(list(decoded_set))

    return decoded_preliminary_sets



def get_prediction_sets_openmax(
    X_train_calib, Y_train_calib, X_test,
    alpha, black_box, calib_size,
    random_state=None
):
    """
    Split-conformal classification with an OpenMax-style open-set classifier.

    The classifier outputs K+1 probabilities (K seen classes + 1 unknown).
    Standard split conformal is applied over this (K+1)-class space.
    Calibration points whose true label is not in Y_train are scored
    against the "unknown" column (index K).

    Returns
    -------
    decoded_sets : list of list
        Each inner list contains original labels and/or '?' (the joker_train).
    Y_train : ndarray
        The training labels (original space) so the caller can evaluate
        coverage relative to joker_train.
    """
    # ---- 1. Split into train / calibration --------------------------------
    calib_num = int(calib_size * len(Y_train_calib))
    X_train, X_calib, Y_train, Y_calib = train_test_split(
        X_train_calib, Y_train_calib,
        test_size=calib_num, random_state=random_state
    )
    n2 = len(Y_calib)

    # ---- 2. Encode training labels to 0..K-1 ------------------------------
    label_encoder = LabelEncoder()
    label_encoder.fit(Y_train)
    Y_train_encoded = label_encoder.transform(Y_train)
    K = len(label_encoder.classes_)
    train_label_set = set(label_encoder.classes_)

    # ---- 3. Fit classifier on encoded training data -----------------------
    fitted_bb = black_box.fit(X_train, Y_train_encoded)

    # ---- 4. Map calibration labels ----------------------------------------
    #   seen  → encoded index (0..K-1)
    #   unseen → K  (the "unknown" column)
    Y_calib_mapped = np.array([
        label_encoder.transform([y])[0] if y in train_label_set else K
        for y in Y_calib
    ])

    # ---- 5. Calibrate -----------------------------------------------------
    p_hat_calib = fitted_bb.predict_proba(X_calib)  # (n_calib, K+1)

    grey_box = ProbAccum(p_hat_calib)
    rng = np.random.default_rng(random_state)
    epsilon = rng.uniform(0.0, 1.0, size=n2)
    alpha_max = grey_box.calibrate_scores(Y_calib_mapped, epsilon=epsilon)
    scores = alpha - alpha_max
    level_adjusted = (1.0 - alpha) * (1.0 + 1.0 / float(n2))
    alpha_correction = mquantiles(scores, prob=level_adjusted)
    alpha_calibrated = alpha - alpha_correction

    # ---- 6. Predict on test data ------------------------------------------
    p_hat_test = fitted_bb.predict_proba(X_test)    # (n_test, K+1)
    grey_box_test = ProbAccum(p_hat_test)
    epsilon_test = rng.uniform(0.0, 1.0, size=len(X_test))
    S_hat = grey_box_test.predict_sets(
        alpha_calibrated, epsilon=epsilon_test, allow_empty=True
    )

    # ---- 7. Decode: 0..K-1 → original labels,  K → '?' ------------------
    decoded_sets = []
    for enc_set in S_hat:
        decoded = []
        for idx in enc_set:
            if idx < K:
                decoded.append(label_encoder.inverse_transform([idx])[0])
            else:
                decoded.append("?")
        decoded_sets.append(decoded)

    return decoded_sets, Y_train


def compute_pvalues_dispatch(
    X_ref, Y_ref, X_test,
    pvalue_method,
    occ=None,
    random_state=None
):
    """Compute the appropriate p-values for the new test points."""
    if random_state is not None:
        np.random.seed(random_state)

    if pvalue_method == 'GT':
        return compute_GT_pvalues_testing_new(X_ref, Y_ref, X_test)
    elif pvalue_method == 'XGT':
        if occ is None:
            raise ValueError("OCC model is required for XGT p-values.")
        return compute_XGT_pvalues_testing_new(X_ref, Y_ref, X_test, occ)
    elif pvalue_method == 'RGT':
        return compute_RGT_pvalues_testing_new(X_ref, Y_ref, X_test)
    else:
        raise ValueError(f"Unknown pvalue_method = {pvalue_method}")

def merge_preliminary_sets_with_pvals(
    decoded_prelim_sets, pvals, alpha_double_prime
):
    """Same merging logic as before."""
    final_sets = []
    for dec_set, pval in zip(decoded_prelim_sets, pvals):
        if pval <= alpha_double_prime:
            final_sets.append(dec_set)
        else:
            final_sets.append(dec_set + ["?"])
    return final_sets






def finalize_prediction_sets(
        decoded_prelim_sets,
        X_ref, Y_ref, X_test,
        pvalue_method,
        alpha_new, alpha_old,  # Updated alpha parameters
        occ=None,
        random_state=None,
        beta = 1.6
):
    """High-level function to compute both p-values and finalize sets."""
    pvals_new, pvals_old = compute_pvalues_dispatch_new(
        X_ref, Y_ref, X_test,
        pvalue_method,
        occ=occ,
        random_state=random_state,
        beta = beta
    )
    return merge_preliminary_sets_with_pvals_new(
        decoded_prelim_sets, pvals_new, pvals_old, alpha_new, alpha_old
    )


def compute_pvalues_dispatch_new(
        X_ref, Y_ref, X_test,
        pvalue_method,
        occ=None,
        random_state=None,
        beta = 1.6
):
    """Compute both p-values (new and old) for the test points."""
    if random_state is not None:
        np.random.seed(random_state)

    # Dispatch on method
    if pvalue_method == "GT":
        p_new = compute_GT_pvalues_testing_new(X_ref, Y_ref, X_test)
        p_old = compute_RGT_pvalues_testing_old(X_ref, Y_ref, X_test, beta)

    elif pvalue_method == "XGT":
        if occ is None:
            raise ValueError("OCC model is required for XGT p-values.")
        p_new = compute_XGT_pvalues_testing_new(X_ref, Y_ref, X_test, occ)
        p_old = compute_RGT_pvalues_testing_old(X_ref, Y_ref, X_test, beta)

    elif pvalue_method == "RGT":
        p_new = compute_RGT_pvalues_testing_new(X_ref, Y_ref, X_test)
        p_old = compute_RGT_pvalues_testing_old(X_ref, Y_ref, X_test, beta)

    else:
        raise ValueError(f"Unknown pvalue_method: {pvalue_method!r}")

    return p_new, p_old


def merge_preliminary_sets_with_pvals_new(
        decoded_prelim_sets, pvals_new, pvals_old, alpha_new, alpha_old
):
    """
    Merge preliminary sets with p-values using the new decision logic.

    According to the new algorithm:
    - If psi_new ≤ alpha_new and psi_old > alpha_old: return prelim set
    - If psi_old ≤ alpha_old and psi_new > alpha_new: return {?}
    - Otherwise: return prelim set ∪ {?}
    """
    final_sets = []
    for dec_set, pval_new, pval_old in zip(decoded_prelim_sets, pvals_new, pvals_old):
        if pval_new <= alpha_new and pval_old > alpha_old:
            # Reject H0_new and fail to reject H0_old
            final_sets.append(dec_set)
        elif pval_old <= alpha_old and pval_new > alpha_new:
            # Reject H0_old and fail to reject H0_new
            final_sets.append(["?"])
        else:
            # Either reject both or fail to reject both
            final_sets.append(dec_set + ["?"])

    return final_sets

















# ---------------------------------------
# Step B: Final sets from p-values
# ---------------------------------------

# Conformal classification with unknown label space
def conformal_classification_unknown_label_space_random(X_train_calib, Y_train_calib, X_test,
                                                        alpha_prime, alpha_double_prime,
                                                        occ, black_box, calib_size,
                                                        pvalue_method='XGT', random_state=2024):
    # Set random seed
    # np.random.seed(random_state)

    prediction_sets = []

    # Map Y_train to the range 1 to C
    label_encoder = LabelEncoder()
    Y_train_calib_encoded = label_encoder.fit_transform(Y_train_calib)
    # Y_train_encoded += 1  # Ensure labels start from 1

    # Compute conformal p-values for all test points
    if pvalue_method == 'GT':
        pvals = compute_GT_pvalues_testing_new(X_train_calib, Y_train_calib, X_test)
    elif pvalue_method == 'XGT':
        pvals = compute_XGT_pvalues_testing_new(X_train_calib, Y_train_calib, X_test, occ)
    elif pvalue_method == 'RGT':
        pvals = compute_RGT_pvalues_testing_new(X_train_calib, Y_train_calib, X_test)

    method_sc = SplitConformal(X_train_calib, Y_train_calib_encoded, black_box, alpha_prime, calib_size)
    preliminary_prediction_sets = method_sc.predict(X_test)
    #     decoded_prediction_set = label_encoder.inverse_transform(preliminary_prediction_set)

    for preliminary_prediction_set, pval in zip(preliminary_prediction_sets, pvals):
        if pval <= alpha_double_prime:
            decoded_prediction_set = label_encoder.inverse_transform(preliminary_prediction_set)
            prediction_sets.append(list(decoded_prediction_set))
        else:
            decoded_prediction_set = label_encoder.inverse_transform(preliminary_prediction_set)
            prediction_sets.append(list(decoded_prediction_set) + ['?'])

    return prediction_sets

def conformal_classification_unknown_label_space_selective(X_train_calib, Y_train_calib, X_test,
                                                           alpha_prime, alpha_double_prime,
                                                           occ, black_box, calib_size,
                                                           pvalue_method='XGT', random_state=2024):

    calib_num = int(calib_size * len(Y_train_calib))

    prediction_sets = []

    # Compute conformal p-values for all test points
    if pvalue_method == 'GT':
        pvals = compute_GT_pvalues_testing_new(X_train_calib, Y_train_calib, X_test)
    elif pvalue_method == 'XGT':
        pvals = compute_XGT_pvalues_testing_new(X_train_calib, Y_train_calib, X_test, occ)
    elif pvalue_method == 'RGT':
        pvals = compute_RGT_pvalues_testing_new(X_train_calib, Y_train_calib, X_test)

    # Selective Split-Conformal Calibration
    method_sc = SelectiveSplitConformal(X_train_calib, Y_train_calib, black_box, alpha_prime, calib_num)
    preliminary_prediction_sets = method_sc.predict(X_test)

    for preliminary_prediction_set, pval in zip(preliminary_prediction_sets, pvals):
        if pval <= alpha_double_prime:
            prediction_sets.append(list(preliminary_prediction_set))
        else:
            prediction_sets.append(list(preliminary_prediction_set) + ['?'])

    return prediction_sets


#Benchmark conformal classification
def conformal_classification_benchmark(X_train_calib, Y_train_calib, X_test,
                                                        alpha_prime, alpha_double_prime,
                                                        occ, black_box, calib_size,
                                                        pvalue_method='XGT', random_state=2024):
    #Calculate alpha
    alpha = alpha_prime + alpha_double_prime

    # Set random seed
    # np.random.seed(random_state)

    # Map Y_train to the range 1 to C
    label_encoder = LabelEncoder()
    Y_train_calib_encoded = label_encoder.fit_transform(Y_train_calib)

    # Compute preliminary prediction sets for all test points
    method_sc = SplitConformal(X_train_calib, Y_train_calib_encoded, black_box, alpha,
                               calib_size, random_state=random_state)
    preliminary_prediction_sets = method_sc.predict(X_test, random_state=random_state)

    prediction_sets = []
    for preliminary_prediction_set in preliminary_prediction_sets:
        decoded_prediction_set = label_encoder.inverse_transform(preliminary_prediction_set)
        prediction_sets.append(list(decoded_prediction_set))

    return prediction_sets


# Benchmark conformal classification without conformal p-value condition
def benchmark_conformal_classification(X_train_calib, Y_train_calib, X_test,
                                       alpha, black_box,
                                       calib_size, random_state=2024):
    # Set random seed
    # np.random.seed(random_state)

    # Map Y_train to the range 1 to C
    label_encoder = LabelEncoder()
    Y_train_calib_encoded = label_encoder.fit_transform(Y_train_calib)

    # Compute preliminary prediction sets for all test points
    method_sc = SplitConformal(X_train_calib, Y_train_calib_encoded, black_box, alpha,
                               calib_size, random_state=random_state)
    preliminary_prediction_sets = method_sc.predict(X_test, random_state=random_state)

    prediction_sets = []
    for preliminary_prediction_set in preliminary_prediction_sets:
        decoded_prediction_set = label_encoder.inverse_transform(preliminary_prediction_set)
        prediction_sets.append(list(decoded_prediction_set))

    return prediction_sets

import numpy as np
import pandas as pd
from collections import Counter

def calculate_frequency_quantiles(Y_ref, method='percentile'):
    """
    Calculate quantiles (q25, q50, q75) of label frequencies using different methods.
    """
    
    # Count how many times each label occurs
    label_counts = Counter(Y_ref)
    counts = np.array(list(label_counts.values()))
    
    if method == 'percentile':
        q25, q50, q75 = np.percentile(counts, [25, 50, 75])
    elif method == 'fixed':
        q25, q50, q75 = 1, 2, 5
    else:
        raise ValueError("Method must be one of: 'percentile', 'fixed'")
    
    return q25, q50, q75

def categorize_by_frequency(Y_test, Y_ref, q1, q2, q3):
    """
    Categorize test labels based on their frequency in the reference set.
    
    Returns:
    dict: Dictionary with category names as keys and indices as values
    """
    # Count frequencies in Y_ref
    label_counts = Counter(Y_ref)
    
    categories = {
        'very_rare': [],
        'rare': [],
        'common': [],
        'very_common': []
    }
    
    for i, label in enumerate(Y_test):
        freq = label_counts.get(label, 0)  # 0 if label not in Y_ref (new label)
        
        if freq == 0 or (freq > 0 and freq <= q1):
            categories['very_rare'].append(i)
        elif freq > q1 and freq <= q2:
            categories['rare'].append(i)
        elif freq > q2 and freq <= q3:
            categories['common'].append(i)
        else:  
            categories['very_common'].append(i)
    
    return categories

def evaluate_prediction_sets(prediction_sets, Y_test, Y_ref, verbose=True):
    # Original metrics calculation
    sizes_with_question = [len(pred_set) for pred_set in prediction_sets]

    seen_labels = np.unique(Y_ref)
    
    coverage_with_question = np.mean([
        1 if (true_label in pred_set or ('?' in pred_set and true_label not in seen_labels))
        else 0
        for pred_set, true_label in zip(prediction_sets, Y_test)
    ])
    
    naive_sizes = [len([label for label in pred_set if label != '?']) for pred_set in prediction_sets]
    
    naive_coverage = np.mean([
        1 if true_label in pred_set else 0
        for pred_set, true_label in zip(prediction_sets, Y_test)
    ])
    
    prop_question = np.mean(['?' in pred_set for pred_set in prediction_sets])
    
    prop_empty = np.mean([
        1 if (len(pred_set) == 0 or (len(pred_set) == 1 and pred_set[0] == '?')) else 0
        for pred_set in prediction_sets
    ])
    
    size_std = np.std(naive_sizes)
    
    # Original seen/unseen metrics
    unseen_indices = [i for i, true_label in enumerate(Y_test) if true_label not in seen_labels]
    seen_indices = [i for i, true_label in enumerate(Y_test) if true_label in seen_labels]
    
    if unseen_indices:
        unseen_sizes = [naive_sizes[i] for i in unseen_indices]
        unseen_avg_size = np.mean(unseen_sizes)
    else:
        unseen_avg_size = np.nan
    
    if seen_indices:
        seen_avg_size = np.mean([naive_sizes[i] for i in seen_indices])
    else:
        seen_avg_size = np.nan
    
    if seen_indices:
        seen_coverage = np.mean([
            1 if Y_test[i] in prediction_sets[i] else 0
            for i in seen_indices
        ])
        seen_coverage_with_question = np.mean([
            1 if Y_test[i] in prediction_sets[i] else 0
            for i in seen_indices
        ])
    else:
        seen_coverage = np.nan
        seen_coverage_with_question = np.nan
    
    if unseen_indices:
        unseen_coverage = np.mean([
            1 if Y_test[i] in prediction_sets[i] else 0
            for i in unseen_indices
        ])
        unseen_coverage_with_question = np.mean([
            1 if (Y_test[i] in prediction_sets[i] or '?' in prediction_sets[i]) else 0
            for i in unseen_indices
        ])
    else:
        unseen_coverage = np.nan
        unseen_coverage_with_question = np.nan
    
    # === NEW: Frequency-based conditional metrics ===
    methods = ['percentile', 'fixed']
    frequency_metrics = {}
    
    for method in methods:
        # Calculate quantiles using the specified method
        q1, q2, q3 = calculate_frequency_quantiles(Y_ref, method=method)
        
        # Categorize test labels
        categories = categorize_by_frequency(Y_test, Y_ref, q1, q2, q3)
        
        # Calculate conditional coverage and size for each category
        for category_name, indices in categories.items():
            if indices:  # If there are samples in this category
                # Coverage (excluding '?')
                category_coverage = np.mean([
                    1 if Y_test[i] in prediction_sets[i] else 0
                    for i in indices
                ])
                
                # Coverage (including '?')
                category_coverage_with_question = np.mean([
                    1 if (Y_test[i] in prediction_sets[i] or 
                          ('?' in prediction_sets[i] and Y_test[i] not in seen_labels)) else 0
                    for i in indices
                ])
                
                # Average size (excluding '?')
                category_avg_size = np.mean([naive_sizes[i] for i in indices])
                
                # Store metrics
                frequency_metrics[f'Coverage ({category_name}) {method}'] = category_coverage
                frequency_metrics[f'Coverage (?) ({category_name}) {method}'] = category_coverage_with_question
                frequency_metrics[f'Size ({category_name}) {method}'] = category_avg_size
                frequency_metrics[f'Count ({category_name}) {method}'] = len(indices)
            else:
                # No samples in this category
                frequency_metrics[f'Coverage ({category_name}) {method}'] = np.nan
                frequency_metrics[f'Coverage (?) ({category_name}) {method}'] = np.nan
                frequency_metrics[f'Size ({category_name}) {method}'] = np.nan
                frequency_metrics[f'Count ({category_name}) {method}'] = 0
    
    # Verbose output
    if verbose:
        print("=== Original Metrics ===")
        print("Average set size (including '?'): {:.2f}.".format(np.mean(sizes_with_question)))
        print("Average coverage (including '?'): {:.2f}.".format(coverage_with_question))
        print("Average set size (excluding '?'): {:.2f}.".format(np.mean(naive_sizes)))
        print("Average coverage (excluding '?'): {:.2f}.".format(naive_coverage))
        print("Proportion of '?': {:.2f}.".format(prop_question))
        print("Proportion of empty sets (or only '?'): {:.2f}.".format(prop_empty))
        print("Standard deviation of set sizes (excluding '?'): {:.2f}.".format(size_std))
        
        if not np.isnan(seen_avg_size):
            print("Average set size for seen test points (excluding '?'): {:.2f}.".format(seen_avg_size))
        else:
            print("Average set size for seen test points (excluding '?'): N/A (no seen labels in test set)")
        
        if not np.isnan(unseen_avg_size):
            print("Average set size for unseen test points (excluding '?'): {:.2f}.".format(unseen_avg_size))
        else:
            print("Average set size for unseen test points (excluding '?'): N/A (no unseen labels in test set)")
        
        if not np.isnan(seen_coverage):
            print("Coverage for seen test labels (excluding '?'): {:.2f}.".format(seen_coverage))
            print("Coverage for seen test labels (including '?'): {:.2f}.".format(seen_coverage_with_question))
        else:
            print("Coverage for seen test labels: N/A (no seen labels in test set)")
        
        if not np.isnan(unseen_coverage):
            print("Coverage for unseen test labels (excluding '?'): {:.2f}.".format(unseen_coverage))
            print("Coverage for unseen test labels (including '?'): {:.2f}.".format(unseen_coverage_with_question))
        else:
            print("Coverage for unseen test labels: N/A (no unseen labels in test set)")
        
        # Print frequency-based metrics
        print("\n=== Frequency-based Conditional Metrics ===")
        for method in methods:
            q1, q2, q3 = calculate_frequency_quantiles(Y_ref, method=method)
            print(f"\nMethod: {method} (q1={q1:.2f}, q2={q2:.2f}, q3={q3:.2f})")
            categories = categorize_by_frequency(Y_test, Y_ref, q1, q2, q3)
            
            for category_name in ['very_rare', 'rare', 'common', 'very_common']:
                count = frequency_metrics[f'Count ({category_name}) {method}']
                if count > 0:
                    coverage = frequency_metrics[f'Coverage ({category_name}) {method}']
                    coverage_q = frequency_metrics[f'Coverage (?) ({category_name}) {method}']
                    size = frequency_metrics[f'Size ({category_name}) {method}']
                    print(f"  {category_name.title()}: {count} samples, Coverage: {coverage:.3f}, Coverage(?): {coverage_q:.3f}, Avg Size: {size:.2f}")
                else:
                    print(f"  {category_name.title()}: 0 samples")
    
    # Create output dataframe
    out_dict = {
        'Size (?)': [np.mean(sizes_with_question)],
        'Coverage (?)': [coverage_with_question],
        'Size': [np.mean(naive_sizes)],
        'Coverage': [naive_coverage],
        'Prop ?': [prop_question],
        'Prop empty': [prop_empty],
        'Size_std': [size_std],
        'Unseen Size': [unseen_avg_size],
        'Seen Size': [seen_avg_size],
        'Seen Coverage': [seen_coverage],
        'Seen Coverage (?)': [seen_coverage_with_question],
        'Unseen Coverage': [unseen_coverage],
        'Unseen Coverage (?)': [unseen_coverage_with_question]
    }
    
    # Add frequency-based metrics to output
    for key, value in frequency_metrics.items():
        out_dict[key] = [value]
    
    out = pd.DataFrame(out_dict)
    
    return out


# Function to evaluate prediction sets without consideration of '?'
def evaluate_prediction_sets_naive(prediction_sets, Y_test, verbose=True):
    # Calculate the size of each prediction set, excluding '?'
    naive_sizes = [len([label for label in pred_set if label != '?']) for pred_set in prediction_sets]

    # Calculate the coverage
    naive_coverage = sum([1 for pred_set, true_label in zip(prediction_sets, Y_test) if true_label in pred_set]) / len(
        Y_test)


    return naive_sizes, naive_coverage


def run_experiment(n_experiment, n_samples, n_test, num_features, sigma, a_zipf, alpha_prime, alpha_double_prime,
                   calib_size):
    sizes_with_question_all = []
    coverage_with_question_all = []
    naive_sizes_all = []
    naive_coverage_all = []

    for _ in range(n_experiment):
        # Generate labels using the Zipf distribution
        zipf_dist = ZipfDist(a_zipf)
        Y_train_calib = zipf_dist.sample(n_samples)

        # Generate features based on the labels using ShiftedNormal
        shifted_normal = ShiftedNormal(num_features, sigma)
        X_train_calib = shifted_normal.sample(Y_train_calib)

        # Generate labels using the Zipf distribution
        Y_test = zipf_dist.sample(n_test)

        # Generate features based on the labels using ShiftedNormal
        X_test = shifted_normal.sample(Y_test)

        # Define the black box classifier and OCC
        black_box = black_boxes.SVC(clip_proba_factor=1e-5)
        occ = LocalOutlierFactor(n_neighbors=1, novelty=True)

        # Conformal classification with unknown label space
        prediction_sets = conformal_classification_unknown_label_space_random(X_train_calib, Y_train_calib, X_test,
                                                                              alpha_prime, alpha_double_prime,
                                                                              occ, black_box, calib_size)

        # Evaluate the prediction sets with consideration of '?'
        seen_labels = np.unique(Y_train_calib)
        sizes_with_question, coverage_with_question = evaluate_prediction_sets(prediction_sets, Y_test, seen_labels)
        sizes_with_question_all.append(np.mean(sizes_with_question))
        coverage_with_question_all.append(coverage_with_question)

        prediction_sets_benchmark = benchmark_conformal_classification(X_train_calib, Y_train_calib, X_test,
                                                                       alpha_prime + alpha_double_prime,
                                                                       black_box, calib_size)

        # Evaluate the naive prediction sets
        naive_sizes, naive_coverage = evaluate_prediction_sets_naive(prediction_sets_benchmark, Y_test)
        naive_sizes_all.append(np.mean(naive_sizes))
        naive_coverage_all.append(naive_coverage)

    return sizes_with_question_all, coverage_with_question_all, naive_sizes_all, naive_coverage_all


def plot_results(sizes_with_question_all, coverage_with_question_all, naive_sizes_all, naive_coverage_all, nominal_coverage):
    plt.figure(figsize=(14, 6))

    # Plot distribution of average prediction set sizes
    plt.subplot(1, 2, 1)
    plt.hist(sizes_with_question_all, bins=20, alpha=0.3, label="With '?'")
    plt.hist(naive_sizes_all, bins=20, alpha=0.3, label="Benchmark")
    plt.xlabel('Average Prediction Set Size')
    plt.ylabel('Frequency')
    plt.title('Distribution of Average Prediction Set Sizes')
    plt.legend()

    # Plot boxplot of coverage
    plt.subplot(1, 2, 2)
    plt.boxplot([coverage_with_question_all, naive_coverage_all], labels=["With '?'", "Benchmark"])
    plt.axhline(y=nominal_coverage, color='r', linestyle='--', label=f'Nominal Coverage ({nominal_coverage:.2f})')
    plt.xlabel('Method')
    plt.ylabel('Coverage')
    plt.title('Comparison of Coverage')
    plt.legend()

    plt.tight_layout()
    plt.show()


def run_experiment_all_four_fixed_calibration(n_experiment, n_samples, n_test, num_features,
                                              sigma, a_zipf,
                                              alpha_prime, alpha_double_prime,
                                              calib_size):
    selective_sizes_all = []
    selective_coverage_all = []
    random_sizes_all = []
    random_coverage_all = []
    naive_sizes_all = []
    naive_coverage_all = []
    benchmark_sizes_all = []
    benchmark_coverage_all = []

    for _ in range(n_experiment):
        # Generate labels using the Zipf distribution
        zipf_dist = ZipfDist(a_zipf)
        Y_train_calib = zipf_dist.sample(n_samples)

        # Generate features based on the labels using ShiftedNormal
        shifted_normal = ShiftedNormal(num_features, sigma)
        X_train_calib = shifted_normal.sample(Y_train_calib)

        # Generate labels using the Zipf distribution
        Y_test = zipf_dist.sample(n_test)

        # Generate features based on the labels using ShiftedNormal
        X_test = shifted_normal.sample(Y_test)

        # Define the black box classifier and OCC
        black_box = black_boxes.SVC(clip_proba_factor=1e-5)
        occ = LocalOutlierFactor(n_neighbors=1, novelty=True)

        # Selective Split-Conformal calibration
        selective_prediction_sets = conformal_classification_unknown_label_space_selective(X_train_calib,
                                                                                           Y_train_calib,
                                                                                           X_test,
                                                                                           alpha_prime,
                                                                                           alpha_double_prime,
                                                                                           occ,
                                                                                           black_box,
                                                                                           int(calib_size * n_samples))
        seen_labels = np.unique(Y_train_calib)
        selective_sizes, selective_coverage = evaluate_prediction_sets(selective_prediction_sets, Y_test, seen_labels)
        selective_sizes_all.append(np.mean(selective_sizes))
        selective_coverage_all.append(selective_coverage)

        # Random Split-Conformal calibration
        random_prediction_sets = conformal_classification_unknown_label_space_random(X_train_calib, Y_train_calib,
                                                                                     X_test,
                                                                                     alpha_prime, alpha_double_prime,
                                                                                     occ, black_box, calib_size)
        seen_labels = np.unique(Y_train_calib)
        random_sizes, random_coverage = evaluate_prediction_sets(random_prediction_sets, Y_test, seen_labels)
        random_sizes_all.append(np.mean(random_sizes))
        random_coverage_all.append(random_coverage)

        # Naive method (use alpha')
        naive_sizes, naive_coverage = evaluate_prediction_sets_naive(random_prediction_sets, Y_test)
        naive_sizes_all.append(np.mean(naive_sizes))
        naive_coverage_all.append(naive_coverage)

        # get_preliminary_sets_benchmark method (use alpha'+alpha'')
        benchmark_prediction_sets = benchmark_conformal_classification(X_train_calib, Y_train_calib, X_test,
                                                                       alpha_prime + alpha_double_prime,
                                                                       black_box, calib_size)
        benchmark_sizes, benchmark_coverage = evaluate_prediction_sets(benchmark_prediction_sets, Y_test, seen_labels)
        benchmark_sizes_all.append(np.mean(benchmark_sizes))
        benchmark_coverage_all.append(benchmark_coverage)

    return (selective_sizes_all, selective_coverage_all,
            random_sizes_all, random_coverage_all,
            naive_sizes_all, naive_coverage_all,
            benchmark_sizes_all, benchmark_coverage_all)


def plot_results_four(selective_sizes_all, selective_coverage_all, random_sizes_all, random_coverage_all,
                      naive_sizes_all, naive_coverage_all, benchmark_sizes_all, benchmark_coverage_all,
                      nominal_coverage):
    plt.figure(figsize=(14, 6))

    # Plot boxplot of average prediction set sizes
    plt.subplot(1, 2, 1)
    plt.boxplot([selective_sizes_all, random_sizes_all, naive_sizes_all, benchmark_sizes_all],
                labels=["Selective Split", "Random Split", "Naive", "Benchmark"])
    plt.xlabel('Method')
    plt.ylabel('Average Prediction Set Size')
    plt.title('Comparison of Average Prediction Set Sizes')

    # Plot boxplot of coverage
    plt.subplot(1, 2, 2)
    plt.boxplot([selective_coverage_all, random_coverage_all, naive_coverage_all, benchmark_coverage_all],
                labels=["Selective", "Random", "Naive", "Benchmark"])
    plt.axhline(y=nominal_coverage, color='r', linestyle='--', label=f'Nominal Coverage ({nominal_coverage:.2f})')
    plt.xlabel('Method')
    plt.ylabel('Coverage')
    plt.title('Comparison of Coverage')
    plt.legend()

    plt.tight_layout()
    plt.show()


def evaluate_prediction_sets_question(prediction_sets, Y_test, seen_labels):
    # Calculate the size of each prediction set
    # This requires we know the seen labels
    sizes = [len(pred_set) for pred_set in prediction_sets]

    # Calculate the coverage
    coverage = sum([
        1 for pred_set, true_label in zip(prediction_sets, Y_test)
        if true_label in pred_set or ('?' in pred_set and true_label not in seen_labels)
    ]) / len(Y_test)

    # Calculate the proportion of prediction sets containing '?'
    prop_question = sum(['?' in pred_set for pred_set in prediction_sets]) / len(prediction_sets)

    return sizes, coverage, prop_question


def run_experiment_all_four_varying_calibration_question(n_experiment, n_train, n_test, num_features,
                                                         sigma, a_zipf,
                                                         alpha_prime, alpha_double_prime,
                                                         calib_sizes):
    # In this function we specify how many training points we want.
    # And we keep it fixed, then the calibration points are increasing.
    results = {
        'selective_sizes_all': [],
        'selective_coverage_all': [],
        'selective_question_all': [],
        'random_sizes_all': [],
        'random_coverage_all': [],
        'random_question_all': [],
        'naive_sizes_all': [],
        'naive_coverage_all': [],
        'benchmark_sizes_all': [],
        'benchmark_coverage_all': [],
    }

    for calib_size in calib_sizes:
        selective_sizes_all = []
        selective_coverage_all = []
        selective_question_all = []
        random_sizes_all = []
        random_coverage_all = []
        random_question_all = []
        naive_sizes_all = []
        naive_coverage_all = []
        benchmark_sizes_all = []
        benchmark_coverage_all = []

        n_samples = int(n_train * (1 + calib_size))

        for _ in range(n_experiment):
            # Generate labels using the Zipf distribution
            zipf_dist = ZipfDist(a_zipf)
            Y_train_calib = zipf_dist.sample(n_samples)

            # Generate features based on the labels using ShiftedNormal
            shifted_normal = ShiftedNormal(num_features, sigma)
            X_train_calib = shifted_normal.sample(Y_train_calib)

            # Generate labels using the Zipf distribution
            Y_test = zipf_dist.sample(n_test)

            # Generate features based on the labels using ShiftedNormal
            X_test = shifted_normal.sample(Y_test)

            # Define the black box classifier and OCC
            black_box = black_boxes.SVC(clip_proba_factor=1e-5)
            occ = LocalOutlierFactor(n_neighbors=1, novelty=True)

            # Selective Split-Conformal calibration
            selective_prediction_sets = conformal_classification_unknown_label_space_selective(X_train_calib,
                                                                                               Y_train_calib,
                                                                                               X_test,
                                                                                               alpha_prime,
                                                                                               alpha_double_prime,
                                                                                               occ,
                                                                                               black_box,
                                                                                               int(calib_size * n_train))
            seen_labels = np.unique(Y_train_calib)
            selective_sizes, selective_coverage, selective_prop_question = evaluate_prediction_sets_question(
                selective_prediction_sets, Y_test, seen_labels)
            selective_sizes_all.append(np.mean(selective_sizes))
            selective_coverage_all.append(selective_coverage)
            selective_question_all.append(selective_prop_question)

            # Random Split-Conformal calibration
            random_prediction_sets = conformal_classification_unknown_label_space_random(X_train_calib, Y_train_calib,
                                                                                         X_test,
                                                                                         alpha_prime,
                                                                                         alpha_double_prime,
                                                                                         occ, black_box,
                                                                                         calib_size / (1 + calib_size))
            seen_labels = np.unique(Y_train_calib)
            random_sizes, random_coverage, random_prop_question = evaluate_prediction_sets_question(
                random_prediction_sets, Y_test, seen_labels)
            random_sizes_all.append(np.mean(random_sizes))
            random_coverage_all.append(random_coverage)
            random_question_all.append(random_prop_question)

            # Naive method (use alpha')
            naive_sizes, naive_coverage = evaluate_prediction_sets_naive(random_prediction_sets, Y_test)
            naive_sizes_all.append(np.mean(naive_sizes))
            naive_coverage_all.append(naive_coverage)

            # Benchmark method (use alpha'+alpha'')
            benchmark_prediction_sets = benchmark_conformal_classification(X_train_calib, Y_train_calib, X_test,
                                                                           alpha_prime + alpha_double_prime,
                                                                           black_box, calib_size / (1 + calib_size))
            benchmark_sizes, benchmark_coverage = evaluate_prediction_sets(benchmark_prediction_sets, Y_test,
                                                                           seen_labels)
            benchmark_sizes_all.append(np.mean(benchmark_sizes))
            benchmark_coverage_all.append(benchmark_coverage)

        results['selective_sizes_all'].append(np.mean(selective_sizes_all))
        results['selective_coverage_all'].append(np.mean(selective_coverage_all))
        results['selective_question_all'].append(np.mean(selective_question_all))
        results['random_sizes_all'].append(np.mean(random_sizes_all))
        results['random_coverage_all'].append(np.mean(random_coverage_all))
        results['random_question_all'].append(np.mean(random_question_all))
        results['naive_sizes_all'].append(np.mean(naive_sizes_all))
        results['naive_coverage_all'].append(np.mean(naive_coverage_all))
        results['benchmark_sizes_all'].append(np.mean(benchmark_sizes_all))
        results['benchmark_coverage_all'].append(np.mean(benchmark_coverage_all))

    return results


def plot_varying_calibration_results(results, calib_sizes, nominal_coverage):
    plt.figure(figsize=(14, 6))

    # Plot average prediction set sizes
    plt.subplot(1, 2, 1)
    plt.plot(calib_sizes, results['selective_sizes_all'], marker='o', label="Selective")
    plt.plot(calib_sizes, results['random_sizes_all'], marker='o', label="Random")
    plt.plot(calib_sizes, results['naive_sizes_all'], marker='o', label="Naive")
    plt.plot(calib_sizes, results['benchmark_sizes_all'], marker='o', label="Benchmark")
    plt.xlabel('Calibration Size')
    plt.ylabel('Average Prediction Set Size')
    plt.title('Average Prediction Set Size vs. Calibration Size')
    plt.legend()

    # Plot coverage
    plt.subplot(1, 2, 2)
    plt.plot(calib_sizes, results['selective_coverage_all'], marker='o', label="Selective")
    plt.plot(calib_sizes, results['random_coverage_all'], marker='o', label="Random")
    plt.plot(calib_sizes, results['naive_coverage_all'], marker='o', label="Naive")
    plt.plot(calib_sizes, results['benchmark_coverage_all'], marker='o', label="Benchmark")
    plt.axhline(y=nominal_coverage, color='r', linestyle='--')
    plt.xlabel('Calibration Size')
    plt.ylabel('Coverage')
    plt.title('Coverage vs. Calibration Size')
    plt.legend()

    plt.tight_layout()
    plt.savefig('varying_calibration_results.pdf', format='pdf', dpi=300)
    plt.show()


def plot_question_mark_proportion(results, calib_sizes):
    plt.figure(figsize=(8, 6))

    # Plot proportion of prediction sets containing '?'
    plt.plot(calib_sizes, results['selective_question_all'], marker='o', label="Selective Split-Conformal")
    plt.plot(calib_sizes, results['random_question_all'], marker='o', label="Random Split-Conformal")
    plt.xlabel('Calibration Size')
    plt.ylabel('Proportion of Prediction Sets with "?"')
    plt.title('Proportion of Prediction Sets with "?" vs. Calibration Size')
    plt.legend()

    plt.tight_layout()
    plt.show()

def compute_proportion_question_mark(n_experiment, n_samples, n_test, num_features,
                                     sigma, a_zipf, alpha_double_prime):
    question_all = []

    for _ in range(n_experiment):
        # Generate labels using the Zipf distribution
        zipf_dist = ZipfDist(a_zipf)
        Y_train_calib = zipf_dist.sample(n_samples)

        # Generate features based on the labels using ShiftedNormal
        shifted_normal = ShiftedNormal(num_features, sigma)
        X_train_calib = shifted_normal.sample(Y_train_calib)

        # Generate labels and features for the test set
        Y_test = zipf_dist.sample(n_test)
        X_test = shifted_normal.sample(Y_test)

        # Define the OCC
        occ = LocalOutlierFactor(n_neighbors=1, novelty=True)

        # Compute conformal p-values for all test points
        pvals = compute_XGT_pvalues_testing_new(X_train_calib, Y_train_calib, X_test, occ)

        # Calculate the proportion of prediction sets containing '?'
        question_count = sum([1 for pval in pvals if pval > alpha_double_prime]) / len(pvals)

        question_all.append(question_count)

    return question_all

def plot_question_mark_proportion_x_label(results, values, x_label):
    mean_values = [np.mean(r) for r in results]
    error_bars = [1.96 * np.std(r) for r in results]  # 95% empirical error bars

    plt.figure(figsize=(7, 6))
    plt.errorbar(values, mean_values, yerr=error_bars, fmt='o', label="Question Mark Proportion", capsize=4)
    plt.plot(values, mean_values, marker='o')  # Connect the points with lines
    plt.xlabel(x_label)
    plt.ylabel('Proportion of Prediction Sets with "?"')
    plt.title(f'Proportion of Prediction Sets with "?" vs. {x_label}')
    plt.legend()
    plt.tight_layout()
    plt.show()
