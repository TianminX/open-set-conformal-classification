import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from functools import partial
import sys
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "20"

sys.path.insert(0, os.path.abspath('../third_party'))
import arc
from arc import black_boxes
from arc import methods

sys.path.insert(0, '../cgtc/')
from conformal_methods import (
    evaluate_prediction_sets, finalize_prediction_sets,
    get_preliminary_sets_naive, get_preliminary_sets_benchmark,
    get_preliminary_sets_Bernoulli,
)
from alpha_tune_plugin import tune_plugin_allocation_cv, compute_plugin_allocation
from distributions_x import ShiftedNormal
from distributions_y import DirichletProcess
from testing import select_beta_cv

#####################
# Define parameters #
#####################

# Updated CV-tuned plug-in missing-mass allocation method.
# The allocation (alpha_class, alpha_unseen, alpha_seen) is derived from the
# plug-in missing-mass estimate mu_hat = (M1 + 1)/(n + 1), with the rule
# hyperparameters (alpha_unseen cap, alpha_seen) selected by cross-validation.
#
# Usage:
#   python synthetic_experiment_dp_mm_plugin.py theta n_ref n_test calib_num \
#          alpha_total lambda_weight batch_num splitting_method_flag [grid_size]
#   splitting_method_flag: 0 = random splitting, 1 = bernoulli splitting (for CV)
#   grid_size: optional cap-grid size G (default 20)

if len(sys.argv) < 9:
    print("Error: incorrect number of parameters.")
    print("Usage: python synthetic_experiment_dp_mm_plugin.py theta n_ref n_test "
          "calib_num alpha_total lambda_weight batch_num splitting_method_flag [grid_size]")
    quit()

theta = int(sys.argv[1])           # DP concentration parameter
n_ref = int(sys.argv[2])           # number of training and calibration samples
n_test = int(sys.argv[3])          # number of test samples
calib_num = int(sys.argv[4])       # number of calibration samples
alpha_total = float(sys.argv[5])   # Total alpha budget
lambda_weight = float(sys.argv[6]) # Loss preference between set size and joker waste, in [0,1]
batch_num = int(sys.argv[7])       # seed
splitting_method_flag = int(sys.argv[8])  # 0 for 'random', 1 for 'bernoulli'
grid_size = int(sys.argv[9]) if len(sys.argv) > 9 else 20  # cap-grid size G

if splitting_method_flag == 0:
    splitting_method = 'random'
elif splitting_method_flag == 1:
    splitting_method = 'bernoulli'
else:
    print(f"Error: splitting_method_flag must be 0 (random) or 1 (bernoulli), "
          f"got '{splitting_method_flag}'")
    quit()


#####################
# Define Parameters #
#####################
# Number of experiments (repetition in each batch, total number is num_exp*num_batch)
num_exp = 5

calib_size = calib_num / n_ref

print(f"splitting_method: {splitting_method} (flag={splitting_method_flag})")
print(f"plug-in CV cap-grid size G: {grid_size}")

# Using power weights to combine the p-values for testing the seen hypothesis.
# If default_beta is not None, use it. If None, use optimal weights.
default_beta = 1.6
# If beta_cv is true, use CV to choose beta
beta_cv = True

# One-class classifier choice: 'lof', 'iforest', 'ocsvm'
occ_name = 'lof'

# Print parsed parameters
print(f"theta: {theta}")
print(f"n_ref: {n_ref}")
print(f"n_test: {n_test}")
print(f"calib_num: {calib_num}")
print(f"calib_size: {calib_size}")
print(f"alpha_total: {alpha_total}")
print(f"lambda_weight: {lambda_weight}")
print(f"batch_num: {batch_num}")


#####################
# Define Output Dir #
#####################

beta_label = "betacv" if beta_cv else f"beta{default_beta}"
output_file = (
    f"results/dp_tuned_mixed_labels_mm_plugin/"
    f"dp_"
    f"occ{occ_name}_"
    f"{beta_label}_"
    f"theta{theta}_"
    f"nref{n_ref}_"
    f"ntest{n_test}_"
    f"cs{calib_num}_"
    f"atotal{alpha_total:.3f}_"
    f"lambda{lambda_weight:.2f}_"
    f"split{splitting_method_flag}_"
    f"G{grid_size}_"
    f"batch{batch_num}.csv"
)

print("Output file name: {:s}".format(output_file))
os.makedirs(os.path.dirname(output_file), exist_ok=True)


#####################
# Define Methods    #
#####################

# All methods to be evaluated after tuning
methods_list = {
    'Method (random splitting)': get_preliminary_sets_naive,
    'Method (benchmark)': get_preliminary_sets_benchmark,
    'Method (Bernoulli)': get_preliminary_sets_Bernoulli,
    'Method (Bernoulli benchmark)': get_preliminary_sets_Bernoulli,
}

# Parameters for data distribution
num_features = 3   # dimension of feature X
sigma = 1e-4       # variance of the shifted normal of the feature distribution

# Number of neighbors to use in KNN
n_neighbors = 5

# Classification and OCC models
classifier = black_boxes.OpenSetKNN(
    calibrate=False,
    n_neighbors=n_neighbors,
    weights='distance',
    algorithm='auto',
    leaf_size=30,
    p=2,
    metric='minkowski',
    metric_params=None,
    n_jobs=-1,
    clip_proba_factor=1e-20
)

# One-class classification model for computing feature-dependent GT p-values
occ_choices = {
    'lof':     LocalOutlierFactor(n_neighbors=1, novelty=True),
    'iforest': IsolationForest(n_estimators=10, contamination='auto', random_state=batch_num),
    'ocsvm':   OneClassSVM(kernel='rbf', nu=0.05),
}

if occ_name not in occ_choices:
    print(f"Error: occ_name must be one of {list(occ_choices.keys())}, got '{occ_name}'")
    quit()

occ = occ_choices[occ_name]


######################################
# Auxiliary functions and Data Model #
######################################

# NOTE: calib_prob_real and calculate_freq_one_proportion implement the
# selective sample-splitting (calibration) rule. Its use of the singleton
# proportion p1 = M1/n (freq_one_prop) is a DIFFERENT quantity from the
# missing-mass plug-in mu_hat = (M1+1)/(n+1); they are intentionally separate.

def calib_prob_real(frequency, exp_prop=calib_size, freq_one_prop=None):
    """
    Given a positive integer 'frequency' representing the frequency count of a label,
    returns the probability of that label being selected into the calibration set.
    """
    if frequency == 0:
        return 0.0
    elif frequency == 1:
        return 0.0
    else:
        if freq_one_prop is not None and freq_one_prop < 1.0:
            adjusted_prob = exp_prop / (1 - freq_one_prop)
            prob_calib = min(adjusted_prob, 1.0)
        else:
            prob_calib = exp_prop
        return prob_calib


def calculate_freq_one_proportion(Y_ref):
    """
    Calculate the proportion of data points that have frequency-1 labels.
    """
    unique_labels, counts = np.unique(Y_ref, return_counts=True)
    freq_one_labels = unique_labels[counts == 1]
    freq_one_mask = np.isin(Y_ref, freq_one_labels)
    freq_one_prop = np.mean(freq_one_mask)
    return freq_one_prop


def split_data(X, Y, n_ref, n_test, random_state=None):
    """
    Splits the dataset into reference (training + calibration) and testing datasets.
    """
    total_samples = len(X)
    if n_ref + n_test > total_samples:
        raise ValueError("n_ref + n_test exceeds the total number of available samples.")
    if random_state is not None:
        np.random.seed(random_state)
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    ref_indices = indices[:n_ref]
    test_indices = indices[n_ref:n_ref + n_test]
    X_ref, Y_ref = X[ref_indices], Y[ref_indices]
    X_test, Y_test = X[test_indices], Y[test_indices]
    return X_ref, Y_ref, X_test, Y_test


def analyze_data(X_ref, Y_ref, X_test, Y_test, methods_list,
                 alpha_unseen, alpha_seen, alpha_class,
                 occ, classifier, calib_size,
                 mu_hat, alpha_unseen_cap, grid_size,
                 random_state=2024):
    """
    Analyze the data using specified methods and evaluate prediction sets.

    The plug-in allocation is already deployed: alpha_class is the inflated
    classification budget (no further missing-mass adjustment is applied here).
    Benchmark methods use the full alpha_total budget.
    """
    # Calculate proportion of frequency-1 labels (for the calibration split rule)
    freq_one_prop = calculate_freq_one_proportion(Y_ref)
    tqdm.write(f"Proportion of frequency-1 labels in reference data: {freq_one_prop:.3f}")

    # Create adjusted calibration probability function (selective split rule)
    calib_prob_adjusted = partial(calib_prob_real,
                                  exp_prop=calib_size,
                                  freq_one_prop=freq_one_prop)

    # List of all unique labels seen in the training set
    seen_labels = np.unique(Y_ref)
    num_unique_labels = len(seen_labels)

    # Calculate proportion of new/unseen test labels
    unseen_mask = ~np.isin(Y_test, seen_labels)
    prop_unseen = np.mean(unseen_mask)
    num_unseen = np.sum(unseen_mask)
    tqdm.write(f"Proportion of unseen test labels: {prop_unseen:.3f} "
               f"({num_unseen}/{len(Y_test)} points)")

    unique_unseen_labels = np.unique(Y_test[unseen_mask])
    num_unique_unseen = len(unique_unseen_labels)
    tqdm.write(f"Number of unique unseen labels in test: {num_unique_unseen}")

    results_df = pd.DataFrame()

    if beta_cv:
        betas = np.linspace(1.0, 3.0, 11)
        best_beta, best_loss = select_beta_cv(X_ref, Y_ref, betas, cv=5, randomized=True)
        tqdm.write(f"Selected beta = {best_beta:.2f} with mean p-value {best_loss:.4f}")
    else:
        best_beta = default_beta
        if default_beta is None:
            tqdm.write(f"Using optimal weights (beta = None)")
        else:
            tqdm.write(f"Using fixed beta = {best_beta:.2f} (CV disabled)")

    # Iterate over each method and evaluate
    for method_name, method_function in methods_list.items():
        tqdm.write(f"Begin running {method_name}")

        if method_name == 'Method (benchmark)' or method_name == 'Method (benchmark full)':
            # Benchmark uses full alpha_total budget
            decoded_prelim_sets = method_function(
                X_ref, Y_ref, X_test,
                alpha=alpha_unseen + alpha_seen + alpha_class,
                black_box=classifier,
                calib_size=calib_size,
                random_state=random_state
            )
        elif method_name == 'Method (Bernoulli benchmark)' or method_name == 'Method (Bernoulli benchmark full)':
            # Benchmark uses full alpha_total budget
            decoded_prelim_sets = method_function(
                X_ref, Y_ref, X_test,
                alpha_prime=alpha_unseen + alpha_seen + alpha_class,
                black_box=classifier,
                calibration_probability=calib_prob_adjusted,
                random_state=random_state
            )
        elif method_name == 'Method (Bernoulli)' or method_name == 'Method (Bernoulli full)' or method_name == 'Method (Bernoulli uniform)':
            # Use the deployed (already-inflated) alpha_class
            decoded_prelim_sets = method_function(
                X_ref, Y_ref, X_test,
                alpha_prime=alpha_class,
                black_box=classifier,
                calibration_probability=calib_prob_adjusted,
                random_state=random_state
            )
        else:
            # Use the deployed (already-inflated) alpha_class
            decoded_prelim_sets = method_function(
                X_ref, Y_ref, X_test,
                alpha_prime=alpha_class,
                black_box=classifier,
                calib_size=calib_size,
                random_state=random_state
            )

        # For each p-value approach
        for pvalue_method in ['GT', 'XGT', 'RGT']:
            # Skip p-value computation for the benchmark methods
            if method_name in ['Method (benchmark)', 'Method (Bernoulli benchmark)',
                               'Method (benchmark full)', 'Method (Bernoulli benchmark full)']:
                final_sets = decoded_prelim_sets
            else:
                # Compute final sets with the dual testing problem
                final_sets = finalize_prediction_sets(
                    decoded_prelim_sets,
                    X_ref, Y_ref, X_test,
                    pvalue_method,
                    alpha_unseen, alpha_seen,
                    occ=occ,
                    random_state=random_state,
                    beta=best_beta
                )

            # Evaluate the prediction sets
            new_results = evaluate_prediction_sets(final_sets, Y_test, Y_ref, verbose=False)

            new_results['method'] = method_name
            new_results['pvalue_method'] = pvalue_method
            new_results['num_unique_labels'] = num_unique_labels
            new_results['prop_unseen_test'] = prop_unseen
            new_results['num_unseen_test'] = num_unseen

            # Plug-in allocation values
            new_results['alpha_class'] = alpha_class
            new_results['alpha_unseen'] = alpha_unseen
            new_results['alpha_seen'] = alpha_seen
            new_results['mu_hat'] = mu_hat
            new_results['alpha_unseen_cap'] = alpha_unseen_cap
            new_results['grid_size'] = grid_size
            new_results['occ'] = occ_name

            if beta_cv:
                new_results['beta'] = best_beta

            results_df = pd.concat([results_df, new_results])

    return results_df


def run_syn_experiment(n_ref, n_test, num_exp, batch_num):
    """
    Run experiments: CV-tune the plug-in hyperparameters, deploy the allocation,
    then evaluate all methods.
    """
    np.random.seed(batch_num)

    all_results = pd.DataFrame()
    all_alpha_class = []
    all_alpha_unseen = []
    all_alpha_seen = []
    all_alpha_unseen_cap = []
    all_mu_hat = []
    all_loss = []

    for i in tqdm(range(num_exp)):

        current_state = batch_num * 1000 + i

        X, Y = data_dist.sample(n_ref + n_test, random_state=current_state)
        X_ref, Y_ref, X_test, Y_test = split_data(X, Y, n_ref, n_test, current_state)

        tqdm.write(f"Loop {i + 1}: Number of data points: {len(Y_ref)}")
        tqdm.write(f"Number of unique classes in Y_ref: {len(np.unique(Y_ref))}")

        # Calculate proportion of frequency-1 labels for the calibration split rule
        freq_one_prop = calculate_freq_one_proportion(Y_ref)
        calib_prob_adjusted = partial(calib_prob_real,
                                      exp_prop=calib_size,
                                      freq_one_prop=freq_one_prop)

        # Use the same data for tuning (as in the synthetic experiments)
        X_tune, Y_tune = X_ref, Y_ref

        # --- Tune: cross-validate the plug-in hyperparameters (cap, alpha_seen)
        best_cap, best_alpha_seen, tuning_results = tune_plugin_allocation_cv(
            X_tune, Y_tune,
            alpha_total=alpha_total,
            lambda_weight=lambda_weight,
            n_splits=10,
            grid_size=grid_size,
            classifier=classifier,
            occ=occ,
            calibration_probability=calib_prob_adjusted,
            calib_size=calib_size,
            pvalue_method='XGT',
            random_state=current_state,
            beta=None,
            splitting_method=splitting_method,
            verbose=(i == 0),
        )
        best_loss = tuning_results['avg_loss'].min()

        # --- Deploy: derive the realized allocation on the full reference sample
        alpha_unseen, alpha_seen, alpha_class, mu_hat = compute_plugin_allocation(
            Y_ref, alpha_total, best_cap, best_alpha_seen
        )
        tqdm.write(f"Deployed plug-in allocation: mu_hat={mu_hat:.4f}, cap={best_cap:.4f}, "
                   f"alpha_unseen={alpha_unseen:.4f}, alpha_seen={alpha_seen:.4f}, "
                   f"alpha_class={alpha_class:.4f}")

        results = analyze_data(
            X_ref, Y_ref, X_test, Y_test, methods_list,
            alpha_unseen, alpha_seen, alpha_class,
            occ, classifier, calib_size,
            mu_hat, best_cap, grid_size,
            random_state=current_state
        )

        all_results = pd.concat([all_results, results], ignore_index=True)
        all_alpha_class.append(alpha_class)
        all_alpha_unseen.append(alpha_unseen)
        all_alpha_seen.append(alpha_seen)
        all_alpha_unseen_cap.append(best_cap)
        all_mu_hat.append(mu_hat)
        all_loss.append(best_loss)

    print("\nSummary of deployed plug-in allocations across experiments:")
    print(f"  Average alpha_class = {np.mean(all_alpha_class):.3f} ± {np.std(all_alpha_class):.3f}")
    print(f"  Average alpha_unseen = {np.mean(all_alpha_unseen):.3f} ± {np.std(all_alpha_unseen):.3f}")
    print(f"  Average alpha_seen = {np.mean(all_alpha_seen):.3f} ± {np.std(all_alpha_seen):.3f}")
    print(f"  Average alpha_unseen_cap = {np.mean(all_alpha_unseen_cap):.3f} ± {np.std(all_alpha_unseen_cap):.3f}")
    print(f"  Average mu_hat = {np.mean(all_mu_hat):.3f} ± {np.std(all_mu_hat):.3f}")
    print(f"  Average loss = {np.mean(all_loss):.3f} ± {np.std(all_loss):.3f}")

    # Add tuning metric to results
    all_results['tuning_loss'] = np.repeat(all_loss, len(all_results) // num_exp)

    return all_results, all_alpha_class, all_alpha_unseen, all_alpha_seen, all_alpha_unseen_cap, all_mu_hat


class DataDistribution_1:
    def __init__(self, label_dist, feature_dist):
        self.label_dist = label_dist
        self.feature_dist = feature_dist

    def sample(self, n, random_state=None):
        Y = self.label_dist.sample(n, random_state=random_state)
        X = self.feature_dist.sample(Y, random_state=random_state)
        return X, Y


########################
# Start Experiments    #
########################

# Define data distribution and generate data
label_dist = DirichletProcess(theta=theta)
feature_dist = ShiftedNormal(num_features, sigma)
data_dist = DataDistribution_1(label_dist, feature_dist)


###################
# Save Results    #
###################
results, alpha_class_list, alpha_unseen_list, alpha_seen_list, alpha_unseen_cap_list, mu_hat_list = \
    run_syn_experiment(n_ref, n_test, num_exp, batch_num)

# Create header with average deployed allocation
header_df = pd.DataFrame({
    "theta": [theta],
    "n_ref": [n_ref],
    "n_test": [n_test],
    "batch_num": [batch_num],
    "alpha_total": [alpha_total],
    "alpha_class_avg": [np.mean(alpha_class_list)],
    "alpha_unseen_avg": [np.mean(alpha_unseen_list)],
    "alpha_seen_avg": [np.mean(alpha_seen_list)],
    "alpha_unseen_cap_avg": [np.mean(alpha_unseen_cap_list)],
    "mu_hat_avg": [np.mean(mu_hat_list)],
    "calib_num": [calib_num],
    "lambda_weight": [lambda_weight],
    "splitting_method": [splitting_method],
    "splitting_method_flag": [splitting_method_flag],
    "method_family": ["plugin"],
})

# Replicate header_df so it has as many rows as the results DataFrame
header_df_expanded = pd.concat([header_df] * len(results), ignore_index=True)

output_df = pd.concat([header_df_expanded, results], axis=1)
output_df.to_csv(output_file, index=False)

print(f"Finished saving final results to:\n{output_file}\nParameters:")
print(f"  n_ref={n_ref}, n_test={n_test}, alpha_total={alpha_total}, "
      f"calib_size={calib_size}, lambda_weight={lambda_weight}, "
      f"batch_num={batch_num}, grid_size={grid_size}")
