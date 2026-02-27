import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import partial
import sys
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "20"

sys.path.insert(0, os.path.abspath('../third_party'))
import arc
from arc import black_boxes

sys.path.insert(0, '../cgtc/')
from conformal_methods import evaluate_prediction_sets, get_prediction_sets_openmax
from distributions_x import ShiftedNormal
from distributions_y import DirichletProcess

#####################
# Define parameters #
#####################

if len(sys.argv) != 7:
    print("Error: incorrect number of parameters.")
    print("Usage: python synthetic_experiment_openmax.py theta n_ref n_test calib_num alpha_total batch_num")
    quit()

# Parse command-line arguments
theta = int(sys.argv[1])        # DP concentration parameter
n_ref = int(sys.argv[2])        # number of reference samples (train + calib)
n_test = int(sys.argv[3])       # number of test samples
calib_num = int(sys.argv[4])    # number of calibration samples
alpha_total = float(sys.argv[5])  # miscoverage probability (full budget)
batch_num = int(sys.argv[6])    # random seed

#####################
# Fixed parameters  #
#####################

# Number of experiment repetitions per batch
num_exp = 5

calib_size = calib_num / n_ref

# Print parsed parameters
print(f"theta: {theta}")
print(f"n_ref: {n_ref}")
print(f"n_test: {n_test}")
print(f"calib_num: {calib_num}")
print(f"calib_size: {calib_size}")
print(f"alpha_total: {alpha_total}")
print(f"batch_num: {batch_num}")

#####################
# Output file       #
#####################

output_file = (
    f"results/dp_openmax/"
    f"dp_"
    f"theta{theta}_"
    f"nref{n_ref}_"
    f"ntest{n_test}_"
    f"cs{calib_num}_"
    f"atotal{alpha_total:.3f}_"
    f"batch{batch_num}.csv"
)

print(f"Output file name: {output_file}")
os.makedirs(os.path.dirname(output_file), exist_ok=True)

#####################
# Data distribution #
#####################

num_features = 3
sigma = 0.000005

class DataDistribution_1:
    def __init__(self, label_dist, feature_dist):
        self.label_dist = label_dist
        self.feature_dist = feature_dist

    def sample(self, n, random_state=None):
        Y = self.label_dist.sample(n, random_state=random_state)
        X = self.feature_dist.sample(Y, random_state=random_state)
        return X, Y

label_dist = DirichletProcess(theta=theta)
feature_dist = ShiftedNormal(num_features, sigma)
data_dist = DataDistribution_1(label_dist, feature_dist)

#####################
# Classifiers       #
#####################

n_neighbors = 5

# OpenMax-KNN: KNN base classifier + Weibull revision on centroid distances
classifier_openmax_knn = black_boxes.OpenMaxKNN(
    n_neighbors=n_neighbors,
    weights='distance',
    algorithm='auto',
    leaf_size=30,
    p=2,
    metric='minkowski',
    metric_params=None,
    n_jobs=-1,
    clip_proba_factor=1e-20,
    tail_size=20,
    alpha_rank=None   # revise all classes
)

# Original OpenMax-MLP: neural network + penultimate-layer activations
classifier_openmax_mlp = black_boxes.OpenMaxMLP(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    max_iter=500,
    random_state=42,
    tail_size=20,
    alpha_rank=None   # revise all classes
)

# Methods to evaluate
methods_list = {
    'Method (OpenMax-KNN)': classifier_openmax_knn,
    'Method (OpenMax-MLP)': classifier_openmax_mlp,
}


#####################
# Helper functions  #
#####################

def split_data(X, Y, n_ref, n_test, random_state=None):
    """Splits dataset into reference and test sets."""
    total_samples = len(X)
    if n_ref + n_test > total_samples:
        raise ValueError("n_ref + n_test exceeds total samples.")
    if random_state is not None:
        np.random.seed(random_state)
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    ref_indices = indices[:n_ref]
    test_indices = indices[n_ref:n_ref + n_test]
    return X[ref_indices], Y[ref_indices], X[test_indices], Y[test_indices]


def analyze_data_openmax(X_ref, Y_ref, X_test, Y_test, methods_list,
                         alpha, calib_size, random_state=2024):
    """
    Run each OpenMax-style method and evaluate prediction sets.

    For each method:
      - Call get_prediction_sets_openmax (which does train/calib split internally)
      - Evaluate with Y_ref  (comparable to CGTC metrics)
      - Also compute joker_train-specific metrics using Y_train
    """

    # Reference-level statistics (based on full Y_ref)
    seen_labels_ref = np.unique(Y_ref)
    unseen_mask_ref = ~np.isin(Y_test, seen_labels_ref)
    prop_unseen_ref = np.mean(unseen_mask_ref)
    num_unseen_ref = np.sum(unseen_mask_ref)

    results_df = pd.DataFrame()

    for method_name, classifier in methods_list.items():
        tqdm.write(f"Running {method_name}")

        prediction_sets, Y_train = get_prediction_sets_openmax(
            X_ref, Y_ref, X_test,
            alpha=alpha, black_box=classifier, calib_size=calib_size,
            random_state=random_state
        )

        # --- Evaluate with Y_ref (for comparison with CGTC) ---
        new_results = evaluate_prediction_sets(
            prediction_sets, Y_test, Y_ref, verbose=False
        )

        # --- Compute joker_train-specific metrics using Y_train ---
        seen_labels_train = np.unique(Y_train)

        # prop of test points whose true label is unseen in training
        unseen_mask_train = ~np.isin(Y_test, seen_labels_train)
        prop_unseen_train = np.mean(unseen_mask_train)
        num_unseen_train = np.sum(unseen_mask_train)

        # Coverage(joker_train): true label in set OR (? in set AND label unseen in train)
        coverage_joker_train = np.mean([
            1 if (yt in ps or ('?' in ps and yt not in seen_labels_train))
            else 0
            for ps, yt in zip(prediction_sets, Y_test)
        ])

        # Conditional coverage for unseen-in-train test points
        unseen_train_idx = [i for i, yt in enumerate(Y_test)
                           if yt not in seen_labels_train]
        if unseen_train_idx:
            unseen_train_coverage = np.mean([
                1 if ('?' in prediction_sets[i] or Y_test[i] in prediction_sets[i])
                else 0
                for i in unseen_train_idx
            ])
        else:
            unseen_train_coverage = np.nan

        # Conditional coverage for seen-in-train test points
        seen_train_idx = [i for i, yt in enumerate(Y_test)
                         if yt in seen_labels_train]
        if seen_train_idx:
            seen_train_coverage = np.mean([
                1 if Y_test[i] in prediction_sets[i] else 0
                for i in seen_train_idx
            ])
        else:
            seen_train_coverage = np.nan

        # Add columns
        new_results['method'] = method_name
        new_results['pvalue_method'] = 'N/A'
        new_results['num_unique_labels'] = len(seen_labels_ref)
        new_results['num_unique_labels_train'] = len(seen_labels_train)
        new_results['prop_unseen_test'] = prop_unseen_ref
        new_results['num_unseen_test'] = num_unseen_ref
        new_results['prop_unseen_train'] = prop_unseen_train
        new_results['num_unseen_train'] = int(num_unseen_train)
        new_results['alpha_class'] = alpha
        new_results['alpha_unseen'] = 0.0
        new_results['alpha_seen'] = 0.0
        new_results['Coverage (joker_train)'] = coverage_joker_train
        new_results['Unseen Coverage (joker_train)'] = unseen_train_coverage
        new_results['Seen Coverage (joker_train)'] = seen_train_coverage

        results_df = pd.concat([results_df, new_results])

    return results_df


########################
# Run experiments      #
########################

def run_openmax_experiment(n_ref, n_test, num_exp, batch_num):
    np.random.seed(batch_num)

    all_results = pd.DataFrame()

    for i in tqdm(range(num_exp)):
        current_state = batch_num * 1000 + i

        X, Y = data_dist.sample(n_ref + n_test, random_state=current_state)
        X_ref, Y_ref, X_test, Y_test = split_data(
            X, Y, n_ref, n_test, current_state
        )

        tqdm.write(f"Loop {i + 1}: n_ref={len(Y_ref)}, "
                   f"unique classes={len(np.unique(Y_ref))}")

        results = analyze_data_openmax(
            X_ref, Y_ref, X_test, Y_test, methods_list,
            alpha=alpha_total, calib_size=calib_size,
            random_state=current_state
        )

        all_results = pd.concat([all_results, results], ignore_index=True)

    return all_results


###################
# Run and save    #
###################

results = run_openmax_experiment(n_ref, n_test, num_exp, batch_num)

# Create header
header_df = pd.DataFrame({
    "theta": [theta],
    "n_ref": [n_ref],
    "n_test": [n_test],
    "batch_num": [batch_num],
    "alpha_total": [alpha_total],
    "calib_num": [calib_num],
})

header_df_expanded = pd.concat([header_df] * len(results), ignore_index=True)
output_df = pd.concat([header_df_expanded, results], axis=1)
output_df.to_csv(output_file, index=False)

print(f"\nFinished saving results to:\n{output_file}")
print(f"  n_ref={n_ref}, n_test={n_test}, alpha_total={alpha_total}, "
      f"calib_size={calib_size}, batch_num={batch_num}")
