import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import LocalOutlierFactor
from functools import partial
import sys
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "20"

sys.path.insert(0, os.path.abspath('../third_party'))
import arc
from arc import black_boxes

sys.path.insert(0, '../cgtc/')
from alpha_tune_function import tune_alpha_allocation_loss_all_optimized
from conformal_methods import evaluate_prediction_sets, finalize_prediction_sets, get_preliminary_sets_naive_full, get_preliminary_sets_naive, get_preliminary_sets_benchmark, get_preliminary_sets_Bernoulli
from distributions_x import ShiftedNormal
from distributions_y import DirichletProcess
from testing import select_beta_cv

#####################
# Define parameters #
#####################

if len(sys.argv) != 9:
    print("Error: incorrect number of parameters.")
    print("Usage: python synthetic_experiment_openmax_cgtc.py theta n_ref n_test calib_num alpha_total lambda_weight batch_num tuning_method_flag")
    quit()

# Parse command-line arguments
theta = int(sys.argv[1])        # DP concentration parameter
n_ref = int(sys.argv[2])        # number of reference samples (train + calib)
n_test = int(sys.argv[3])       # number of test samples
calib_num = int(sys.argv[4])    # number of calibration samples
alpha_total = float(sys.argv[5])  # Total alpha budget
lambda_weight = float(sys.argv[6])  # Tune alpha allocation using loss function
batch_num = int(sys.argv[7])    # seed
tuning_method_flag = int(sys.argv[8])  # 0 for 'random', 1 for 'bernoulli', -1 for fixed

# Convert flag to string
if tuning_method_flag == 0:
    tuning_method = 'random'
elif tuning_method_flag == 1:
    tuning_method = 'bernoulli'
elif tuning_method_flag == -1:
    tuning_method = 'fixed'
    alpha_class_fixed = alpha_total / 3
    alpha_unseen_fixed = alpha_total / 3
    alpha_seen_fixed = alpha_total / 3
else:
    print(f"Error: tuning_method must be 0 (random) or 1 (bernoulli) or -1 (fixed), got '{tuning_method_flag}'")
    quit()

#####################
# Fixed parameters  #
#####################

num_exp = 5

calib_size = calib_num / n_ref

print(f"alpha tuning_method: {tuning_method} (flag={tuning_method_flag})")

# Power weights for combining p-values for seen hypothesis testing
default_beta = 1.6
beta_cv = False

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
# Output file       #
#####################

output_file = (
    f"results/dp_openmax_cgtc/"
    f"dp_"
    f"theta{theta}_"
    f"nref{n_ref}_"
    f"ntest{n_test}_"
    f"cs{calib_num}_"
    f"atotal{alpha_total:.3f}_"
    f"lambda{lambda_weight:.2f}_"
    f"tune{tuning_method_flag}_"
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

# OpenSetKNNOpenMax: KNN + Weibull revision to replace Good-Turing for unseen labels
classifier_knn = black_boxes.OpenSetKNNOpenMax(
    calibrate=False,
    n_neighbors=n_neighbors,
    weights='distance',
    algorithm='auto',
    leaf_size=30,
    p=2,
    metric='minkowski',
    metric_params=None,
    n_jobs=-1,
    clip_proba_factor=1e-20,
    noise_scale=1e-6,
    tail_size=20,
    alpha_rank=None
)

# OpenSetMLPOpenMax: MLP + Weibull revision on penultimate-layer activations
classifier_mlp = black_boxes.OpenSetMLPOpenMax(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    max_iter=500,
    random_state=42,
    clip_proba_factor=1e-20,
    noise_scale=1e-6,
    tail_size=20,
    alpha_rank=None
)

# OCC model for computing feature-dependent GT p-values (XGT)
occ = LocalOutlierFactor(n_neighbors=1, novelty=True)

#####################
# Methods to tune   #
#####################

# For alpha tuning, use random splitting full with the KNN classifier
methods_list_tuning = {
    'Method (random splitting full)': get_preliminary_sets_naive_full,
}

#####################
# Helper functions  #
#####################

def calib_prob_real(frequency, exp_prop=calib_size, freq_one_prop=None):
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
    unique_labels, counts = np.unique(Y_ref, return_counts=True)
    freq_one_labels = unique_labels[counts == 1]
    freq_one_mask = np.isin(Y_ref, freq_one_labels)
    return np.mean(freq_one_mask)


def split_data(X, Y, n_ref, n_test, random_state=None):
    total_samples = len(X)
    if n_ref + n_test > total_samples:
        raise ValueError("n_ref + n_test exceeds the total number of available samples.")
    if random_state is not None:
        np.random.seed(random_state)
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    ref_indices = indices[:n_ref]
    test_indices = indices[n_ref:n_ref + n_test]
    return X[ref_indices], Y[ref_indices], X[test_indices], Y[test_indices]


def analyze_data(X_ref, Y_ref, X_test, Y_test, classifiers_dict,
                 alpha_unseen, alpha_seen, alpha_class,
                 occ, calib_size, random_state=2024):
    """
    Run each OpenMax-CGTC classifier through the full CGTC pipeline
    (preliminary sets + dual hypothesis testing) and evaluate.
    """
    freq_one_prop = calculate_freq_one_proportion(Y_ref)
    tqdm.write(f"Proportion of frequency-1 labels: {freq_one_prop:.3f}")

    calib_prob_adjusted = partial(calib_prob_real,
                                  exp_prop=calib_size,
                                  freq_one_prop=freq_one_prop)

    seen_labels = np.unique(Y_ref)
    num_unique_labels = len(seen_labels)

    unseen_mask = ~np.isin(Y_test, seen_labels)
    prop_unseen = np.mean(unseen_mask)
    num_unseen = np.sum(unseen_mask)
    tqdm.write(f"Proportion of unseen test labels: {prop_unseen:.3f} ({num_unseen}/{len(Y_test)} points)")

    results_df = pd.DataFrame()

    if beta_cv:
        betas = np.linspace(1.0, 3.0, 11)
        best_beta, best_loss = select_beta_cv(X_ref, Y_ref, betas, cv=5, randomized=True)
        tqdm.write(f"Selected beta = {best_beta:.2f} with mean p-value {best_loss:.4f}")
    else:
        best_beta = default_beta
        if default_beta is None:
            tqdm.write("Using optimal weights (beta = None)")
        else:
            tqdm.write(f"Using fixed beta = {best_beta:.2f} (CV disabled)")

    # For each classifier, run methods through the CGTC pipeline
    for classifier_name, classifier in classifiers_dict.items():
        tqdm.write(f"\n=== Classifier: {classifier_name} ===")

        # Methods to evaluate with this classifier
        methods_list = {
            'Method (random splitting)': get_preliminary_sets_naive,
            'Method (benchmark)': get_preliminary_sets_benchmark,
            'Method (Bernoulli)': get_preliminary_sets_Bernoulli,
            'Method (Bernoulli benchmark)': get_preliminary_sets_Bernoulli,
        }

        for method_name, method_function in methods_list.items():
            tqdm.write(f"  Running {method_name}")

            if method_name == 'Method (benchmark)':
                decoded_prelim_sets = method_function(
                    X_ref, Y_ref, X_test,
                    alpha=alpha_unseen + alpha_seen + alpha_class,
                    black_box=classifier,
                    calib_size=calib_size,
                    random_state=random_state
                )
            elif method_name == 'Method (Bernoulli benchmark)':
                decoded_prelim_sets = method_function(
                    X_ref, Y_ref, X_test,
                    alpha_prime=alpha_unseen + alpha_seen + alpha_class,
                    black_box=classifier,
                    calibration_probability=calib_prob_adjusted,
                    random_state=random_state
                )
            elif method_name == 'Method (Bernoulli)':
                decoded_prelim_sets = method_function(
                    X_ref, Y_ref, X_test,
                    alpha_prime=alpha_class,
                    black_box=classifier,
                    calibration_probability=calib_prob_adjusted,
                    random_state=random_state
                )
            else:
                decoded_prelim_sets = method_function(
                    X_ref, Y_ref, X_test,
                    alpha_prime=alpha_class,
                    black_box=classifier,
                    calib_size=calib_size,
                    random_state=random_state
                )

            for pvalue_method in ['GT', 'XGT', 'RGT']:
                if method_name in ['Method (benchmark)', 'Method (Bernoulli benchmark)']:
                    final_sets = decoded_prelim_sets
                else:
                    final_sets = finalize_prediction_sets(
                        decoded_prelim_sets,
                        X_ref, Y_ref, X_test,
                        pvalue_method,
                        alpha_unseen, alpha_seen,
                        occ=occ,
                        random_state=random_state,
                        beta=best_beta
                    )

                new_results = evaluate_prediction_sets(final_sets, Y_test, Y_ref, verbose=False)

                new_results['method'] = f"{method_name} [{classifier_name}]"
                new_results['pvalue_method'] = pvalue_method
                new_results['num_unique_labels'] = num_unique_labels
                new_results['prop_unseen_test'] = prop_unseen
                new_results['num_unseen_test'] = num_unseen
                new_results['alpha_class'] = alpha_class
                new_results['alpha_unseen'] = alpha_unseen
                new_results['alpha_seen'] = alpha_seen
                new_results['classifier'] = classifier_name

                results_df = pd.concat([results_df, new_results])

    return results_df


########################
# Run experiments      #
########################

def run_experiment(n_ref, n_test, num_exp, batch_num):
    np.random.seed(batch_num)

    classifiers_dict = {
        'OpenMax-KNN': classifier_knn,
        'OpenMax-MLP': classifier_mlp,
    }

    all_results = pd.DataFrame()
    all_alpha_class = []
    all_alpha_unseen = []
    all_alpha_seen = []
    all_loss = []
    all_normalized_size = []
    all_joker_waste = []

    for i in tqdm(range(num_exp)):
        current_state = batch_num * 1000 + i

        X, Y = data_dist.sample(n_ref + n_test, random_state=current_state)
        X_ref, Y_ref, X_test, Y_test = split_data(
            X, Y, n_ref, n_test, current_state
        )

        tqdm.write(f"Loop {i + 1}: n_ref={len(Y_ref)}, unique classes={len(np.unique(Y_ref))}")

        freq_one_prop = calculate_freq_one_proportion(Y_ref)
        calib_prob_adjusted = partial(calib_prob_real,
                                      exp_prop=calib_size,
                                      freq_one_prop=freq_one_prop)

        X_tune, Y_tune = X_ref, Y_ref

        if tuning_method_flag != -1:
            alpha_class_tuned, alpha_unseen_tuned, alpha_seen_tuned, tuning_results = tune_alpha_allocation_loss_all_optimized(
                X_tune, Y_tune,
                alpha_total=alpha_total,
                lambda_weight=lambda_weight,
                n_splits=10,
                alpha_step=0.005,
                classifier=classifier_knn,
                occ=occ,
                calibration_probability=calib_prob_adjusted,
                calib_size=calib_size,
                pvalue_method='XGT',
                random_state=current_state,
                beta=None,
                splitting_method=tuning_method,
                verbose=(i == 0)
            )
        elif tuning_method_flag == -1:
            alpha_class_tuned = alpha_class_fixed
            alpha_unseen_tuned = alpha_unseen_fixed
            alpha_seen_tuned = alpha_seen_fixed

        best_idx = tuning_results['avg_loss'].idxmin()
        best_loss = tuning_results.loc[best_idx, 'avg_loss']
        best_normalized_size = tuning_results.loc[best_idx, 'avg_normalized_size']
        best_joker_waste = tuning_results.loc[best_idx, 'avg_joker_waste']

        results = analyze_data(
            X_ref, Y_ref, X_test, Y_test, classifiers_dict,
            alpha_unseen_tuned, alpha_seen_tuned, alpha_class_tuned,
            occ, calib_size,
            random_state=current_state
        )

        all_results = pd.concat([all_results, results], ignore_index=True)
        all_alpha_class.append(alpha_class_tuned)
        all_alpha_unseen.append(alpha_unseen_tuned)
        all_alpha_seen.append(alpha_seen_tuned)
        all_loss.append(best_loss)
        all_normalized_size.append(best_normalized_size)
        all_joker_waste.append(best_joker_waste)

    print("\nSummary of tuned alphas across experiments:")
    print(f"  Average alpha_class = {np.mean(all_alpha_class):.3f} +/- {np.std(all_alpha_class):.3f}")
    print(f"  Average alpha_unseen = {np.mean(all_alpha_unseen):.3f} +/- {np.std(all_alpha_unseen):.3f}")
    print(f"  Average alpha_seen = {np.mean(all_alpha_seen):.3f} +/- {np.std(all_alpha_seen):.3f}")
    print(f"  Average loss = {np.mean(all_loss):.3f} +/- {np.std(all_loss):.3f}")

    n_results_per_exp = len(all_results) // num_exp
    all_results['tuning_loss'] = np.repeat(all_loss, n_results_per_exp)
    all_results['tuning_normalized_size'] = np.repeat(all_normalized_size, n_results_per_exp)
    all_results['tuning_joker_waste'] = np.repeat(all_joker_waste, n_results_per_exp)

    return all_results, all_alpha_class, all_alpha_unseen, all_alpha_seen


###################
# Run and save    #
###################

results, alpha_class_list, alpha_unseen_list, alpha_seen_list = run_experiment(
    n_ref, n_test, num_exp, batch_num
)

header_df = pd.DataFrame({
    "theta": [theta],
    "n_ref": [n_ref],
    "n_test": [n_test],
    "batch_num": [batch_num],
    "alpha_total": [alpha_total],
    "alpha_class_avg": [np.mean(alpha_class_list)],
    "alpha_unseen_avg": [np.mean(alpha_unseen_list)],
    "alpha_seen_avg": [np.mean(alpha_seen_list)],
    "calib_num": [calib_num],
    "lambda_weight": [lambda_weight],
    "tuning_method": [tuning_method],
    "tuning_method_flag": [tuning_method_flag]
})

header_df_expanded = pd.concat([header_df] * len(results), ignore_index=True)
output_df = pd.concat([header_df_expanded, results], axis=1)
output_df.to_csv(output_file, index=False)

print(f"\nFinished saving results to:\n{output_file}")
print(f"  n_ref={n_ref}, n_test={n_test}, alpha_total={alpha_total}, "
      f"calib_size={calib_size}, lambda_weight={lambda_weight}, "
      f"batch_num={batch_num}")
