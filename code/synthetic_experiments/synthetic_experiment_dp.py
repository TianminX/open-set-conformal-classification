import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import LocalOutlierFactor
from tqdm import tqdm
from functools import partial
import sys
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "20"

sys.path.insert(0, os.path.abspath('../third_party'))
import arc 
from arc import black_boxes
from arc import methods

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
    print("Usage: python experiment_dp_alpha_tune.py theta n_ref batch_num calib_num alpha_total lambda_weight")
    quit()

# 1) Parse command-line arguments
theta = int(sys.argv[1]) # DP concentration parameter
n_ref = int(sys.argv[2]) # number of training and calibration samples 
n_test = int(sys.argv[3]) # number of test samples
calib_num = int(sys.argv[4])  # number of calibration samples
alpha_total = float(sys.argv[5])  # Total alpha budget
lambda_weight = float(sys.argv[6])     # Tune alpha allocation using loss function, between 0 and 1
batch_num = int(sys.argv[7]) # seed 
tuning_method_flag = int(sys.argv[8])  # data-driven alpha allocation paramter. Which datasplit is used during cross validation? 0 for 'random', 1 for 'bernoulli', -1 for using fixed alpha allocation

# Convert flag to string for internal use
if tuning_method_flag == 0:
    tuning_method = 'random'
elif tuning_method_flag == 1:
    tuning_method = 'bernoulli'

# Optional: not used in experiments. Using fixed alpha allocation. 
elif tuning_method_flag == -1:
    tuning_method = 'fixed'
    alpha_class_fixed = alpha_total / 3
    alpha_unseen_fixed = alpha_total / 3
    alpha_seen_fixed = alpha_total / 3

else:
    print(f"Error: tuning_method must be 0 (random) or 1 (bernoulli) or -1 (using fixed alpha), got '{tuning_method_flag}'")
    quit()



#####################
# Define Parameters #
#####################
# Number of experiments (repetition in each batch, total number is num_exp*num_batch)
num_exp = 5


calib_size = calib_num / n_ref

print(f"alpha tuning_method: {tuning_method} (flag={tuning_method_flag})")

# Using power weights to combine the p-values for testing individual hypothesis for seen hypothesis
# If default_beta is not none, use it. If default_beta is None, use optimal weights
default_beta = 1.6
# If beta_cv is true, use CV to choose beta
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
# Define Output Dir #
#####################



# Output file - Updated filename format 
output_file = (
    f"results/dp_tuned_mixed_labels/"
    f"dp_"
    f"theta{theta}_"
    f"nref{n_ref}_"
    f"ntest{n_test}_"
    f"cs{calib_num}_"
    f"atotal{alpha_total:.3f}_"
    f"lambda{lambda_weight:.2f}_"
    f"tune{tuning_method_flag}_"  # Using the flag (0 or 1) in filename
    f"batch{batch_num}.csv"
)

# Display the filename
print("Output file name: {:s}".format(output_file))

os.makedirs(os.path.dirname(output_file), exist_ok=True)


#####################
# Define Methods    #
##################### 

# Method to be used for tuning - using random splitting full as specified
methods_list_tuning = {
    'Method (random splitting full)': get_preliminary_sets_naive_full,
}

# All methods to be evaluated after tuning
methods_list = {
    'Method (random splitting)': get_preliminary_sets_naive,
    'Method (benchmark)': get_preliminary_sets_benchmark,
    'Method (Bernoulli)': get_preliminary_sets_Bernoulli,
    'Method (Bernoulli benchmark)': get_preliminary_sets_Bernoulli
  }


# Parameters for data distribution
# dimension of feature X
num_features = 3
# variance of the shifted normal of the feature distribution (around labels from DP)
sigma = 0.000005

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
occ = LocalOutlierFactor(n_neighbors=1, novelty=True)



######################################
# Auxiliary functions and Data Model #
######################################

def calib_prob_real(frequency, exp_prop=calib_size, freq_one_prop=None):
    """
    Given a positive integer 'frequency' representing the frequency count of a label,
    returns the probability of that label being selected into the calibration set.

    Args:
        frequency: Frequency count of the label
        exp_prop: Base expected proportion (calib_size)
        freq_one_prop: Proportion of data points with frequency-1 labels
    """
    if frequency == 0:
        return 0.0
    elif frequency == 1:
        return 0.0
    else:
        # If freq_one_prop is provided, adjust the probability to maintain expected calibration size
        if freq_one_prop is not None and freq_one_prop < 1.0:
            # Adjusted probability to compensate for excluded frequency-1 labels
            # We need: adjusted_prob * (1 - freq_one_prop) = exp_prop
            adjusted_prob = exp_prop / (1 - freq_one_prop)
            # Cap at 1.0 to ensure it's a valid probability
            prob_calib = min(adjusted_prob, 1.0)
        else:
            # Fallback to base proportion if no adjustment needed
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

    # Set random seed for reproducibility
    if random_state is not None:
        np.random.seed(random_state)

    # Randomly shuffle indices
    indices = np.arange(total_samples)
    np.random.shuffle(indices)

    # Split indices for reference and testing
    ref_indices = indices[:n_ref]
    test_indices = indices[n_ref:n_ref + n_test]

    # Create reference (training + calibration) and test datasets
    X_ref, Y_ref = X[ref_indices], Y[ref_indices]
    X_test, Y_test = X[test_indices], Y[test_indices]

    return X_ref, Y_ref, X_test, Y_test


def analyze_data(X_ref, Y_ref, X_test, Y_test, methods_list,
                 alpha_unseen, alpha_seen, alpha_class,  # Updated parameters
                 occ, classifier, calib_size,
                 random_state=2024):
    """
    Analyze the data using specified methods and evaluate prediction sets.
    """

    # Calculate proportion of frequency-1 labels
    freq_one_prop = calculate_freq_one_proportion(Y_ref)
    tqdm.write(f"Proportion of frequency-1 labels in reference data: {freq_one_prop:.3f}")

    # Create adjusted calibration probability function
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
    tqdm.write(f"Proportion of unseen test labels: {prop_unseen:.3f} ({num_unseen}/{len(Y_test)} points)")

    # Also calculate number of unique unseen labels if you're interested
    unique_unseen_labels = np.unique(Y_test[unseen_mask])
    num_unique_unseen = len(unique_unseen_labels)
    tqdm.write(f"Number of unique unseen labels in test: {num_unique_unseen}")

    # Initialize an empty data frame to store results
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
            decoded_prelim_sets = method_function(
                X_ref, Y_ref, X_test,
                alpha=alpha_unseen + alpha_seen + alpha_class,
                black_box=classifier,
                calib_size=calib_size,
                random_state=random_state
            )
        elif method_name == 'Method (Bernoulli benchmark)' or method_name == 'Method (Bernoulli benchmark full)':
            decoded_prelim_sets = method_function(
                X_ref, Y_ref, X_test,
                alpha_prime=alpha_unseen + alpha_seen + alpha_class,
                black_box=classifier,
                calibration_probability=calib_prob_adjusted,  # Use adjusted function
                random_state=random_state
            )
        elif method_name == 'Method (Bernoulli)' or method_name == 'Method (Bernoulli full)' or method_name == 'Method (Bernoulli uniform)':
            decoded_prelim_sets = method_function(
                X_ref, Y_ref, X_test,
                alpha_prime=alpha_class,
                black_box=classifier,
                calibration_probability=calib_prob_adjusted,  # Use adjusted function
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

        # For each p-value approach
        for pvalue_method in ['GT', 'XGT', 'RGT']:
            # Skip p-value computation for the benchmark methods
            if method_name in ['Method (benchmark)', 'Method (Bernoulli benchmark)', 'Method (benchmark full)',
                               'Method (Bernoulli benchmark full)']:
                final_sets = decoded_prelim_sets
            else:
                # Compute final sets with dual testing problem
                final_sets = finalize_prediction_sets(
                    decoded_prelim_sets,
                    X_ref, Y_ref, X_test,
                    pvalue_method,
                    alpha_unseen, alpha_seen,  # Pass both alpha parameters
                    occ=occ,
                    random_state=random_state,
                    beta=best_beta
                )

            # Evaluate the prediction sets
            new_results = evaluate_prediction_sets(final_sets, Y_test, Y_ref, verbose=False)

            # Add the method name to the results
            new_results['method'] = method_name
            new_results['pvalue_method'] = pvalue_method
            new_results['num_unique_labels'] = num_unique_labels

            # Add proportion of unseen test labels to results
            new_results['prop_unseen_test'] = prop_unseen
            new_results['num_unseen_test'] = num_unseen

            # Add alpha values to results
            new_results['alpha_class'] = alpha_class
            new_results['alpha_unseen'] = alpha_unseen
            new_results['alpha_seen'] = alpha_seen

            # Append the results to the DataFrame
            results_df = pd.concat([results_df, new_results])

    return results_df


def run_syn_experiment(n_ref, n_test, num_exp, batch_num):
    """
    Run experiments with alpha tuning for all three alphas.
    """
    np.random.seed(batch_num)


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

        tqdm.write(f"Loop {i + 1}: Number of data points: {len(Y_ref)}")
        tqdm.write(f"Number of unique classes in Y_ref: {len(np.unique(Y_ref))}")

        # Calculate proportion of frequency-1 labels for tuning
        freq_one_prop = calculate_freq_one_proportion(Y_ref)

        # Create adjusted calibration probability function for tuning
        calib_prob_adjusted = partial(calib_prob_real,
                                      exp_prop=calib_size,
                                      freq_one_prop=freq_one_prop)

        # Use the same data for tuning (as in synthetic experiments)
        X_tune, Y_tune = X_ref, Y_ref

        if tuning_method_flag != -1:
            # Tune all three alphas using loss-based approach
            alpha_class_tuned, alpha_unseen_tuned, alpha_seen_tuned, tuning_results = tune_alpha_allocation_loss_all_optimized(
                X_tune, Y_tune,
                alpha_total=alpha_total,
                lambda_weight=lambda_weight,
                n_splits=10,
                alpha_step=0.005,
                classifier=classifier,
                occ=occ,
                calibration_probability=calib_prob_adjusted,  # Use adjusted function
                calib_size=calib_size,
                pvalue_method='XGT',
                random_state=current_state,
                beta=None,
                splitting_method=tuning_method,  # Uses the converted string value
                verbose=(i == 0)  # Only verbose for first experiment
            )
        elif tuning_method_flag == -1:
            alpha_class_tuned = alpha_class_fixed
            alpha_unseen_tuned = alpha_unseen_fixed
            alpha_seen_tuned = alpha_seen_fixed

        # Extract the best loss metrics from tuning results
        best_idx = tuning_results['avg_loss'].idxmin()
        best_loss = tuning_results.loc[best_idx, 'avg_loss']
        best_normalized_size = tuning_results.loc[best_idx, 'avg_normalized_size']
        best_joker_waste = tuning_results.loc[best_idx, 'avg_joker_waste']

        # Analyze with tuned alphas
        results = analyze_data(
            X_ref, Y_ref, X_test, Y_test, methods_list,
            alpha_unseen_tuned, alpha_seen_tuned, alpha_class_tuned,
            occ, classifier, calib_size,
            random_state=current_state
        )

        all_results = pd.concat([all_results, results], ignore_index=True)
        all_alpha_class.append(alpha_class_tuned)
        all_alpha_unseen.append(alpha_unseen_tuned)
        all_alpha_seen.append(alpha_seen_tuned)
        all_loss.append(best_loss)
        all_normalized_size.append(best_normalized_size)
        all_joker_waste.append(best_joker_waste)

    # Print summary of tuned alphas
    print("\nSummary of tuned alphas across experiments:")
    print(f"  Average alpha_class = {np.mean(all_alpha_class):.3f} ± {np.std(all_alpha_class):.3f}")
    print(f"  Average alpha_unseen = {np.mean(all_alpha_unseen):.3f} ± {np.std(all_alpha_unseen):.3f}")
    print(f"  Average alpha_seen = {np.mean(all_alpha_seen):.3f} ± {np.std(all_alpha_seen):.3f}")
    print(f"  Average loss = {np.mean(all_loss):.3f} ± {np.std(all_loss):.3f}")

    # Add tuning metrics to results
    all_results['tuning_loss'] = np.repeat(all_loss, len(all_results) // num_exp)
    all_results['tuning_normalized_size'] = np.repeat(all_normalized_size, len(all_results) // num_exp)
    all_results['tuning_joker_waste'] = np.repeat(all_joker_waste, len(all_results) // num_exp)

    return all_results, all_alpha_class, all_alpha_unseen, all_alpha_seen



class DataDistribution_1:
    def __init__(self, label_dist, feature_dist):
        self.label_dist = label_dist
        self.feature_dist = feature_dist

    def sample(self, n, random_state=None):
        # Generate labels
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
results, alpha_class_list, alpha_unseen_list, alpha_seen_list = run_syn_experiment(
    n_ref, n_test, num_exp, batch_num
)

# Create header with average tuned alphas
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
    "tuning_method": [tuning_method],  # String name for clarity in results
    "tuning_method_flag": [tuning_method_flag]  # Numeric flag
})

# Replicate header_df so it has as many rows as your results DataFrame
header_df_expanded = pd.concat([header_df] * len(results), ignore_index=True)

output_df = pd.concat([header_df_expanded, results], axis=1)
output_df.to_csv(output_file, index=False)

print(f"Finished saving final results to:\n{output_file}\nParameters:")
print(f"  n_ref={n_ref}, n_test={n_test}, alpha_total={alpha_total}, "
      f"calib_size={calib_size}, lambda_weight={lambda_weight}, "
      f"batch_num={batch_num}")
