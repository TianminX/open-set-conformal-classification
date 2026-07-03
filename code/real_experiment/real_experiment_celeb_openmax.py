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
from conformal_methods import (
    evaluate_prediction_sets,
    get_prediction_sets_openmax,
    get_prediction_sets_openmax_bernoulli,
)

#####################
# Define Parameters #
#####################

# Direct conformalization of off-the-shelf open-set classifiers (OpenMax) on
# the CelebA real data. Real-data counterpart of synthetic_experiment_openmax.py
# and the baseline comparison for real_experiment_celeb_mm_plugin.py.
#
# The classifier outputs K+1 probabilities (K training classes + 1 unknown) and
# standard split conformal is applied over the (K+1)-class space; calibration
# points whose label is unseen in training are scored against the unknown column.
# There is no Good-Turing testing step and no alpha-budget split: a single
# alpha_total is used.
#
# Usage:
#   python real_experiment_celeb_openmax.py n_ref n_test calib_num alpha_total \
#          n_label_total k_top k_bot batch_num

if len(sys.argv) != 9:
    print("Error: incorrect number of parameters.")
    print(
        "Usage: python real_experiment_celeb_openmax.py n_ref n_test calib_num "
        "alpha_total n_label_total k_top k_bot batch_num")
    quit()

# 1) Parse command-line arguments
n_ref = int(sys.argv[1])            # number of training and calibration samples
n_test = int(sys.argv[2])           # number of test samples
calib_num = int(sys.argv[3])        # number of calibration samples
alpha_total = float(sys.argv[4])    # miscoverage probability (single budget)
n_label_total = int(sys.argv[5])    # "uniform" sampling scheme - Number of labels (0 to skip)
k_top = int(sys.argv[6])            # "top_bottom" scheme - keep top k1 celebrities (0 to skip)
k_bot = int(sys.argv[7])            # "top_bottom" scheme - keep bottom k2 celebrities (0 to skip)
batch_num = int(sys.argv[8])        # seed

num_exp = 1  # number of repetition within each batch

# Determine subsampling scheme based on input parameters
if n_label_total > 0:
    subsampling_scheme = "uniform"
    seed = 42  # Random seed for uniform sampling
elif k_top > 0 or k_bot > 0:
    subsampling_scheme = "top_bottom"
else:
    subsampling_scheme = "none"  # Use full data

calib_size = calib_num / n_ref

# Print parsed parameters
print(f"n_ref: {n_ref}")
print(f"n_test: {n_test}")
print(f"calib_num: {calib_num}")
print(f"calib_size: {calib_size}")
print(f"alpha_total: {alpha_total}")
print(f"n_label_total: {n_label_total}")
print(f"k_top: {k_top}")
print(f"k_bot: {k_bot}")
print(f"batch_num: {batch_num}")
print(f"subsampling_scheme: {subsampling_scheme}")


#####################
# Define Output Dir #
#####################

output_file = (
    f"results/celeb_openmax/"
    f"celeb_openmax_"
    f"nref{n_ref}_"
    f"ntest{n_test}_"
    f"cs{calib_num}_"
    f"atotal{alpha_total:.3f}_"
    f"nlabel{n_label_total}_"
    f"ktop{k_top}_"
    f"kbot{k_bot}_"
    f"batch_{batch_num}.csv"
)

print("Output file name: {:s}".format(output_file))
os.makedirs(os.path.dirname(output_file), exist_ok=True)


#####################
# Classifiers       #
#####################

# Number of neighbors in KNN: same as the CGTC real experiment so that the
# seen-class model matches real_experiment_celeb_mm_plugin.py exactly.
n_neighbors = 10

# OpenMax-KNN: same KNN base as the CGTC pipeline (cosine metric on FaceNet
# embeddings) + Weibull revision on per-class centroid distances.
classifier_openmax_knn = black_boxes.OpenMaxKNN(
    n_neighbors=n_neighbors,
    weights='distance',
    algorithm='auto',
    leaf_size=30,
    p=2,
    metric='cosine',
    metric_params=None,
    n_jobs=-1,
    clip_proba_factor=1e-20,
    tail_size=20,
    alpha_rank=None   # revise all classes
)

# Original OpenMax-MLP (Bendale & Boult, 2016): neural network on the FaceNet
# embeddings + Weibull revision on penultimate-layer MAV distances.
classifier_openmax_mlp = black_boxes.OpenMaxMLP(
    hidden_layer_sizes=(256, 128),
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
    'Method (Bernoulli OpenMax-KNN)': classifier_openmax_knn,
    'Method (Bernoulli OpenMax-MLP)': classifier_openmax_mlp,
}


########################
# Auxiliary functions  #
########################

#################### Selective Splitting ############################
def calib_prob_real(frequency, exp_prop=calib_size, freq_one_prop=None):
    """
    Given a positive integer 'frequency' representing the frequency count of a label,
    returns the probability of that label being selected into the calibration set.
    Frequency-1 labels are never calibrated.
    """
    if frequency <= 1:
        return 0.0
    if freq_one_prop is not None and freq_one_prop < 1.0:
        adjusted_prob = exp_prop / (1 - freq_one_prop)
        return min(adjusted_prob, 1.0)
    return exp_prop


def calculate_freq_one_proportion(Y_ref):
    """Calculate the proportion of data points that have frequency-1 labels."""
    unique_labels, counts = np.unique(Y_ref, return_counts=True)
    freq_one_labels = unique_labels[counts == 1]
    freq_one_mask = np.isin(Y_ref, freq_one_labels)
    return np.mean(freq_one_mask)


#################### Sampling techniques ############################

def sample_labels_and_filter(X_full, Y_full, image_names_full, n_label_total, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    unique_labels = np.unique(Y_full)

    if n_label_total > len(unique_labels):
        raise ValueError(f"Requested {n_label_total} labels but only {len(unique_labels)} unique labels available")

    selected_labels = np.random.choice(unique_labels, size=n_label_total, replace=False)

    mask = np.isin(Y_full, selected_labels)

    X_filtered = X_full[mask]
    Y_filtered = Y_full[mask]
    image_names_filtered = image_names_full[mask]

    return X_filtered, Y_filtered, image_names_filtered, selected_labels


def subsample_top_and_bottom_classes(X, Y, image_names, k1, k2):
    """
    Keep only samples corresponding to the top k1 classes with the most images
    and the bottom k2 classes with the fewest images.
    """
    unique_labels, counts = np.unique(Y, return_counts=True)

    sorted_indices = np.argsort(-counts)  # descending order
    sorted_labels = unique_labels[sorted_indices]

    top_k1_labels = set(sorted_labels[:k1]) if k1 > 0 else set()
    bottom_k2_labels = set(sorted_labels[-k2:]) if k2 > 0 else set()

    selected_labels = top_k1_labels.union(bottom_k2_labels)

    mask = np.array([label in selected_labels for label in Y])

    X_sub = X[mask]
    Y_sub = Y[mask]
    names_sub = image_names[mask]

    return X_sub, Y_sub, names_sub


######################### Split and Analyze Data #######################

def split_data(X, Y, image_names, n_ref, n_test, random_state=None):
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

    X_ref, Y_ref, names_ref = X[ref_indices], Y[ref_indices], image_names[ref_indices]
    X_test, Y_test, names_test = X[test_indices], Y[test_indices], image_names[test_indices]

    return X_ref, Y_ref, names_ref, X_test, Y_test, names_test


def analyze_data_openmax(X_ref, Y_ref, X_test, Y_test, methods_list,
                         alpha, calib_size, random_state=2024):
    """
    Run each OpenMax-style method and evaluate prediction sets.

    For each method:
      - Call get_prediction_sets_openmax (which does the train/calib split
        internally) or the Bernoulli variant with the selective split rule.
      - Evaluate with Y_ref (comparable to CGTC metrics).
      - Compute joker_train-specific metrics using Y_train.
      - Compute joker-adjusted set sizes: whenever '?' is in a prediction set,
        charge it the number of calibration-only classes (labels in the
        calibration set but not in training), since the CGTC methods enumerate
        those labels explicitly. This is the exact per-set version of the
        synthetic-experiment accounting
        Size + `Prop ?` * (num_unique_labels - num_unique_labels_train).
    """

    # Reference-level statistics (based on full Y_ref)
    seen_labels_ref = np.unique(Y_ref)
    unseen_mask_ref = ~np.isin(Y_test, seen_labels_ref)
    prop_unseen_ref = np.mean(unseen_mask_ref)
    num_unseen_ref = np.sum(unseen_mask_ref)

    # Compute adjusted calibration probability for Bernoulli methods
    freq_one_prop = calculate_freq_one_proportion(Y_ref)
    tqdm.write(f"Proportion of frequency-1 labels in reference data: {freq_one_prop:.3f}")
    calib_prob_adjusted = partial(calib_prob_real,
                                  exp_prop=calib_size,
                                  freq_one_prop=freq_one_prop)

    results_df = pd.DataFrame()

    for method_name, classifier in methods_list.items():
        tqdm.write(f"Running {method_name}")

        if 'Bernoulli' in method_name:
            prediction_sets, Y_train, Y_calib = get_prediction_sets_openmax_bernoulli(
                X_ref, Y_ref, X_test,
                alpha=alpha, black_box=classifier,
                calibration_probability=calib_prob_adjusted,
                random_state=random_state
            )
        else:
            prediction_sets, Y_train, Y_calib = get_prediction_sets_openmax(
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

        # --- Compute calib-not-train metrics ---
        # Test points whose label is in calibration but not in training
        seen_labels_calib = np.unique(Y_calib)
        calib_not_train_mask = (~np.isin(Y_test, seen_labels_train)) & np.isin(Y_test, seen_labels_calib)
        prop_calib_not_train = np.mean(calib_not_train_mask)
        num_calib_not_train = np.sum(calib_not_train_mask)

        # Coverage for calib-not-train test points (joker covers these)
        calib_not_train_idx = [i for i, m in enumerate(calib_not_train_mask) if m]
        if calib_not_train_idx:
            calib_not_train_coverage = np.mean([
                1 if ('?' in prediction_sets[i] or Y_test[i] in prediction_sets[i])
                else 0
                for i in calib_not_train_idx
            ])
        else:
            calib_not_train_coverage = np.nan

        # --- Joker-adjusted set sizes ---
        # Number of classes that appear in calibration but not in training:
        # these are collapsed into '?' by the direct method but enumerated
        # explicitly by the CGTC methods, so '?' is charged their count.
        num_labels_calib_only = len(np.setdiff1d(seen_labels_calib, seen_labels_train))

        naive_sizes = [len([lab for lab in ps if lab != '?']) for ps in prediction_sets]
        joker_adj_sizes = [
            sz + (num_labels_calib_only if '?' in ps else 0)
            for sz, ps in zip(naive_sizes, prediction_sets)
        ]
        size_joker_adj = np.mean(joker_adj_sizes)

        # Conditional joker-adjusted sizes (seen/unseen relative to Y_ref,
        # matching the seen/unseen split used by evaluate_prediction_sets)
        unseen_idx_ref = [i for i, m in enumerate(unseen_mask_ref) if m]
        seen_idx_ref = [i for i, m in enumerate(unseen_mask_ref) if not m]
        unseen_size_joker_adj = (np.mean([joker_adj_sizes[i] for i in unseen_idx_ref])
                                 if unseen_idx_ref else np.nan)
        seen_size_joker_adj = (np.mean([joker_adj_sizes[i] for i in seen_idx_ref])
                               if seen_idx_ref else np.nan)

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
        new_results['prop_calib_not_train'] = prop_calib_not_train
        new_results['num_calib_not_train'] = int(num_calib_not_train)
        new_results['num_labels_calib_only'] = int(num_labels_calib_only)
        new_results['Coverage (joker_train)'] = coverage_joker_train
        new_results['Unseen Coverage (joker_train)'] = unseen_train_coverage
        new_results['Seen Coverage (joker_train)'] = seen_train_coverage
        new_results['Calib-not-train Coverage (joker_train)'] = calib_not_train_coverage
        new_results['Size (joker_adj)'] = size_joker_adj
        new_results['Unseen Size (joker_adj)'] = unseen_size_joker_adj
        new_results['Seen Size (joker_adj)'] = seen_size_joker_adj

        results_df = pd.concat([results_df, new_results])

    return results_df


def run_real_experiment(X, Y, image_names, n_ref, n_test, num_exp, batch_num):
    """
    Run direct-conformalization experiments with the OpenMax classifiers.
    """
    np.random.seed(batch_num)

    all_results = pd.DataFrame()

    for i in tqdm(range(num_exp)):
        current_state = batch_num * 1000 + i
        X_ref, Y_ref, names_ref, X_test, Y_test, names_test = split_data(
            X, Y, image_names, n_ref, n_test, current_state
        )

        tqdm.write(f"Loop {i + 1}: Number of data points: {len(Y_ref)}")
        tqdm.write(f"Number of unique classes in Y_ref: {len(np.unique(Y_ref))}")

        results = analyze_data_openmax(
            X_ref, Y_ref, X_test, Y_test, methods_list,
            alpha=alpha_total, calib_size=calib_size,
            random_state=current_state
        )

        all_results = pd.concat([all_results, results], ignore_index=True)

    return all_results


########################
# Start Experiments    #
########################

# Load the data
data = np.load("combined_data.npz")
X, Y, image_names = data['X'], data['Y'], data['image_name']

# Subsampling
print("Original dataset size:", len(Y))
print(f"Original number of unique labels: {len(np.unique(Y))}")

if subsampling_scheme == "uniform":
    X, Y, image_names, sampled_labels = sample_labels_and_filter(
        X, Y, image_names,
        n_label_total=n_label_total,
        random_state=seed
    )
    print(f"Uniform sampling: selected {n_label_total} labels")
    print(f"New dataset size: {len(Y)}")
    print(f"Total unique labels = {len(np.unique(Y))}")

    unique_classes, counts = np.unique(Y, return_counts=True)
    print(f"Class count statistics - Min: {counts.min()}, Max: {counts.max()}, Mean: {counts.mean():.2f}")

elif subsampling_scheme == "top_bottom":
    X, Y, image_names = subsample_top_and_bottom_classes(X, Y, image_names, k_top, k_bot)
    print(f"Top-bottom sampling: selected top {k_top} and bottom {k_bot} classes")
    print(f"New dataset size: {len(Y)}")
    print(f"Number of unique classes in the new dataset: {len(np.unique(Y))}")

    unique_classes, counts = np.unique(Y, return_counts=True)
    print(f"All class counts: {np.unique(counts)}")

else:  # subsampling_scheme == "none"
    print("Using full dataset without subsampling")
    print(f"Dataset size: {len(Y)}")
    print(f"Number of unique classes in the dataset: {len(np.unique(Y))}")

    unique_classes, counts = np.unique(Y, return_counts=True)
    print(f"Class count statistics - Min: {counts.min()}, Max: {counts.max()}, Mean: {counts.mean():.2f}")


###################
# Save Results    #
###################

results = run_real_experiment(X, Y, image_names, n_ref, n_test, num_exp, batch_num)

# Create header with experiment parameters
header_df = pd.DataFrame({
    "n_ref": [n_ref],
    "n_test": [n_test],
    "batch_num": [batch_num],
    "alpha_total": [alpha_total],
    "calib_num": [calib_num],
    "n_label_total": [n_label_total],
    "k_top": [k_top],
    "k_bot": [k_bot],
    "method_family": ["openmax_direct"],
})

# Replicate header_df so it has as many rows as the results DataFrame
header_df_expanded = pd.concat([header_df] * len(results), ignore_index=True)

output_df = pd.concat([header_df_expanded, results], axis=1)
output_df.to_csv(output_file, index=False)

print(f"Finished saving final results to:\n{output_file}\nParameters:")
print(f"  n_ref={n_ref}, n_test={n_test}, alpha_total={alpha_total}, "
      f"calib_size={calib_size}, n_label_total={n_label_total}, "
      f"k_top={k_top}, k_bot={k_bot}, batch_num={batch_num}")
