import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
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
)
from distributions_x import ShiftedNormal
from distributions_y import DirichletProcess

#####################
# Define Parameters #
#####################

# Open-set benchmarks on the Dirichlet-process synthetic data: the synthetic
# counterpart of real_experiment_celeb_openset_bench.py (directly
# conformalized open-set classifiers, Algorithm "open-set plug-in" of the
# paper), with the SAME data model, seeds and repetitions as
# synthetic_experiment_dp_mm_plugin.py so that every benchmark row pairs with
# a CGTC+ row: labels from a Dirichlet process with concentration theta,
# features in R^3 Gaussian around the label with standard deviation 1e-4,
# num_exp = 5 repetitions per batch with split seed batch_num * 1000 + i.
# The seen-class component is the same KNN as the CGTC+ pipeline on this data
# (k = 5, Euclidean metric, distance weighting).
#
# Families (methods run in one job):
#   naive : Naive baseline, the feature-blind Good-Turing constant
#           (black_boxes.GTOpenSetKNN, as in synthetic_experiment_gt_knn.py);
#           the mode argument is ignored (output folder dp_bench_naive)
#   occ   : OCC (LOF) raw + recalibrated: LocalOutlierFactor(n_neighbors=1),
#           centered at its inlier value 1 and divided by 2 (as on CelebA)
#   knn1  : KNN-dist (k=1) raw + recalibrated: distance to the nearest
#           training point, divided by sqrt(3) (the largest possible distance
#           between two class centroids, whose coordinates all equal a
#           Uniform(0,1) draw) and clipped to [0, 1]; the raw variant depends
#           on this scale, the recalibrated ones do not (scale: up to the cap;
#           isotonic: rank-based)
# Recalibration mode (see black_boxes.GTRecalOpenSet):
#   scale    - multiplicative p_unk = min(c s, cap)            ("Recal-")
#   center   - subtract the pseudo-seen median, then rescale
#   isotonic - isotonic estimate of P(novel | s), then rescale  ("Iso-")
# In every mode the average unknown probability is anchored at the
# Good-Turing estimate M1/n of the training split.
#
# Output: results/dp_bench_<family>_<mode>/dp_bench_<family>_<mode>_theta..._batchN.csv
#         (results/dp_bench_naive/dp_bench_naive_theta..._batchN.csv for naive)
# Method rows: 'Method (GT-KNN)', 'Method (OCC lof)', 'Method (Recal OCC lof)',
#              'Method (KNN-dist k=1)', 'Method (Recal KNN-dist k=1)'
#
# Usage:
#   python synthetic_experiment_dp_openset_bench.py theta n_ref n_test calib_num \
#          alpha_total batch_num family mode

FAMILIES = ('naive', 'occ', 'knn1')
MODES = ('scale', 'center', 'isotonic')

if len(sys.argv) != 9:
    print("Error: incorrect number of parameters.")
    print("Usage: python synthetic_experiment_dp_openset_bench.py theta n_ref n_test "
          "calib_num alpha_total batch_num family mode")
    print(f"  family in {FAMILIES}; mode in {MODES} (ignored for naive)")
    quit()

theta = int(sys.argv[1])           # DP concentration parameter
n_ref = int(sys.argv[2])           # number of training and calibration samples
n_test = int(sys.argv[3])          # number of test samples
calib_num = int(sys.argv[4])       # number of calibration samples
alpha_total = float(sys.argv[5])   # miscoverage probability (single budget)
batch_num = int(sys.argv[6])       # seed
family = sys.argv[7]               # benchmark family (see FAMILIES)
recal_mode = sys.argv[8]           # recalibration mode (see MODES)

if family not in FAMILIES:
    print(f"Error: family must be one of {FAMILIES}, got '{family}'")
    quit()
if family == 'naive':
    recal_mode = 'none'
elif recal_mode not in MODES:
    print(f"Error: mode must be one of {MODES}, got '{recal_mode}'")
    quit()

# Number of repetitions per batch (total = num_exp * number of batches), as in
# synthetic_experiment_dp_mm_plugin.py
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
print(f"family: {family}")
print(f"recal_mode: {recal_mode}")


#####################
# Define Output Dir #
#####################

tag = "dp_bench_naive" if family == 'naive' else f"dp_bench_{family}_{recal_mode}"
output_file = (
    f"results/{tag}/"
    f"{tag}_"
    f"theta{theta}_"
    f"nref{n_ref}_"
    f"ntest{n_test}_"
    f"cs{calib_num}_"
    f"atotal{alpha_total:.3f}_"
    f"batch{batch_num}.csv"
)

print("Output file name: {:s}".format(output_file))
os.makedirs(os.path.dirname(output_file), exist_ok=True)


#####################
# Data distribution #
#####################

# Same feature model as synthetic_experiment_dp_mm_plugin.py
num_features = 3   # dimension of feature X
sigma = 1e-4       # standard deviation of the shifted normal of the feature distribution


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

# Number of neighbors in KNN: same as the CGTC+ synthetic experiment so that
# the seen-class model matches synthetic_experiment_dp_mm_plugin.py exactly.
n_neighbors = 5

KNN_KW = dict(
    n_neighbors=n_neighbors,
    weights='distance',
    algorithm='auto',
    leaf_size=30,
    p=2,
    metric='minkowski',
    metric_params=None,
    n_jobs=-1,
    clip_proba_factor=1e-20,
)

RECAL_KW = dict(recal_frac=0.2, cap=0.9, random_state=42, mode=recal_mode)

# Scale of the raw KNN-dist score: the largest possible distance between two
# class centroids (all coordinates equal to a Uniform(0,1) draw).
DIST_SCALE = float(np.sqrt(num_features))


def recal(base):
    """Wrap a base open-set classifier in the selected recalibration."""
    return black_boxes.GTRecalOpenSet(base=base, **RECAL_KW)


def make_knn_dist(k_score):
    return black_boxes.KNNDistOpenSet(k_score=k_score, dist_scale=DIST_SCALE, **KNN_KW)


def make_occ():
    # Same LOF as inside the CGTC+ XGT p-values (occ_choices['lof'] of
    # synthetic_experiment_dp_mm_plugin.py), centered and scaled as on CelebA.
    return black_boxes.OCCOpenSet(occ=LocalOutlierFactor(n_neighbors=1, novelty=True),
                                  occ_offset=1.0, occ_scale=2.0, **KNN_KW)


# Methods to evaluate (random internal split only). Method names are the CSV
# row keys read by dp_compare_mm_plugin_vs_openset_bench.R.
if family == 'naive':
    methods_list = {
        'Method (GT-KNN)': black_boxes.GTOpenSetKNN(**KNN_KW),
    }
elif family == 'occ':
    methods_list = {
        'Method (OCC lof)': make_occ(),
        'Method (Recal OCC lof)': recal(make_occ()),
    }
else:  # 'knn1'
    methods_list = {
        'Method (KNN-dist k=1)': make_knn_dist(1),
        'Method (Recal KNN-dist k=1)': recal(make_knn_dist(1)),
    }


########################
# Auxiliary functions  #
########################

def split_data(X, Y, n_ref, n_test, random_state=None):
    """
    Splits the dataset into reference (training + calibration) and testing
    datasets (identical to synthetic_experiment_dp_mm_plugin.py).
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


def analyze_data_bench(X_ref, Y_ref, X_test, Y_test, methods_list,
                       alpha, calib_size, random_state=2024):
    """
    Run every method of the family and evaluate its prediction sets
    (identical to real_experiment_celeb_openset_bench.py):
      - get_prediction_sets_openmax does the train/calib split internally;
      - reference-level metrics via evaluate_prediction_sets (Y_ref);
      - training-level (joker_train) coverage metrics via Y_train;
      - decoded sizes: '?' charged the number of calibration-only classes;
      - raw-classifier diagnostics on the (n_test, K+1) probability matrix.
    """

    # Reference-level statistics (based on full Y_ref)
    seen_labels_ref = np.unique(Y_ref)
    unseen_mask_ref = ~np.isin(Y_test, seen_labels_ref)
    prop_unseen_ref = np.mean(unseen_mask_ref)
    num_unseen_ref = np.sum(unseen_mask_ref)

    results_df = pd.DataFrame()

    for method_name, classifier in methods_list.items():
        tqdm.write(f"Running {method_name}")

        prediction_sets, Y_train, Y_calib, P_test = get_prediction_sets_openmax(
            X_ref, Y_ref, X_test,
            alpha=alpha, black_box=classifier, calib_size=calib_size,
            random_state=random_state, return_probs=True
        )

        # fit() mutates the wrapper before returning its deep copy, so the
        # recalibration diagnostics are available on the original instance.
        is_recal = isinstance(classifier, black_boxes.GTRecalOpenSet)
        if is_recal:
            tqdm.write(
                f"  [{method_name}] mode={classifier.mode}, p_gt={classifier.p_gt_:.4f}, "
                f"s0={classifier.s0_:.4f}, s_bar={classifier.s_bar_:.4f}, "
                f"holdout novelty={classifier.holdout_novelty_rate_:.3f}, "
                f"scale={'fallback-constant' if classifier.scale_ is None else f'{classifier.scale_:.4f}'}")

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

        # --- Raw classifier accuracy (pre-conformal) ---
        # Computed on the (n_test, K+1) probability matrix. Columns 0..K-1 are
        # in np.unique(Y_train) order (LabelEncoder order); column K = unknown.
        top1_acc = top5_acc = rank_median = rank_mean = np.nan
        p_unk_beats_true = p_open_auc = np.nan
        p_open_seen = p_open_calib_only = p_open_novel = np.nan
        if P_test is not None:
            K = len(seen_labels_train)
            p_open_test = P_test[:, K]
            idx_seen = np.where(~unseen_mask_train)[0]
            if len(idx_seen) > 0:
                y_enc = np.searchsorted(seen_labels_train, Y_test[idx_seen])
                p_seen = P_test[idx_seen, :K]
                p_true = p_seen[np.arange(len(idx_seen)), y_enc]
                # rank of the true label among the K seen columns (1 = best)
                ranks = np.sum(p_seen > p_true[:, None], axis=1) + 1
                top1_acc = np.mean(ranks == 1)
                top5_acc = np.mean(ranks <= 5)
                rank_median = np.median(ranks)
                rank_mean = np.mean(ranks)
                p_unk_beats_true = np.mean(p_open_test[idx_seen] > p_true)
            # p_open by test-point group
            calib_only_mask = unseen_mask_train & ~unseen_mask_ref
            if (~unseen_mask_train).sum() > 0:
                p_open_seen = np.mean(p_open_test[~unseen_mask_train])
            if calib_only_mask.sum() > 0:
                p_open_calib_only = np.mean(p_open_test[calib_only_mask])
            if unseen_mask_ref.sum() > 0:
                p_open_novel = np.mean(p_open_test[unseen_mask_ref])
            # AUC of p_open separating not-in-train from seen-in-train
            if 0 < unseen_mask_train.sum() < len(Y_test):
                p_open_auc = roc_auc_score(unseen_mask_train, p_open_test)
            tqdm.write(
                f"  [{method_name}] p_open mean={np.mean(p_open_test):.4f}, "
                f"top1={top1_acc:.3f} (chance={1.0 / K:.4f}), "
                f"top5={top5_acc:.3f}, p_open AUC={p_open_auc:.3f}")

        # --- Joker-adjusted (decoded) set sizes ---
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
        new_results['Top1 Accuracy (train)'] = top1_acc
        new_results['Top5 Accuracy (train)'] = top5_acc
        new_results['True Label Rank (median)'] = rank_median
        new_results['True Label Rank (mean)'] = rank_mean
        new_results['P(p_open > p_true)'] = p_unk_beats_true
        new_results['p_open AUC'] = p_open_auc
        new_results['p_open mean (seen_train)'] = p_open_seen
        new_results['p_open mean (calib_only)'] = p_open_calib_only
        new_results['p_open mean (novel)'] = p_open_novel
        new_results['recal mode'] = classifier.mode if is_recal else 'none'
        new_results['recal p_gt'] = classifier.p_gt_ if is_recal else np.nan
        new_results['recal s0'] = classifier.s0_ if is_recal else np.nan
        new_results['recal s_bar'] = classifier.s_bar_ if is_recal else np.nan
        new_results['recal scale'] = (classifier.scale_
                                      if is_recal and classifier.scale_ is not None
                                      else np.nan)
        new_results['recal holdout novelty'] = (classifier.holdout_novelty_rate_
                                                if is_recal else np.nan)

        results_df = pd.concat([results_df, new_results])

    return results_df


def run_syn_experiment(n_ref, n_test, num_exp, batch_num):
    """
    Run the experiment num_exp times with the same data seeds as
    synthetic_experiment_dp_mm_plugin.py (sample and split seed
    batch_num * 1000 + i).
    """
    np.random.seed(batch_num)

    all_results = pd.DataFrame()

    for i in tqdm(range(num_exp)):
        current_state = batch_num * 1000 + i

        X, Y = data_dist.sample(n_ref + n_test, random_state=current_state)
        X_ref, Y_ref, X_test, Y_test = split_data(X, Y, n_ref, n_test, current_state)

        tqdm.write(f"Loop {i + 1}: Number of data points: {len(Y_ref)}")
        tqdm.write(f"Number of unique classes in Y_ref: {len(np.unique(Y_ref))}")

        results = analyze_data_bench(
            X_ref, Y_ref, X_test, Y_test, methods_list,
            alpha=alpha_total, calib_size=calib_size,
            random_state=current_state
        )

        all_results = pd.concat([all_results, results], ignore_index=True)

    return all_results


###################
# Save Results    #
###################

results = run_syn_experiment(n_ref, n_test, num_exp, batch_num)

# Create header with experiment parameters
header_df = pd.DataFrame({
    "theta": [theta],
    "n_ref": [n_ref],
    "n_test": [n_test],
    "batch_num": [batch_num],
    "alpha_total": [alpha_total],
    "calib_num": [calib_num],
    "method_family": [f"bench_{family}_{recal_mode}"],
})

# Replicate header_df so it has as many rows as the results DataFrame
header_df_expanded = pd.concat([header_df] * len(results), ignore_index=True)

output_df = pd.concat([header_df_expanded, results], axis=1)
output_df.to_csv(output_file, index=False)

print(f"Finished saving final results to:\n{output_file}\nParameters:")
print(f"  theta={theta}, n_ref={n_ref}, n_test={n_test}, alpha_total={alpha_total}, "
      f"calib_size={calib_size}, batch_num={batch_num}, family={family}, mode={recal_mode}")
