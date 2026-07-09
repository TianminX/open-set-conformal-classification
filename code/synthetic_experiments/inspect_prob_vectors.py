"""
Diagnostic script to inspect predicted probability vectors from
OpenSetKNN (Good-Turing) vs OpenSetKNNwithMLPOpenMax (Hybrid)
vs OpenSetKNNOpenMax (pure OpenMax) vs OpenSetKNNwithGTOpenMaxHybrid (GT-anchored).

Uses the same data generation and classifier setup as the experiments.
Run from the synthetic_experiments/ directory.
"""

import numpy as np
import sys
import os
from sklearn.preprocessing import LabelEncoder

os.environ["LOKY_MAX_CPU_COUNT"] = "20"

sys.path.insert(0, os.path.abspath('../third_party'))
import arc
from arc import black_boxes

sys.path.insert(0, '../cgtc/')
from distributions_x import ShiftedNormal
from distributions_y import DirichletProcess

# ============================================================
# Parameters — match your experiment settings
# ============================================================
theta = 600          # DP concentration (try 100, 500, 1000)
n_ref = 2000
n_test = 500          # small for inspection
calib_size = 0.1
batch_num = 1
n_neighbors = 5
num_features = 3
sigma = 0.000005

np.random.seed(batch_num)
current_state = batch_num * 1000

# ============================================================
# Generate data (same as experiments)
# ============================================================
label_dist = DirichletProcess(theta=theta)
feature_dist = ShiftedNormal(num_features, sigma)


class DataDistribution_1:
    def __init__(self, label_dist, feature_dist):
        self.label_dist = label_dist
        self.feature_dist = feature_dist

    def sample(self, n, random_state=None):
        Y = self.label_dist.sample(n, random_state=random_state)
        X = self.feature_dist.sample(Y, random_state=random_state)
        return X, Y


data_dist = DataDistribution_1(label_dist, feature_dist)
X, Y = data_dist.sample(n_ref + n_test, random_state=current_state)

# DirichletProcess samples labels from Uniform(0,1) producing floats.
# The conformal_methods pipeline uses LabelEncoder to convert to ints before
# fitting sklearn classifiers. We do the same here.
le = LabelEncoder()
Y_encoded = le.fit_transform(Y)

indices = np.arange(len(X))
np.random.seed(current_state)
np.random.shuffle(indices)
X_ref, Y_ref = X[indices[:n_ref]], Y_encoded[indices[:n_ref]]
X_test, Y_test = X[indices[n_ref:n_ref + n_test]], Y_encoded[indices[n_ref:n_ref + n_test]]

# Split ref into train / calib
calib_num = int(n_ref * calib_size)
train_num = n_ref - calib_num
unique_ref, counts_ref = np.unique(Y_ref, return_counts=True)
singletons = np.sum(counts_ref == 1)

print(f"theta={theta}, n_ref={n_ref}, n_test={n_test}")
print(f"Unique ref labels: {len(unique_ref)}")
print(f"Singletons in ref: {singletons}")

# Identify seen vs unseen test labels
seen_labels = set(unique_ref)
unseen_mask = np.array([y not in seen_labels for y in Y_test])
print(f"Test: {n_test} points, {unseen_mask.sum()} unseen, {(~unseen_mask).sum()} seen")

# ============================================================
# Build classifiers (same params as experiments)
# ============================================================

# 1. OpenSetKNN (Good-Turing baseline)
classifier_gt = black_boxes.OpenSetKNN(
    calibrate=False,
    n_neighbors=n_neighbors,
    weights='distance',
    algorithm='auto',
    leaf_size=30,
    p=2,
    metric='minkowski',
    n_jobs=-1,
    clip_proba_factor=1e-20,
    noise_scale=1e-6,
)

# 2. OpenSetKNNwithMLPOpenMax (Hybrid)
classifier_hybrid = black_boxes.OpenSetKNNwithMLPOpenMax(
    calibrate=False,
    n_neighbors=n_neighbors,
    weights='distance',
    algorithm='auto',
    leaf_size=30,
    p=2,
    metric='minkowski',
    n_jobs=-1,
    clip_proba_factor=1e-20,
    noise_scale=1e-6,
    hidden_layer_sizes=(64, 32),
    activation='relu',
    max_iter=500,
    mlp_random_state=42,
    tail_size=20,
    alpha_rank=10,
)

# 3. OpenSetKNNOpenMax (pure OpenMax p_open for unseen)
classifier_openmax = black_boxes.OpenSetKNNOpenMax(
    calibrate=False,
    n_neighbors=n_neighbors,
    weights='distance',
    algorithm='auto',
    leaf_size=30,
    p=2,
    metric='minkowski',
    n_jobs=-1,
    clip_proba_factor=1e-20,
    noise_scale=1e-6,
    tail_size=20,
    alpha_rank=10,
)

# 4. OpenSetKNNEVM (EVM margin-based Weibull)
classifier_evm = black_boxes.OpenSetKNNEVM(
    calibrate=False,
    n_neighbors=n_neighbors,
    weights='distance',
    algorithm='auto',
    leaf_size=30,
    p=2,
    metric='minkowski',
    n_jobs=-1,
    clip_proba_factor=1e-20,
    noise_scale=1e-6,
    tail_size=20,
)

# 5. OpenSetKNNwithGTOpenMaxHybrid (GT-anchored + OpenMax modulation)
classifier_gt_openmax = black_boxes.OpenSetKNNwithGTOpenMaxHybrid(
    calibrate=False,
    n_neighbors=n_neighbors,
    weights='distance',
    algorithm='auto',
    leaf_size=30,
    p=2,
    metric='minkowski',
    n_jobs=-1,
    clip_proba_factor=1e-20,
    noise_scale=1e-6,
    hidden_layer_sizes=(64, 32),
    activation='relu',
    max_iter=500,
    mlp_random_state=42,
    tail_size=20,
    alpha_rank=10,
    p_unseen_cap=0.5,
)

# ============================================================
# Split ref into train / calib (like the actual pipeline)
# ============================================================
np.random.seed(current_state + 999)
ref_indices = np.arange(n_ref)
np.random.shuffle(ref_indices)
train_indices = ref_indices[calib_num:]
calib_indices = ref_indices[:calib_num]

X_train, Y_train = X_ref[train_indices], Y_ref[train_indices]
X_calib, Y_calib = X_ref[calib_indices], Y_ref[calib_indices]

train_labels = set(np.unique(Y_train))
calib_labels = set(np.unique(Y_calib))
unseen_calib_labels = calib_labels - train_labels

print(f"\nTrain: {len(Y_train)} points, {len(train_labels)} unique labels")
print(f"Calib: {len(Y_calib)} points, {len(calib_labels)} unique labels")
print(f"Unseen calib labels (in calib but not train): {len(unseen_calib_labels)}")

# ============================================================
# Fit on TRAINING data only (not full ref)
# ============================================================
print("\nFitting classifiers on training data...")
classifier_gt.fit(X_train, Y_train)
classifier_hybrid.fit(X_train, Y_train)
classifier_openmax.fit(X_train, Y_train)
classifier_evm.fit(X_train, Y_train)
classifier_gt_openmax.fit(X_train, Y_train)

# Calibrate the GT+OpenMax hybrid's normaliser using calibration data
classifier_gt_openmax.calibrate_p_open(X_calib)

# ============================================================
# Predict probabilities (with y_calib to trigger unseen allocation)
# ============================================================
# Pass Y_calib so the classifier knows about unseen calib labels
prob_gt = classifier_gt.predict_proba(X_test, y_calib=Y_calib)
prob_hybrid = classifier_hybrid.predict_proba(X_test, y_calib=Y_calib)
prob_openmax = classifier_openmax.predict_proba(X_test, y_calib=Y_calib)
prob_evm = classifier_evm.predict_proba(X_test, y_calib=Y_calib)
prob_gt_openmax = classifier_gt_openmax.predict_proba(X_test, y_calib=Y_calib)

# Also get raw p_open from hybrid, openmax, evm, and gt_openmax
p_open_hybrid = classifier_hybrid._compute_p_open(X_test)
# OpenSetKNNOpenMax._compute_p_open needs p_seen; recompute base KNN probs
_p_seen_openmax = classifier_openmax.model_fit.predict_proba(X_test)
_p_seen_openmax = np.clip(_p_seen_openmax, classifier_openmax.factor / classifier_openmax.num_classes, 1.0)
_p_seen_openmax = _p_seen_openmax / _p_seen_openmax.sum(axis=1)[:, None]
p_open_openmax = classifier_openmax._compute_p_open(X_test, _p_seen_openmax)
# EVM._compute_p_open also needs p_seen
_p_seen_evm = classifier_evm.model_fit.predict_proba(X_test)
_p_seen_evm = np.clip(_p_seen_evm, classifier_evm.factor / classifier_evm.num_classes, 1.0)
_p_seen_evm = _p_seen_evm / _p_seen_evm.sum(axis=1)[:, None]
p_open_evm = classifier_evm._compute_p_open(X_test, _p_seen_evm)
p_open_gt_openmax = classifier_gt_openmax._compute_p_open(X_test)

print(f"\nProb matrix shape — GT: {prob_gt.shape}, Hybrid: {prob_hybrid.shape}, OpenMax: {prob_openmax.shape}, EVM: {prob_evm.shape}, GT+OpenMax: {prob_gt_openmax.shape}")
print(f"Full classes (GT): {len(classifier_gt.full_classes)}")
print(f"Full classes (Hybrid): {len(classifier_hybrid.full_classes)}")
print(f"Full classes (OpenMax): {len(classifier_openmax.full_classes)}")
print(f"Full classes (EVM): {len(classifier_evm.full_classes)}")
print(f"Full classes (GT+OpenMax): {len(classifier_gt_openmax.full_classes)}")
num_unseen_in_model = len(classifier_gt.full_classes) - len(classifier_gt.classes_)
print(f"Unseen classes in model: {num_unseen_in_model}")
p_gt_value = (1 + classifier_gt_openmax.n_singletons) / (1 + classifier_gt_openmax.n_train)
print(f"Good-Turing anchor p_gt: {p_gt_value:.6f}")
print(f"Mean p_open (calib): {classifier_gt_openmax.mean_p_open_calib_:.6f}")

# ============================================================
# Inspect individual test points
# ============================================================
# Get unseen indices in the full class array
unseen_class_mask_gt = np.ones(len(classifier_gt.full_classes), dtype=bool)
for c in classifier_gt.classes_:
    idx = np.where(classifier_gt.full_classes == c)[0][0]
    unseen_class_mask_gt[idx] = False

unseen_class_mask_hybrid = np.ones(len(classifier_hybrid.full_classes), dtype=bool)
for c in classifier_hybrid.classes_:
    idx = np.where(classifier_hybrid.full_classes == c)[0][0]
    unseen_class_mask_hybrid[idx] = False

unseen_class_mask_om = np.ones(len(classifier_openmax.full_classes), dtype=bool)
for c in classifier_openmax.classes_:
    idx = np.where(classifier_openmax.full_classes == c)[0][0]
    unseen_class_mask_om[idx] = False

unseen_class_mask_evm = np.ones(len(classifier_evm.full_classes), dtype=bool)
for c in classifier_evm.classes_:
    idx = np.where(classifier_evm.full_classes == c)[0][0]
    unseen_class_mask_evm[idx] = False

unseen_class_mask_gto = np.ones(len(classifier_gt_openmax.full_classes), dtype=bool)
for c in classifier_gt_openmax.classes_:
    idx = np.where(classifier_gt_openmax.full_classes == c)[0][0]
    unseen_class_mask_gto[idx] = False

print("\n" + "=" * 80)
print("PROBABILITY VECTOR DIAGNOSTICS")
print("=" * 80)

for i in range(min(n_test, 10)):
    true_label = Y_test[i]
    is_unseen = unseen_mask[i]

    print(f"\n--- Test point {i} | True label: {true_label} | {'UNSEEN' if is_unseen else 'SEEN'} ---")

    # GT
    p_gt = prob_gt[i]
    gt_seen_probs = p_gt[~unseen_class_mask_gt]
    gt_unseen_probs = p_gt[unseen_class_mask_gt]
    gt_top5 = np.sort(p_gt)[::-1][:5]

    print(f"  [GT]     p_unseen = {1 - gt_seen_probs.sum():.6f} (via Good-Turing)")
    print(f"           sum(seen) = {gt_seen_probs.sum():.6f}, sum(unseen) = {gt_unseen_probs.sum():.6f}")
    print(f"           max(seen) = {gt_seen_probs.max():.6f}, min(seen) = {gt_seen_probs.min():.2e}")
    print(f"           max(unseen) = {gt_unseen_probs.max():.2e}, min(unseen) = {gt_unseen_probs.min():.2e}")
    print(f"           top-5 probs: {gt_top5}")

    # Hybrid
    p_hy = prob_hybrid[i]
    hy_seen_probs = p_hy[~unseen_class_mask_hybrid]
    hy_unseen_probs = p_hy[unseen_class_mask_hybrid]
    hy_top5 = np.sort(p_hy)[::-1][:5]

    print(f"  [Hybrid] p_open (raw OpenMax) = {p_open_hybrid[i]:.6f}")
    print(f"           sum(seen) = {hy_seen_probs.sum():.6f}, sum(unseen) = {hy_unseen_probs.sum():.6f}")
    print(f"           max(seen) = {hy_seen_probs.max():.6f}, min(seen) = {hy_seen_probs.min():.2e}")
    print(f"           max(unseen) = {hy_unseen_probs.max():.2e}, min(unseen) = {hy_unseen_probs.min():.2e}")
    print(f"           top-5 probs: {hy_top5}")

    # OpenMax (pure)
    p_om = prob_openmax[i]
    om_seen_probs = p_om[~unseen_class_mask_om]
    om_unseen_probs = p_om[unseen_class_mask_om]
    om_top5 = np.sort(p_om)[::-1][:5]

    print(f"  [OpenMax] p_open (raw) = {p_open_openmax[i]:.6f}")
    print(f"           sum(seen) = {om_seen_probs.sum():.6f}, sum(unseen) = {om_unseen_probs.sum():.6f}")
    print(f"           max(seen) = {om_seen_probs.max():.6f}, min(seen) = {om_seen_probs.min():.2e}")
    print(f"           max(unseen) = {om_unseen_probs.max():.2e}, min(unseen) = {om_unseen_probs.min():.2e}")
    print(f"           top-5 probs: {om_top5}")

    # EVM
    p_ev = prob_evm[i]
    ev_seen_probs = p_ev[~unseen_class_mask_evm]
    ev_unseen_probs = p_ev[unseen_class_mask_evm]
    ev_top5 = np.sort(p_ev)[::-1][:5]

    print(f"  [EVM]    p_open (raw) = {p_open_evm[i]:.6f}")
    print(f"           sum(seen) = {ev_seen_probs.sum():.6f}, sum(unseen) = {ev_unseen_probs.sum():.6f}")
    print(f"           max(seen) = {ev_seen_probs.max():.6f}, min(seen) = {ev_seen_probs.min():.2e}")
    print(f"           max(unseen) = {ev_unseen_probs.max():.2e}, min(unseen) = {ev_unseen_probs.min():.2e}")
    print(f"           top-5 probs: {ev_top5}")

    # GT+OpenMax
    p_gto = prob_gt_openmax[i]
    gto_seen_probs = p_gto[~unseen_class_mask_gto]
    gto_unseen_probs = p_gto[unseen_class_mask_gto]
    gto_top5 = np.sort(p_gto)[::-1][:5]
    ratio_i = p_open_gt_openmax[i] / classifier_gt_openmax.mean_p_open_calib_

    print(f"  [GT+OM]  p_open={p_open_gt_openmax[i]:.6f}, ratio={ratio_i:.4f}, p_unseen={gto_unseen_probs.sum():.6f}")
    print(f"           sum(seen) = {gto_seen_probs.sum():.6f}, sum(unseen) = {gto_unseen_probs.sum():.6f}")
    print(f"           max(seen) = {gto_seen_probs.max():.6f}, min(seen) = {gto_seen_probs.min():.2e}")
    print(f"           top-5 probs: {gto_top5}")

# ============================================================
# Summary statistics across all test points
# ============================================================
print("\n" + "=" * 80)
print("SUMMARY ACROSS ALL TEST POINTS")
print("=" * 80)

gt_total_unseen = prob_gt[:, unseen_class_mask_gt].sum(axis=1)
hy_total_unseen = prob_hybrid[:, unseen_class_mask_hybrid].sum(axis=1)
om_total_unseen = prob_openmax[:, unseen_class_mask_om].sum(axis=1)
ev_total_unseen = prob_evm[:, unseen_class_mask_evm].sum(axis=1)
gto_total_unseen = prob_gt_openmax[:, unseen_class_mask_gto].sum(axis=1)

print(f"\nGood-Turing total unseen prob:  mean={gt_total_unseen.mean():.6f}, "
      f"std={gt_total_unseen.std():.6f}, min={gt_total_unseen.min():.6f}, max={gt_total_unseen.max():.6f}")
print(f"Hybrid p_open (raw):            mean={p_open_hybrid.mean():.6f}, "
      f"std={p_open_hybrid.std():.6f}, min={p_open_hybrid.min():.6f}, max={p_open_hybrid.max():.6f}")
print(f"Hybrid total unseen prob:       mean={hy_total_unseen.mean():.6f}, "
      f"std={hy_total_unseen.std():.6f}, min={hy_total_unseen.min():.6f}, max={hy_total_unseen.max():.6f}")
print(f"OpenMax p_open (raw):           mean={p_open_openmax.mean():.6f}, "
      f"std={p_open_openmax.std():.6f}, min={p_open_openmax.min():.6f}, max={p_open_openmax.max():.6f}")
print(f"OpenMax total unseen prob:      mean={om_total_unseen.mean():.6f}, "
      f"std={om_total_unseen.std():.6f}, min={om_total_unseen.min():.6f}, max={om_total_unseen.max():.6f}")
print(f"EVM p_open (raw):               mean={p_open_evm.mean():.6f}, "
      f"std={p_open_evm.std():.6f}, min={p_open_evm.min():.6f}, max={p_open_evm.max():.6f}")
print(f"EVM total unseen prob:          mean={ev_total_unseen.mean():.6f}, "
      f"std={ev_total_unseen.std():.6f}, min={ev_total_unseen.min():.6f}, max={ev_total_unseen.max():.6f}")
print(f"GT+OpenMax total unseen prob:   mean={gto_total_unseen.mean():.6f}, "
      f"std={gto_total_unseen.std():.6f}, min={gto_total_unseen.min():.6f}, max={gto_total_unseen.max():.6f}")

# Breakdown by seen vs unseen test points
if unseen_mask.any():
    print(f"\n  For UNSEEN test points ({unseen_mask.sum()}):")
    print(f"    GT     unseen prob: mean={gt_total_unseen[unseen_mask].mean():.6f}")
    print(f"    Hybrid unseen prob: mean={hy_total_unseen[unseen_mask].mean():.6f}")
    print(f"    OpenMax unseen prob: mean={om_total_unseen[unseen_mask].mean():.6f}")
    print(f"    EVM    unseen prob: mean={ev_total_unseen[unseen_mask].mean():.6f}")
    print(f"    GT+OM  unseen prob: mean={gto_total_unseen[unseen_mask].mean():.6f}")
    print(f"    Hybrid p_open:      mean={p_open_hybrid[unseen_mask].mean():.6f}")
    print(f"    OpenMax p_open:     mean={p_open_openmax[unseen_mask].mean():.6f}")
    print(f"    EVM    p_open:      mean={p_open_evm[unseen_mask].mean():.6f}")
    print(f"    GT+OM  p_open:      mean={p_open_gt_openmax[unseen_mask].mean():.6f}")
if (~unseen_mask).any():
    print(f"\n  For SEEN test points ({(~unseen_mask).sum()}):")
    print(f"    GT     unseen prob: mean={gt_total_unseen[~unseen_mask].mean():.6f}")
    print(f"    Hybrid unseen prob: mean={hy_total_unseen[~unseen_mask].mean():.6f}")
    print(f"    OpenMax unseen prob: mean={om_total_unseen[~unseen_mask].mean():.6f}")
    print(f"    EVM    unseen prob: mean={ev_total_unseen[~unseen_mask].mean():.6f}")
    print(f"    GT+OM  unseen prob: mean={gto_total_unseen[~unseen_mask].mean():.6f}")
    print(f"    Hybrid p_open:      mean={p_open_hybrid[~unseen_mask].mean():.6f}")
    print(f"    OpenMax p_open:     mean={p_open_openmax[~unseen_mask].mean():.6f}")
    print(f"    EVM    p_open:      mean={p_open_evm[~unseen_mask].mean():.6f}")
    print(f"    GT+OM  p_open:      mean={p_open_gt_openmax[~unseen_mask].mean():.6f}")
