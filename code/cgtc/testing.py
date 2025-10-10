import numpy as np

from utils import random_choice, dictToList, listToDict
from sklearn.model_selection import KFold
from tqdm import tqdm


def compute_GT_pvalues_testing_new(X_ref, Y_ref, X_test):
    """Compute the Good-Turing p-values, without looking at the features
    Deterministic version
    """
    n = len(Y_ref)
    m = len(X_test)

    # Count the number of species in the reference data that appear exactly once
    label_frequencies = np.fromiter(listToDict(Y_ref).values(), dtype=int)
    M1 = np.sum(label_frequencies==1)

    # Compute the p-values
    pvals = np.ones((m,))
    for i in range(m):
        pvals[i] = (1.0+M1)/(1.0+n)

    return pvals


def compute_XGT_pvalues_testing_new(X_ref, Y_ref, X_test, occ):
    """Compute the Good-Turing p-values, using the features via OCC
    """
    n = len(Y_ref)
    m = len(X_test)

    # Count the number of species in the reference data that appear exactly once
    ref_frequency = np.array([np.sum(Y_ref==y) for y in Y_ref])
    idx_train = np.where(ref_frequency>1)[0]
    idx_cal = np.where(ref_frequency==1)[0]

    # Train the OCC
    occ.fit(X_ref[idx_train])

    # Apply the occ to compute scores
    scores_ref = -occ.score_samples(X_ref)
    scores_cal = scores_ref[idx_cal]
    scores_test = -occ.score_samples(X_test)

    # Compute the p-values
    pvals = np.ones((m,))
    for i in range(m):
        tmp = np.sum(scores_cal <= scores_test[i])
        pvals[i] = (1.0+tmp) / (1.0+n)
    return pvals


def compute_RGT_pvalues_testing_new(X_ref, Y_ref, X_test):
    """
    Compute the Good-Turing p-values with a randomized component.
    Randomized version using uniformly distributed over {0, 1, ..., M_n^1}.
    """
    n = len(Y_ref)
    m = len(X_test)

    # Count the number of species in the reference data that appear exactly once
    label_frequencies = np.fromiter(listToDict(Y_ref).values(), dtype=int)
    M1 = np.sum(label_frequencies==1)

    # Compute the randomized p-values
    pvals = np.ones((m,))
    for i in range(m):
        U = np.random.randint(0, M1 + 1)  # Randomly choose U from {0, 1, ..., M1}
        pvals[i] = (U + 1) / (n + 1)

    return pvals


def compute_GT_pvalues_testing_old(X_ref, Y_ref, X_test, beta=1.6):
    """
    Compute the Good-Turing p-values for testing H0^old: Y_{n+1} ∈ Y_ref,
    using the formula ψ_old = max_{f in K} ((M_n^{f+1} + f+1) / (n + 1)),
    where K is the set of observed label frequencies in Y_ref,
    and for each frequency f, M_n^f = (# labels with frequency f) * f.

    If beta is None, uses optimal (though not theoretically valid) weights proportional to candidate values.
    Otherwise, uses weights = (1 / f ** beta) / normalization_factor.
    This p-value is computed deterministically and applied to each test point.
    """

    n = len(Y_ref)
    m = len(X_test)

    # Compute frequency counts for each label.
    # Assume listToDict(Y_ref) returns a dictionary {label: frequency}
    freq_dict = listToDict(Y_ref)
    frequencies = list(freq_dict.values())

    # K: unique frequency levels observed
    unique_freq = np.unique(frequencies)

    if beta is None:
        # When weights are proportional to candidate values,
        # all weighted candidates equal sum(candidate_values)
        candidate_values = []
        for f in unique_freq:
            # Count the number of labels that appear exactly f times.
            num_labels_at_f = sum(1 for val in frequencies if val == f)
            num_labels_at_f1 = sum(1 for val in frequencies if val == f + 1)
            # M_n^f1: total number of points from labels with frequency f+1.
            M_nf1 = num_labels_at_f1 * (f + 1)

            candidate = (M_nf1 + f + 1) / (n + 1)
            candidate_values.append(candidate)

        # Since all weighted candidates are equal when using optimal weights,
        # psi_old is just the sum
        psi_old = sum(candidate_values)

    else:
        # Original beta-weighted approach
        # Calculate normalization factor: sum(1/i^beta)
        normalization_factor = sum(1 / f ** beta for f in range(1, n + 1))

        candidate_values = []
        for f in unique_freq:
            # Count the number of labels that appear exactly f times.
            num_labels_at_f = sum(1 for val in frequencies if val == f)
            num_labels_at_f1 = sum(1 for val in frequencies if val == f + 1)
            # M_n^f1: total number of points from labels with frequency f+1.
            M_nf1 = num_labels_at_f1 * (f + 1)

            # Apply inverse beta weighting (1/f^beta) normalized
            weight = (1 / f ** beta) / normalization_factor

            # Calculate candidate with weighting scheme
            candidate = (M_nf1 + f + 1) / (n + 1) / weight

            candidate_values.append(candidate)

        # psi_old is the maximum over all frequency levels.
        psi_old = max(candidate_values)

    pvals = np.ones((m,))
    for i in range(m):
        pvals[i] = psi_old

    return pvals



def compute_RGT_pvalues_testing_old(X_ref, Y_ref, X_test, beta=1.6):
    """
    Compute the Good-Turing p-values for testing H0^old: Y_{n+1} ∈ Y_ref,
    using the formula ψ_old = max_{f in K} ((U+1) / (n + 1)),
    where U is a uniform rv in (0, M_n^{f+1} + f)
    where K is the set of observed label frequencies in Y_ref,
    and for each frequency f, M_n^f = (# labels with frequency f) * f.

    If beta is None, uses optimal (though not theoretically valid) weights proportional to candidate_p_value.
    Otherwise, uses weights = (1 / f ** beta) / normalization_factor.
    """

    n = len(Y_ref)
    m = len(X_test)

    # Compute frequency counts for each label.
    # Assume listToDict(Y_ref) returns a dictionary {label: frequency}
    freq_dict = listToDict(Y_ref)
    frequencies = list(freq_dict.values())

    # K: unique frequency levels observed
    unique_freq = np.unique(frequencies)

    # Calculate normalization factor only if beta is not None
    if beta is not None:
        normalization_factor = sum(1 / f ** beta for f in unique_freq)

    pvals = np.ones((m,))
    for i in range(m):
        if beta is None:
            # When weights are proportional to candidate_p_values,
            # all weighted candidates equal sum(candidate_p_values)
            candidate_p_values = []
            for f in unique_freq:
                # Count the number of labels that appear exactly f+1 times.
                num_labels_at_f = sum(1 for val in frequencies if val == f)
                num_labels_at_f1 = sum(1 for val in frequencies if val == f + 1)

                # M_n^f1: total number of points from labels with frequency f+1.
                M_nf1 = num_labels_at_f1 * (f + 1)
                U = np.random.randint(0, M_nf1 + f + 1)  # Randomly choose U from {0, 1, ..., M_nf1+f}

                candidate_p_value = (U + 1) / (n + 1)
                candidate_p_values.append(candidate_p_value)

            # Since all weighted candidates are equal, psi_old is just the sum
            psi_old = sum(candidate_p_values)

        else:
            # Original beta-weighted approach
            weighted_candidate_values = []
            for f in unique_freq:
                # Count the number of labels that appear exactly f+1 times.
                num_labels_at_f = sum(1 for val in frequencies if val == f)
                num_labels_at_f1 = sum(1 for val in frequencies if val == f + 1)

                # M_n^f1: total number of points from labels with frequency f+1.
                M_nf1 = num_labels_at_f1 * (f + 1)
                U = np.random.randint(0, M_nf1 + f + 1)  # Randomly choose U from {0, 1, ..., M_nf1+f}

                weight = (1 / f ** beta) / normalization_factor

                candidate_p_value = (U + 1) / (n + 1)
                weighted_candidate = candidate_p_value / weight

                weighted_candidate_values.append(weighted_candidate)

            # psi_old is the maximum over all frequency levels.
            psi_old = max(weighted_candidate_values)

        pvals[i] = psi_old

    return pvals


def select_beta_cv(X_ref, Y_ref, betas=np.linspace(1.0, 2.0, 21), cv=5, randomized=True):
    """
    Cross-validate beta ∈ betas to minimize mean p-value on held-out folds.

    Parameters
    ----------
    X_ref, Y_ref : reference features & labels
    betas        : array-like of candidate betas
    cv           : number of folds
    randomized   : if True, use compute_RGT_... else compute_GT_...

    Returns
    -------
    best_beta, best_score
    """
    tqdm.write(f"[CV] Starting beta cross-validation over {len(betas)} values using {cv}-fold...")
    kf = KFold(n_splits=cv, shuffle=True, random_state=0)
    best_beta, best_score = None, np.inf

    for beta in betas:
        fold_scores = []
        # tqdm.write(f"[CV] Testing beta = {beta:.3f}")
        for fold_id, (train_idx, test_idx) in enumerate(kf.split(X_ref)):
            X_train = X_ref[train_idx]
            Y_train = [Y_ref[i] for i in train_idx]
            X_val = X_ref[test_idx]

            if randomized:
                pvals = compute_RGT_pvalues_testing_old(X_train, Y_train, X_val, beta=beta)
            else:
                pvals = compute_GT_pvalues_testing_old(X_train, Y_train, X_val, beta=beta)

            fold_score = pvals.mean()
            fold_scores.append(fold_score)
            # tqdm.write(f"  Fold {fold_id + 1}/{cv} → Mean p-value: {fold_score:.6f}")

        mean_score = np.mean(fold_scores)
        tqdm.write(f"[CV] Beta = {beta:.3f}, Average CV p-value = {mean_score:.6f}")

        if mean_score < best_score:
            best_score, best_beta = mean_score, beta

    tqdm.write(f"[CV] Best beta selected: {best_beta:.3f} with score {best_score:.6f}")
    return best_beta, best_score







