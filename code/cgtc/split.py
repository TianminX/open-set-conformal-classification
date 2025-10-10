import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from scipy.stats.mstats import mquantiles
from sklearn.preprocessing import LabelEncoder
import sys
from tqdm import tqdm
import math

from utils import weighted_quantile


import pdb

from arc.classification import ProbabilityAccumulator as ProbAccum


class BernoulliSplitConformal:
    def __init__(self, X, Y, black_box, alpha, calibration_probability, random_state=2020, allow_empty=True, verbose=False):
        """
        Parameters:
          X : array-like, feature matrix.
          Y : array-like, labels.
          black_box : classifier with .fit() and .predict_proba().
          alpha : float, desired miscoverage level.
          calibration_probability : callable, calibration frequency function.
               Given an integer (the frequency of a label in Y), returns a probability in [0,1].
          random_state : int, random seed.
          allow_empty : bool, whether empty sets are allowed.
          verbose : bool, (unused here).
        """
        self.X = X
        self.Y = Y
        self.black_box = black_box
        self.alpha = alpha
        self.allow_empty = allow_empty
        self.calibration_probability = calibration_probability
        self.random_state = random_state
        self.verbose = verbose

        # Unique candidate labels and their counts.
        # self.unique_labels = np.unique(self.Y)
        # self.label_counts = {label: np.sum(self.Y == label) for label in self.unique_labels}
        self.unique_labels, counts = np.unique(self.Y, return_counts=True)
        self.label_counts = dict(zip(self.unique_labels, counts))

        n = len(Y)
        rng = np.random.default_rng(random_state)

        # For each data point i, sample I_i ~ Bernoulli( π( N(Y_i) ) )
        # I is Boolean
        I = np.empty(n, dtype=bool)
        for i in range(n):
            count_i = self.label_counts[self.Y[i]]
            I[i] = (rng.uniform() < self.calibration_probability(count_i))

        # Define calibration and training sets based on I.
        calib_idx = np.where(I)[0]       # indices with I == True
        train_idx = np.where(~I)[0]        # indices with I == False
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_calib, Y_calib = X[calib_idx], Y[calib_idx]
        n2 = len(Y_calib)
        tqdm.write(f"Bernoulli: Size of all data: {n}. Size of calibration set: {n2}. ")

        # Fit the black_box classifier on the training set.
        self.black_box = black_box
        self.black_box.fit(X_train, Y_train)

        # Form prediction sets on calibration data.
        # Pass y_calib so that extra unseen labels (if any) are included.
        p_hat_calib = self.black_box.predict_proba(X_calib, y_calib=Y_calib)
        grey_box = ProbAccum(p_hat_calib)

        epsilon_calib = rng.uniform(low=0.0, high=1.0, size=n2)

        # Here we compute calibration scores (only once) (using the standard ProbAccum procedure)
        alpha_max = grey_box.calibrate_scores(Y_calib, epsilon=epsilon_calib)

        # In standard split conformal, we set scores = alpha - alpha_max.
        # In the weighted version, these scores serve as our conformity values.
        cal_scores = self.alpha - alpha_max  # array of calibration conformity scores

        # Compute baseline unpermuted joint probability (p_(n+1)):
        p_unpermuted = 1.0

        # --- Begin computing candidate-specific weights and calibrated levels ---
        # For each candidate, compute permuted weights and calibrated level.
        self.alpha_calibrated = {}
        for candidate in self.unique_labels:
            permuted_prob = np.empty(n2)
            # Cache computed probability for each unique swap label in Y_calib
            computed_probs = {}
            for j in range(n2):
                swap_label = Y_calib[j]
                if swap_label not in computed_probs:
                    computed_probs[swap_label] = self.compute_prob(Y_train, Y_calib,
                                                                   calib_pi=calibration_probability,
                                                                   test_label=candidate,
                                                                   swap_label=swap_label)
                permuted_prob[j] = computed_probs[swap_label]

            # Compute candidate-specific α-correction using the weighted quantile.
            alpha_correction = weighted_quantile(values = cal_scores,
                                                 quantiles = (1.0 - self.alpha),
                                                 calib_weight=permuted_prob,
                                                 test_weight = p_unpermuted,
                                                 alpha = self.alpha,
                                                 sorted=False)

            # Calibrated level for candidate.
            self.alpha_calibrated[candidate] = self.alpha - alpha_correction
        # --- End candidate-specific calibration ---

        # In BernoulliSplitConformal.__init__, after the loop that computes alpha_calibrated

        # Basic monitoring
        tqdm.write(f"  Alpha: {self.alpha:.4f}")
        tqdm.write(f"  Calibration scores range: [{np.min(cal_scores):.4f}, {np.max(cal_scores):.4f}]")
        tqdm.write(f"  Calibration scores mean: {np.mean(cal_scores):.4f}, std: {np.std(cal_scores):.4f}")
        tqdm.write(f"  Alpha_max range: [{np.min(alpha_max):.4f}, {np.max(alpha_max):.4f}]")
        tqdm.write(f"  Alpha_max mean: {np.mean(alpha_max):.4f}, std: {np.std(alpha_max):.4f}")
        tqdm.write(f"  Number of unique labels: {len(self.unique_labels)}")
        tqdm.write(f"  Number of calibration points: {n2}")

        # Monitor permuted probabilities for each candidate
        # Store statistics for all candidates
        permuted_prob_stats = {}
        for candidate in self.unique_labels:
            permuted_prob = np.empty(n2)
            computed_probs = {}
            for j in range(n2):
                swap_label = Y_calib[j]
                if swap_label not in computed_probs:
                    computed_probs[swap_label] = self.compute_prob(Y_train, Y_calib,
                                                                   calib_pi=calibration_probability,
                                                                   test_label=candidate,
                                                                   swap_label=swap_label)
                permuted_prob[j] = computed_probs[swap_label]

            # Store stats for this candidate
            permuted_prob_stats[candidate] = {
                'sum': np.sum(permuted_prob),
                'mean': np.mean(permuted_prob),
                'std': np.std(permuted_prob),
                'min': np.min(permuted_prob),
                'max': np.max(permuted_prob)
            }

        # Display statistics for first few candidates
        tqdm.write(f"\n  Permuted probability statistics (first 5 candidates):")
        for i, candidate in enumerate(list(self.unique_labels)[:5]):
            stats = permuted_prob_stats[candidate]
            tqdm.write(f"    Candidate {candidate}: sum={stats['sum']:.4f}, mean={stats['mean']:.4f}, "
                       f"std={stats['std']:.4f}, range=[{stats['min']:.4f}, {stats['max']:.4f}], "
                       f"alpha_calib={self.alpha_calibrated[candidate]:.4f}")

        # Overall statistics across all candidates
        all_sums = [stats['sum'] for stats in permuted_prob_stats.values()]
        all_alpha_calib = list(self.alpha_calibrated.values())

        tqdm.write(f"\n  Across all candidates:")
        tqdm.write(f"    Permuted prob sums: mean={np.mean(all_sums):.4f}, std={np.std(all_sums):.4f}, "
                   f"range=[{np.min(all_sums):.4f}, {np.max(all_sums):.4f}]")
        tqdm.write(f"    Alpha calibrated: mean={np.mean(all_alpha_calib):.4f}, std={np.std(all_alpha_calib):.4f}, "
                   f"range=[{np.min(all_alpha_calib):.4f}, {np.max(all_alpha_calib):.4f}]")

        # Check how many candidates have calibrated alpha > 0.01 (problematic)
        high_alpha_count = sum(1 for a in all_alpha_calib if a < 0.01)
        tqdm.write(f"    Candidates with alpha_calibrated < 0.01: {high_alpha_count}/{len(self.unique_labels)}")

        # Check deviation from ideal (sum should be close to n2 for uniform weights)
        tqdm.write(f"    Expected sum for uniform weights: {n2:.1f}")
        tqdm.write(f"    Actual mean sum: {np.mean(all_sums):.4f} (ratio: {np.mean(all_sums) / n2:.4f})")

    def compute_prob(self, Y_train, Y_calib, calib_pi, test_label, swap_label):
        """
        Compute the candidate-specific probability ratio
        (tilde{p}^{(j)} = p^{(j)}/p^{(n+1)}) for a given candidate (test_label)
        and a calibration point (swap_label).

        If test_label == swap_label, no swap is performed and the ratio is 1.

        Parameters:
          Y_train : array-like
              Array of training labels.
          Y_calib : array-like
              Array of calibration labels.
          calib_pi : callable
              Calibration probability function, which takes an integer (frequency count)
              and returns a probability in [0,1].
          test_label : label type (e.g. int, str)
              The candidate label for the test point (i.e. (y_{n+1})).
          swap_label : label type (e.g. int, str)
              The label of the calibration point that is being swapped (i.e. (y_j)).

        Returns:
          float
              The computed ratio (tilde{p}^{(j)}).
        """
        # If the candidate label equals the calibration point's label,
        # the swap has no effect.
        if test_label == swap_label:
            return 1.0

        # Compute frequency counts for test_label.
        f_calib_test = np.sum(Y_calib == test_label)
        f_train_test = np.sum(Y_train == test_label)
        N_test = f_calib_test + f_train_test  # total count of test_label in Y_train and Y_calib

        # Compute frequency counts for swap_label.
        f_calib_swap = np.sum(Y_calib == swap_label)
        f_train_swap = np.sum(Y_train == swap_label)
        N_swap = f_calib_swap + f_train_swap  # total count of swap_label in Y_train and Y_calib

        # Compute the difference factor for p^(j).

        # Compute the factors for swap_label.
        # For swap_label (which is the calibration label y_j), the frequency decreases by one.
        prob_j = (calib_pi(N_swap - 1)) ** (f_calib_swap - 1)
        prob_j *= ((1 - calib_pi(N_swap - 1))) ** (f_train_swap)

        #Note that here prob_j could be 0

        # Compute the factors for test_label.
        # For test_label (which is the candidate label y_{n+1}), the frequency increases by one.
        prob_j *= (calib_pi(N_test + 1)) ** (f_calib_test + 1)
        prob_j *= ((1 - calib_pi(N_test + 1))) ** (f_train_test)

        # Compute the difference factor for p^(n+1).
        # Compute the factors for swap_label.
        # For swap_label (which is the calibration label y_j), the frequency decreases by one.
        prob_new = (calib_pi(N_swap)) ** (f_calib_swap)
        prob_new *= ((1 - calib_pi(N_swap)) ) ** (f_train_swap)

        # Compute the factors for test_label.
        # For test_label (which is the candidate label y_{n+1}), the frequency increases by one.
        prob_new *= (calib_pi(N_test)) ** (f_calib_test)
        prob_new *= ((1 - calib_pi(N_test))) ** (f_train_test)

        # The final candidate-specific ratio is the product of the two ratios.
        return prob_j / prob_new

    def predict(self, X, random_state=2020):
        """
        Predict conformal sets for new data X.

        Parameters:
          X : array-like, new data.
          random_state : int, random seed.

        Returns:
          S_hat : list of prediction sets (each is a list/array of class indices).
        """
        n = X.shape[0]
        rng = np.random.default_rng(random_state)
        epsilon = rng.uniform(low=0.0, high=1.0, size=n)

        # When predicting new data, the union of classes computed during calibration is reused.
        p_hat = self.black_box.predict_proba(X)
        grey_box = ProbAccum(p_hat)

        # Initialize temporary prediction sets as lists.
        S_hat_temp = [[] for _ in range(n)]
        # For each candidate, use its candidate-specific calibrated level.
        for candidate in self.unique_labels:
            candidate_threshold = self.alpha_calibrated[candidate]
            candidate_sets = grey_box.predict_sets(candidate_threshold, epsilon=epsilon, allow_empty=self.allow_empty)
            for i, s in enumerate(candidate_sets):
                # If the candidate appears in the candidate's prediction set, include it.
                if candidate in s:
                    S_hat_temp[i].append(candidate)
        # Order each prediction set according to the ranking in grey_box.order.
        S_hat = []
        for i in range(n):
            ordered = [c for c in grey_box.order[i] if c in S_hat_temp[i]]
            S_hat.append(np.array(ordered))
        return S_hat

class BernoulliSplitConformalFull:
    def __init__(self, X, Y, black_box, alpha, calibration_probability, random_state=2020, allow_empty=True, verbose=False):
        """
        Parameters:
          X : array-like, feature matrix.
          Y : array-like, labels.
          black_box : classifier with .fit() and .predict_proba().
          alpha : float, desired miscoverage level.
          calibration_probability : callable, calibration frequency function.
               Given an integer (the frequency of a label in Y), returns a probability in [0,1].
          random_state : int, random seed.
          allow_empty : bool, whether empty sets are allowed.
          verbose : bool, (unused here).
        """
        self.X = X
        self.Y = Y
        self.black_box = black_box
        self.alpha = alpha
        self.allow_empty = allow_empty
        self.calibration_probability = calibration_probability
        self.random_state = random_state
        self.verbose = verbose

        # Unique candidate labels and their counts.
        # self.unique_labels = np.unique(self.Y)
        # self.label_counts = {label: np.sum(self.Y == label) for label in self.unique_labels}
        self.unique_labels, counts = np.unique(self.Y, return_counts=True)
        self.label_counts = dict(zip(self.unique_labels, counts))

        n = len(Y)
        rng = np.random.default_rng(random_state)

        # For each data point i, sample I_i ~ Bernoulli( π( N(Y_i) ) )
        # I is Boolean
        I = np.empty(n, dtype=bool)
        for i in range(n):
            count_i = self.label_counts[self.Y[i]]
            I[i] = (rng.uniform() < self.calibration_probability(count_i))

        # Define calibration and training sets based on I.
        calib_idx = np.where(I)[0]       # indices with I == True
        train_idx = np.where(~I)[0]        # indices with I == False
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_calib, Y_calib = X[calib_idx], Y[calib_idx]
        n2 = len(Y_calib)
        X_train, Y_train = X, Y
        tqdm.write(f"Bernoulli: Size of all data: {n}. Size of calibration set: {n2}. ")

        # Fit the black_box classifier on the training set.
        self.black_box = black_box
        self.black_box.fit(X_train, Y_train)

        # Form prediction sets on calibration data.
        # Pass y_calib so that extra unseen labels (if any) are included.
        p_hat_calib = self.black_box.predict_proba(X_calib, y_calib=Y_calib)
        grey_box = ProbAccum(p_hat_calib)

        epsilon_calib = rng.uniform(low=0.0, high=1.0, size=n2)

        # Here we compute calibration scores (only once) (using the standard ProbAccum procedure)
        alpha_max = grey_box.calibrate_scores(Y_calib, epsilon=epsilon_calib)

        # In standard split conformal, we set scores = alpha - alpha_max.
        # In the weighted version, these scores serve as our conformity values.
        cal_scores = self.alpha - alpha_max  # array of calibration conformity scores

        # Compute baseline unpermuted joint probability (p_(n+1)):
        p_unpermuted = 1.0

        # --- Begin computing candidate-specific weights and calibrated levels ---
        # For each candidate, compute permuted weights and calibrated level.
        self.alpha_calibrated = {}
        for candidate in self.unique_labels:
            permuted_prob = np.empty(n2)
            # Cache computed probability for each unique swap label in Y_calib
            computed_probs = {}
            for j in range(n2):
                swap_label = Y_calib[j]
                if swap_label not in computed_probs:
                    computed_probs[swap_label] = self.compute_prob(Y_train, Y_calib,
                                                                   calib_pi=calibration_probability,
                                                                   test_label=candidate,
                                                                   swap_label=swap_label)
                permuted_prob[j] = computed_probs[swap_label]

            # Compute candidate-specific α-correction using the weighted quantile.
            alpha_correction = weighted_quantile(values = cal_scores,
                                                 quantiles = (1.0 - self.alpha),
                                                 calib_weight=permuted_prob,
                                                 test_weight = p_unpermuted,
                                                 alpha = self.alpha,
                                                 sorted=False)

            # Calibrated level for candidate.
            self.alpha_calibrated[candidate] = self.alpha - alpha_correction
        # --- End candidate-specific calibration ---

    def compute_prob(self, Y_train, Y_calib, calib_pi, test_label, swap_label):
        """
        Compute the candidate-specific probability ratio
        (tilde{p}^{(j)} = p^{(j)}/p^{(n+1)}) for a given candidate (test_label)
        and a calibration point (swap_label).

        If test_label == swap_label, no swap is performed and the ratio is 1.

        Parameters:
          Y_train : array-like
              Array of training labels.
          Y_calib : array-like
              Array of calibration labels.
          calib_pi : callable
              Calibration probability function, which takes an integer (frequency count)
              and returns a probability in [0,1].
          test_label : label type (e.g. int, str)
              The candidate label for the test point (i.e. (y_{n+1})).
          swap_label : label type (e.g. int, str)
              The label of the calibration point that is being swapped (i.e. (y_j)).

        Returns:
          float
              The computed ratio (tilde{p}^{(j)}).
        """
        # If the candidate label equals the calibration point's label,
        # the swap has no effect.
        if test_label == swap_label:
            return 1.0

        # Compute frequency counts for test_label.
        f_calib_test = np.sum(Y_calib == test_label)
        f_train_test = np.sum(Y_train == test_label)
        N_test = f_calib_test + f_train_test  # total count of test_label in Y_train and Y_calib

        # Compute frequency counts for swap_label.
        f_calib_swap = np.sum(Y_calib == swap_label)
        f_train_swap = np.sum(Y_train == swap_label)
        N_swap = f_calib_swap + f_train_swap  # total count of swap_label in Y_train and Y_calib

        # Compute the difference factor for p^(j).

        # Compute the factors for swap_label.
        # For swap_label (which is the calibration label y_j), the frequency decreases by one.
        prob_j = (calib_pi(N_swap - 1)) ** (f_calib_swap - 1)
        prob_j *= ((1 - calib_pi(N_swap - 1))) ** (f_train_swap)

        #Note that here prob_j could be 0

        # Compute the factors for test_label.
        # For test_label (which is the candidate label y_{n+1}), the frequency increases by one.
        prob_j *= (calib_pi(N_test + 1)) ** (f_calib_test + 1)
        prob_j *= ((1 - calib_pi(N_test + 1))) ** (f_train_test)

        # Compute the difference factor for p^(n+1).
        # Compute the factors for swap_label.
        # For swap_label (which is the calibration label y_j), the frequency decreases by one.
        prob_new = (calib_pi(N_swap)) ** (f_calib_swap)
        prob_new *= ((1 - calib_pi(N_swap)) ) ** (f_train_swap)

        # Compute the factors for test_label.
        # For test_label (which is the candidate label y_{n+1}), the frequency increases by one.
        prob_new *= (calib_pi(N_test)) ** (f_calib_test)
        prob_new *= ((1 - calib_pi(N_test))) ** (f_train_test)

        # The final candidate-specific ratio is the product of the two ratios.
        return prob_j / prob_new

    def predict(self, X, random_state=2020):
        """
        Predict conformal sets for new data X.

        Parameters:
          X : array-like, new data.
          random_state : int, random seed.

        Returns:
          S_hat : list of prediction sets (each is a list/array of class indices).
        """
        n = X.shape[0]
        rng = np.random.default_rng(random_state)
        epsilon = rng.uniform(low=0.0, high=1.0, size=n)

        # When predicting new data, the union of classes computed during calibration is reused.
        p_hat = self.black_box.predict_proba(X)
        grey_box = ProbAccum(p_hat)

        # Initialize temporary prediction sets as lists.
        S_hat_temp = [[] for _ in range(n)]
        # For each candidate, use its candidate-specific calibrated level.
        for candidate in self.unique_labels:
            candidate_threshold = self.alpha_calibrated[candidate]
            candidate_sets = grey_box.predict_sets(candidate_threshold, epsilon=epsilon, allow_empty=self.allow_empty)
            for i, s in enumerate(candidate_sets):
                # If the candidate appears in the candidate's prediction set, include it.
                if candidate in s:
                    S_hat_temp[i].append(candidate)
        # Order each prediction set according to the ranking in grey_box.order.
        S_hat = []
        for i in range(n):
            ordered = [c for c in grey_box.order[i] if c in S_hat_temp[i]]
            S_hat.append(np.array(ordered))
        return S_hat



class TestBernoulliSplitConformal:
    def __init__(self, X, Y, black_box, alpha, calibration_probability, random_state=2020, allow_empty=True, verbose=False):
        """
        Parameters:
          X : array-like, feature matrix.
          Y : array-like, labels.
          black_box : classifier with .fit() and .predict_proba().
          alpha : float, desired miscoverage level.
          calibration_probability : callable, calibration frequency function.
               Given an integer (the frequency of a label in Y), returns a probability in [0,1].
          random_state : int, random seed.
          allow_empty : bool, whether empty sets are allowed.
          verbose : bool, (unused here).

          Use uniform weights for all indices.
        """
        self.X = X
        self.Y = Y
        self.black_box = black_box
        self.alpha = alpha
        self.allow_empty = allow_empty
        self.calibration_probability = calibration_probability
        self.random_state = random_state
        self.verbose = verbose

        # Unique candidate labels and their counts.
        # self.unique_labels = np.unique(self.Y)
        # self.label_counts = {label: np.sum(self.Y == label) for label in self.unique_labels}
        self.unique_labels, counts = np.unique(self.Y, return_counts=True)
        self.label_counts = dict(zip(self.unique_labels, counts))

        n = len(Y)
        rng = np.random.default_rng(random_state)

        # For each data point i, sample I_i ~ Bernoulli( π( N(Y_i) ) )
        # I is Boolean
        I = np.empty(n, dtype=bool)
        for i in range(n):
            count_i = self.label_counts[self.Y[i]]
            I[i] = (rng.uniform() < self.calibration_probability(count_i))

        # Define calibration and training sets based on I.
        calib_idx = np.where(I)[0]       # indices with I == True
        train_idx = np.where(~I)[0]        # indices with I == False
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_calib, Y_calib = X[calib_idx], Y[calib_idx]
        n2 = len(Y_calib)
        tqdm.write(f"Bernoulli: Size of all data: {n}. Size of calibration set: {n2}. ")

        # Fit the black_box classifier on the training set.
        self.black_box = black_box
        self.black_box.fit(X_train, Y_train)

        # Form prediction sets on calibration data.
        # Pass y_calib so that extra unseen labels (if any) are included.
        p_hat_calib = self.black_box.predict_proba(X_calib, y_calib=Y_calib)
        grey_box = ProbAccum(p_hat_calib)

        epsilon_calib = rng.uniform(low=0.0, high=1.0, size=n2)

        # Here we compute calibration scores (only once) (using the standard ProbAccum procedure)
        alpha_max = grey_box.calibrate_scores(Y_calib, epsilon=epsilon_calib)

        # In standard split conformal, we set scores = alpha - alpha_max.
        # In the weighted version, these scores serve as our conformity values.
        cal_scores = self.alpha - alpha_max  # array of calibration conformity scores

        # Compute baseline unpermuted joint probability (p_(n+1)):
        p_unpermuted = 1.0

        # --- Begin computing candidate-specific weights and calibrated levels ---
        # For each candidate, compute permuted weights and calibrated level.
        self.alpha_calibrated = {}
        for candidate in self.unique_labels:
            permuted_prob = np.empty(n2)
            # Cache computed probability for each unique swap label in Y_calib
            computed_probs = {}
            for j in range(n2):
                swap_label = Y_calib[j]
                if swap_label not in computed_probs:
                    computed_probs[swap_label] = 1.0
                permuted_prob[j] = computed_probs[swap_label]

            # Compute candidate-specific α-correction using the weighted quantile.
            alpha_correction = weighted_quantile(values = cal_scores,
                                                 quantiles = (1.0 - self.alpha),
                                                 calib_weight=permuted_prob,
                                                 test_weight = p_unpermuted,
                                                 alpha = self.alpha,
                                                 sorted=False)

            # Calibrated level for candidate.
            self.alpha_calibrated[candidate] = self.alpha - alpha_correction
        # --- End candidate-specific calibration ---

    def compute_prob(self, Y_train, Y_calib, calib_pi, test_label, swap_label):
        """
        Compute the candidate-specific probability ratio
        (tilde{p}^{(j)} = p^{(j)}/p^{(n+1)}) for a given candidate (test_label)
        and a calibration point (swap_label).

        If test_label == swap_label, no swap is performed and the ratio is 1.

        Parameters:
          Y_train : array-like
              Array of training labels.
          Y_calib : array-like
              Array of calibration labels.
          calib_pi : callable
              Calibration probability function, which takes an integer (frequency count)
              and returns a probability in [0,1].
          test_label : label type (e.g. int, str)
              The candidate label for the test point (i.e. (y_{n+1})).
          swap_label : label type (e.g. int, str)
              The label of the calibration point that is being swapped (i.e. (y_j)).

        Returns:
          float
              The computed ratio (tilde{p}^{(j)}).
        """
        # If the candidate label equals the calibration point's label,
        # the swap has no effect.
        if test_label == swap_label:
            return 1.0

        # Compute frequency counts for test_label.
        f_calib_test = np.sum(Y_calib == test_label)
        f_train_test = np.sum(Y_train == test_label)
        N_test = f_calib_test + f_train_test  # total count of test_label in Y_train and Y_calib

        # Compute frequency counts for swap_label.
        f_calib_swap = np.sum(Y_calib == swap_label)
        f_train_swap = np.sum(Y_train == swap_label)
        N_swap = f_calib_swap + f_train_swap  # total count of swap_label in Y_train and Y_calib

        # Compute the difference factor for p^(j).

        # Compute the factors for swap_label.
        # For swap_label (which is the calibration label y_j), the frequency decreases by one.
        prob_j = (calib_pi(N_swap - 1)) ** (f_calib_swap - 1)
        prob_j *= ((1 - calib_pi(N_swap - 1))) ** (f_train_swap)

        #Note that here prob_j could be 0

        # Compute the factors for test_label.
        # For test_label (which is the candidate label y_{n+1}), the frequency increases by one.
        prob_j *= (calib_pi(N_test + 1)) ** (f_calib_test + 1)
        prob_j *= ((1 - calib_pi(N_test + 1))) ** (f_train_test)

        # Compute the difference factor for p^(n+1).
        # Compute the factors for swap_label.
        # For swap_label (which is the calibration label y_j), the frequency decreases by one.
        prob_new = (calib_pi(N_swap)) ** (f_calib_swap)
        prob_new *= ((1 - calib_pi(N_swap)) ) ** (f_train_swap)

        # Compute the factors for test_label.
        # For test_label (which is the candidate label y_{n+1}), the frequency increases by one.
        prob_new *= (calib_pi(N_test)) ** (f_calib_test)
        prob_new *= ((1 - calib_pi(N_test))) ** (f_train_test)

        # The final candidate-specific ratio is the product of the two ratios.
        return prob_j / prob_new

    def predict(self, X, random_state=2020):
        """
        Predict conformal sets for new data X.

        Parameters:
          X : array-like, new data.
          random_state : int, random seed.

        Returns:
          S_hat : list of prediction sets (each is a list/array of class indices).
        """
        n = X.shape[0]
        rng = np.random.default_rng(random_state)
        epsilon = rng.uniform(low=0.0, high=1.0, size=n)

        # When predicting new data, the union of classes computed during calibration is reused.
        p_hat = self.black_box.predict_proba(X)
        grey_box = ProbAccum(p_hat)

        # Initialize temporary prediction sets as lists.
        S_hat_temp = [[] for _ in range(n)]
        # For each candidate, use its candidate-specific calibrated level.
        for candidate in self.unique_labels:
            candidate_threshold = self.alpha_calibrated[candidate]
            candidate_sets = grey_box.predict_sets(candidate_threshold, epsilon=epsilon, allow_empty=self.allow_empty)
            for i, s in enumerate(candidate_sets):
                # If the candidate appears in the candidate's prediction set, include it.
                if candidate in s:
                    S_hat_temp[i].append(candidate)
        # Order each prediction set according to the ranking in grey_box.order.
        S_hat = []
        for i in range(n):
            ordered = [c for c in grey_box.order[i] if c in S_hat_temp[i]]
            S_hat.append(np.array(ordered))
        return S_hat





class SplitConformal:
    def __init__(self, X, Y, black_box, alpha, calib_size, random_state=2020, allow_empty=True, verbose=True):
        self.allow_empty = allow_empty

        # Split data into training/calibration sets
        X_train, X_calib, Y_train, Y_calib = train_test_split(X, Y, test_size=calib_size, random_state=random_state)
        n2 = X_calib.shape[0]

        tqdm.write(f"Random split: Size of all data: {len(Y)}. Size of calibration set: {n2}. ")

        self.black_box = black_box

        # Fit model
        self.black_box.fit(X_train, Y_train)

        # Form prediction sets on calibration data
        # For calibration, pass y_calib so that extra unseen labels (if any) are included.
        p_hat_calib = self.black_box.predict_proba(X_calib, y_calib=Y_calib)

        # # ============= MONITOR p_hat_calib HERE =============
        # tqdm.write(f"\n=== Monitoring p_hat_calib ===")
        # tqdm.write(f"  Shape: {p_hat_calib.shape}")
        # tqdm.write(f"  Min probability: {np.min(p_hat_calib):.6f}")
        # tqdm.write(f"  Max probability: {np.max(p_hat_calib):.6f}")
        # tqdm.write(f"  Mean probability: {np.mean(p_hat_calib):.6f}")
        # tqdm.write(f"  Std probability: {np.std(p_hat_calib):.6f}")
        #
        # # Check if probabilities sum to 1 for each sample
        # row_sums = np.sum(p_hat_calib, axis=1)
        # tqdm.write(f"  Row sums (should be ~1.0): min={np.min(row_sums):.6f}, max={np.max(row_sums):.6f}")
        #
        # # Get the max probability for each sample (confidence in top prediction)
        # max_probs = np.max(p_hat_calib, axis=1)
        # tqdm.write(
        #     f"  Max prob per sample: min={np.min(max_probs):.4f}, max={np.max(max_probs):.4f}, mean={np.mean(max_probs):.4f}")
        #
        # # NEW: Count number of classes with probability > 0.001 for each sample
        # num_classes_above_threshold = np.sum(p_hat_calib > 0.001, axis=1)
        # tqdm.write(f"  Classes with prob > 0.001 per sample:")
        # tqdm.write(f"    Min: {np.min(num_classes_above_threshold)}")
        # tqdm.write(f"    Max: {np.max(num_classes_above_threshold)}")
        # tqdm.write(f"    Mean: {np.mean(num_classes_above_threshold):.2f}")
        # tqdm.write(f"    Median: {np.median(num_classes_above_threshold):.0f}")
        # tqdm.write(f"    Distribution: {np.bincount(num_classes_above_threshold)}")
        #
        # # Get predicted classes and their probabilities
        # predicted_classes = np.argmax(p_hat_calib, axis=1)
        # tqdm.write(f"  Predicted classes distribution: {np.bincount(predicted_classes)}")
        #
        # # Show true labels vs predicted for comparison
        # tqdm.write(f"  True labels distribution: {np.bincount(Y_calib)}")
        # accuracy = np.mean(predicted_classes == Y_calib)
        # tqdm.write(f"  Calibration set accuracy: {accuracy:.4f}")
        #
        # # Optional: Save full probability matrix for external analysis
        # if verbose:
        #     np.save('p_hat_calib.npy', p_hat_calib)
        #     tqdm.write(f"  Saved full p_hat_calib to 'p_hat_calib.npy'")
        #
        #     # Print first few samples as examples
        #     tqdm.write(f"\n  First 5 samples of p_hat_calib:")
        #     for i in range(min(5, p_hat_calib.shape[0])):
        #         tqdm.write(f"    Sample {i}: True label={Y_calib[i]}, Predicted={predicted_classes[i]}")
        #         tqdm.write(f"    Probabilities: {p_hat_calib[i]}")
        #
        # # Store for later reference if needed
        # self.p_hat_calib = p_hat_calib
        # # ============= END OF p_hat_calib MONITORING =============



        grey_box = ProbAccum(p_hat_calib)

        rng = np.random.default_rng(random_state)
        epsilon = rng.uniform(low=0.0, high=1.0, size=n2)
        alpha_max = grey_box.calibrate_scores(Y_calib, epsilon=epsilon)
        scores = alpha - alpha_max
        level_adjusted = (1.0-alpha)*(1.0+1.0/float(n2))
        alpha_correction = mquantiles(scores, prob=level_adjusted)

        # Store calibrate level
        self.alpha_calibrated = alpha - alpha_correction

        # # MONITORING HERE
        # tqdm.write(f"  Alpha: {alpha:.4f}")
        # tqdm.write(f"  Level adjusted: {level_adjusted:.4f}")
        # tqdm.write(f"  Alpha correction: {float(alpha_correction):.4f}")
        # tqdm.write(f"  Alpha calibrated: {float(self.alpha_calibrated):.4f}")
        # tqdm.write(f"  Scores range: [{np.min(scores):.4f}, {np.max(scores):.4f}]")
        # tqdm.write(f"  Scores mean: {np.mean(scores):.4f}, std: {np.std(scores):.4f}")
        #
        # # Additional monitoring for calibration scores
        # tqdm.write(f"  Alpha_max range: [{np.min(alpha_max):.4f}, {np.max(alpha_max):.4f}]")
        # tqdm.write(f"  Alpha_max mean: {np.mean(alpha_max):.4f}, std: {np.std(alpha_max):.4f}")
        #
        # # Print the full alpha_max vector for external analysis
        # tqdm.write(f"\n  Full alpha_max vector (n={len(alpha_max)}):")
        # tqdm.write(f"  alpha_max = {alpha_max.tolist()}")
        # np.savetxt('alpha_max_values.txt', alpha_max)
        #
        # # Also print scores vector if you want to analyze it
        # tqdm.write(f"\n  Full scores vector (n={len(scores)}):")
        # tqdm.write(f"  scores = {scores.tolist()}")

    def predict(self, X, random_state=2020):
        n = X.shape[0]
        rng = np.random.default_rng(random_state)
        epsilon = rng.uniform(low=0.0, high=1.0, size=n)

        # Note: When predicting new data, we do not have y_calib;
        # the union (and extra columns) computed during calibration is reused.
        p_hat = self.black_box.predict_proba(X)
        grey_box = ProbAccum(p_hat)
        S_hat = grey_box.predict_sets(self.alpha_calibrated, epsilon=epsilon, allow_empty=self.allow_empty)
        return S_hat



class SelectiveSplitConformal:
    # This class may run into issues
    # if there are not enough samples after filtering labels that appear at least twice.
    # It assumes a fixed number of calibration samples.
    def __init__(self, X, Y, black_box, alpha, calib_num, random_state=2020, allow_empty=True, verbose=False):
        self.allow_empty = allow_empty

        # Map Y to the range 0 to C-1
        label_encoder = LabelEncoder()
        Y_encoded = label_encoder.fit_transform(Y)

        # Identify unique labels and their counts
        unique_labels, counts = np.unique(Y_encoded, return_counts=True)

        # Identify labels that appear only once
        labels_once = unique_labels[counts == 1]

        # Initialize lists to hold training and calibration indices
        D_train_once = []
        D_other = []

        for label in unique_labels:
            indices = np.where(Y_encoded == label)[0]
            if label in labels_once:
                D_train_once.extend(indices)
            else:
                D_other.extend(indices)

        # Convert lists to numpy arrays
        D_train_once = np.array(D_train_once, dtype=int)
        D_other = np.array(D_other, dtype=int)

        if calib_num > len(D_other):
            raise ValueError(f"calib_num ({calib_num}) exceeds the number of samples in D_other ({len(D_other)}).")

        # Split D_other into training and calibration sets
        D_train_other, D_calib = train_test_split(D_other, test_size=calib_num, random_state=random_state)

        # Combine the training sets
        D_train = np.concatenate((D_train_once, D_train_other))

        X_train, Y_train = X[D_train], Y_encoded[D_train]
        X_calib, Y_calib = X[D_calib], Y_encoded[D_calib]

        n2 = X_calib.shape[0]
        tqdm.write(f"Nondynamic selective split: Size of all data: {len(Y)}. Size of calibration set: {n2}. ")

        self.black_box = black_box

        # Fit model
        self.black_box.fit(X_train, Y_train)

        # Form prediction sets on calibration data
        p_hat_calib = self.black_box.predict_proba(X_calib, y_calib=Y_calib)
        grey_box = ProbAccum(p_hat_calib)

        rng = np.random.default_rng(random_state)
        epsilon = rng.uniform(low=0.0, high=1.0, size=n2)
        alpha_max = grey_box.calibrate_scores(Y_calib, epsilon=epsilon)
        scores = alpha - alpha_max
        level_adjusted = (1.0 - alpha) * (1.0 + 1.0 / float(n2))
        alpha_correction = mquantiles(scores, prob=level_adjusted)

        # Store calibrate level
        self.alpha_calibrated = alpha - alpha_correction
        self.label_encoder = label_encoder

    def predict(self, X, random_state=2020):
        n = X.shape[0]
        rng = np.random.default_rng(random_state)
        epsilon = rng.uniform(low=0.0, high=1.0, size=n)
        p_hat = self.black_box.predict_proba(X)
        grey_box = ProbAccum(p_hat)
        S_hat_encoded = grey_box.predict_sets(self.alpha_calibrated, epsilon=epsilon, allow_empty=self.allow_empty)

        # Decode the prediction sets
        S_hat = [self.label_encoder.inverse_transform(pred_set) for pred_set in S_hat_encoded]
        return S_hat

class SelectiveSplitConformalDynamic:
    # This class addresses the potential issue in the previous class by dynamically adjusting
    # the calibration set size based on the available data, ensuring a better balance between
    # training and calibration samples.
    def __init__(self, X, Y, black_box, alpha, calib_num, random_state=2020, allow_empty=True, verbose=False):
        self.allow_empty = allow_empty

        # Map Y to the range 0 to C-1
        label_encoder = LabelEncoder()
        Y_encoded = label_encoder.fit_transform(Y)

        # Identify unique labels and their counts
        unique_labels, counts = np.unique(Y_encoded, return_counts=True)

        # Identify labels that appear only once
        labels_once = unique_labels[counts == 1]

        # Initialize lists to hold training and calibration indices
        D_train_once = []
        D_other = []

        for label in unique_labels:
            indices = np.where(Y_encoded == label)[0]
            if label in labels_once:
                D_train_once.extend(indices)
            else:
                D_other.extend(indices)

        # Convert lists to numpy arrays
        D_train_once = np.array(D_train_once, dtype=int)
        D_other = np.array(D_other, dtype=int)

        calib_num_dynamic = int(np.floor(calib_num * len(D_other) / len(Y)))


        # Split D_other into training and calibration sets
        D_train_other, D_calib = train_test_split(D_other, test_size=calib_num_dynamic, random_state=random_state)

        # Combine the training sets
        D_train = np.concatenate((D_train_once, D_train_other))

        X_train, Y_train = X[D_train], Y_encoded[D_train]
        X_calib, Y_calib = X[D_calib], Y_encoded[D_calib]

        n2 = X_calib.shape[0]
        tqdm.write(f"Dynamic selective split: Size of all data: {len(Y)}. Size of calibration set: {n2}. ")


        self.black_box = black_box

        # Fit model
        self.black_box.fit(X_train, Y_train)

        # Form prediction sets on calibration data
        p_hat_calib = self.black_box.predict_proba(X_calib, y_calib=Y_calib)
        grey_box = ProbAccum(p_hat_calib)

        rng = np.random.default_rng(random_state)
        epsilon = rng.uniform(low=0.0, high=1.0, size=n2)
        alpha_max = grey_box.calibrate_scores(Y_calib, epsilon=epsilon)
        scores = alpha - alpha_max
        level_adjusted = (1.0 - alpha) * (1.0 + 1.0 / float(n2))
        alpha_correction = mquantiles(scores, prob=level_adjusted)

        # Store calibrate level
        self.alpha_calibrated = alpha - alpha_correction
        self.label_encoder = label_encoder

    def predict(self, X, random_state=2020):
        n = X.shape[0]
        rng = np.random.default_rng(random_state)
        epsilon = rng.uniform(low=0.0, high=1.0, size=n)
        p_hat = self.black_box.predict_proba(X)
        grey_box = ProbAccum(p_hat)
        S_hat_encoded = grey_box.predict_sets(self.alpha_calibrated, epsilon=epsilon, allow_empty=self.allow_empty)

        # Decode the prediction sets
        S_hat = [self.label_encoder.inverse_transform(pred_set) for pred_set in S_hat_encoded]
        return S_hat


class InclusiveSplitConformal:
    """
    This class splits each label's samples so that approximately half (ceiling)
    go into the training set and the rest go to the calibration set.

    For example:
      - If a label appears once, that single sample goes to training.
      - If a label appears twice, 1 (ceiling(2/2)) goes to training, 1 goes to calibration.
      - If a label appears three times, 2 (ceiling(3/2) = 2) go to training, 1 goes to calibration.
      - etc.

    After the split, it proceeds with fitting the black-box model, then calibration
    via the usual conformal approach (similar to the original code).
    """

    def __init__(
            self,
            X,
            Y,
            black_box,
            alpha,
            calib_num,
            random_state=2020,
            allow_empty=True,
            verbose=False
    ):
        """
        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        Y : array-like
            Labels.
        black_box : object
            A classifier with a .fit(X, y) and .predict_proba(X) method.
        alpha : float
            Miscoverage level for conformal prediction.
        random_state : int
            Random seed for reproducibility.
        allow_empty : bool
            Whether empty sets are allowed if confidence is extremely low.
        verbose : bool
            Verbosity control (not used heavily here).
        """

        self.allow_empty = allow_empty
        self.verbose = verbose

        # Encode the labels from 0 to C-1
        label_encoder = LabelEncoder()
        Y_encoded = label_encoder.fit_transform(Y)
        unique_labels = np.unique(Y_encoded)

        # ------------------------------------------------
        # Label-wise splitting: half of each label (ceil)
        # ------------------------------------------------
        rng = np.random.default_rng(random_state)
        D_train = []
        D_calib = []

        for label in unique_labels:
            indices = np.where(Y_encoded == label)[0]
            rng.shuffle(indices)

            c = len(indices)
            # number of samples going to calibration
            # calib_ratio = 0.5
            calib_ratio = calib_num / float(len(Y))
            t = math.floor(c * calib_ratio)

            # first t go to calib, the rest go to training
            D_calib.extend(indices[:t])
            D_train.extend(indices[t:])

        D_train = np.array(D_train, dtype=int)
        D_calib = np.array(D_calib, dtype=int)

        # Prepare final train/calib arrays
        X_train, Y_train = X[D_train], Y_encoded[D_train]
        X_calib, Y_calib = X[D_calib], Y_encoded[D_calib]

        n2 = X_calib.shape[0]
        tqdm.write(f"Inclusive split: Size of all data: {len(Y)}. Size of calibration set: {n2}. ")

        self.black_box = black_box

        # Fit model
        self.black_box.fit(X_train, Y_train)

        # ------------------------------------------------
        # Form prediction sets on calibration data
        # ------------------------------------------------
        # Probability estimates on calibration data
        p_hat_calib = self.black_box.predict_proba(X_calib, y_calib=Y_calib)
        grey_box = ProbAccum(p_hat_calib)

        # Randomization for tie-breaking
        rng = np.random.default_rng(random_state)
        epsilon = rng.uniform(low=0.0, high=1.0, size=n2)

        # alpha_max is the threshold used for each calibration sample
        alpha_max = grey_box.calibrate_scores(Y_calib, epsilon=epsilon)

        # The difference from the target alpha
        scores = alpha - alpha_max

        # The standard correction for finite sample
        # Using alpha_correction = quantile of (scores) at (1 - alpha)*(1 + 1/n2)
        level_adjusted = (1.0 - alpha) * (1.0 + 1.0 / float(n2))
        alpha_correction = mquantiles(scores, prob=level_adjusted)

        # Final calibrated alpha
        self.alpha_calibrated = alpha - alpha_correction
        self.label_encoder = label_encoder

    def predict(self, X, random_state=2020):
        """
        Predict conformal sets for a new batch of points X.

        Parameters
        ----------
        X : np.ndarray
            New data to predict sets for.
        random_state : int
            Random seed for reproducibility.

        Returns
        -------
        S_hat : list of arrays
            List of predicted label sets for each sample in X.
        """

        n = X.shape[0]
        rng = np.random.default_rng(random_state)
        epsilon = rng.uniform(low=0.0, high=1.0, size=n)

        p_hat = self.black_box.predict_proba(X)
        grey_box = ProbAccum(p_hat)
        S_hat_encoded = grey_box.predict_sets(
            self.alpha_calibrated,
            epsilon=epsilon,
            allow_empty=self.allow_empty
        )

        # Decode the prediction sets
        S_hat = [self.label_encoder.inverse_transform(pred_set) for pred_set in S_hat_encoded]
        return S_hat




class RealInclusiveSplitConformal:
    """
    This class splits each label's samples so that approximately half (ceiling)
    go into the training set and the rest go to the calibration set.

    For example:
      - If a label appears once, that single sample goes to training.
      - If a label appears twice, 1 (ceiling(2/2)) goes to training, 1 goes to calibration.
      - If a label appears three times, 2 (ceiling(3/2) = 2) go to training, 1 goes to calibration.
      - etc.

    After the split, it proceeds with fitting the black-box model, then calibration
    via the usual conformal approach (similar to the original code).
    """

    def __init__(
            self,
            X,
            Y,
            black_box,
            alpha,
            calib_num,
            random_state=2020,
            allow_empty=True,
            verbose=False
    ):
        """
        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        Y : array-like
            Labels.
        black_box : object
            A classifier with a .fit(X, y) and .predict_proba(X) method.
        alpha : float
            Miscoverage level for conformal prediction.
        random_state : int
            Random seed for reproducibility.
        allow_empty : bool
            Whether empty sets are allowed if confidence is extremely low.
        verbose : bool
            Verbosity control (not used heavily here).
        """

        self.allow_empty = allow_empty
        self.verbose = verbose

        # Encode the labels from 0 to C-1
        label_encoder = LabelEncoder()
        Y_encoded = label_encoder.fit_transform(Y)
        unique_labels = np.unique(Y_encoded)

        # ------------------------------------------------
        # Label-wise splitting: half of each label (ceil)
        # ------------------------------------------------
        rng = np.random.default_rng(random_state)
        D_train = []
        D_calib = []

        for label in unique_labels:
            indices = np.where(Y_encoded == label)[0]
            rng.shuffle(indices)

            c = len(indices)
            # number of samples going to calibration
            # calib_ratio = 0.5
            calib_ratio = 1.5 * calib_num / float(len(Y))
            t = math.floor(c * calib_ratio)

            # first t go to calib, the rest go to training
            D_calib.extend(indices[:t])
            D_train.extend(indices[t:])

        D_train = np.array(D_train, dtype=int)
        D_calib = np.array(D_calib, dtype=int)

        # Prepare final train/calib arrays
        X_train, Y_train = X[D_train], Y_encoded[D_train]
        X_calib, Y_calib = X[D_calib], Y_encoded[D_calib]

        n2 = X_calib.shape[0]
        tqdm.write(f"Inclusive split: Size of all data: {len(Y)}. Size of calibration set: {n2}. ")

        self.black_box = black_box

        # Fit model
        self.black_box.fit(X_train, Y_train)

        # ------------------------------------------------
        # Form prediction sets on calibration data
        # ------------------------------------------------
        # Probability estimates on calibration data
        p_hat_calib = self.black_box.predict_proba(X_calib, y_calib=Y_calib)
        grey_box = ProbAccum(p_hat_calib)

        # Randomization for tie-breaking
        rng = np.random.default_rng(random_state)
        epsilon = rng.uniform(low=0.0, high=1.0, size=n2)

        # alpha_max is the threshold used for each calibration sample
        alpha_max = grey_box.calibrate_scores(Y_calib, epsilon=epsilon)

        # The difference from the target alpha
        scores = alpha - alpha_max

        # The standard correction for finite sample
        # Using alpha_correction = quantile of (scores) at (1 - alpha)*(1 + 1/n2)
        level_adjusted = (1.0 - alpha) * (1.0 + 1.0 / float(n2))
        alpha_correction = mquantiles(scores, prob=level_adjusted)

        # Final calibrated alpha
        self.alpha_calibrated = alpha - alpha_correction
        self.label_encoder = label_encoder

    def predict(self, X, random_state=2020):
        """
        Predict conformal sets for a new batch of points X.

        Parameters
        ----------
        X : np.ndarray
            New data to predict sets for.
        random_state : int
            Random seed for reproducibility.

        Returns
        -------
        S_hat : list of arrays
            List of predicted label sets for each sample in X.
        """

        n = X.shape[0]
        rng = np.random.default_rng(random_state)
        epsilon = rng.uniform(low=0.0, high=1.0, size=n)

        p_hat = self.black_box.predict_proba(X)
        grey_box = ProbAccum(p_hat)
        S_hat_encoded = grey_box.predict_sets(
            self.alpha_calibrated,
            epsilon=epsilon,
            allow_empty=self.allow_empty
        )

        # Decode the prediction sets
        S_hat = [self.label_encoder.inverse_transform(pred_set) for pred_set in S_hat_encoded]
        return S_hat


class SplitConformalFull:
    def __init__(self, X, Y, black_box, alpha, calib_size, random_state=2020, allow_empty=True, verbose=False):
        self.allow_empty = allow_empty

        # Split data into training/calibration sets
        X_train, X_calib, Y_train, Y_calib = train_test_split(X, Y, test_size=calib_size, random_state=random_state)
        X_train, Y_train = X, Y
        n2 = X_calib.shape[0]

        tqdm.write(f"Random split: Size of all data: {len(Y)}. Size of calibration set: {n2}. ")

        self.black_box = black_box

        # Fit model
        self.black_box.fit(X_train, Y_train)

        # Form prediction sets on calibration data
        # For calibration, pass y_calib so that extra unseen labels (if any) are included.
        p_hat_calib = self.black_box.predict_proba(X_calib, y_calib=Y_calib)
        grey_box = ProbAccum(p_hat_calib)

        rng = np.random.default_rng(random_state)
        epsilon = rng.uniform(low=0.0, high=1.0, size=n2)
        alpha_max = grey_box.calibrate_scores(Y_calib, epsilon=epsilon)
        scores = alpha - alpha_max
        level_adjusted = (1.0-alpha)*(1.0+1.0/float(n2))
        alpha_correction = mquantiles(scores, prob=level_adjusted)

        # Store calibrate level
        self.alpha_calibrated = alpha - alpha_correction


        # MONITORING HERE
        tqdm.write(f"  Alpha: {alpha:.4f}")
        tqdm.write(f"  Level adjusted: {level_adjusted:.4f}")
        tqdm.write(f"  Alpha correction: {float(alpha_correction):.4f}")
        tqdm.write(f"  Alpha calibrated: {float(self.alpha_calibrated):.4f}")
        tqdm.write(f"  Scores range: [{np.min(scores):.4f}, {np.max(scores):.4f}]")
        tqdm.write(f"  Scores mean: {np.mean(scores):.4f}, std: {np.std(scores):.4f}")

        # Additional monitoring for calibration scores
        tqdm.write(f"  Alpha_max range: [{np.min(alpha_max):.4f}, {np.max(alpha_max):.4f}]")
        tqdm.write(f"  Alpha_max mean: {np.mean(alpha_max):.4f}, std: {np.std(alpha_max):.4f}")


    def predict(self, X, random_state=2020):
        n = X.shape[0]
        rng = np.random.default_rng(random_state)
        epsilon = rng.uniform(low=0.0, high=1.0, size=n)

        # Note: When predicting new data, we do not have y_calib;
        # the union (and extra columns) computed during calibration is reused.
        p_hat = self.black_box.predict_proba(X)
        grey_box = ProbAccum(p_hat)
        S_hat = grey_box.predict_sets(self.alpha_calibrated, epsilon=epsilon, allow_empty=self.allow_empty)
        return S_hat





class SplitConformalUnseen:
    def __init__(self, X, Y, black_box, alpha, calib_size, random_state=2020, allow_empty=True, verbose=True):
        self.allow_empty = allow_empty

        # Split data into training/calibration sets
        X_train, X_calib, Y_train, Y_calib = train_test_split(X, Y, test_size=calib_size, random_state=random_state)

        # Find unique labels in training set
        train_labels = set(Y_train)

        # Find calibration points with labels not seen in training
        unseen_mask = ~np.isin(Y_calib, list(train_labels))
        X_unseen = X_calib[unseen_mask]
        Y_unseen = Y_calib[unseen_mask]

        # Augment training set with unseen label points
        if len(Y_unseen) > 0:
            X_train = np.vstack([X_train, X_unseen])
            Y_train = np.concatenate([Y_train, Y_unseen])

            if verbose:
                tqdm.write(f"Added {len(Y_unseen)} calibration points with unseen labels to training set")

        n2 = X_calib.shape[0]

        tqdm.write(f"Random split: Size of all data: {len(Y)}. Size of calibration set: {n2}. ")

        self.black_box = black_box

        # Fit model
        self.black_box.fit(X_train, Y_train)

        # Form prediction sets on calibration data
        # For calibration, pass y_calib so that extra unseen labels (if any) are included.
        p_hat_calib = self.black_box.predict_proba(X_calib, y_calib=Y_calib)
        grey_box = ProbAccum(p_hat_calib)

        rng = np.random.default_rng(random_state)
        epsilon = rng.uniform(low=0.0, high=1.0, size=n2)
        alpha_max = grey_box.calibrate_scores(Y_calib, epsilon=epsilon)
        scores = alpha - alpha_max
        level_adjusted = (1.0-alpha)*(1.0+1.0/float(n2))
        alpha_correction = mquantiles(scores, prob=level_adjusted)

        # Store calibrate level
        self.alpha_calibrated = alpha - alpha_correction

        # MONITORING HERE
        tqdm.write(f"  Alpha: {alpha:.4f}")
        tqdm.write(f"  Level adjusted: {level_adjusted:.4f}")
        tqdm.write(f"  Alpha correction: {float(alpha_correction):.4f}")
        tqdm.write(f"  Alpha calibrated: {float(self.alpha_calibrated):.4f}")
        tqdm.write(f"  Scores range: [{np.min(scores):.4f}, {np.max(scores):.4f}]")
        tqdm.write(f"  Scores mean: {np.mean(scores):.4f}, std: {np.std(scores):.4f}")

        # Additional monitoring for calibration scores
        tqdm.write(f"  Alpha_max range: [{np.min(alpha_max):.4f}, {np.max(alpha_max):.4f}]")
        tqdm.write(f"  Alpha_max mean: {np.mean(alpha_max):.4f}, std: {np.std(alpha_max):.4f}")

    def predict(self, X, random_state=2020):
        n = X.shape[0]
        rng = np.random.default_rng(random_state)
        epsilon = rng.uniform(low=0.0, high=1.0, size=n)

        # Note: When predicting new data, we do not have y_calib;
        # the union (and extra columns) computed during calibration is reused.
        p_hat = self.black_box.predict_proba(X)
        grey_box = ProbAccum(p_hat)
        S_hat = grey_box.predict_sets(self.alpha_calibrated, epsilon=epsilon, allow_empty=self.allow_empty)
        return S_hat


class SplitConformalOnePerLabel:
    def __init__(self, X, Y, black_box, alpha, calib_size, random_state=2020, allow_empty=True, verbose=True):
        self.allow_empty = allow_empty

        # Split data into training/calibration sets
        X_train, X_calib, Y_train, Y_calib = train_test_split(X, Y, test_size=calib_size, random_state=random_state)

        # Find unique labels in training set
        train_labels = set(Y_train)

        # Find labels that are in calibration set but not in training set
        all_labels = set(Y)
        unseen_labels = all_labels - train_labels

        # For each unseen label, randomly select one data point from calibration set
        if len(unseen_labels) > 0:
            rng = np.random.default_rng(random_state)

            selected_X = []
            selected_Y = []

            for label in unseen_labels:
                # Find all calibration points with this label
                label_mask = (Y_calib == label)
                if np.any(label_mask):  # Check if this label exists in calibration set
                    label_indices = np.where(label_mask)[0]

                    # Randomly select one index
                    chosen_idx = rng.choice(label_indices)

                    selected_X.append(X_calib[chosen_idx])
                    selected_Y.append(Y_calib[chosen_idx])

            # Augment training set with selected points
            if len(selected_Y) > 0:
                selected_X = np.array(selected_X)
                selected_Y = np.array(selected_Y)

                X_train = np.vstack([X_train, selected_X])
                Y_train = np.concatenate([Y_train, selected_Y])

                if verbose:
                    tqdm.write(f"Added {len(selected_Y)} calibration points (one per unseen label) to training set")
                    tqdm.write(f"Unseen labels added: {sorted(selected_Y)}")

        n2 = X_calib.shape[0]

        tqdm.write(f"Random split: Size of all data: {len(Y)}. Size of calibration set: {n2}.")
        tqdm.write(f"Final training set size: {len(Y_train)}")

        self.black_box = black_box

        # Fit model on augmented training set
        self.black_box.fit(X_train, Y_train)

        # Form prediction sets on calibration data
        # For calibration, pass y_calib so that extra unseen labels (if any) are included.
        p_hat_calib = self.black_box.predict_proba(X_calib, y_calib=Y_calib)
        grey_box = ProbAccum(p_hat_calib)

        rng = np.random.default_rng(random_state)
        epsilon = rng.uniform(low=0.0, high=1.0, size=n2)
        alpha_max = grey_box.calibrate_scores(Y_calib, epsilon=epsilon)
        scores = alpha - alpha_max
        level_adjusted = (1.0 - alpha) * (1.0 + 1.0 / float(n2))
        alpha_correction = mquantiles(scores, prob=level_adjusted)

        # Store calibrate level
        self.alpha_calibrated = alpha - alpha_correction

        # MONITORING HERE
        tqdm.write(f"  Alpha: {alpha:.4f}")
        tqdm.write(f"  Level adjusted: {level_adjusted:.4f}")
        tqdm.write(f"  Alpha correction: {float(alpha_correction):.4f}")
        tqdm.write(f"  Alpha calibrated: {float(self.alpha_calibrated):.4f}")
        tqdm.write(f"  Scores range: [{np.min(scores):.4f}, {np.max(scores):.4f}]")
        tqdm.write(f"  Scores mean: {np.mean(scores):.4f}, std: {np.std(scores):.4f}")

        # Additional monitoring for calibration scores
        tqdm.write(f"  Alpha_max range: [{np.min(alpha_max):.4f}, {np.max(alpha_max):.4f}]")
        tqdm.write(f"  Alpha_max mean: {np.mean(alpha_max):.4f}, std: {np.std(alpha_max):.4f}")

    def predict(self, X, random_state=2020):
        n = X.shape[0]
        rng = np.random.default_rng(random_state)
        epsilon = rng.uniform(low=0.0, high=1.0, size=n)

        # Note: When predicting new data, we do not have y_calib;
        # the union (and extra columns) computed during calibration is reused.
        p_hat = self.black_box.predict_proba(X)
        grey_box = ProbAccum(p_hat)
        S_hat = grey_box.predict_sets(self.alpha_calibrated, epsilon=epsilon, allow_empty=self.allow_empty)
        return S_hat






class SplitConformalUnseenGuess:
    def __init__(self, X, Y, black_box, alpha, calib_size, random_state=2020, allow_empty=True, verbose=False):
        self.allow_empty = allow_empty

        # Split data into training/calibration sets
        X_train, X_calib, Y_train, Y_calib = train_test_split(X, Y, test_size=calib_size, random_state=random_state)
        n2 = X_calib.shape[0]

        # Find labels that are in the full dataset but not in training set
        all_labels = set(Y)
        train_labels = set(Y_train)
        unseen_labels = all_labels - train_labels

        if len(unseen_labels) > 0 and len(X_train) > 0:
            # Generate synthetic features for each unseen label
            rng = np.random.default_rng(random_state)

            synthetic_X = []
            synthetic_Y = []

            for label in unseen_labels:
                # Randomly select one feature vector from X_train
                random_idx = rng.integers(0, len(X_train))
                synthetic_features = X_train[random_idx].copy()

                synthetic_X.append(synthetic_features)
                synthetic_Y.append(label)

            # Convert to arrays
            synthetic_X = np.array(synthetic_X)
            synthetic_Y = np.array(synthetic_Y)

            # Augment training set
            X_train = np.vstack([X_train, synthetic_X])
            Y_train = np.concatenate([Y_train, synthetic_Y])

            if verbose:
                tqdm.write(f"Added {len(unseen_labels)} synthetic points for unseen labels: {sorted(unseen_labels)}")

        tqdm.write(f"Random split: Size of all data: {len(Y)}. Size of calibration set: {n2}. ")

        self.black_box = black_box

        # Fit model
        self.black_box.fit(X_train, Y_train)

        # Form prediction sets on calibration data
        # For calibration, pass y_calib so that extra unseen labels (if any) are included.
        p_hat_calib = self.black_box.predict_proba(X_calib, y_calib=Y_calib)
        grey_box = ProbAccum(p_hat_calib)

        rng = np.random.default_rng(random_state)
        epsilon = rng.uniform(low=0.0, high=1.0, size=n2)
        alpha_max = grey_box.calibrate_scores(Y_calib, epsilon=epsilon)
        scores = alpha - alpha_max
        level_adjusted = (1.0-alpha)*(1.0+1.0/float(n2))
        alpha_correction = mquantiles(scores, prob=level_adjusted)

        # Store calibrate level
        self.alpha_calibrated = alpha - alpha_correction


        # MONITORING HERE
        tqdm.write(f"  Alpha: {alpha:.4f}")
        tqdm.write(f"  Level adjusted: {level_adjusted:.4f}")
        tqdm.write(f"  Alpha correction: {float(alpha_correction):.4f}")
        tqdm.write(f"  Alpha calibrated: {float(self.alpha_calibrated):.4f}")
        tqdm.write(f"  Scores range: [{np.min(scores):.4f}, {np.max(scores):.4f}]")
        tqdm.write(f"  Scores mean: {np.mean(scores):.4f}, std: {np.std(scores):.4f}")

        # Additional monitoring for calibration scores
        tqdm.write(f"  Alpha_max range: [{np.min(alpha_max):.4f}, {np.max(alpha_max):.4f}]")
        tqdm.write(f"  Alpha_max mean: {np.mean(alpha_max):.4f}, std: {np.std(alpha_max):.4f}")


    def predict(self, X, random_state=2020):
        n = X.shape[0]
        rng = np.random.default_rng(random_state)
        epsilon = rng.uniform(low=0.0, high=1.0, size=n)

        # Note: When predicting new data, we do not have y_calib;
        # the union (and extra columns) computed during calibration is reused.
        p_hat = self.black_box.predict_proba(X)
        grey_box = ProbAccum(p_hat)
        S_hat = grey_box.predict_sets(self.alpha_calibrated, epsilon=epsilon, allow_empty=self.allow_empty)
        return S_hat


class SplitConformalUnseenGuessMeanNoise:
    """
    SplitConformal with synthetic augmentation:
    After the train/calib split, add ONE synthetic point for each label that
    appears in Y (global) but is missing from Y_train (post-split).

    Synthetic feature x* for each unseen label is sampled as:
        x* ~ Normal(mean(X_train), (noise_scale * std(X_train))^2) elementwise

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    Y : np.ndarray, shape (n_samples,)
    black_box : object with .fit(X, y) and .predict_proba(X, y_calib=None)
    alpha : float in (0,1)
    calib_size : float or int
        If float in (0,1), treated as fraction for test_size in train_test_split.
        If int, treated as absolute calibration size.
    random_state : int
    allow_empty : bool
    verbose : bool
    noise_scale : float
        Scales per-feature std when sampling synthetic points.
    samples_per_unseen : int
        How many synthetic samples to add per unseen label.
    """
    def __init__(
        self,
        X, Y,
        black_box,
        alpha,
        calib_size,
        random_state=2020,
        allow_empty=True,
        verbose=True,
        noise_scale=0.5,
        samples_per_unseen=1,
    ):
        self.allow_empty = allow_empty
        self.alpha = alpha
        self.noise_scale = float(noise_scale)
        self.samples_per_unseen = int(samples_per_unseen)

        # Split data into training/calibration sets
        X_train, X_calib, Y_train, Y_calib = train_test_split(
            X, Y, test_size=calib_size, random_state=random_state
        )
        n2 = X_calib.shape[0]

        if verbose:
            tqdm.write(f"Random split: Size of all data: {len(Y)}. "
                       f"Size of calibration set: {n2}.")

        rng = np.random.default_rng(random_state)

        # Find labels present overall but missing from training
        all_labels = np.unique(Y)
        train_labels = np.unique(Y_train)
        unseen_labels = np.setdiff1d(all_labels, train_labels, assume_unique=False)

        # If needed, synthesize one (or more) training point(s) per unseen label
        if unseen_labels.size > 0:
            mu = X_train.mean(axis=0)
            std = X_train.std(axis=0, ddof=1)
            # Avoid zero std; use 1.0 as a fallback scale for constant features
            std_safe = np.where(std > 0, std, 1.0)

            synth_X = []
            synth_Y = []
            for y_new in unseen_labels:
                for _ in range(self.samples_per_unseen):
                    x_new = rng.normal(loc=mu, scale=self.noise_scale * std_safe)
                    synth_X.append(x_new)
                    synth_Y.append(y_new)

            synth_X = np.vstack(synth_X)
            synth_Y = np.asarray(synth_Y, dtype=Y_train.dtype)

            X_train = np.vstack([X_train, synth_X])
            Y_train = np.concatenate([Y_train, synth_Y])

            if verbose:
                tqdm.write(f"Augmented training set with "
                           f"{len(synth_Y)} synthetic point(s) "
                           f"for {len(unseen_labels)} unseen label(s): {unseen_labels.tolist()}")

        self.black_box = black_box

        # Fit model on augmented training data
        self.black_box.fit(X_train, Y_train)

        # Calibrate on calibration set:
        # Pass y_calib so predict_proba can expand to the union of labels if needed.
        p_hat_calib = self.black_box.predict_proba(X_calib, y_calib=Y_calib)
        grey_box = ProbAccum(p_hat_calib)

        epsilon = rng.uniform(low=0.0, high=1.0, size=n2)
        alpha_max = grey_box.calibrate_scores(Y_calib, epsilon=epsilon)
        scores = alpha - alpha_max
        level_adjusted = (1.0 - alpha) * (1.0 + 1.0 / float(n2))
        alpha_correction = mquantiles(scores, prob=level_adjusted)

        # Store monitoring quantities
        self.level_adjusted = float(level_adjusted)
        self.alpha_correction = float(alpha_correction)
        self.alpha_calibrated = float(alpha - alpha_correction)

        if verbose:
            tqdm.write(f"level_adjusted={self.level_adjusted:.6f} | "
                       f"alpha_correction={self.alpha_correction:.6f} | "
                       f"alpha_calibrated={self.alpha_calibrated:.6f}")

    def predict(self, X, random_state=2020):
        n = X.shape[0]
        rng = np.random.default_rng(random_state)
        epsilon = rng.uniform(low=0.0, high=1.0, size=n)

        # Note: When predicting new data, we do not have y_calib;
        # the union (and extra columns) computed during calibration is reused.
        p_hat = self.black_box.predict_proba(X)
        grey_box = ProbAccum(p_hat)
        S_hat = grey_box.predict_sets(
            self.alpha_calibrated, epsilon=epsilon, allow_empty=self.allow_empty
        )
        return S_hat