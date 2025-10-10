import numpy as np
import pdb
from collections import defaultdict, OrderedDict

def dictToList(d):
    reg_list = [[x] * d[x] for x in d.keys()]
    flat_list = np.array([item for sublist in reg_list for item in sublist])
    return flat_list

def listToDict(v):
    d = defaultdict(lambda: 0)
    for x in v:
        d[x] += 1
    return d

def sort_dict(d):
    return dict(sorted(d.items(), key=lambda x: x[1], reverse=True))

def random_choice(probs, rng=None):
    if rng is None:
        x = np.random.rand()
    else:
        x = rng.random()
    cum = 0
    for i,p in enumerate(probs):
        cum += p
        if x < cum:
            break
    return i


def weighted_quantile(values, quantiles, calib_weight, test_weight, alpha, sorted=False):
    """
    Compute weighted quantiles incorporating an extra value.

    Parameters:
      values : array-like
          Calibration values.
      quantiles : array-like
          Quantile levels to compute (each between 0 and 1).
      calib_weight : array-like
          Weights for the calibration values (same length as values).
      test_weight : float
          Weight for the extra value.
      alpha : float
          The extra value to include (e.g., the target threshold).
      sorted : bool, optional
          Flag indicating whether 'values' is already sorted (default: False).

    Returns:
      The computed weighted quantile(s) based on the combined dataset of
      calibration values (weighted by calib_weight) and the extra value (weighted by test_weight).
    """
    # Convert inputs to numpy arrays
    values = np.array(values)
    quantiles = np.array(quantiles)
    calib_weight = np.array(calib_weight)

    # Combine calibration values with the extra value 'alpha'
    combined_values = np.concatenate([values, [alpha]])
    combined_weights = np.concatenate([calib_weight, [test_weight]])

    # Sort if not already sorted
    if not sorted:
        sorter = np.argsort(combined_values)
        combined_values = combined_values[sorter]
        combined_weights = combined_weights[sorter]

    # Compute cumulative weights and normalize to obtain the weighted CDF
    cum_weights = np.cumsum(combined_weights)
    total_weight = cum_weights[-1]
    weighted_cdf = cum_weights / total_weight

    # Interpolate to get the quantile values
    return np.interp(quantiles, weighted_cdf, combined_values)


def calibration_probability_rate(frequency, lambda_param=0.1, max_prob=0.5):
    """
    Given a positive integer 'frequency' representing the frequency count of a label,
    returns the probability of that label being selected into the calibration set.

    This function is designed to be an increasing function of frequency so that rare labels
    (frequency 1) are always used in training (p(1) = 0), while as the frequency increases,
    the probability of being placed in the calibration set increases but is capped at 0.5.

    Parameters:
        frequency (int): The frequency count of the label (must be >= 1).
        lambda_param (float): A positive parameter controlling the rate of increase.

    Returns:
        float: The probability in [0,0.5] of selection.
    """
    if frequency == 0:
        # this is the placeholder when no observations of such label
        # 0 ** 0 = 1 so zero does not cause an issue
        return 0.0
    else:
        prob_calib =  max_prob * (1 - np.exp(-lambda_param * (frequency - 1)))
        return prob_calib


def calibration_probability_level(frequency, prespecified_level=0.01, max_prob=0.5):
    """
    Given a non-negative integer 'frequency' representing the frequency count of a label,
    returns the probability of that label being selected into the calibration set.

    This function follows these rules:
    - For frequency 0 or 1: returns 0.0 (always use in training)
    - For frequency > 1: returns (prespecified_level)^(1/frequency)

    As frequency increases, the probability approaches 1.0

    Parameters:
        frequency (int): The frequency count of the label (must be >= 0).
        prespecified_level (float): A positive parameter between 0 and 1 (default 0.01).
                                   Lower values result in lower probabilities.

    Returns:
        float: The probability of selection into the calibration set.
    """
    if frequency <= 1:
        return 0.0
    else:
        prob = prespecified_level ** (1 / frequency)
        return min(prob, max_prob)




def tune_calibration_params(data_dist, n_ref, calib_num, max_prob=None, num_trials=10, random_state=2025, tol=1e-3, max_iter=10):
    """
    Simulate multiple reference datasets to approximate lambda_param and max_prob.

    Parameters:
    - data_dist: object with .sample(n, random_state) -> (X, Y)
    - n_ref: int, number of reference samples per simulation
    - calib_num: desired number of calibration points
    - max_prob: float or None; if None, set to calib_num / n_ref
    - num_trials: int, number of simulated reference draws
    - random_state: int, seed for reproducibility
    - tol: float, tolerance for lambda bisection
    - max_iter: int, maximum bisection iterations

    Returns:
    - lambda_param: float, rate parameter for calibration_probability
    - max_prob: float, maximum inclusion probability
    """
    print(f"[tune] Starting tuning: n_ref={n_ref}, calib_num={calib_num}, num_trials={num_trials}")

    # Initialize max_prob if not provided
    if max_prob is None:
        max_prob = min(2 * calib_num / n_ref, 1)
    print(f"[tune] Using max_prob initial guess = {max_prob:.4f}")

    # Prepare simulation
    rng = np.random.default_rng(random_state)
    freqs_trials = []

    # Simulate label frequencies across trials
    for _ in range(num_trials):
        _, Y_ref = data_dist.sample(n_ref, random_state=rng.integers(1e9))
        unique, counts = np.unique(Y_ref, return_counts=True)
        # Reconstruct per-point frequencies
        freqs = np.repeat(counts, counts)
        freqs_trials.append(freqs)
        print(f"[tune] Trial {t+1}/{num_trials}: unique_labels={len(unique)}, "
              f"avg_freq={freqs.mean():.2f}, max_freq={freqs.max()}")

    # Expected calibration count given lambda
    def expected_calib(lam):
        total = 0.0
        for freqs in freqs_trials:
            total += np.sum(max_prob * (1 - np.exp(-lam * (freqs - 1))))
        exp_val = total / num_trials
        print(f"[tune] expected_calib at lambda={lam:.6f} => {exp_val:.2f}")
        return exp_val

    # Bracket for bisection
    lo, hi = 1e-6, 1.0
    print(f"[tune] Bracketing: initial lo={lo}, hi={hi}")
    while expected_calib(hi) < calib_num and hi < 1e3:
        hi *= 2
        print(f"[tune] Increasing hi: now hi={hi}")

    # Bisection to find lambda so expected_calib ≈ calib_num
    for it in range(1, max_iter+1):
        mid = 0.5 * (lo + hi)
        exp_mid = expected_calib(mid)
        if exp_mid < calib_num:
            lo = mid
        else:
            hi = mid
        print(f"[tune] Iter {it}: mid={mid:.6f}, exp_calib={exp_mid:.2f}, lo={lo:.6f}, hi={hi:.6f}")
        if abs(hi - lo) < tol:
            print(f"[tune] Converged after {it} iterations")
            break

    lambda_param = 0.5 * (lo + hi)
    print(f"[tune] Final lambda_param={lambda_param:.6f}, max_prob={max_prob:.4f}")

    return lambda_param, max_prob

