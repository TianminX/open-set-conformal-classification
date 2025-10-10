import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.special import erf


def random_choice(prob, rng):
    """Helper function for weighted random choice"""
    cumsum = np.cumsum(prob)
    return np.searchsorted(cumsum, rng.random())


def dictToList(data):
    """Convert dictionary to list"""
    result = []
    for key, count in data.items():
        result.extend([key] * count)
    return np.array(result)


class DirichletProcessDiscrete:
    """
    Sample labels according to a Dirichlet process with parameter theta
    The base distribution samples discrete labels 1, 2, 3, ...
    """

    def __init__(self, theta):
        assert (theta > 0)
        self.theta = theta
        self.next_new_label = 1
        self.set_random_state()

    def set_random_state(self, random_state=None):
        # Initialize random number generator
        self.rng = np.random.Generator(np.random.PCG64(random_state))

    def _sample_new_label(self):
        """Sample a new label from the base distribution (sequential integers)"""
        label = self.next_new_label
        self.next_new_label += 1
        return label

    def _prob_vec(self, counts):
        # Compute probability of observing a new label or a previously seen label
        m = np.sum(counts)
        k = len(counts)
        # Add place holder for new label
        counts = np.concatenate([[self.theta], counts])
        # Compute probabilities for new and old labels
        prob = counts / (self.theta + m)
        assert np.abs(np.sum(prob) - 1) < 1e-3
        return prob / np.sum(prob)

    def _sample_step(self, data):
        counts = np.fromiter(data.values(), dtype=int)
        prob = self._prob_vec(counts)
        j = random_choice(prob, rng=self.rng)
        if j == 0:
            x = self._sample_new_label()
        else:
            x = list(data.keys())[j - 1]
        return x

    def sample(self, n=1, random_state=None):
        if random_state is not None:
            self.set_random_state(random_state)
            self.next_new_label = 1  # Reset label counter

        # Initialize empty data container
        data = defaultdict(lambda: 0)

        # Sample sequentially
        labels = []
        for i in range(n):
            x = self._sample_step(data)
            data[x] = data[x] + 1
            labels.append(x)

        # Return as numpy array WITHOUT shuffling
        return np.array(labels)


class ClusterProbabilityFeatures:
    """
    Generate features based on P(K_n >= k) where k is the label value
    Using asymptotic approximations for simplicity
    """

    def __init__(self, num_features, sigma, theta):
        self.num_features = num_features
        self.sigma = sigma
        self.theta = theta

    def prob_at_least_k_clusters_approx(self, n, k):
        """
        Approximate P(K_n >= k) using asymptotic results

        For DP, the expected number of clusters is E[K_n] ≈ θ * log(n) for large n
        The distribution is approximately normal for large n
        """
        if k <= 1:
            return 1.0

        # Expected number of clusters
        expected_k = self.theta * np.log(1 + n / self.theta)

        # Approximate variance (rough approximation)
        var_k = expected_k
        std_k = np.sqrt(var_k)

        # Use normal approximation
        # P(K_n >= k) ≈ P(Z >= (k - expected_k)/std_k)
        z_score = (k - 0.5 - expected_k) / std_k  # continuity correction

        # Standard normal CDF
        prob = 0.5 * (1 - erf(z_score / np.sqrt(2)))

        return np.clip(prob, 0.0, 1.0)

    def sample(self, Y, random_state=None):
        n_total = len(Y)
        rng = np.random.Generator(np.random.PCG64(random_state))
        X = np.zeros((n_total, self.num_features))

        # First feature: P(K_n >= label)
        for i, label in enumerate(Y):
            X[i, 0] = self.prob_at_least_k_clusters_approx(n_total, int(label))

        # Additional features: just add Gaussian noise
        if self.num_features > 1:
            noise = rng.normal(loc=0, scale=self.sigma, size=(n_total, self.num_features - 1))
            X[:, 1:] = noise

        return X


class DataDistributionClusterProb:
    """
    Data distribution using cluster probability features
    """

    def __init__(self, theta, num_features, sigma):
        self.theta = theta
        self.label_dist = DirichletProcessDiscrete(theta)
        self.feature_dist = ClusterProbabilityFeatures(num_features, sigma, theta)

    def sample(self, n, random_state=None):
        # Generate labels
        Y = self.label_dist.sample(n, random_state=random_state)

        # Generate features based on cluster probabilities
        X = self.feature_dist.sample(Y, random_state=random_state)

        return X, Y


# Example usage
if __name__ == "__main__":
    # Test the implementation
    theta = 10.0
    n_samples = 2000
    num_features = 1
    sigma = 0.0001

    # Create distribution
    data_dist = DataDistributionClusterProb(theta, num_features, sigma)

    # Sample data
    X, Y = data_dist.sample(n_samples, random_state=42)

    print(f"Data shape: X={X.shape}, Y={Y.shape}")
    print(f"Unique labels: {len(np.unique(Y))}")
    print(f"Label range: [{Y.min()}, {Y.max()}]")

    # Show first 10 samples
    print(f"\nFirst 10 samples:")
    print("Label | P(K_n >= k)")
    print("-" * 45)
    for i in range(min(10, len(Y))):
        print(f"{Y[i]:5d} | {X[i, 0]:11.4f}")

    # Show the approximation
    print(f"\nApproximate P(K_n >= k) for n={n_samples}, θ={theta}:")
    print("-" * 40)
    cf = ClusterProbabilityFeatures(num_features, sigma, theta)
    for k in range(1, 11):
        prob = cf.prob_at_least_k_clusters_approx(n_samples, k)
        print(f"P(K_{n_samples} >= {k}) ≈ {prob:.4f}")