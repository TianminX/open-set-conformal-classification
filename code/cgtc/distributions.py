import numpy as np
import pdb
from collections import defaultdict, OrderedDict

from utils import random_choice, dictToList, listToDict

class DistributionXY:
    "Class for generating classification data with infinitely many possible labels."

    def __init__(self, dist_labels, dist_features):
        self.dist_labels = dist_labels
        self.dist_features = dist_features

    def sample(self, n, random_state=None):
        Y = self.dist_labels.sample(n, random_state=random_state)
        X = self.dist_features.sample(Y, random_state=random_state)
        return X,Y


class ShiftedZipfNormal:
    """
    Assumes Y is a vector of labels sampled from a Zipf distribution
    Conditional on Y=y, sample X from a multivariate normal distribution with independent components,
    and one component shifted by y
    Parameters:
     - num_features: feature dimensions
     - sigma: standard deviation
     - a: parameter of the Zipf distribution
    """

    def __init__(self, num_features, sigma, a):
        self.num_features = num_features
        self.sigma = sigma
        self.a = a

    def sample_labels(self, n, random_state=None):
        rng = np.random.Generator(np.random.PCG64(random_state))
        return rng.zipf(a=self.a, size=n)

    def sample_features(self, Y, random_state=None):
        n = len(Y)
        rng = np.random.Generator(np.random.PCG64(random_state))
        X = rng.normal(loc=0, scale=self.sigma, size=(n, self.num_features))
        for k in range(X.shape[1]):
            X[:, k] += Y.astype(float)
        return X

    def sample(self, n, random_state=None):
        Y = self.sample_labels(n, random_state)
        X = self.sample_features(Y, random_state)
        return X, Y

