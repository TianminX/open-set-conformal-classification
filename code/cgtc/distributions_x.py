import numpy as np
import pdb

class ShiftedNormal:
    """
    Assumes Y is a vector of labels
    Assumes each label is a real number within the interval [0,1]
    Conditional on Y=y, sample X from a multivariate normal distribution with independent components, and one component shifted by y
    Parameters:
     - num_features: feature dimensions
     - sigma: standard deviation
    """

    def __init__(self, num_features, sigma):
        self.num_features = num_features
        self.sigma = sigma

    def sample(self, Y, random_state=None):
        n = len(Y)
        rng = np.random.Generator(np.random.PCG64(random_state))
        X = rng.normal(loc=0, scale=self.sigma, size=(n,self.num_features))
        for k in range(X.shape[1]):
            X[:,k] += Y.astype(float)
        return X


