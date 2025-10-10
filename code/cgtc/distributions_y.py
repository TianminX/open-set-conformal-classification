import numpy as np
import pdb
from collections import defaultdict, OrderedDict

from utils import random_choice, dictToList, listToDict

class DirichletProcess:
    """
    Sample labels according to a Dirichlet process with parameter theta
    The base distribution is Uniform(0,1)
    """
    def __init__(self, theta):
        assert (theta>0)
        self.theta = theta
        self.set_random_state()

    def set_random_state(self, random_state=None):
        # Initialize random number generator
        self.rng = np.random.Generator(np.random.PCG64(random_state))

        # Initialize the base distribution
        self.P0 = self.rng.uniform

    def _prob_vec(self, counts):
        # Compute probability of observing a new label or a previously seen label
        m = np.sum(counts)
        k = len(counts)
        # Add place holder for new label
        counts = np.concatenate([[self.theta], counts])
        # Compute probabilities for new and old labels
        prob = counts / (self.theta + m)
        assert np.abs(np.sum(prob) - 1) < 1e-3
        return prob/np.sum(prob)

    def _sample_step(self, data):
        counts = np.fromiter(data.values(), dtype=int)
        prob = self._prob_vec(counts)
        #j = np.random.choice(len(prob), 1, p=prob)[0]
        j = random_choice(prob, rng=self.rng)
        if j == 0:
            x = np.round(self.P0(), 6)
        else:
            x = list(data.keys())[j-1]
        return x

    def sample(self, n=1, random_state=None):
        if random_state is not None:
            self.set_random_state(random_state)

        # Initialize empty data container
        data = defaultdict(lambda: 0)

        # Sample sequentially
        for i in range(n):
            x = self._sample_step(data)
            data[x] = data[x]+1
        if n==1:
            data = x

        # Convert to vector
        data_vec = dictToList(data)
        self.rng.shuffle(data_vec)

        return data_vec

class ZipfDist:
    """
    Sample labels according to a Zipf distribution with parameter a.
    """

    def __init__(self, a):
        assert (a > 0)
        self.a = a
        self.set_random_state()

    def set_random_state(self, random_state=None):
        # Initialize random number generator
        self.rng = np.random.Generator(np.random.PCG64(random_state))

    def _sample_step(self, data):
        # Sample from the Zipf distribution
        x = self.rng.zipf(a=self.a)
        return x

    def sample(self, n=1, random_state=None):
        if random_state is not None:
            self.set_random_state(random_state)

        # Initialize empty data container
        data = defaultdict(lambda: 0)

        # Sample sequentially
        for i in range(n):
            x = self._sample_step(data)
            data[x] = data[x] + 1
        if n == 1:
            data = x

        # Convert to vector
        data_vec = dictToList(data)
        self.rng.shuffle(data_vec)

        return data_vec
