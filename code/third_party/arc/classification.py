import numpy as np

class ProbabilityAccumulator:
    def __init__(self, prob):
        self.n, self.K = prob.shape
        self.order = np.argsort(-prob, axis=1)
        self.ranks = np.empty_like(self.order)
        for i in range(self.n):
            self.ranks[i, self.order[i]] = np.arange(len(self.order[i]))
        self.prob_sort = -np.sort(-prob, axis=1)
        self.epsilon = np.random.uniform(low=0.0, high=1.0, size=self.n)
        self.Z = np.round(self.prob_sort.cumsum(axis=1),9)        
        
    def predict_sets(self, alpha, epsilon=None, allow_empty=True):
        if alpha>0:
            L = np.argmax(self.Z >= 1.0-alpha, axis=1).flatten()
        else:
            L = (self.Z.shape[1]-1)*np.ones((self.Z.shape[0],)).astype(int)
        if epsilon is not None:
            Z_excess = np.array([ self.Z[i, L[i]] for i in range(self.n) ]) - (1.0-alpha)
            p_remove = Z_excess / np.array([ self.prob_sort[i, L[i]] for i in range(self.n) ])
            remove = epsilon <= p_remove
            for i in np.where(remove)[0]:
                if not allow_empty:
                    L[i] = np.maximum(0, L[i] - 1)  # Note: avoid returning empty sets
                else:
                    L[i] = L[i] - 1

        # Return prediction set
        S = [ self.order[i,np.arange(0, L[i]+1)] for i in range(self.n) ]
        return(S)

    def calibrate_scores(self, Y, epsilon=None):
        Y = np.atleast_1d(Y)
        n2 = len(Y)
        ranks = np.array([ self.ranks[i,Y[i]] for i in range(n2) ])
        prob_cum = np.array([ self.Z[i,ranks[i]] for i in range(n2) ])
        prob = np.array([ self.prob_sort[i,ranks[i]] for i in range(n2) ])
        alpha_max = 1.0 - prob_cum
        if epsilon is not None:
            alpha_max += np.multiply(prob, epsilon)
        else:
            alpha_max += prob
        alpha_max = np.minimum(alpha_max, 1)
        return alpha_max

    # def calibrate_scores(self, Y, epsilon=None):
    #     Y = np.atleast_1d(Y)
    #     n2 = len(Y)
    #     ranks = np.zeros(n2)
    #
    #     # For unseen labels we add a small positive noise and later adjust the cumulative prob.
    #     noise = np.zeros(n2)
    #     for i in range(n2):
    #         if Y[i] < self.K:
    #             ranks[i] = self.ranks[i, Y[i]]
    #             noise[i] = 0.0  # No noise needed for seen labels.
    #         else:
    #             # For labels not seen in training, set rank to self.K (out-of-bound)
    #             # and assign a small positive noise.
    #             ranks[i] = self.K
    #             noise[i] = np.random.uniform(low=1e-6, high=1e-5)
    #
    #     for i in range(n2):
    #         if Y[i] < self.K:
    #             ranks[i] = self.ranks[i, Y[i]]
    #         else:
    #             ranks[i] = self.K  # Use the highest rank (out of bounds)
    #     prob_cum = np.array([self.Z[i, int(ranks[i])] if ranks[i] < self.K else 1.0 for i in range(n2)])
    #     prob = np.array([self.prob_sort[i, int(ranks[i])] if ranks[i] < self.K else 0.0 for i in range(n2)])
    #
    #     # for i in range(n2):
    #     #     if Y[i] < self.K:
    #     #         ranks[i] = self.ranks[i, Y[i]]
    #     #     else:
    #     #         ranks[i] = self.K - 1  # Use the last rank for unseen labels
    #     # prob_cum = np.array([self.Z[i, int(ranks[i])] for i in range(n2)])
    #     # prob = np.array([self.prob_sort[i, int(ranks[i])] for i in range(n2)])
    #
    #     alpha_max = 1.0 - prob_cum
    #     if epsilon is not None:
    #         alpha_max += np.multiply(prob, epsilon)
    #     else:
    #         alpha_max += prob
    #     alpha_max = np.minimum(alpha_max, 1)
    #     return alpha_max


# class ProbabilityAccumulatorPerturbed:
#     def __init__(self, prob):
#         """
#         prob: an (n x K_orig) array of predicted probabilities,
#               where K_orig is the number of classes seen in training.
#         For each row we add an extra entry (column) to represent the probability
#         assigned to any label that was unseen in training. The extra entry is
#         drawn from a uniform distribution whose upper bound is given by the smallest
#         predicted probability (among seen labels) in that row.
#         """
#         n, K_orig = prob.shape
#         new_prob = np.empty((n, K_orig + 1))
#         for i in range(n):
#             p = prob[i].copy()
#             # Compute the smallest positive predicted probability among seen classes.
#             # (If none are positive, use a tiny default value.)
#             positive_probs = p[p > 0]
#             if positive_probs.size > 0:
#                 p_min = positive_probs.min()
#             else:
#                 p_min = 1e-12
#             # Draw a small positive noise for the unseen label.
#             # (You can adjust the low bound if you wish.)
#             noise = np.random.uniform(low=1e-12, high=p_min)
#             # Append the noise to the seen probabilities.
#             p_aug = np.concatenate([p, [noise]])
#             # Renormalize so that the new probability vector sums to 1.
#             p_aug = p_aug / p_aug.sum()
#             new_prob[i] = p_aug
#
#         self.prob = new_prob
#         # Note: self.K is now K_orig + 1, with the last column for unseen labels.
#         self.n, self.K = new_prob.shape
#
#         # Compute the order, ranks, sorted probabilities, and cumulative sum.
#         self.order = np.argsort(-new_prob, axis=1)
#         self.ranks = np.empty_like(self.order)
#         for i in range(self.n):
#             # For each row, the element with the highest probability gets rank 0, etc.
#             self.ranks[i, self.order[i]] = np.arange(len(self.order[i]))
#         self.prob_sort = -np.sort(-new_prob, axis=1)
#         self.Z = np.round(self.prob_sort.cumsum(axis=1), 9)
#
#     def predict_sets(self, alpha, epsilon=None, allow_empty=True):
#         if alpha > 0:
#             L = np.argmax(self.Z >= 1.0 - alpha, axis=1).flatten()
#         else:
#             L = (self.Z.shape[1] - 1) * np.ones((self.Z.shape[0],)).astype(int)
#         if epsilon is not None:
#             Z_excess = np.array([self.Z[i, L[i]] for i in range(self.n)]) - (1.0 - alpha)
#             p_remove = Z_excess / np.array([self.prob_sort[i, L[i]] for i in range(self.n)])
#             remove = epsilon <= p_remove
#             for i in np.where(remove)[0]:
#                 if not allow_empty:
#                     L[i] = np.maximum(0, L[i] - 1)  # avoid returning empty sets
#                 else:
#                     L[i] = L[i] - 1
#
#         # Return the prediction set for each sample.
#         S = [self.order[i, np.arange(0, L[i] + 1)] for i in range(self.n)]
#         return S
#
#     def calibrate_scores(self, Y, epsilon=None):
#         """
#         Y: an array of true labels (of length n2). For any Y[i] that is
#            among the seen classes (i.e. less than K_orig), we use its corresponding
#            rank. For any Y[i] that is not seen (i.e. >= K_orig), we use the extra
#            (last) column. Note that self.K = K_orig + 1.
#         """
#         Y = np.atleast_1d(Y)
#         n2 = len(Y)
#         ranks = np.zeros(n2)
#
#         # For seen labels, use their rank; for unseen ones, use the extra column.
#         for i in range(n2):
#             if Y[i] < self.K - 1:  # seen labels (since self.K - 1 equals original K)
#                 ranks[i] = self.ranks[i, Y[i]]
#             else:
#                 ranks[i] = self.ranks[i, self.K - 1]  # unseen label
#         # Look up the cumulative probability and the probability of the “selected” element.
#         prob_cum = np.array([self.Z[i, int(ranks[i])] for i in range(n2)])
#         prob = np.array([self.prob_sort[i, int(ranks[i])] for i in range(n2)])
#
#         alpha_max = 1.0 - prob_cum
#         if epsilon is not None:
#             alpha_max += np.multiply(prob, epsilon)
#         else:
#             alpha_max += prob
#         alpha_max = np.minimum(alpha_max, 1)
#         return alpha_max
