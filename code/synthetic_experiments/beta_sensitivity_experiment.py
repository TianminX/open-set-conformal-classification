"""
Sensitivity analysis for the power-law exponent beta in the combination
p-value psi_seen = max_{k in K_n} psi_k / c_k with
c_k = k^{-beta} / sum_{j=1}^n j^{-beta}  (Eq. combination-pval-power-law).

For each label distribution and sample size, we draw Y_ref repeatedly,
compute the deterministic psi_seen(beta) over a grid of beta values
(same formula as testing.compute_GT_pvalues_testing_old), and record:
  - psi_seen(beta) for every beta on the grid,
  - psi_oracle = sum_k psi_k, the (infeasible) empirically optimal
    combination value, which lower-bounds psi_seen over all fixed
    weight vectors with sum c_k <= 1.

Smaller psi_seen = more powerful test of H_seen, so the curve
E[psi_seen(beta)] vs beta is exactly the population version of the CV
criterion used by testing.select_beta_cv.

Usage:
    python beta_sensitivity_experiment.py [n_reps] [seed] [mode]
Defaults: n_reps=100, seed=2026, mode='all'.
mode='all': DP + Zipf + geometric + uniform (broad robustness sweep).
mode='dp':  DP only, with a fine grid of theta values around the
            sparse (informative) regime.
Output: results/beta_sensitivity/beta_sensitivity[_{mode}]_reps{R}_seed{S}.csv
"""

import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, '../cgtc/')

#####################
# Define parameters #
#####################

n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 100
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 2026
mode = sys.argv[3] if len(sys.argv) > 3 else 'all'

# Grid of beta values; extends below the CV grid [1.0, 3.0] used in the
# main experiments because CV there almost always selects the boundary 1.0
betas = np.round(np.arange(0.0, 3.01, 0.1), 2)

# Sample sizes (n_ref = 2000 matches most synthetic experiments,
# 20000 matches the plugin experiments); the dp mode focuses on the
# smaller sizes where the seen-test is informative
if mode == 'dp':
    n_values = [500, 1000, 1500, 2000]
else:
    n_values = [1000, 2000, 5000, 20000]

# Label distributions: (family, parameter value)
if mode == 'dp':
    # Fine theta grid around the sparse regime where the seen-test is
    # informative (roughly theta comparable to or larger than n)
    dist_configs = [('dp', theta) for theta in
                    [750, 1000, 1250, 1500, 1750, 2000, 2500, 3000]]
elif mode == 'all':
    dist_configs = (
        [('dp', theta) for theta in [12, 100, 500, 1500]] +
        [('zipf', a) for a in [1.5, 2.0, 3.0]] +
        [('geometric', p) for p in [0.01, 0.001]] +
        [('uniform', K) for K in [100, 1000]]
    )
else:
    raise ValueError(f"Unknown mode: {mode}")

mode_label = "" if mode == 'all' else f"_{mode}"
output_file = (f"results/beta_sensitivity/"
               f"beta_sensitivity{mode_label}_reps{n_reps}_seed{seed}.csv")
os.makedirs(os.path.dirname(output_file), exist_ok=True)
print(f"Output file name: {output_file}")


########################
# Label-count samplers #
########################

def sample_crp_counts(n, theta, rng):
    """Cluster sizes of a Chinese restaurant process (Polya urn) sample of
    size n with concentration theta. Same label-frequency law as
    distributions_y.DirichletProcess (the base-measure draws are a.s.
    distinct), but O(n) instead of O(n*K)."""
    table = np.empty(n, dtype=np.int64)
    table[0] = 0
    num_tables = 1
    u = rng.uniform(size=n)
    # Individual i starts a new table w.p. theta/(theta+i); otherwise it
    # joins the table of a uniformly chosen previous individual, which is
    # equivalent to choosing an existing table with prob proportional to size
    for i in range(1, n):
        if u[i] < theta / (theta + i):
            table[i] = num_tables
            num_tables += 1
        else:
            table[i] = table[rng.integers(0, i)]
    return np.bincount(table)


def sample_counts(family, param, n, rng):
    if family == 'dp':
        return sample_crp_counts(n, param, rng)
    if family == 'zipf':
        labels = rng.zipf(a=param, size=n)
    elif family == 'geometric':
        labels = rng.geometric(p=param, size=n)
    elif family == 'uniform':
        labels = rng.integers(0, int(param), size=n)
    else:
        raise ValueError(f"Unknown family: {family}")
    # np.unique instead of np.bincount: heavy-tailed Zipf labels can be ~1e8
    _, counts = np.unique(labels, return_counts=True)
    return counts


#############################
# psi_seen(beta) evaluation #
#############################

def psi_seen_curve(counts, n, betas, rng, n_rand=20, log_j_cache={}):
    """psi_seen(beta) for every beta on the grid, using both the
    deterministic GT p-values and the randomized RGT p-values, plus the
    oracle value. Mirrors testing.compute_GT_pvalues_testing_old:
      candidate_f = (M_{f+1} + f + 1) / (n + 1) for observed frequency f,
      c_f = f^{-beta} / sum_{j=1}^n j^{-beta},
      psi_seen(beta) = max_f candidate_f / c_f,
      psi_oracle = sum_f candidate_f.
    The RGT variant replaces candidate_f with (U_f + 1)/(n + 1),
    U_f ~ Uniform{0, ..., (f+1) M_{f+1} + f} independently across f
    (paper Eq. p-value-RGT), and both normalizations use sum over [n]
    as in the paper. Returns:
      psi_gt (per beta), psi_oracle,
      rgt_trunc (per beta): mean over n_rand draws of min(psi_rgt, 1),
      rgt_raw   (per beta): mean over n_rand draws of psi_rgt.
    """
    if n not in log_j_cache:
        log_j_cache[n] = np.log(np.arange(1, n + 1))
    log_j = log_j_cache[n]

    count_of_count = np.bincount(counts, minlength=counts.max() + 2)
    unique_f = np.flatnonzero(count_of_count[:-1])  # observed frequencies
    M_f1 = count_of_count[unique_f + 1] * (unique_f + 1)
    candidates = (M_f1 + unique_f + 1) / (n + 1)

    # log Z(beta) = log sum_j j^{-beta}, computed stably for all betas at once
    log_terms = -betas[:, None] * log_j[None, :]
    log_Z = np.logaddexp.reduce(log_terms, axis=1)

    # Deterministic GT: psi(beta) = max_f candidates_f * f^beta * Z(beta)
    log_f_beta = betas[:, None] * np.log(unique_f)[None, :]
    log_psi = (np.log(candidates)[None, :] + log_f_beta).max(axis=1) + log_Z
    psi_gt = np.exp(log_psi)

    # Randomized RGT: U_f in {0, ..., (f+1) M_{f+1} + f}, n_rand draws
    U = rng.integers(0, M_f1 + unique_f + 1, size=(n_rand, len(unique_f)))
    log_cand_rand = np.log((U + 1) / (n + 1))
    log_psi_rand = (log_cand_rand[:, None, :] + log_f_beta[None, :, :]
                    ).max(axis=2) + log_Z[None, :]
    psi_rgt = np.exp(log_psi_rand)  # shape (n_rand, n_beta)
    rgt_trunc = np.minimum(psi_rgt, 1.0).mean(axis=0)
    rgt_raw = psi_rgt.mean(axis=0)

    return psi_gt, candidates.sum(), rgt_trunc, rgt_raw


#################
# Run the study #
#################

def main():
    rows = []
    for family, param in tqdm(dist_configs, desc="distributions"):
        for n in n_values:
            rng = np.random.Generator(np.random.PCG64([seed, hash((family, param, n)) % 2**31]))
            for rep in range(n_reps):
                counts = sample_counts(family, param, n, rng)
                psi, psi_oracle, rgt_trunc, rgt_raw = psi_seen_curve(
                    counts, n, betas, rng)
                best_beta = betas[np.argmin(psi)]
                for i, b in enumerate(betas):
                    rows.append({
                        'family': family,
                        'param': param,
                        'n': n,
                        'rep': rep,
                        'beta': b,
                        'psi_seen': psi[i],
                        'psi_rgt_trunc': rgt_trunc[i],
                        'psi_rgt_raw': rgt_raw[i],
                        'psi_oracle': psi_oracle,
                        'best_beta': best_beta,
                        'num_labels': len(counts),
                        'M1': int(np.sum(counts == 1)),
                    })

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"Wrote {len(df)} rows to {output_file}")

    # Console summary: minimizing beta and psi at beta = 1.0 / 1.6 vs oracle
    summary = (df.groupby(['family', 'param', 'n', 'beta'])['psi_seen']
                 .mean().reset_index())
    for (family, param, n), g in summary.groupby(['family', 'param', 'n']):
        g = g.set_index('beta')['psi_seen']
        b_star = g.idxmin()
        oracle = df[(df.family == family) & (df.param == param)
                    & (df.n == n)]['psi_oracle'].mean()
        print(f"{family}(param={param}), n={n}: "
              f"argmin beta={b_star:.1f} (psi={g[b_star]:.4f}), "
              f"psi(1.0)={g[1.0]:.4f}, psi(1.6)={g[1.6]:.4f}, "
              f"oracle={oracle:.4f}")


if __name__ == '__main__':
    main()
