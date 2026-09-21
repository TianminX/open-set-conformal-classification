"""Effective sample size of the selective-splitting conformalization weights.

For each candidate label y in Y_n, the weighted closed-set p-value of Section 4
(selective splitting) uses normalized weights w_j(y), j in the calibration set,
computed from the ratios ptilde_j(y) = p^{(j)}(y) / p^{(n+1)}(y) of the
computational shortcut appendix.  This script evaluates, on the label profiles
of the paper's experiments (no model fitting is involved, the weights depend on
the labels and the split only):

  * Kish's effective sample size  n_eff(y) = (sum_j w_j(y))^2 / sum_j w_j(y)^2,
    reported as a fraction of the realized calibration size n_cal;
  * the share of calibration points receiving weight zero (doubleton labels
    with both copies in the calibration set);
  * the lower bound (1 - c)^2 (1 - zero share) of Appendix "ESS" ;
  * the test-point effective size sum_j ptilde_j(y) = 1/w_{n+1}(y) - 1.

Label profiles:
  * Dirichlet-process labels exactly as in the synthetic experiments
    (cgtc.distributions_y.DirichletProcess, seeds batch*1000 + i);
  * CelebA identity labels exactly as in the real-data experiments
    (2,000 identities sampled with seed 42, reference sets drawn with
    seeds batch*1000 + i as in split_data).

Usage (from code/synthetic_experiments/):
  python ess_selective_splitting.py [--dp-reps 20] [--celeb-reps 20]
                                    [--no-celeb] [--out results/ess_selective_splitting.csv]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "cgtc"))
from distributions_y import DirichletProcess  # noqa: E402

# ----------------------------------------------------------------------------
# Selective splitting rule of the paper: pi(1) = 0, pi(k) = c for k >= 2,
# c = min(calprop / (1 - p1), 1), p1 = M_1 / n (see calib_prob_real in the
# experiment scripts).
# ----------------------------------------------------------------------------


def make_pi(c):
    def pi(k):
        k = np.asarray(k)
        return np.where(k >= 2, c, 0.0)
    return pi


def ratios_for_label(y, Yc, N, f_cal, f_tr, pi):
    """ptilde_j(y) for every calibration point j (general pi, App. shortcut)."""
    Ns, fcs, fts = N[Yc], f_cal[Yc], f_tr[Yc]
    Nt, fct, ftt = N[y], f_cal[y], f_tr[y]
    with np.errstate(divide="ignore", invalid="ignore"):
        num = (pi(Ns - 1) ** (fcs - 1)) * ((1 - pi(Ns - 1)) ** fts)
        num = num * (pi(Nt + 1) ** (fct + 1)) * ((1 - pi(Nt + 1)) ** ftt)
        den = (pi(Ns) ** fcs) * ((1 - pi(Ns)) ** fts)
        den = den * (pi(Nt) ** fct) * ((1 - pi(Nt)) ** ftt)
        r = num / den
    return np.where(Yc == y, 1.0, r)


def ess_one_replicate(Y, calprop, rng):
    """Split Y selectively and summarize the weights over all candidate labels."""
    _, Yi, N_by_lab = np.unique(Y, return_inverse=True, return_counts=True)
    N = N_by_lab
    n = len(Yi)
    p1 = np.mean(N[Yi] == 1)
    c = min(calprop / (1 - p1), 1.0)
    pi = make_pi(c)
    I = rng.uniform(size=n) < pi(N[Yi])
    cal_idx = np.where(I)[0]
    n_cal = len(cal_idx)
    Yc = Yi[cal_idx]
    f_cal = np.bincount(Yc, minlength=len(N))
    f_tr = N - f_cal
    M1 = int(np.sum(N == 1))
    M2 = int(np.sum(N == 2))

    kish, zero, sptilde, sing = [], [], [], []
    for y in range(len(N)):
        r = ratios_for_label(y, Yc, N, f_cal, f_tr, pi)
        s, s2 = r.sum(), (r ** 2).sum()
        kish.append(s ** 2 / s2)
        zero.append(n_cal - int(np.sum(r > 0)))
        sptilde.append(s)
        sing.append(N[y] == 1)
    kish, zero, sptilde, sing = map(np.asarray, (kish, zero, sptilde, sing))
    zero_share = zero.mean() / n_cal
    return dict(
        n=n, n_labels=len(N), M1=M1, M2=M2, p1=p1, c=c, n_cal=n_cal,
        zero_share=zero_share,
        ess_ratio_mean=np.mean(kish) / n_cal,
        ess_ratio_min=np.min(kish) / n_cal,
        ess_lower_bound=(1 - c) ** 2 * (1 - zero_share),
        test_ess_ratio_singleton=np.mean(sptilde[sing]) / n_cal if sing.any() else np.nan,
        test_ess_ratio_recurring=np.mean(sptilde[~sing]) / n_cal if (~sing).any() else np.nan,
        expected_zero_share=2 * M2 * c / max(n - M1, 1),
    )


def dp_profiles(theta, n, reps):
    """Replicates the label draws of the synthetic experiments (num_exp = 5 per batch)."""
    dist = DirichletProcess(theta=theta)
    k = 0
    for batch in range(1, 10 ** 6):
        for i in range(5):
            if k >= reps:
                return
            yield batch * 1000 + i, np.asarray(dist.sample(n, random_state=batch * 1000 + i))
            k += 1


def celeb_profiles(npz_path, n_ref, reps, n_label_total=2000):
    """Replicates sample_labels_and_filter (seed 42) and split_data of the real experiments."""
    Y_full = np.load(npz_path)["Y"]
    np.random.seed(42)
    unique_labels = np.unique(Y_full)
    selected = np.random.choice(unique_labels, size=n_label_total, replace=False)
    Y = Y_full[np.isin(Y_full, selected)]
    for batch in range(1, reps + 1):
        rs = batch * 1000 + 0
        np.random.seed(rs)
        idx = np.arange(len(Y))
        np.random.shuffle(idx)
        yield rs, Y[idx[:n_ref]]


def summarize(rows):
    df = pd.DataFrame(rows)
    keys = ["setting", "theta", "n", "calprop"]
    agg = df.groupby(keys, dropna=False).agg(
        reps=("c", "size"), c=("c", "mean"), n_labels=("n_labels", "mean"),
        p1=("p1", "mean"), n_cal=("n_cal", "mean"),
        zero_share=("zero_share", "mean"), zero_share_sd=("zero_share", "std"),
        ess_ratio=("ess_ratio_mean", "mean"), ess_ratio_sd=("ess_ratio_mean", "std"),
        ess_ratio_min=("ess_ratio_min", "mean"),
        ess_lower_bound=("ess_lower_bound", "mean"),
        test_ess_singleton=("test_ess_ratio_singleton", "mean"),
        test_ess_recurring=("test_ess_ratio_recurring", "mean"),
        expected_zero_share=("expected_zero_share", "mean"),
    ).reset_index()
    return df, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dp-reps", type=int, default=20)
    ap.add_argument("--celeb-reps", type=int, default=20)
    ap.add_argument("--no-celeb", action="store_true")
    ap.add_argument("--thetas", type=str, default="12,25,50,100,200,300,400,500,600,700,800,900,1000")
    ap.add_argument("--out", type=str, default=os.path.join(HERE, "results", "ess_selective_splitting.csv"))
    ap.add_argument("--npz", type=str, default=os.path.join(HERE, "..", "real_experiment", "combined_data.npz"))
    args = ap.parse_args()

    rng = np.random.default_rng(2026)
    rows = []
    thetas = [int(t) for t in args.thetas.split(",") if t]

    # (a) main synthetic grid: n = 2000, 10% calibration
    for theta in thetas:
        for rs, Y in dp_profiles(theta, 2000, args.dp_reps):
            d = ess_one_replicate(Y, 0.10, rng)
            d.update(setting="DP", theta=theta, calprop=0.10, seed=rs)
            rows.append(d)
        print(f"DP theta={theta} done", flush=True)

    # (b) calibration-size sweep of the appendix: theta = 1000, cs in {100, 200, 400, 1000}
    for cs in [100, 200, 400, 1000]:
        for rs, Y in dp_profiles(1000, 2000, args.dp_reps):
            d = ess_one_replicate(Y, cs / 2000, rng)
            d.update(setting="DP-varycal", theta=1000, calprop=cs / 2000, seed=rs)
            rows.append(d)
        print(f"DP vary-cal cs={cs} done", flush=True)

    # (c) CelebA: 2,000 identities, n_ref in {2000, ..., 6000}, 10% calibration
    if not args.no_celeb and os.path.exists(args.npz):
        for n_ref in [2000, 3000, 4000, 5000, 6000]:
            for rs, Y in celeb_profiles(args.npz, n_ref, args.celeb_reps):
                d = ess_one_replicate(Y, int(0.1 * n_ref) / n_ref, rng)
                d.update(setting="CelebA", theta=np.nan, calprop=0.10, seed=rs)
                rows.append(d)
            print(f"CelebA n_ref={n_ref} done", flush=True)

    df, agg = summarize(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out.replace(".csv", "_replicates.csv"), index=False)
    agg.to_csv(args.out, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(agg.round(4).to_string(index=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
