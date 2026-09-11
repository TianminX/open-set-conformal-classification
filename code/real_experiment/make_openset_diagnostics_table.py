"""Generate the diagnostics table of the open-set appendix (LaTeX).

For each raw benchmark score it aggregates, over the same batches as
make_openset_benchmark_table.py and at n_ref = 2000 and 6000:
  AUC     - area under the ROC curve of the unknown-class probability as a
            detector of training-level novelty (test identity absent from the
            training split); stored column `p_open AUC`
  p_seen  - average unknown probability over test points whose identity is
            present in the training split; column `p_open mean (seen_train)`
  ratio   - average unknown probability over test points whose identity is
            absent from the whole reference sample (`p_open mean (novel)`)
            divided by p_seen
  top1    - top-1 accuracy of the seen-class component on test points whose
            identity is present in the training split; column
            `Top1 Accuracy (train)`
The recalibrated variants share the AUC and (up to the cap) the ratio of
their raw score, so only raw scores are tabulated.

Output: ../../table_openset_diagnostics.tex (repository root) and a copy in
../../Open_Set_Conformal_Classification/tables/ when that folder exists.
Run: python make_openset_diagnostics_table.py  (from code/real_experiment)
"""

import glob
import math
import os
import re

import pandas as pd

ALPHA = 0.2
NLABEL = 2000
NREFS = [2000, 6000]
HERE = os.path.dirname(os.path.abspath(__file__))
OUTS = [os.path.join(HERE, "..", "..", "table_openset_diagnostics.tex"),
        os.path.join(HERE, "..", "..", "Open_Set_Conformal_Classification",
                     "tables", "table_openset_diagnostics.tex")]


def load(folder, method):
    files = glob.glob(os.path.join(HERE, "results_hpc", folder, "*.csv"))
    if not files:
        raise FileNotFoundError(f"no CSVs in results_hpc/{folder}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df[(abs(df["alpha_total"] - ALPHA) < 1e-10)
              & (df["n_label_total"] == NLABEL)
              & (df["k_top"] == 0) & (df["k_bot"] == 0)
              & (df["method"] == method)]


RECAL_MODE = os.environ.get("OPENSET_RECAL_MODE", "center")


def bench(family):
    return f"celeb_bench_{family}_{RECAL_MODE}"


def available(folder, method):
    try:
        df = load(folder, method)
    except FileNotFoundError:
        return None
    if any((df["n_ref"] == nr).sum() == 0 for nr in NREFS):
        return None
    return df


# (label, candidates [(folder, method), ...]); the first complete candidate is
# used, unified-runner folders first (see make_openset_benchmark_table.py).
ROWS = [
    ("OpenMax (simplified)", [("celeb_openmax",        "Method (OpenMax-MLP)")]),
    ("OpenMax (faithful)",   [("celeb_openmax_osdn",   "Method (OpenMax-MLP)")]),
    ("OpenMax-KNN",          [("celeb_openmax",        "Method (OpenMax-KNN)")]),
    ("KNN-dist",             [("celeb_knn_scores_raw", "Method (KNN-dist)")]),
    ("KNN-dist (k=1)",       [(bench("knn1"),          "Method (KNN-dist k=1)")]),
    ("KNN-MSP",              [("celeb_knn_scores_raw", "Method (KNN-MSP)")]),
    ("PROSER",               [("celeb_proser",         "Method (PROSER)")]),
    ("OCC (centered LOF)",   [(bench("occ"), "Method (OCC lof)"), ("celeb_occ", "Method (OCC)")]),
    ("OCC (IF)",             [(bench("occ"), "Method (OCC iforest)")]),
    ("OCC (OCSVM)",          [(bench("occ"), "Method (OCC ocsvm)")]),
    ("OCC (OCSVM, $\\gamma{=}20$)", [(bench("occ"), "Method (OCC ocsvm20)")]),
    ("Naive (GT constant)",  [("celeb_gt_knn",         "Method (GT-KNN)")]),
]
resolved = []
for name, cands in ROWS:
    for folder, method in cands:
        df = available(folder, method)
        if df is not None:
            resolved.append((name, df))
            break
    else:
        print(f"skipping {name}: no complete results in {[c[0] for c in cands]}")
names = [n for n, _ in resolved]
if "KNN-dist (k=1)" in names:
    resolved = [("KNN-dist (k=10)" if n == "KNN-dist" else n, d) for n, d in resolved]

cells = []
ses_auc, ses_p = [], []
for name, df in resolved:
    row = [name]
    for nr in NREFS:
        g = df[df["n_ref"] == nr]
        n = len(g)
        auc = g["p_open AUC"].mean()
        p_seen = g["p_open mean (seen_train)"].mean()
        p_novel = g["p_open mean (novel)"].mean()
        top1 = g["Top1 Accuracy (train)"].mean()
        beats = g["P(p_open > p_true)"].mean()
        # the ratio is left blank when the seen-point level is itself zero to
        # the displayed precision (degenerate faithful OpenMax), where a ratio
        # of two near-zero means carries no information
        ratio = p_novel / p_seen if p_seen >= 0.005 else float("nan")
        ses_auc.append(g["p_open AUC"].std() / math.sqrt(n))
        ses_p.append(g["p_open mean (seen_train)"].std() / math.sqrt(n))
        ratio_tex = "" if math.isnan(ratio) else (f"{ratio:.1f}" if ratio < 100 else f"{ratio:.0f}")
        row += [f"{auc:.2f}", f"{p_seen:.2f}", ratio_tex, f"{beats:.2f}", f"{top1:.2f}"]
    cells.append(row)

max_se_auc = math.ceil(max(ses_auc) * 100) / 100
max_se_p = math.ceil(max(ses_p) * 100) / 100

lines = []
lines.append("% Auto-generated by code/real_experiment/make_openset_diagnostics_table.py")
lines.append("% Do not edit by hand; rerun the script after refreshing results.")
lines.append(r"\begin{table}[!htb]")
lines.append(r"\centering")
lines.append(
    r"\caption{Diagnostics of the raw unknown-class probabilities on CelebA ($\alpha = 0.2$, $2{,}000$ sampled identities, averages over the same batches as Table~\ref{tab:app-openset-benchmarks}) at the smallest and largest reference sample sizes. "
    r"\emph{AUC}: area under the ROC curve of the unknown probability as a detector of novelty at the training level, that is, for separating test points whose identity is absent from the training split from those whose identity is present. "
    r"\emph{$\bar{p}_{\mathrm{unk}}$}: average unknown probability over test points whose identity is present in the training split. "
    r"\emph{Ratio}: average unknown probability over test points whose identity is absent from the entire reference sample, divided by the previous column; a multiplicative recalibration preserves this ratio up to the effect of the cap, so each recalibrated variant shares the AUC and the ratio of its raw score. "
    r"\emph{$p_{\mathrm{unk}} > p_{Y}$}: fraction of test points whose identity is present in the training split for which the unknown probability exceeds the probability of the true identity. "
    r"\emph{Top-1}: accuracy of the seen-class component on test points whose identity is present in the training split. "
    f"Monte Carlo standard errors are at most ${max_se_auc:.2f}$ for the AUC and at most ${max_se_p:.2f}$ for $\\bar{{p}}_{{\\mathrm{{unk}}}}$.}}"
)
lines.append(r"\label{tab:app-openset-diagnostics}")
lines.append(r"\small")
lines.append(r"\setlength{\tabcolsep}{3.5pt}")
lines.append(r"\begin{tabular}{lcccccccccc}")
lines.append(r"\toprule")
lines.append(r"& \multicolumn{5}{c}{$n_{\mathrm{ref}} = 2{,}000$} & \multicolumn{5}{c}{$n_{\mathrm{ref}} = 6{,}000$} \\")
lines.append(r"\cmidrule(lr){2-6} \cmidrule(lr){7-11}")
lines.append(r"Score & AUC & $\bar{p}_{\mathrm{unk}}$ & Ratio & $p_{\mathrm{unk}} > p_{Y}$ & Top-1 & AUC & $\bar{p}_{\mathrm{unk}}$ & Ratio & $p_{\mathrm{unk}} > p_{Y}$ & Top-1 \\")
lines.append(r"\midrule")
for row in cells:
    if row[0].startswith("Naive"):
        lines.append(r"\midrule")
    lines.append(" & ".join(row) + r" \\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

tex = "\n".join(lines) + "\n"
for out in OUTS:
    if os.path.isdir(os.path.dirname(out)):
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            f.write(tex)
        print("wrote", os.path.normpath(out))
print(tex)
