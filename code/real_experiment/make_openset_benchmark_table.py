"""Generate the full open-set benchmark summary table (LaTeX).

Aggregates every benchmark of the open-set appendix (raw, OpenMax family,
recalibrated) together with CGTC-MM at n_ref = 2000 and 6000, and writes a
self-contained table file that the appendix pulls in via
\\input{table_openset_benchmarks}. Output: ../../table_openset_benchmarks.tex
(repository root; upload next to the appendix tex).

Conventions match real_celeb_compare_mm_plugin_vs_openmax.R:
  - CGTC-MM: selective Bernoulli split, XGT p-values, lambda = 0.5, scored at
    the reference level (Coverage (?), Size, Prop ?, Unseen Coverage (?)).
  - Benchmarks: training-level coverage (Coverage (joker_train)) and decoded
    size (Size (joker_adj)).
Run: python make_openset_benchmark_table.py  (from code/real_experiment)
"""

import glob
import math
import os

import pandas as pd

ALPHA = 0.2
NLABEL = 2000
NREFS = [2000, 6000]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "..", "table_openset_benchmarks.tex")


def load(folder, method):
    files = glob.glob(f"results_hpc/{folder}/*.csv")
    if not files:
        raise FileNotFoundError(f"no CSVs in results_hpc/{folder}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[(abs(df["alpha_total"] - ALPHA) < 1e-10)
            & (df["n_label_total"] == NLABEL)
            & (df["k_top"] == 0) & (df["k_bot"] == 0)
            & (df["method"] == method)]
    return df


def agg(df, cov, size, joker, unseen):
    out = {}
    for nr in NREFS:
        g = df[df["n_ref"] == nr]
        n = len(g)
        out[nr] = {
            "cov": g[cov].mean(), "cov_se": g[cov].std() / math.sqrt(n),
            "size": g[size].mean(), "size_se": g[size].std() / math.sqrt(n),
            "joker": g[joker].mean(), "joker_se": g[joker].std() / math.sqrt(n),
            "uns": g[unseen].mean(), "uns_se": g[unseen].std() / math.sqrt(n),
        }
    return out


# ---- CGTC-MM (reference-level scoring) -------------------------------------
mm = load("celeb_mm_plugin", "Method (Bernoulli)")
mm = mm[(mm["pvalue_method"] == "XGT")
        & (abs(mm["lambda_weight"] - 0.5) < 1e-10)
        & (mm["splitting_method_flag"] == 0)]
rows = [("CGTC-MM", agg(mm, "Coverage (?)", "Size", "Prop ?",
                        "Unseen Coverage (?)"))]

# ---- benchmarks (training-level scoring, decoded size) ---------------------
# (name, results folder, CSV method row, group): group "direct" = raw and
# off-the-shelf scores, "recal" = Good-Turing recalibrated variants.
# Folders that are absent or incomplete are skipped with a warning, so
# pending benchmarks (e.g. the OCC rows) appear automatically once their
# results land in results_hpc/.
BENCH = [
    ("OpenMax (simplified)", "celeb_openmax",      "Method (OpenMax-MLP)", "direct"),
    ("OpenMax (faithful)",   "celeb_openmax_osdn", "Method (OpenMax-MLP)", "direct"),
    ("OpenMax-KNN",          "celeb_openmax",      "Method (OpenMax-KNN)", "direct"),
    ("KNN-dist (raw)",       "celeb_knn_scores_raw", "Method (KNN-dist)",  "direct"),
    ("KNN-MSP (raw)",        "celeb_knn_scores_raw", "Method (KNN-MSP)",   "direct"),
    ("PROSER (raw)",         "celeb_proser",       "Method (PROSER)",      "direct"),
    ("OCC (raw)",            "celeb_occ",          "Method (OCC)",         "direct"),
    ("Naive (GT constant)",  "celeb_gt_knn",       "Method (GT-KNN)",      "direct"),
    ("Recal-KNN-dist",       "celeb_knn_scores",   "Method (Recal KNN-dist)", "recal"),
    ("Recal-KNN-MSP",        "celeb_knn_scores",   "Method (Recal KNN-MSP)",  "recal"),
    ("Recal-PROSER",         "celeb_proser_recal", "Method (Recal PROSER)",   "recal"),
    ("Recal-OCC",            "celeb_occ",          "Method (Recal OCC)",      "recal"),
    ("Recal-OpenMax",        "celeb_openmax_recal", "Method (Recal OpenMax-MLP)", "recal"),
    ("Recal-OpenMax-KNN",    "celeb_openmax_recal", "Method (Recal OpenMax-KNN)", "recal"),
]
included = []
for name, folder, method, group in BENCH:
    try:
        df = load(folder, method)
        if any((df["n_ref"] == nr).sum() == 0 for nr in NREFS):
            raise FileNotFoundError(f"missing n_ref values for {method}")
    except FileNotFoundError as err:
        print(f"skipping {name}: {err}")
        continue
    included.append((name, group))
    rows.append((name, agg(df, "Coverage (joker_train)", "Size (joker_adj)",
                           "Prop ?", "Unseen Coverage (joker_train)")))

# ---- true novelty rates ----------------------------------------------------
om = load("celeb_openmax", "Method (OpenMax-MLP)")
truth_train = {nr: om.loc[om["n_ref"] == nr, "prop_unseen_train"].mean()
               for nr in NREFS}
truth_ref = {nr: mm.loc[mm["n_ref"] == nr, "prop_unseen_test"].mean()
             for nr in NREFS}

# ---- standard-error summary for the caption --------------------------------
prop_ses, size_ses = [], []
for name, a in rows:
    for nr in NREFS:
        prop_ses += [a[nr]["cov_se"], a[nr]["joker_se"], a[nr]["uns_se"]]
        size_ses.append((a[nr]["size_se"], name, nr))
max_prop_se = math.ceil(max(prop_ses) * 100) / 100
size_ses.sort(reverse=True)
(top_se, top_name, top_nr), (second_se, _, _) = size_ses[0], size_ses[1]
if top_se > 3 * second_se:
    nr_tex = f"{top_nr:,}".replace(",", "{,}")
    se_note = (f"Monte Carlo standard errors are at most ${max_prop_se:.2f}$ "
               f"for the proportion columns and below ${math.ceil(second_se)}$ "
               f"for the size columns, except {top_name} at "
               f"$n_{{\\mathrm{{ref}}}} = {nr_tex}$"
               f" (standard error ${top_se:.0f}$).")
else:
    se_note = (f"Monte Carlo standard errors are at most ${max_prop_se:.2f}$ "
               f"for the proportion columns and at most "
               f"${math.ceil(top_se)}$ for the size columns.")

# ---- emit ------------------------------------------------------------------
def cells(a):
    return " & ".join(
        f"{a[nr]['cov']:.2f} & {a[nr]['size']:.1f} & "
        f"{a[nr]['joker']:.2f} & {a[nr]['uns']:.2f}" for nr in NREFS)


GROUPS = [["CGTC-MM"],
          [n for n, g in included if g == "direct"],
          [n for n, g in included if g == "recal"]]
by_name = dict(rows)

caption = (
    "Full benchmark summary on CelebA ($\\alpha = 0.2$, $2{,}000$ sampled "
    "identities, averages over independent batches) at the smallest and "
    "largest reference sample sizes; the figures show the full "
    "$n_{\\mathrm{ref}}$ grid for a readable subset of these methods. "
    "\\emph{Cov.}: marginal coverage of each method's own guaranteed notion, "
    "the reference-level target for CGTC-MM and the weaker training-level "
    "event~\\eqref{eq:training-level-coverage} for all benchmarks. "
    "\\emph{Size}: decoded prediction set size $|\\mathrm{dec}(\\hat{C})|$ "
    "of~\\eqref{eq:decode-map} for the benchmarks and the nominal "
    "(coinciding) size for CGTC-MM; the joker symbol is never counted. "
    "\\emph{Joker}: fraction of prediction sets containing the joker or "
    "unknown class, to be compared with the true novelty rate in the bottom "
    "rows (the training-level rate for the benchmarks, the reference-level "
    "rate for CGTC-MM). \\emph{Uns.\\ Cov.}: coverage conditional on the "
    "corresponding unseen event (label outside the training split for the "
    "benchmarks, outside the reference sample for CGTC-MM). " + se_note)

lines = [
    "% Auto-generated by code/real_experiment/make_openset_benchmark_table.py",
    "% Do not edit by hand; rerun the script after refreshing results.",
    "\\begin{table}[!htb]",
    "\\centering",
    f"\\caption{{{caption}}}",
    "\\label{tab:app-openset-benchmarks}",
    "\\small",
    "\\setlength{\\tabcolsep}{4.5pt}",
    "\\begin{tabular}{lcccccccc}",
    "\\toprule",
    "& \\multicolumn{4}{c}{$n_{\\mathrm{ref}} = 2{,}000$}"
    " & \\multicolumn{4}{c}{$n_{\\mathrm{ref}} = 6{,}000$} \\\\",
    "\\cmidrule(lr){2-5} \\cmidrule(lr){6-9}",
    "Method & Cov. & Size & Joker & Uns.\\ Cov."
    " & Cov. & Size & Joker & Uns.\\ Cov. \\\\",
]
for group in GROUPS:
    lines.append("\\midrule")
    for name in group:
        lines.append(f"{name} & {cells(by_name[name])} \\\\")
lines += [
    "\\midrule",
    "True novelty rate (training level) & & & "
    f"{truth_train[2000]:.2f} & & & & {truth_train[6000]:.2f} & \\\\",
    "True novelty rate (reference level) & & & "
    f"{truth_ref[2000]:.2f} & & & & {truth_ref[6000]:.2f} & \\\\",
    "\\bottomrule",
    "\\end{tabular}",
    "\\end{table}",
    "",
]

with open(OUT, "w") as f:
    f.write("\n".join(lines))
print(f"wrote {os.path.normpath(OUT)}")
print(f"SE note: {se_note}")
