"""Generate the size-accounting table of the response letter (LaTeX).

Referee 3 (comment 3) asked whether counting the joker as one element
flattered the method. The table reports, for CGTC+ and for the standard
closed-set method (both with selective splitting and XGT p-values), the
coverage, the prediction set size under the revision's convention (named
labels only), under the former convention (joker counted as one element),
and restricted to test points with a previously seen label, together with
the joker rate, at two settings of each experiment of Section 5.

Data (same selections as the manuscript figures):
  - CelebA: results_hpc/celeb_mm_plugin, files celeb_betacv_*_lambda0.50_*_split0_*,
    batches 1-20 (as celeb_mm_plugin_paper_plots.R), alpha 0.2, n_ref 2000 and 6000.
  - Synthetic: ../synthetic_experiments/results_hpc/dp_tuned_mixed_labels_mm_plugin,
    files *betacv*_nref2000_*_lambda0.50_split0_*, alpha 0.1, theta 1000 and 1500
    (as dp_mm_plugin_paper_plots.R).
Rows: 'Method (Bernoulli)' = CGTC+ with selective splitting,
'Method (Bernoulli benchmark)' = standard method with selective splitting;
pvalue_method XGT. Columns: Coverage (?) (joker credited), Size (named labels
only), Size (?) (former convention), Seen Size (test points with a seen label),
Prop ? (joker rate); the unseen rate in the row label is prop_unseen_test.

Output: ../../table_size_accounting.tex (repository root) and
../../Open_Set_Conformal_Classification/tables/table_size_accounting.tex when
that folder exists; response.tex pulls it in via \\input{tables/table_size_accounting}.
Run: python make_size_accounting_table.py  (any working directory)
"""

import glob
import math
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CELEB_DIR = os.path.join(HERE, "results_hpc", "celeb_mm_plugin")
DP_DIR = os.path.join(HERE, "..", "synthetic_experiments", "results_hpc",
                      "dp_tuned_mixed_labels_mm_plugin")
OUTS = [os.path.join(HERE, "..", "..", "table_size_accounting.tex"),
        os.path.join(HERE, "..", "..", "Open_Set_Conformal_Classification",
                     "tables", "table_size_accounting.tex")]

METHODS = {"Method (Bernoulli)": "CGTC+",
           "Method (Bernoulli benchmark)": "standard"}
COLS = {"cov": "Coverage (?)", "named": "Size", "former": "Size (?)",
        "seen": "Seen Size", "joker": "Prop ?"}
SIZE_KEYS = ["named", "former", "seen"]
PROP_KEYS = ["cov", "joker"]


def load(folder, file_re, batch_max=None):
    files = [f for f in glob.glob(os.path.join(folder, "*.csv"))
             if re.search(file_re, os.path.basename(f))]
    if batch_max is not None:
        def batch_no(f):
            m = re.search(r"batch_?(\d+)\.csv$", os.path.basename(f))
            return int(m.group(1)) if m else None
        files = [f for f in files
                 if batch_no(f) is not None and 1 <= batch_no(f) <= batch_max]
    if not files:
        raise FileNotFoundError(f"no CSVs matching {file_re} in {os.path.normpath(folder)}")
    frames = []
    for f in files:
        df = pd.read_csv(f)
        frames.append(df.loc[:, ~df.columns.duplicated()])
    df = pd.concat(frames, ignore_index=True)
    return df[(df["pvalue_method"] == "XGT") & df["method"].isin(METHODS)]


celeb = load(CELEB_DIR, r"^celeb_betacv_.*_lambda0\.50_.*_split0_", batch_max=20)
celeb = celeb[(abs(celeb["alpha_total"] - 0.2) < 1e-10)
              & (celeb["n_label_total"] == 2000)
              & (celeb["k_top"] == 0) & (celeb["k_bot"] == 0)]
dp = load(DP_DIR, r"betacv.*_nref2000_.*_lambda0\.50_split0_")
dp = dp[(abs(dp["alpha_total"] - 0.1) < 1e-10) & (dp["n_ref"] == 2000)]

SETTINGS = [
    ("CelebA, $n_{\\mathrm{ref}} = 2{,}000$", celeb[celeb["n_ref"] == 2000]),
    ("CelebA, $n_{\\mathrm{ref}} = 6{,}000$", celeb[celeb["n_ref"] == 6000]),
    ("Synthetic, $\\theta = 1000$", dp[dp["theta"] == 1000]),
    ("Synthetic, $\\theta = 1500$", dp[dp["theta"] == 1500]),
]

rows = []
max_se = {"size": 0.0, "prop": 0.0}
for setting, df in SETTINGS:
    if df.empty:
        raise ValueError(f"no rows for setting {setting}")
    unseen = df["prop_unseen_test"].mean()
    for i, (method, label) in enumerate(METHODS.items()):
        g = df[df["method"] == method]
        n = len(g)
        if n == 0:
            raise ValueError(f"no rows for {method} in setting {setting}")
        mean = {k: g[c].mean() for k, c in COLS.items()}
        se = {k: g[c].std() / math.sqrt(n) for k, c in COLS.items()}
        max_se["size"] = max(max_se["size"], *(se[k] for k in SIZE_KEYS))
        max_se["prop"] = max(max_se["prop"], *(se[k] for k in PROP_KEYS))
        head = setting if i == 0 else f"(unseen rate ${unseen:.2f}$)"
        rows.append(
            f"{head} & {label} & {mean['cov']:.2f} & {mean['named']:.2f} & "
            f"{mean['former']:.2f} & {mean['seen']:.2f} & {mean['joker']:.2f} \\\\")
        print(f"{setting:40s} {label:9s} n={n:3d}  "
              + "  ".join(f"{k}={mean[k]:.3f}({se[k]:.3f})" for k in COLS))
    if setting != SETTINGS[-1][0]:
        rows.append("\\midrule")


def ceil2(x):
    return math.ceil(x * 100 - 1e-9) / 100


se_note = (f"Monte Carlo standard errors are at most ${ceil2(max_se['size']):.2f}$ "
           f"for the size columns and ${ceil2(max_se['prop']):.2f}$ for the proportions.")

caption = (
    "Prediction set size for the deployed method (CGTC+) and the standard "
    "closed-set method, both with selective splitting and feature-based "
    "$p$-values. \\emph{Named labels} is the convention adopted throughout the "
    "revision; \\emph{former convention} additionally counted $\\joker$ as one "
    "element; \\emph{seen-label points} restricts the first column to test "
    "points whose label appears in the reference sample. The two size "
    "conventions differ by exactly the joker rate. The nominal coverage level "
    "is $0.80$ for CelebA and $0.90$ for the synthetic experiment. " + se_note)

lines = [
    "% Auto-generated by code/real_experiment/make_size_accounting_table.py",
    "% Do not edit by hand; rerun the script after refreshing results.",
    "\\begin{table}[!htb]",
    "\\centering",
    f"\\caption{{{caption}}}",
    "\\label{tab:r3-size-accounting}",
    "\\begin{tabular}{llccccc}",
    "\\toprule",
    "& & & \\multicolumn{3}{c}{Prediction set size} & \\\\",
    "\\cmidrule(lr){4-6}",
    "Setting & Method & Coverage & Named labels & Former convention & "
    "Seen-label points & Joker rate \\\\",
    "\\midrule",
    *rows,
    "\\bottomrule",
    "\\end{tabular}",
    "\\end{table}",
    "",
]

for out in OUTS:
    if not os.path.isdir(os.path.dirname(out)):
        print(f"skipped {os.path.normpath(out)} (folder missing)")
        continue
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {os.path.normpath(out)}")
