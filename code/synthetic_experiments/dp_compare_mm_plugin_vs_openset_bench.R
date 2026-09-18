library(data.table)
library(tidyverse)
library(ggh4x)

# ============================================================
# Dirichlet-process counterpart of the CelebA headline benchmark figure
# (real_experiment/real_celeb_compare_mm_plugin_vs_openmax.R, setting "main"):
# CGTC+ against the naive baseline and the two strongest directly
# conformalized benchmarks, as a function of the concentration parameter
# theta at n_ref = 2000. Small theta is the (almost) closed-set regime, where
# the methods should approach each other; large theta the open-set regime.
#
# Sources (same data seeds, so the rows pair batch by batch):
#   CGTC+              results_hpc/dp_tuned_mixed_labels_mm_plugin/
#                        (synthetic_experiment_dp_mm_plugin.py: betacv,
#                        lambda 0.50, random CV split, XGT p-values,
#                        selective Bernoulli split; reference-level scoring)
#   Naive              results_hpc/dp_bench_naive/              [Method (GT-KNN)]
#   Recal-OCC (LOF)    results_hpc/dp_bench_occ_scale/          [Method (Recal OCC lof)]
#   Iso-KNN-dist (k=1) results_hpc/dp_bench_knn1_isotonic/      [Method (Recal KNN-dist k=1)]
#                        (synthetic_experiment_dp_openset_bench.py; training-
#                        level coverage and decoded size, as on CelebA)
# Panels: Coverage | Nominal Set Size | Decoded Set Size | Joker Proportion
# (see the CelebA script for the definitions). A source that lacks some
# theta values simply stops there (union of the theta grids).
# Output: dp_mm_plugin_vs_openset_bench_four_panel_paper.pdf
# ============================================================

# ---- knobs ----------------------------------------------------------------
MM_DIR      <- "results_hpc/dp_tuned_mixed_labels_mm_plugin/"
PVAL        <- "XGT"
ALPHA_TOT   <- 0.10
LAMBDA      <- 0.50
NREF        <- 2000
THETA_MIN   <- 0
THETA_MAX   <- 100      # closed-set end only (Inf for the full sweep)
X_BREAKS    <- c(12, 25, 50, 100)   # c(12, 50, 200, 500, 1500) for the full sweep
LOG_SIZE_Y  <- FALSE    # TRUE for the full sweep (Recal-OCC (LOF) spans the dictionary)
BENCH_SOURCES <- list(
  "Naive"              = c("results_hpc/dp_bench_naive/",         "Method (GT-KNN)"),
  "Recal-OCC (LOF)"    = c("results_hpc/dp_bench_occ_scale/",     "Method (Recal OCC lof)"),
  "Iso-KNN-dist (k=1)" = c("results_hpc/dp_bench_knn1_isotonic/", "Method (Recal KNN-dist k=1)")
)
OUT_FILE <- "dp_mm_plugin_vs_openset_bench_four_panel_paper.pdf"

# ---- load -----------------------------------------------------------------
read_dir <- function(path, pattern = "\\.csv$") {
  files <- list.files(path, pattern = pattern, full.names = TRUE)
  if (length(files) == 0) return(NULL)
  map_dfr(files, ~ {
    dt <- fread(.x)
    dt[, which(!duplicated(names(dt))), with = FALSE]
  })
}

cat("=== Loading CGTC+ (", MM_DIR, ") ===\n")
df_mm_all <- read_dir(MM_DIR, pattern = "^dp_occlof_betacv_.*\\.csv$")
if (is.null(df_mm_all)) stop("No CGTC+ CSV files found in ", MM_DIR)
df_mm <- df_mm_all %>%
  filter(method == "Method (Bernoulli)", pvalue_method == PVAL,
         abs(alpha_total - ALPHA_TOT) < 1e-10,
         abs(lambda_weight - LAMBDA) < 1e-10,
         splitting_method_flag == 0,
         n_ref == NREF, calib_num == NREF * 0.1) %>%
  transmute(theta, batch_num,
            cov_marginal = `Coverage (?)`,
            cov_seen     = `Seen Coverage (?)`,
            cov_unseen   = `Unseen Coverage (?)`,
            size_raw     = Size,
            size_adj     = Size,
            prop_joker   = `Prop ?`,
            prop_unseen  = prop_unseen_test,
            source = "CGTC+")
cat("  rows:", nrow(df_mm), " thetas:", paste(sort(unique(df_mm$theta)), collapse = ", "), "\n")

# Benchmark rows: training-level coverage (the event the direct method's
# guarantee applies to) and decoded size ('?' charged the calibration-only
# classes), exactly as in the CelebA figures.
load_bench <- function(label, cand) {
  df <- read_dir(cand[1])
  if (is.null(df)) {
    cat(sprintf("  source '%s': no results in %s, omitted\n", label, cand[1]))
    return(NULL)
  }
  df <- df %>%
    filter(method == cand[2], abs(alpha_total - ALPHA_TOT) < 1e-10,
           n_ref == NREF, calib_num == NREF * 0.1) %>%
    transmute(theta, batch_num,
              cov_marginal = `Coverage (joker_train)`,
              cov_seen     = `Seen Coverage (joker_train)`,
              cov_unseen   = `Unseen Coverage (joker_train)`,
              size_raw     = Size,
              size_adj     = `Size (joker_adj)`,
              prop_joker   = `Prop ?`,
              prop_unseen  = prop_unseen_test,
              source = label)
  cat(sprintf("  source '%s' <- %s [%s]: %d rows, thetas: %s\n", label, cand[1], cand[2],
              nrow(df), paste(sort(unique(df$theta)), collapse = ", ")))
  df
}
bench_dfs <- imap(BENCH_SOURCES, ~ load_bench(.y, .x))

dat <- bind_rows(c(list(df_mm), unname(bench_dfs))) %>%
  filter(theta >= THETA_MIN, theta <= THETA_MAX)

# ---- summarise: mean +/- se over batches x repetitions ---------------------
source_levels <- c("CGTC+", "Recal-OCC (LOF)", "Iso-KNN-dist (k=1)", "Naive")
source_shapes <- c("CGTC+" = 18, "Recal-OCC (LOF)" = 3, "Iso-KNN-dist (k=1)" = 20, "Naive" = 15)
source_colors <- c("CGTC+" = "#000000", "Recal-OCC (LOF)" = "#88CCEE",
                   "Iso-KNN-dist (k=1)" = "#E7298A", "Naive" = "#D55E00")

se <- function(x) sd(x, na.rm = TRUE) / sqrt(sum(!is.na(x)))
agg <- dat %>%
  mutate(source = factor(source, levels = source_levels)) %>%
  group_by(theta, source) %>%
  summarise(n = n(),
            across(c(cov_marginal, cov_seen, cov_unseen,
                     size_raw, size_adj, prop_joker, prop_unseen),
                   list(m = ~mean(.x, na.rm = TRUE), se = ~se(.x)),
                   .names = "{.col}.{.fn}"),
            .groups = "drop")

cat("\n=== Summary (mean over repetitions) ===\n")
agg %>%
  transmute(theta, source, n, coverage = round(cov_marginal.m, 3),
            nominal = round(size_raw.m, 2), decoded = round(size_adj.m, 2),
            joker = round(prop_joker.m, 3), unseen_cov = round(cov_unseen.m, 3),
            true_unseen = round(prop_unseen.m, 3)) %>%
  arrange(source, theta) %>%
  as.data.frame() %>%
  print(row.names = FALSE)

# ---- four-panel paper figure ------------------------------------------------
theme_paper <- theme_bw() +
  theme(
    text = element_text(size = 14),
    axis.title = element_text(size = 18),
    axis.text = element_text(size = 14),
    legend.title = element_text(size = 18),
    legend.text = element_text(size = 16),
    panel.grid.major = element_line(linewidth = 0.5),
    panel.grid.minor = element_blank(),
    strip.text = element_text(size = 16, face = "plain"),
    strip.background = element_rect(fill = "grey90", color = "black"),
    legend.position = "top"
  )

metric_levels <- c("Coverage", "Nominal Set Size", "Decoded Set Size", "Joker Proportion")
long <- bind_rows(
  agg %>% transmute(theta, source, metric = "Coverage", m = cov_marginal.m, se = cov_marginal.se),
  agg %>% transmute(theta, source, metric = "Nominal Set Size", m = size_raw.m, se = size_raw.se),
  agg %>% transmute(theta, source, metric = "Decoded Set Size", m = size_adj.m, se = size_adj.se),
  agg %>% transmute(theta, source, metric = "Joker Proportion", m = prop_joker.m, se = prop_joker.se)
) %>%
  mutate(metric = factor(metric, levels = metric_levels))

# Reference line: true fraction of test labels unseen in the reference sample
ref_joker <- agg %>%
  group_by(theta) %>%
  summarise(m = mean(prop_unseen.m, na.rm = TRUE), .groups = "drop") %>%
  mutate(metric = factor("Joker Proportion", levels = metric_levels))
cov_target <- tibble(metric = factor("Coverage", levels = metric_levels),
                     yintercept = 1 - ALPHA_TOT)

p <- ggplot(long, aes(theta, m, color = source, fill = source, shape = source)) +
  geom_errorbar(aes(ymin = m - 1.96 * se, ymax = m + 1.96 * se),
                width = 0.04, linewidth = 0.7) +   # width in log10 units
  geom_line(linewidth = 0.8) + geom_point(size = 2) +
  geom_hline(data = cov_target, aes(yintercept = yintercept),
             linetype = "dotted", color = "black") +
  geom_line(data = ref_joker, aes(theta, m), inherit.aes = FALSE,
            linetype = "dashed", color = "grey40") +
  facet_wrap(~ metric, scales = "free_y", nrow = 1) +
  facetted_pos_scales(y = list(
    metric == "Coverage" ~ scale_y_continuous(limits = c(0, 1)),
    metric %in% c("Nominal Set Size", "Decoded Set Size") ~
      (if (LOG_SIZE_Y) scale_y_log10() else scale_y_continuous()))) +
  scale_x_log10(breaks = X_BREAKS,
                labels = function(x) format(x, scientific = FALSE, trim = TRUE)) +
  scale_shape_manual(values = source_shapes) +
  scale_color_manual(values = source_colors) +
  scale_fill_manual(values = source_colors) +
  guides(color = guide_legend(nrow = 1), fill = guide_legend(nrow = 1),
         shape = guide_legend(nrow = 1)) +
  labs(x = "Dirichlet concentration parameter", y = NULL,
       color = NULL, fill = NULL, shape = NULL) +
  theme_paper

ggsave(OUT_FILE, p, width = 14, height = 4.2)
cat(">>> wrote", OUT_FILE, "\n")
