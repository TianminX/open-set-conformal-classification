library(data.table)
library(tidyverse)

# ============================================================
# Compare, on CelebA, the NEW plug-in missing-mass method
#     results_hpc/celeb_mm_plugin/  (real_experiment_celeb_mm_plugin.py)
# against the ORIGINAL loss-based CGTC method
#     results_hpc/celeb/            (real_experiment_celeb.py)
#
# Real-data counterpart of dp_compare_mm_plugin_vs_cgtc.R, with n_ref on the
# x-axis instead of theta. Three outputs per split setting:
#   1) coverage vs n_ref      (marginal / seen / unseen)
#   2) set size & joker proportion vs n_ref
#   3) alpha allocation vs n_ref (class / seen / unseen, with mu_hat reference)
# plus a console table of differences (MM-plugin - Original).
#
# The script runs once per row of SPLIT_SETTINGS below (bernoulli and random),
# keeping the method row and the tune/split flags consistent, and suffixes the
# output PDFs with the split name.
# ============================================================

# ---- knobs ----------------------------------------------------------------
ORIG_DIR   <- "results_hpc/celeb/"
NEW_DIR    <- "results_hpc/celeb_mm_plugin/"
PVAL       <- "XGT"                   # GT / RGT / XGT
ALPHA_TOT  <- 0.20
LAMBDA     <- 0.50
NLABEL     <- 2000                    # subsampling: uniform-sampling label count
KTOP       <- 0
KBOT       <- 0
COV_COL    <- "Coverage (?)"          # marginal coverage (counts joker "?")

# One row per split setting. `flag` filters tuning_method_flag (original) and
# splitting_method_flag (plug-in), so both methods are tuned under the same CV
# split (apples-to-apples); `method_row` picks the matching method row.
# To compare the OLD random-split data instead, point ORIG_DIR to
# "results_hpc/celeb_copy_from_old/" and keep only the random row.
SPLIT_SETTINGS <- tribble(
  ~split_name, ~method_row,                 ~flag,
  "bernoulli", "Method (Bernoulli)",         1,
  "random",    "Method (Bernoulli)",  0
  # "random",    "Method (random splitting)",  0
)

# ---- load -----------------------------------------------------------------
read_dir <- function(path) {
  files <- list.files(path, pattern = "\\.csv$", full.names = TRUE)
  if (length(files) == 0)
    stop("No CSV files found in ", path, ". Run the experiment / download results first.")
  map_dfr(files, ~ {
    dt <- fread(.x)
    dt[, which(!duplicated(names(dt))), with = FALSE]
  })
}

# Older runs (e.g. species celeb_tuned_mixed_labels) named the unseen/seen
# budgets alpha_new/alpha_old; rename to the current alpha_unseen/alpha_seen.
normalize_cols <- function(df) {
  ren <- c(alpha_new = "alpha_unseen", alpha_old = "alpha_seen",
           alpha_new_avg = "alpha_unseen_avg", alpha_old_avg = "alpha_seen_avg")
  for (old in names(ren)) {
    if (old %in% names(df) && !(ren[[old]] %in% names(df)))
      names(df)[names(df) == old] <- ren[[old]]
  }
  df
}

cat("=== Loading original CGTC (", ORIG_DIR, ") ===\n")
df_orig_all <- read_dir(ORIG_DIR) %>% normalize_cols()
cat("  rows:", nrow(df_orig_all), "\n")

cat("=== Loading mm_plugin (", NEW_DIR, ") ===\n")
# Keep only the CV-selected-beta runs and batches 1-20, matching
# celeb_mm_plugin_paper_plots.R (the folder also holds beta1.6 runs and
# extra batches for some settings).
new_files <- list.files(NEW_DIR, pattern = "^celeb_betacv_.*\\.csv$", full.names = TRUE)
new_batch <- as.integer(str_match(basename(new_files), "_batch_(\\d+)\\.csv$")[, 2])
new_files <- new_files[!is.na(new_batch) & new_batch >= 1 & new_batch <= 20]
cat("  reading", length(new_files), "betacv files (batches 1-20)\n")
df_new_all <- map_dfr(new_files, ~ {
  dt <- fread(.x)
  dt[, which(!duplicated(names(dt))), with = FALSE]
}) %>% normalize_cols()
cat("  rows:", nrow(df_new_all), "\n\n")

# mu_hat only exists in the plug-in output; add a placeholder to the original
if (!"mu_hat" %in% names(df_orig_all))
  df_orig_all <- df_orig_all %>% mutate(mu_hat = NA_real_)

# ---- shared plotting bits ---------------------------------------------------
theme_cmp <- theme_bw() + theme(legend.position = "bottom")

# Paper version formatting follows the old-repo figure conventions
# (dp_mixed_final_conditional.R): larger fonts, grey facet strips,
# thicker lines/points, no title and no explanatory caption
# (those belong in the LaTeX figure caption instead).
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

# Distinguish the two methods by shape as well as colour (cf. custom_shapes
# in the old dp_mixed_final_conditional.R), so they remain readable in B/W.
source_shapes <- c("Original CGTC" = 16, "CGTC+" = 18)

# ============================================================
# One full comparison (all plots + diff table) for a split setting
# ============================================================
run_comparison <- function(split_name, method_row, flag) {
  cat(sprintf("\n########## split = %s (method row: %s, flag = %d) ##########\n",
              split_name, method_row, flag))

  df_orig <- df_orig_all %>% filter(tuning_method_flag == flag)
  df_new  <- df_new_all  %>% filter(splitting_method_flag == flag)
  cat("  original rows:", nrow(df_orig), " | mm_plugin rows:", nrow(df_new), "\n")
  if (nrow(df_orig) == 0 || nrow(df_new) == 0) {
    warning("No rows with flag ", flag, " for split '", split_name,
            "'; skipping this setting.", call. = FALSE)
    return(invisible(NULL))
  }

  # ---- common n_ref ---------------------------------------------------------
  common_nref <- sort(intersect(unique(df_orig$n_ref), unique(df_new$n_ref)))
  cat("  common n_ref:", paste(common_nref, collapse = ", "), "\n")
  if (length(common_nref) == 0) {
    warning("No overlapping n_ref between the two result folders for split '",
            split_name, "'; skipping.", call. = FALSE)
    return(invisible(NULL))
  }

  prep <- function(df, label) {
    df %>%
      filter(method == method_row, pvalue_method == PVAL,
             abs(alpha_total - ALPHA_TOT) < 1e-10,
             abs(lambda_weight - LAMBDA) < 1e-10,
             n_label_total == NLABEL, k_top == KTOP, k_bot == KBOT,
             n_ref %in% common_nref) %>%
      transmute(
        n_ref, batch_num,
        cov_marginal = .data[[COV_COL]],
        cov_seen     = `Seen Coverage (?)`,
        cov_unseen   = `Unseen Coverage (?)`,
        size         = Size,              # set size EXCLUDING the joker "?"
        prop_joker   = `Prop ?`,          # proportion of test points assigned "?"
        prop_unseen  = prop_unseen_test,  # true prop. of test labels unseen in ref
        mu_hat,                            # missing-mass estimate (plug-in only)
        alpha_class, alpha_seen, alpha_unseen,
        source = label)
  }

  dat <- bind_rows(prep(df_orig, "Original CGTC"),
                   prep(df_new,  "CGTC+")) %>%
    mutate(source = factor(source, levels = c("Original CGTC", "CGTC+")))

  if (nrow(dat) == 0) {
    warning("No rows match the selected knobs for split '", split_name,
            "'. Check method row / filters; skipping.", call. = FALSE)
    return(invisible(NULL))
  }

  # mean +/- se across batches, per n_ref x source
  se <- function(x) sd(x, na.rm = TRUE) / sqrt(sum(!is.na(x)))
  agg <- dat %>%
    group_by(n_ref, source) %>%
    summarise(across(c(cov_marginal, cov_seen, cov_unseen,
                       size, prop_joker, prop_unseen, mu_hat,
                       alpha_class, alpha_seen, alpha_unseen),
                     list(m = ~mean(.x, na.rm = TRUE), se = ~se(.x)),
                     .names = "{.col}.{.fn}"),
              .groups = "drop")

  out_file <- function(stem)
    sprintf("real_celeb_mm_plugin_vs_original_%s_%s.pdf", stem, split_name)

  # ============================================================
  # Plot 1: coverage vs n_ref
  # ============================================================
  cov_long <- agg %>%
    select(n_ref, source, starts_with("cov_")) %>%
    pivot_longer(starts_with("cov_"),
                 names_to = c("metric", ".value"), names_sep = "\\.") %>%
    mutate(metric = recode(metric,
      cov_marginal = "Marginal", cov_seen = "Seen", cov_unseen = "Unseen") %>%
      factor(levels = c("Marginal", "Seen", "Unseen")))

  p_cov <- ggplot(cov_long, aes(n_ref, m, color = source, fill = source)) +
    geom_hline(yintercept = 1 - ALPHA_TOT, linetype = "dashed", color = "grey40") +
    geom_errorbar(aes(ymin = m - 1.96 * se, ymax = m + 1.96 * se), width = 150) +
    geom_line() + geom_point(size = 1.2) +
    facet_wrap(~ metric) +
    labs(x = "Number of reference observations (n_ref)", y = "Coverage",
         color = NULL, fill = NULL,
         title = sprintf("CelebA coverage vs n_ref  (%s, %s)", method_row, PVAL)) +
    theme_cmp

  ggsave(out_file("coverage"), p_cov, width = 10, height = 4)
  cat(">>> wrote", out_file("coverage"), "\n")

  # ============================================================
  # Plot 2: coverage (first panel), set size (excl. "?") and joker proportion
  # ============================================================
  metric_levels <- c("Coverage", "Set size (excl. ?)", "Joker proportion (?)")

  size_long <- agg %>%
    select(n_ref, source,
           cov_marginal.m, cov_marginal.se,
           size.m, size.se, prop_joker.m, prop_joker.se) %>%
    pivot_longer(c(cov_marginal.m, cov_marginal.se,
                   size.m, size.se, prop_joker.m, prop_joker.se),
                 names_to = c("metric", ".value"), names_sep = "\\.") %>%
    mutate(metric = recode(metric,
      cov_marginal = "Coverage",
      size = "Set size (excl. ?)",
      prop_joker = "Joker proportion (?)") %>%
      factor(levels = metric_levels))

  # Reference line: true prop. of unseen test labels, shown only in the joker panel
  ref_joker <- agg %>%
    group_by(n_ref) %>%
    summarise(m = mean(prop_unseen.m, na.rm = TRUE), .groups = "drop") %>%
    mutate(metric = factor("Joker proportion (?)", levels = metric_levels))

  # Reference line: nominal coverage target 1 - alpha, shown only in the Coverage panel
  cov_target <- tibble(metric = factor("Coverage", levels = metric_levels),
                       yintercept = 1 - ALPHA_TOT)

  p_size <- ggplot(size_long, aes(n_ref, m, color = source, fill = source)) +
    geom_errorbar(aes(ymin = m - 1.96 * se, ymax = m + 1.96 * se), width = 150) +
    geom_line() + geom_point(size = 1.2) +
    geom_hline(data = cov_target, aes(yintercept = yintercept),
               linetype = "dotted", color = "black") +
    geom_line(data = ref_joker, aes(n_ref, m), inherit.aes = FALSE,
              linetype = "dashed", color = "grey40") +
    facet_wrap(~ metric, scales = "free_y") +
    labs(x = "Number of reference observations (n_ref)", y = NULL,
         color = NULL, fill = NULL,
         title = sprintf("CelebA coverage, set size & joker proportion vs n_ref  (%s, %s)",
                         method_row, PVAL),
         caption = "dotted line (Coverage panel) = 1 - alpha; dashed line (Joker panel) = true prop. of test labels unseen in reference") +
    theme_cmp

  ggsave(out_file("size_joker"), p_size, width = 12, height = 4)
  cat(">>> wrote", out_file("size_joker"), "\n")

  # Plot 2 (paper version): same data, restyled for inclusion in the paper.
  p_size_paper <- ggplot(size_long,
                         aes(n_ref, m, color = source, fill = source,
                             shape = source)) +
    geom_errorbar(aes(ymin = m - 1.96 * se, ymax = m + 1.96 * se),
                  width = 150, linewidth = 0.7) +
    geom_line(linewidth = 1.2) + geom_point(size = 3) +
    geom_hline(data = cov_target, aes(yintercept = yintercept),
               linetype = "dotted", color = "black") +
    geom_line(data = ref_joker, aes(n_ref, m), inherit.aes = FALSE,
              linetype = "dashed", color = "grey40") +
    facet_wrap(~ metric, scales = "free_y",
               labeller = as_labeller(c(
                 "Coverage"             = "Coverage",
                 "Set size (excl. ?)"   = "Prediction set size",
                 "Joker proportion (?)" = "Joker proportion"))) +
    scale_shape_manual(values = source_shapes) +
    labs(x = "Number of reference observations", y = NULL,
         color = NULL, fill = NULL, shape = NULL) +
    theme_paper

  ggsave(out_file("size_joker_paper"), p_size_paper, width = 12, height = 3.8)
  cat(">>> wrote", out_file("size_joker_paper"), "\n")

  # ============================================================
  # Plot 3: alpha allocation vs n_ref
  # ============================================================
  alpha_long <- agg %>%
    select(n_ref, source, starts_with("alpha_")) %>%
    pivot_longer(starts_with("alpha_"),
                 names_to = c("metric", ".value"), names_sep = "\\.") %>%
    mutate(metric = recode(metric,
      alpha_class = "alpha_class", alpha_seen = "alpha_seen",
      alpha_unseen = "alpha_unseen"))

  # Reference line: mu_hat (missing-mass estimate, plug-in only), alpha_class facet
  ref_mu <- agg %>%
    filter(source == "CGTC+") %>%
    transmute(n_ref, m = mu_hat.m, metric = "alpha_class")

  p_alpha <- ggplot(alpha_long, aes(n_ref, m, color = source, fill = source)) +
    geom_errorbar(aes(ymin = m - 1.96 * se, ymax = m + 1.96 * se), width = 150) +
    geom_line() + geom_point(size = 1.2) +
    geom_line(data = ref_mu, aes(n_ref, m), inherit.aes = FALSE,
              linetype = "dashed", color = "grey40") +
    facet_wrap(~ metric, scales = "free_y") +
    labs(x = "Number of reference observations (n_ref)", y = "alpha",
         color = NULL, fill = NULL,
         title = sprintf("CelebA alpha allocation vs n_ref  (%s, %s)", method_row, PVAL),
         caption = "dashed line (alpha_class panel) = mu_hat, the missing-mass estimate used for inflation") +
    theme_cmp

  ggsave(out_file("alpha_allocation"), p_alpha, width = 10, height = 4)
  cat(">>> wrote", out_file("alpha_allocation"), "\n")

  # ============================================================
  # Console table of differences (MM-plugin - Original)
  # ============================================================
  diff_tbl <- agg %>%
    select(n_ref, source, cov_marginal.m, size.m, prop_joker.m,
           alpha_class.m, alpha_seen.m, alpha_unseen.m) %>%
    pivot_wider(names_from = source,
                values_from = c(cov_marginal.m, size.m, prop_joker.m,
                                alpha_class.m, alpha_seen.m, alpha_unseen.m)) %>%
    mutate(
      d_cov          = `cov_marginal.m_CGTC+` - `cov_marginal.m_Original CGTC`,
      d_size         = `size.m_CGTC+`         - `size.m_Original CGTC`,
      d_prop_joker   = `prop_joker.m_CGTC+`   - `prop_joker.m_Original CGTC`,
      d_alpha_class  = `alpha_class.m_CGTC+`  - `alpha_class.m_Original CGTC`,
      d_alpha_seen   = `alpha_seen.m_CGTC+`   - `alpha_seen.m_Original CGTC`,
      d_alpha_unseen = `alpha_unseen.m_CGTC+` - `alpha_unseen.m_Original CGTC`) %>%
    select(n_ref, d_cov, d_size, d_prop_joker,
           d_alpha_class, d_alpha_seen, d_alpha_unseen)

  cat(sprintf("\n=== Differences (MM-plugin - Original CGTC), split = %s ===\n",
              split_name))
  print(as.data.frame(diff_tbl), digits = 4)

  invisible(agg)
}

# ---- run all split settings -------------------------------------------------
pwalk(SPLIT_SETTINGS, run_comparison)
