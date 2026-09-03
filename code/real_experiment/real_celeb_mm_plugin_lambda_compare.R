library(data.table)
library(tidyverse)

# ============================================================
# Compare the CV-tuned plug-in missing-mass allocation on CelebA
# across the loss preferences lambda in {0.10, 0.20, 0.25, 0.30,
# 0.40, 0.50, 0.60, 0.70}.
#
# Both run families are beta-CV ("betacv" in the filename) with
# random-splitting tuning CV (split0); the older fixed-beta files
# (beta1.6) in the same directory are excluded by the filename filter.
#
# Reads:  results_hpc/celeb_mm_plugin/celeb_betacv_*.csv
# Writes:
#   1. real_celeb_mm_plugin_lambda_compare_performance.pdf
#        Coverage / Set Size / Joker Proportion vs n_ref,
#        CGTC methods only, color = lambda, facet grid metric x method
#   2. real_celeb_mm_plugin_lambda_compare_performance_selective.pdf
#        Same metrics but CGTC (selective) only, one row of 3 panels
#   3. real_celeb_mm_plugin_lambda_compare_allocation.pdf
#        alpha_class / alpha_unseen / cap / alpha_seen / mu_hat vs n_ref,
#        faceted by lambda (2 rows x 4 cols)
# ============================================================

# ---- Parameters ----
RESULTS_DIR   <- "results_hpc/celeb_mm_plugin/"
PVAL_SELECT   <- "XGT"
ALPHA_TOTAL   <- 0.20
# LAMBDA_LIST   <- c(0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70)
LAMBDA_LIST   <- c(0.30, 0.50, 0.70)

NLABEL_SELECT <- 2000
KTOP_SELECT   <- 0
KBOT_SELECT   <- 0
SPLIT_FLAG    <- 0        # random-splitting tuning CV (the only split with both lambdas)
GRID_SELECT   <- 20
MIN_RUNS      <- 5        # drop (lambda, n_ref) cells with fewer runs (partial download)

# ============================================================
# 1. Load (betacv files only)
# ============================================================
files <- list.files(RESULTS_DIR, pattern = "^celeb_betacv_.*\\.csv$", full.names = TRUE)
if (length(files) == 0) stop("No celeb_betacv_*.csv files found in ", RESULTS_DIR)

df_raw <- map_dfr(files, ~ {
  dt <- fread(.x)
  dt[, which(!duplicated(names(dt))), with = FALSE]
})
cat("Loaded", nrow(df_raw), "rows from", length(files), "betacv files\n")

df_raw <- df_raw %>%
  mutate(method = recode(method,
    "Method (random splitting)"    = "CGTC (random)",
    "Method (benchmark)"           = "standard (random)",
    "Method (Bernoulli)"           = "CGTC (selective)",
    "Method (Bernoulli benchmark)" = "standard (selective)"))

df <- df_raw %>%
  filter(
    pvalue_method == PVAL_SELECT,
    abs(alpha_total - ALPHA_TOTAL) < 1e-10,
    lambda_weight %in% LAMBDA_LIST,
    n_label_total == NLABEL_SELECT,
    k_top == KTOP_SELECT, k_bot == KBOT_SELECT,
    splitting_method_flag == SPLIT_FLAG,
    grid_size == GRID_SELECT
  ) %>%
  mutate(lambda_lab = sprintf("lambda == %.2f", lambda_weight))

# Report run counts per cell, then drop under-filled cells
run_counts <- df %>%
  filter(method == "CGTC (selective)") %>%
  count(lambda_weight, n_ref, name = "n_runs")
cat("\nRuns per (lambda, n_ref):\n")
print(as.data.frame(run_counts))

keep_cells <- run_counts %>% filter(n_runs >= MIN_RUNS) %>% select(lambda_weight, n_ref)
df <- df %>% semi_join(keep_cells, by = c("lambda_weight", "n_ref"))
dropped <- run_counts %>% filter(n_runs < MIN_RUNS)
if (nrow(dropped) > 0) {
  cat("\nDropped cells with <", MIN_RUNS, "runs:\n")
  print(as.data.frame(dropped))
}

# ============================================================
# 2. Performance summary (CGTC methods; benchmarks are lambda-invariant)
# ============================================================
df_perf <- df %>%
  filter(method %in% c("CGTC (random)", "CGTC (selective)")) %>%
  group_by(lambda_weight, lambda_lab, n_ref, method) %>%
  summarise(
    mean_cov    = mean(`Coverage (?)`, na.rm = TRUE),
    se_cov      = sd(`Coverage (?)`, na.rm = TRUE) / sqrt(n()),
    mean_size   = mean(Size, na.rm = TRUE),
    se_size     = sd(Size, na.rm = TRUE) / sqrt(n()),
    mean_propq  = mean(`Prop ?`, na.rm = TRUE),
    se_propq    = sd(`Prop ?`, na.rm = TRUE) / sqrt(n()),
    mean_unseen = mean(prop_unseen_test, na.rm = TRUE),
    .groups = "drop"
  )

df_perf_long <- bind_rows(
  df_perf %>% transmute(lambda_weight, lambda_lab, n_ref, method,
                        metric = "Coverage", mean = mean_cov, se = se_cov),
  df_perf %>% transmute(lambda_weight, lambda_lab, n_ref, method,
                        metric = "Prediction Set Size", mean = mean_size, se = se_size),
  df_perf %>% transmute(lambda_weight, lambda_lab, n_ref, method,
                        metric = "Joker Proportion", mean = mean_propq, se = se_propq)
) %>%
  mutate(metric = factor(metric,
           levels = c("Coverage", "Prediction Set Size", "Joker Proportion")),
         lambda_f = factor(sprintf("%.2f", lambda_weight)),
         lci = mean - 1.96 * se, uci = mean + 1.96 * se)

ref_unseen <- df_perf %>%
  group_by(n_ref) %>%
  summarise(mean = mean(mean_unseen, na.rm = TRUE), .groups = "drop") %>%
  mutate(metric = factor("Joker Proportion", levels = levels(df_perf_long$metric)))

# Fixed panel y-ranges: free_y zooms each panel to its data, which makes
# stable curves look noisy. Pin Coverage to [0, 1] and start Set Size at 0
# (upper bound stays automatic); Joker Proportion remains fully automatic.
COV_YLIM  <- c(0, 1)
# SIZE_YMIN <- 0
ylim_blank <- bind_rows(
  tibble(metric = "Coverage",            mean = COV_YLIM),
  # tibble(metric = "Prediction Set Size", mean = SIZE_YMIN)
) %>%
  mutate(metric = factor(metric, levels = levels(df_perf_long$metric)),
         n_ref  = min(df_perf_long$n_ref))

# ============================================================
# 3. Aesthetics (matches the existing mm_plugin plot)
# ============================================================
lambda_colors <- c("0.10" = "#A50026",
                   "0.20" = "#D73027",
                   "0.25" = "#F46D43",
                   "0.30" = "#FDAE61",
                   "0.40" = "#ABD9E9",
                   "0.50" = "#74ADD1",
                   "0.60" = "#4575B4",
                   "0.70" = "#313695")
lambda_shapes <- c("0.10" = 16, "0.20" = 17, "0.25" = 15, "0.30" = 18,
                   "0.40" = 16, "0.50" = 17, "0.60" = 15, "0.70" = 18)

theme_paper <- theme_bw() +
  theme(
    text = element_text(size = 14),
    axis.title = element_text(size = 18),
    axis.text = element_text(size = 14),
    legend.title = element_text(size = 18),
    legend.text = element_text(size = 16),
    panel.grid.minor = element_blank(),
    strip.text = element_text(size = 16),
    strip.background = element_rect(fill = "grey90", color = "black"),
    legend.position = "right"
  )

# ============================================================
# 4. Performance plot: color = lambda, facet grid metric x method
#    (free_y per metric row works in facet_grid since metrics are rows)
# ============================================================
p_perf <- ggplot(df_perf_long,
                 aes(x = n_ref, y = mean, color = lambda_f, shape = lambda_f,
                     group = lambda_f)) +
  geom_line(linewidth = 1.0) +
  geom_errorbar(aes(ymin = lci, ymax = uci), width = 150, show.legend = FALSE) +
  geom_point(size = 2.2) +
  geom_line(data = ref_unseen, aes(x = n_ref, y = mean), inherit.aes = FALSE,
            linetype = "dashed", color = "grey40") +
  geom_blank(data = ylim_blank, aes(x = n_ref, y = mean), inherit.aes = FALSE) +
  facet_grid(metric ~ method, scales = "free_y") +
  scale_color_manual(name = expression(lambda), values = lambda_colors) +
  scale_shape_manual(name = expression(lambda), values = lambda_shapes) +
  geom_hline(
    data = tibble(metric = factor("Coverage", levels = levels(df_perf_long$metric)),
                  yintercept = 1 - ALPHA_TOTAL),
    aes(yintercept = yintercept), linetype = "dotted", color = "black") +
  labs(x = "Number of reference observations (n_ref)", y = "",
       title = sprintf("CelebA plug-in CGTC across lambda (alpha=%.2f, %s, split%d, beta-CV)",
                       ALPHA_TOTAL, PVAL_SELECT, SPLIT_FLAG),
       caption = "dashed grey line (Joker row) = true prop. of unseen test labels") +
  theme_paper

print(p_perf)
ggsave("real_celeb_mm_plugin_lambda_compare_performance.pdf", p_perf,
       width = 11, height = 10, units = "in")
cat("Plot saved: real_celeb_mm_plugin_lambda_compare_performance.pdf\n")

# ============================================================
# 4b. Single-method performance plot: one CGTC variant only,
#     metrics side by side (1 row x 3 panels)
# ============================================================
METHOD_SINGLE <- "CGTC (selective)"   # Bernoulli splitting; or "CGTC (random)"

p_perf_single <- ggplot(df_perf_long %>% filter(method == METHOD_SINGLE),
                        aes(x = n_ref, y = mean, color = lambda_f, shape = lambda_f,
                            group = lambda_f)) +
  geom_line(linewidth = 1.0) +
  geom_errorbar(aes(ymin = lci, ymax = uci), width = 150, show.legend = FALSE) +
  geom_point(size = 2.2) +
  geom_line(data = ref_unseen, aes(x = n_ref, y = mean), inherit.aes = FALSE,
            linetype = "dashed", color = "grey40") +
  geom_blank(data = ylim_blank, aes(x = n_ref, y = mean), inherit.aes = FALSE) +
  facet_wrap(~ metric, scales = "free_y", nrow = 1) +
  scale_color_manual(name = expression(lambda), values = lambda_colors) +
  scale_shape_manual(name = expression(lambda), values = lambda_shapes) +
  geom_hline(
    data = tibble(metric = factor("Coverage", levels = levels(df_perf_long$metric)),
                  yintercept = 1 - ALPHA_TOTAL),
    aes(yintercept = yintercept), linetype = "dotted", color = "black") +
  labs(x = "Number of reference observations", y = "",
       # title = sprintf("CelebA plug-in %s across lambda (alpha=%.2f, %s, split%d, beta-CV)",
       #                 METHOD_SINGLE, ALPHA_TOTAL, PVAL_SELECT, SPLIT_FLAG),
       # caption = "dashed grey line (Joker panel) = prop. of unseen test labels"
       ) +
  theme_paper

print(p_perf_single)
ggsave("real_celeb_mm_plugin_lambda_compare_performance_selective.pdf", p_perf_single,
       width = 14, height = 3.2, units = "in")
cat("Plot saved: real_celeb_mm_plugin_lambda_compare_performance_selective.pdf\n")

# ============================================================
# 5. Allocation plot: faceted by lambda (includes the CV-chosen cap)
# ============================================================
df_alpha <- df %>%
  filter(method == "CGTC (selective)") %>%
  group_by(lambda_weight, lambda_lab, n_ref) %>%
  summarise(
    across(c(alpha_class, alpha_unseen, alpha_unseen_cap, alpha_seen, mu_hat),
           list(mean = ~mean(.x, na.rm = TRUE),
                se   = ~sd(.x, na.rm = TRUE) / sqrt(n()))),
    .groups = "drop"
  ) %>%
  pivot_longer(-c(lambda_weight, lambda_lab, n_ref),
               names_to = c("param", "stat"),
               names_pattern = "(.*)_(mean|se)") %>%
  pivot_wider(names_from = stat, values_from = value) %>%
  mutate(param = recode(param,
           "alpha_class"      = "alpha_class (effective)",
           "alpha_unseen"     = "alpha_unseen",
           "alpha_unseen_cap" = "cap (CV-chosen)",
           "alpha_seen"       = "alpha_seen",
           "mu_hat"           = "mu-hat (missing mass)"),
         lci = mean - 1.96 * se, uci = mean + 1.96 * se)

alpha_colors <- c("alpha_class (effective)" = "#984EA3",
                  "alpha_unseen"            = "#377EB8",
                  "cap (CV-chosen)"         = "#E41A1C",
                  "alpha_seen"              = "#4DAF4A",
                  "mu-hat (missing mass)"   = "#FF7F00")

p_alpha <- ggplot(df_alpha,
                  aes(x = n_ref, y = mean, color = param, shape = param)) +
  geom_line(linewidth = 1.1) +
  geom_errorbar(aes(ymin = lci, ymax = uci), width = 150) +
  geom_point(size = 2.2) +
  facet_wrap(~ lambda_lab, labeller = label_parsed, ncol = 4) +
  scale_color_manual(name = "Parameter", values = alpha_colors) +
  scale_shape_manual(name = "Parameter", values = c(17, 15, 4, 18, 16)) +
  geom_hline(yintercept = ALPHA_TOTAL, linetype = "dotted", color = "black") +
  labs(x = "Number of reference observations (n_ref)", y = "Alpha value",
       title = sprintf("CelebA plug-in allocation by lambda (alpha=%.2f, split%d, beta-CV)",
                       ALPHA_TOTAL, SPLIT_FLAG)) +
  theme_paper

print(p_alpha)
ggsave("real_celeb_mm_plugin_lambda_compare_allocation.pdf", p_alpha,
       width = 16, height = 8, units = "in")
cat("Plot saved: real_celeb_mm_plugin_lambda_compare_allocation.pdf\n")

# ============================================================
# 6. Numerical summary table
# ============================================================
cat("\n=== Performance by (lambda, n_ref), CGTC (selective) ===\n")
df_perf %>%
  filter(method == "CGTC (selective)") %>%
  transmute(lambda = lambda_weight, n_ref,
            coverage = sprintf("%.3f (%.3f)", mean_cov, se_cov),
            size     = sprintf("%.2f (%.2f)", mean_size, se_size),
            prop_joker = sprintf("%.3f (%.3f)", mean_propq, se_propq)) %>%
  as.data.frame() %>% print()
