library(data.table)
library(tidyverse)
library(scales)
library(ggh4x)

# ============================================================
# Varying-calibration-proportion figures for the manuscript appendix,
# in the layout of the vary-OCC figures (dp_vary_occ_plots.R):
#   dp_three_panel_s2_lof_varyCalib.pdf     (x = nominal calibration size)
#   dp_three_panel_realized_calib.pdf       (x = realized calibration size,
#                                            annotated per method)
#   dp_cond_cov_four_levels_s2_lof_varyCalib.pdf
# The central panel of the three-panel figures reports the average
# number of previously seen labels in the prediction set (the joker
# symbol is excluded from the count).
# Data: runs of synthetic_experiment_dp.py with theta = 1000,
# n_ref = 2000, the LOF one-class classifier, and calibration sizes
# calib_num in {100, 200, 400, 1000}; the CSVs record the realized
# calibration size (n_calib_realized) under each splitting scheme.
# Expected under results_hpc/dp_tuned_mixed_labels/vary_calprop/.
# Run from code/synthetic_experiments/; outputs are written to figures/.
# NOTE: reconstruction. The original plotting script for these figures
# was not preserved; this file re-derives them from the same summary
# machinery as dp_vary_occ_plots.R.
# ============================================================

idir <- "results_hpc/dp_tuned_mixed_labels/vary_calprop"
fig.dir <- "figures"
dir.create(fig.dir, showWarnings = FALSE)

# 1. Load data
df_all <- list.files(idir, pattern = "\\.csv$", full.names = TRUE) %>%
  map_dfr(~ {
    dt <- fread(.x)
    dt[, which(!duplicated(names(dt))), with = FALSE]
  })

# 2. Recode method names
df_all <- df_all %>%
  mutate(method = recode(method,
                         "Method (random splitting)" = "CGTC (random)",
                         "Method (benchmark)" = "standard (random)",
                         "Method (Bernoulli)" = "CGTC (selective)",
                         "Method (Bernoulli benchmark)" = "standard (selective)"))

methods_to_keep <- c("CGTC (random)",
                     "CGTC (selective)",
                     "standard (random)",
                     "standard (selective)")

# Styling shared with the manuscript appendix figures
custom_shapes <- c("CGTC (random)" = 16,
                   "standard (random)" = 15,
                   "CGTC (selective)" = 18,
                   "standard (selective)" = 8)

custom_colors <- c("CGTC (random)" = "#E41A1C",
                   "standard (random)" = "#4DAF4A",
                   "CGTC (selective)" = "#377EB8",
                   "standard (selective)" = "#FF7F00")

theme_main <- theme_bw() +
  theme(
    text = element_text(size = 15),
    axis.title = element_text(size = 18),
    axis.text = element_text(size = 15),
    legend.title = element_text(size = 18),
    legend.text = element_text(size = 18),
    panel.grid.major = element_line(linewidth = 0.5),
    panel.grid.minor = element_blank(),
    strip.text = element_text(size = 15),
    legend.position = "right",
    legend.direction = "vertical"
  )

cond_method <- "fixed"

# 3. Summarize across batches (theta and n_ref fixed; x = calib_num)
df_summary <- df_all %>%
  filter(method %in% methods_to_keep,
         tuning_method_flag == 0,
         abs(alpha_total - 0.1) < 1e-10,
         theta == 1000,
         n_ref == 2000) %>%
  group_by(calib_num, method, pvalue_method) %>%
  summarise(
    mean_cov_jk = mean(`Coverage (?)`, na.rm = TRUE),
    mean_seen_size = mean(`Seen Size`, na.rm = TRUE),
    mean_prop_q = mean(`Prop ?`, na.rm = TRUE),
    mean_prop_unseen = mean(prop_unseen_test, na.rm = TRUE),
    mean_calib_realized = mean(n_calib_realized, na.rm = TRUE),
    se_cov_jk = sd(`Coverage (?)`, na.rm = TRUE)/sqrt(n()),
    se_seen_size = sd(`Seen Size`, na.rm = TRUE)/sqrt(n()),
    se_prop_q = sd(`Prop ?`, na.rm = TRUE)/sqrt(n()),
    mean_cov_very_rare = mean(.data[[paste0("Coverage (?) (very_rare) ", cond_method)]], na.rm = TRUE),
    se_cov_very_rare = sd(.data[[paste0("Coverage (?) (very_rare) ", cond_method)]], na.rm = TRUE)/sqrt(sum(!is.na(.data[[paste0("Coverage (?) (very_rare) ", cond_method)]]))),
    mean_cov_rare = mean(.data[[paste0("Coverage (?) (rare) ", cond_method)]], na.rm = TRUE),
    se_cov_rare = sd(.data[[paste0("Coverage (?) (rare) ", cond_method)]], na.rm = TRUE)/sqrt(sum(!is.na(.data[[paste0("Coverage (?) (rare) ", cond_method)]]))),
    mean_cov_common = mean(.data[[paste0("Coverage (?) (common) ", cond_method)]], na.rm = TRUE),
    se_cov_common = sd(.data[[paste0("Coverage (?) (common) ", cond_method)]], na.rm = TRUE)/sqrt(sum(!is.na(.data[[paste0("Coverage (?) (common) ", cond_method)]]))),
    mean_cov_very_common = mean(.data[[paste0("Coverage (?) (very_common) ", cond_method)]], na.rm = TRUE),
    se_cov_very_common = sd(.data[[paste0("Coverage (?) (very_common) ", cond_method)]], na.rm = TRUE)/sqrt(sum(!is.na(.data[[paste0("Coverage (?) (very_common) ", cond_method)]]))),
    n_batches = n(),
    .groups = "drop"
  ) %>%
  mutate(
    lci_cov_jk = mean_cov_jk - 1.96*se_cov_jk,
    uci_cov_jk = mean_cov_jk + 1.96*se_cov_jk,
    lci_seen_size = mean_seen_size - 1.96*se_seen_size,
    uci_seen_size = mean_seen_size + 1.96*se_seen_size,
    lci_prop_q = mean_prop_q - 1.96*se_prop_q,
    uci_prop_q = mean_prop_q + 1.96*se_prop_q,
    lci_cov_very_rare = mean_cov_very_rare - 1.96*se_cov_very_rare,
    uci_cov_very_rare = mean_cov_very_rare + 1.96*se_cov_very_rare,
    lci_cov_rare = mean_cov_rare - 1.96*se_cov_rare,
    uci_cov_rare = mean_cov_rare + 1.96*se_cov_rare,
    lci_cov_common = mean_cov_common - 1.96*se_cov_common,
    uci_cov_common = mean_cov_common + 1.96*se_cov_common,
    lci_cov_very_common = mean_cov_very_common - 1.96*se_cov_very_common,
    uci_cov_very_common = mean_cov_very_common + 1.96*se_cov_very_common
  )

df_xgt <- df_summary %>% filter(pvalue_method == "XGT")

cat("--- batches per calib_num (XGT, CGTC (random)) ---\n")
df_xgt %>%
  filter(method == "CGTC (random)") %>%
  select(calib_num, n_batches) %>% as.data.frame() %>% print()

# 4. Three-panel figure builder (Coverage / Seen Labels in Set / Joker Prop)
metric_levels <- c("Coverage", "Seen Labels in Set", "Joker Proportion")

make_three_panel <- function(df, x_var, x_lab, errw, annotate_realized = FALSE) {
  df_three <- bind_rows(
    df %>% select(all_of(x_var), method, mean = mean_cov_jk, lci = lci_cov_jk, uci = uci_cov_jk) %>%
      mutate(metric = "Coverage"),
    df %>% select(all_of(x_var), method, mean = mean_seen_size, lci = lci_seen_size, uci = uci_seen_size) %>%
      mutate(metric = "Seen Labels in Set"),
    df %>% select(all_of(x_var), method, mean = mean_prop_q, lci = lci_prop_q, uci = uci_prop_q) %>%
      mutate(metric = "Joker Proportion")
  ) %>%
    mutate(metric = factor(metric, levels = metric_levels))

  ref_unseen <- df %>%
    group_by(.data[[x_var]]) %>%
    summarise(mean = mean(mean_prop_unseen, na.rm = TRUE), .groups = "drop") %>%
    mutate(metric = factor("Joker Proportion", levels = metric_levels))

  p <- ggplot(df_three, aes(x = .data[[x_var]], y = mean, color = method, shape = method)) +
    geom_line(linewidth = 1.2) +
    geom_point(size = 3) +
    geom_errorbar(aes(ymin = lci, ymax = uci), width = errw, linewidth = 1) +
    geom_line(data = ref_unseen, aes(x = .data[[x_var]], y = mean),
              inherit.aes = FALSE, linetype = "dashed", color = "grey40") +
    facet_wrap(~ metric, scales = "free_y", nrow = 1, ncol = 3) +
    ggh4x::facetted_pos_scales(
      y = list(metric == "Seen Labels in Set" ~ scale_y_log10())
    ) +
    scale_color_manual(name = "Method", values = custom_colors, guide = guide_legend(order = 1)) +
    scale_shape_manual(name = "Method", values = custom_shapes, guide = guide_legend(order = 1)) +
    geom_hline(
      data = tibble(metric = factor("Coverage", levels = metric_levels),
                    yintercept = 0.9),
      aes(yintercept = yintercept),
      linetype = "dashed", color = "black"
    ) +
    labs(x = x_lab, y = "") +
    theme_main

  if (annotate_realized) {
    df_lab <- df_three %>% filter(metric == "Coverage")
    p <- p + geom_text(data = df_lab,
                       aes(label = round(.data[[x_var]])),
                       vjust = -1.1, size = 3.2, check_overlap = TRUE,
                       show.legend = FALSE)
  }
  p
}

# 5. Nominal calibration size on the x-axis
p_nominal <- make_three_panel(df_xgt, "calib_num", "Calibration sample size", 30)
print(p_nominal)
ofile <- "dp_three_panel_s2_lof_varyCalib.pdf"
ggsave(file.path(fig.dir, ofile), p_nominal, width = 13.5, height = 3.5, units = "in")
cat(sprintf(">>> wrote %s\n", file.path(fig.dir, ofile)))

# 6. Realized (effective) calibration size on the x-axis, annotated
p_realized <- make_three_panel(df_xgt %>% mutate(calib_realized = mean_calib_realized),
                               "calib_realized",
                               "Realized calibration sample size", 30,
                               annotate_realized = TRUE)
print(p_realized)
ofile <- "dp_three_panel_realized_calib.pdf"
ggsave(file.path(fig.dir, ofile), p_realized, width = 13.5, height = 3.5, units = "in")
cat(sprintf(">>> wrote %s\n", file.path(fig.dir, ofile)))

# 7. Conditional coverage stratified by label frequency
df_cond <- df_xgt %>%
  select(calib_num, method,
         mean_cov_very_rare, lci_cov_very_rare, uci_cov_very_rare,
         mean_cov_rare, lci_cov_rare, uci_cov_rare,
         mean_cov_common, lci_cov_common, uci_cov_common,
         mean_cov_very_common, lci_cov_very_common, uci_cov_very_common) %>%
  pivot_longer(cols = -c(calib_num, method),
               names_to = c("stat", "frequency_type"),
               names_pattern = "(mean|lci|uci)_cov_(very_rare|rare|common|very_common)",
               values_to = "value") %>%
  pivot_wider(names_from = stat, values_from = value) %>%
  mutate(frequency_type = factor(frequency_type,
                                 levels = c("very_rare", "rare", "common", "very_common"),
                                 labels = c("Very Rare", "Rare", "Common", "Very Common")))

p_cond <- ggplot(df_cond, aes(x = calib_num, y = mean, color = method, shape = method)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  geom_errorbar(aes(ymin = lci, ymax = uci), width = 30, linewidth = 0.7) +
  facet_wrap(~ frequency_type, scales = "fixed", nrow = 1) +
  scale_color_manual(name = "Method", values = custom_colors, guide = guide_legend(order = 1)) +
  scale_shape_manual(name = "Method", values = custom_shapes, guide = guide_legend(order = 1)) +
  geom_hline(yintercept = 0.9, linetype = "dashed", color = "black", alpha = 0.5) +
  labs(x = "Calibration sample size", y = "Coverage") +
  theme_main +
  theme(strip.text = element_text(size = 16),
        axis.text = element_text(size = 14),
        legend.text = element_text(size = 14),
        legend.title = element_text(size = 16),
        legend.position = "top",
        legend.direction = "horizontal")

print(p_cond)
ofile <- "dp_cond_cov_four_levels_s2_lof_varyCalib.pdf"
ggsave(file.path(fig.dir, ofile), p_cond, width = 11.5, height = 3.5, units = "in")
cat(sprintf(">>> wrote %s\n", file.path(fig.dir, ofile)))

cat("\nAll done.\n")
