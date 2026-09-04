library(data.table)
library(tidyverse)
library(scales)
library(ggh4x)

# ============================================================
# Vary-OCC figures (Isolation Forest / OCSVM) for the response
# letter (Referee 1 Q5) and Appendix app:occ-sensitivity, in the
# revised layout of the manuscript appendix figures
# (dp_original_appendix_plots.R):
#   - the "Unseen Test Label Proportion" panel is dropped; the true
#     unseen proportion appears instead as a dashed grey reference
#     line inside the Joker Proportion panel;
#   - the p-value comparison figure gains the same reference line.
# Data: results_hpc/dp_tuned_mixed_labels/vary_occ/ (CV-beta rerun via
# synthetic_experiment_dp_vary_occ.py: adaptive-bandwidth OCSVM and
# 100-tree Isolation Forest, tune0 + tune-1; only tune0 is plotted).
# Run from code/synthetic_experiments/. Outputs (written to figures/):
#   dp_four_panel_s1_<occ>.pdf           (3 panels)
#   dp_pvalue_propjoker_s1_<occ>.pdf     (reference line added)
#   dp_cond_cov_four_levels_s1_<occ>.pdf
# ============================================================

idir <- "results_hpc/dp_tuned_mixed_labels/vary_occ"
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

pvalue_colors <- c("GT" = "#1b9e77", "RGT" = "#d95f02", "XGT" = "#7570b3")
pvalue_shapes <- c("GT" = 16, "RGT" = 17, "XGT" = 15)

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

for (occ_method in c("iforest", "ocsvm")) {

  # 3. Summarize across batches
  df_summary <- df_all %>%
    filter(occ == occ_method,
           method %in% methods_to_keep,
           tuning_method_flag == 0) %>%
    group_by(theta, alpha_total, method, pvalue_method, calib_num, n_ref) %>%
    summarise(
      mean_cov_jk = mean(`Coverage (?)`, na.rm = TRUE),
      mean_size = mean(`Size`, na.rm = TRUE),
      mean_prop_q = mean(`Prop ?`, na.rm = TRUE),
      mean_prop_unseen = mean(prop_unseen_test, na.rm = TRUE),
      se_cov_jk = sd(`Coverage (?)`, na.rm = TRUE)/sqrt(n()),
      se_size = sd(`Size`, na.rm = TRUE)/sqrt(n()),
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
      lci_size = mean_size - 1.96*se_size,
      uci_size = mean_size + 1.96*se_size,
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

  df_theta <- df_summary %>%
    filter(abs(alpha_total - 0.1) < 1e-10,
           calib_num == n_ref * 0.1,
           n_ref == 2000,
           theta != 25)

  cat(sprintf("--- %s: batches per theta (XGT, CGTC (random)) ---\n", occ_method))
  df_theta %>%
    filter(pvalue_method == "XGT", method == "CGTC (random)") %>%
    select(theta, n_batches) %>% as.data.frame() %>% print()

  # 4. Three-panel performance figure (Coverage / Size / Joker Proportion)
  df_xgt <- df_theta %>% filter(pvalue_method == "XGT")
  metric_levels <- c("Coverage", "Prediction Set Size", "Joker Proportion")

  df_three <- bind_rows(
    df_xgt %>% select(theta, method, mean = mean_cov_jk, lci = lci_cov_jk, uci = uci_cov_jk) %>%
      mutate(metric = "Coverage"),
    df_xgt %>% select(theta, method, mean = mean_size, lci = lci_size, uci = uci_size) %>%
      mutate(metric = "Prediction Set Size"),
    df_xgt %>% select(theta, method, mean = mean_prop_q, lci = lci_prop_q, uci = uci_prop_q) %>%
      mutate(metric = "Joker Proportion")
  ) %>%
    mutate(metric = factor(metric, levels = metric_levels))

  ref_unseen <- df_xgt %>%
    group_by(theta) %>%
    summarise(mean = mean(mean_prop_unseen, na.rm = TRUE), .groups = "drop") %>%
    mutate(metric = factor("Joker Proportion", levels = metric_levels))

  p_three <- ggplot(df_three, aes(x = theta, y = mean, color = method, shape = method)) +
    geom_line(linewidth = 1.2) +
    geom_point(size = 3) +
    geom_errorbar(aes(ymin = lci, ymax = uci), width = 20, linewidth = 1) +
    geom_line(data = ref_unseen, aes(x = theta, y = mean),
              inherit.aes = FALSE, linetype = "dashed", color = "grey40") +
    facet_wrap(~ metric, scales = "free_y", nrow = 1, ncol = 3) +
    ggh4x::facetted_pos_scales(
      y = list(metric == "Prediction Set Size" ~ scale_y_log10())
    ) +
    scale_color_manual(name = "Method", values = custom_colors, guide = guide_legend(order = 1)) +
    scale_shape_manual(name = "Method", values = custom_shapes, guide = guide_legend(order = 1)) +
    geom_hline(
      data = tibble(metric = factor("Coverage", levels = metric_levels),
                    yintercept = 0.9),
      aes(yintercept = yintercept),
      linetype = "dashed", color = "black"
    ) +
    labs(x = "Dirichlet concentration parameter", y = "") +
    theme_main

  print(p_three)
  ofile <- sprintf("dp_four_panel_s1_%s.pdf", occ_method)
  ggsave(file.path(fig.dir, ofile), p_three, width = 13.5, height = 3.5, units = "in")
  cat(sprintf(">>> wrote %s (3-panel layout)\n", file.path(fig.dir, ofile)))

  # 5. P-value comparison (joker proportion) with reference line
  df_single <- df_theta %>% filter(method == "CGTC (random)")

  ref_unseen_pj <- df_theta %>%
    group_by(theta) %>%
    summarise(mean = mean(mean_prop_unseen, na.rm = TRUE), .groups = "drop")

  p_pj <- ggplot(df_single,
                 aes(x = theta, y = mean_prop_q,
                     color = pvalue_method, shape = pvalue_method)) +
    geom_line(linewidth = 1.2) +
    geom_point(size = 3) +
    geom_errorbar(aes(ymin = lci_prop_q, ymax = uci_prop_q), width = 2, linewidth = 0.7) +
    geom_line(data = ref_unseen_pj, aes(x = theta, y = mean),
              inherit.aes = FALSE, linetype = "dashed", color = "grey40") +
    scale_color_manual(name = "P-value Method", values = pvalue_colors,
                       guide = guide_legend(order = 1)) +
    scale_shape_manual(name = "P-value Method", values = pvalue_shapes,
                       guide = guide_legend(order = 1)) +
    labs(x = "Dirichlet concentration parameter", y = "Proportion of joker") +
    theme_bw() +
    theme(
      text = element_text(size = 14),
      axis.title = element_text(size = 15),
      axis.text = element_text(size = 14),
      legend.title = element_text(size = 14),
      legend.text = element_text(size = 14),
      panel.grid.major = element_line(linewidth = 0.5),
      panel.grid.minor = element_blank()
    )

  print(p_pj)
  ofile <- sprintf("dp_pvalue_propjoker_s1_%s.pdf", occ_method)
  ggsave(file.path(fig.dir, ofile), p_pj, width = 6.5, height = 3, units = "in")
  cat(sprintf(">>> wrote %s (reference line added)\n", file.path(fig.dir, ofile)))

  # 6. Conditional coverage stratified by label frequency
  df_cond <- df_theta %>%
    filter(pvalue_method == "XGT") %>%
    select(theta, method,
           mean_cov_very_rare, lci_cov_very_rare, uci_cov_very_rare,
           mean_cov_rare, lci_cov_rare, uci_cov_rare,
           mean_cov_common, lci_cov_common, uci_cov_common,
           mean_cov_very_common, lci_cov_very_common, uci_cov_very_common) %>%
    pivot_longer(cols = -c(theta, method),
                 names_to = c("stat", "frequency_type"),
                 names_pattern = "(mean|lci|uci)_cov_(very_rare|rare|common|very_common)",
                 values_to = "value") %>%
    pivot_wider(names_from = stat, values_from = value) %>%
    mutate(frequency_type = factor(frequency_type,
                                   levels = c("very_rare", "rare", "common", "very_common"),
                                   labels = c("Very Rare", "Rare", "Common", "Very Common")))

  p_cond <- ggplot(df_cond, aes(x = theta, y = mean, color = method, shape = method)) +
    geom_line(linewidth = 1) +
    geom_point(size = 2) +
    geom_errorbar(aes(ymin = lci, ymax = uci), width = 2, linewidth = 0.7) +
    facet_wrap(~ frequency_type, scales = "fixed", nrow = 1) +
    scale_color_manual(name = "Method", values = custom_colors, guide = guide_legend(order = 1)) +
    scale_shape_manual(name = "Method", values = custom_shapes, guide = guide_legend(order = 1)) +
    geom_hline(yintercept = 0.9, linetype = "dashed", color = "black", alpha = 0.5) +
    labs(x = "Dirichlet concentration parameter", y = "Coverage") +
    theme_main +
    theme(strip.text = element_text(size = 16),
          axis.text = element_text(size = 14),
          legend.text = element_text(size = 14),
          legend.title = element_text(size = 16),
          legend.position = "top",
          legend.direction = "horizontal")

  print(p_cond)
  ofile <- sprintf("dp_cond_cov_four_levels_s1_%s.pdf", occ_method)
  ggsave(file.path(fig.dir, ofile), p_cond, width = 11.5, height = 3.5, units = "in")
  cat(sprintf(">>> wrote %s\n", file.path(fig.dir, ofile)))
}

cat("\nAll done.\n")
