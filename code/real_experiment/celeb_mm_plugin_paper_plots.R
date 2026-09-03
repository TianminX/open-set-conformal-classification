library(data.table)
library(tidyverse)
library(scales)
library(ggh4x)

# ============================================================
# Main-paper CelebA figures for the UPDATED method: CGTC with the
# plug-in missing-mass adjustment and CV-selected beta.
#   data: results_hpc/celeb_mm_plugin/
#
# IMPORTANT filters (per 2026-07-31 instructions):
#   - only the CV-selected-beta runs  (files named celeb_betacv_*)
#   - only batches 1-20               (some settings have extra batches)
#
# Real-data counterparts of the synthetic dp_mm_* figures
# (dp_mm_plugin_paper_plots.R), same style and dimensions:
#   celeb_mm_four_panel_80_joker_size.pdf   <- celeb_four_panel_80_joker_size.pdf
#     (3 panels: Coverage / Prediction Set Size / Joker Proportion,
#      dashed grey line in the joker panel = true unseen proportion,
#      method legend on the right)
#   celeb_mm_cond_cov_four_levels.pdf       (appendix)
#   celeb_mm_pvalue_propjoker.pdf           (appendix)
#   celeb_mm_pvalue_full.pdf                (appendix)
#   celeb_mm_tuned_alphas.pdf               (appendix)
# Unsuffixed = split0 (CV tuning under random splitting, matching the
# old figure's tune0); *_split1.pdf = CV tuning under Bernoulli splitting.
# ============================================================

RESULTS_DIR <- "results_hpc/celeb_mm_plugin/"
fig.dir <- "."
cond_method <- "fixed"
ALPHA_TOT <- 0.20
LAMBDA <- 0.50
NLABEL <- 2000
KTOP <- 0
KBOT <- 0
BATCH_MAX <- 20

# 1. Load: betacv files only, batches 1-20 only (filtered on the FILENAME,
#    since the beta column alone cannot distinguish betacv from beta1.6 runs)
files <- list.files(RESULTS_DIR, pattern = "^celeb_betacv_.*\\.csv$", full.names = TRUE)
batch_no <- as.integer(str_match(basename(files), "_batch_(\\d+)\\.csv$")[, 2])
files <- files[!is.na(batch_no) & batch_no >= 1 & batch_no <= BATCH_MAX]
cat("Reading", length(files), "betacv files (batches 1-", BATCH_MAX, ")\n")

df_mm <- map_dfr(files, ~ {
  dt <- fread(.x)
  dt[, which(!duplicated(names(dt))), with = FALSE]
})

df_mm <- df_mm %>%
  mutate(method = recode(method,
                         "Method (random splitting)" = "CGTC+ (random)",
                         "Method (benchmark)" = "standard (random)",
                         "Method (Bernoulli)" = "CGTC+ (selective)",
                         "Method (Bernoulli benchmark)" = "standard (selective)"))

methods_to_keep <- c("CGTC+ (random)",
                     "CGTC+ (selective)",
                     "standard (random)",
                     "standard (selective)")

# 2. Summarize across batches, per split flag
summarize_mm <- function(df) {
  df %>%
    filter(method %in% methods_to_keep,
           abs(alpha_total - ALPHA_TOT) < 1e-10,
           abs(lambda_weight - LAMBDA) < 1e-10,
           n_label_total == NLABEL, k_top == KTOP, k_bot == KBOT,
           calib_num == n_ref * 0.1) %>%
    group_by(n_ref, method, pvalue_method) %>%
    summarise(
      mean_alpha_class = mean(alpha_class, na.rm = TRUE),
      mean_alpha_new = mean(alpha_unseen, na.rm = TRUE),
      mean_alpha_old = mean(alpha_seen, na.rm = TRUE),
      se_alpha_class = sd(alpha_class, na.rm = TRUE)/sqrt(n()),
      se_alpha_new = sd(alpha_unseen, na.rm = TRUE)/sqrt(n()),
      se_alpha_old = sd(alpha_seen, na.rm = TRUE)/sqrt(n()),
      mean_cov_jk = mean(`Coverage (?)`, na.rm = TRUE),
      mean_size = mean(`Size`, na.rm = TRUE),
      mean_prop_q = mean(`Prop ?`, na.rm = TRUE),
      mean_prop_unseen = mean(prop_unseen_test, na.rm = TRUE),
      se_cov_jk = sd(`Coverage (?)`, na.rm = TRUE)/sqrt(n()),
      se_size = sd(`Size`, na.rm = TRUE)/sqrt(n()),
      se_prop_q = sd(`Prop ?`, na.rm = TRUE)/sqrt(n()),
      se_prop_unseen = sd(prop_unseen_test, na.rm = TRUE)/sqrt(n()),
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
      lci_alpha_class = mean_alpha_class - 1.96*se_alpha_class,
      uci_alpha_class = mean_alpha_class + 1.96*se_alpha_class,
      lci_alpha_new = mean_alpha_new - 1.96*se_alpha_new,
      uci_alpha_new = mean_alpha_new + 1.96*se_alpha_new,
      lci_alpha_old = mean_alpha_old - 1.96*se_alpha_old,
      uci_alpha_old = mean_alpha_old + 1.96*se_alpha_old,
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
}

# Styling shared with the synthetic mm figures
custom_shapes <- c("CGTC+ (random)" = 16,
                   "standard (random)" = 15,
                   "CGTC+ (selective)" = 18,
                   "standard (selective)" = 8)

custom_colors <- c("CGTC+ (random)" = "#E41A1C",
                   "standard (random)" = "#4DAF4A",
                   "CGTC+ (selective)" = "#377EB8",
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

NOMINAL <- 1 - ALPHA_TOT
ERRW <- 150
X_LAB <- "Number of reference observations"

make_three_panel <- function(df, out_file) {
  metric_levels <- c("Coverage", "Prediction Set Size", "Joker Proportion")
  df_three <- bind_rows(
    df %>% select(n_ref, method, mean = mean_cov_jk, lci = lci_cov_jk, uci = uci_cov_jk) %>%
      mutate(metric = "Coverage"),
    df %>% select(n_ref, method, mean = mean_size, lci = lci_size, uci = uci_size) %>%
      mutate(metric = "Prediction Set Size"),
    df %>% select(n_ref, method, mean = mean_prop_q, lci = lci_prop_q, uci = uci_prop_q) %>%
      mutate(metric = "Joker Proportion")
  ) %>%
    mutate(metric = factor(metric, levels = metric_levels))

  ref_unseen <- df %>%
    group_by(n_ref) %>%
    summarise(mean = mean(mean_prop_unseen, na.rm = TRUE), .groups = "drop") %>%
    mutate(metric = factor("Joker Proportion", levels = metric_levels))

  p <- ggplot(df_three, aes(x = n_ref, y = mean, color = method, shape = method)) +
    geom_line(linewidth = 1.2) +
    geom_point(size = 3) +
    geom_errorbar(aes(ymin = lci, ymax = uci), width = ERRW, linewidth = 1) +
    geom_line(data = ref_unseen, aes(x = n_ref, y = mean),
              inherit.aes = FALSE, linetype = "dashed", color = "grey40") +
    facet_wrap(~ metric, scales = "free_y", nrow = 1, ncol = 3) +
    ggh4x::facetted_pos_scales(
      y = list(metric == "Prediction Set Size" ~ scale_y_log10())
    ) +
    scale_color_manual(name = "Method", values = custom_colors, guide = guide_legend(order = 1)) +
    scale_shape_manual(name = "Method", values = custom_shapes, guide = guide_legend(order = 1)) +
    geom_hline(
      data = tibble(metric = factor("Coverage", levels = metric_levels),
                    yintercept = NOMINAL),
      aes(yintercept = yintercept),
      linetype = "dashed", color = "black"
    ) +
    labs(x = X_LAB, y = "") +
    theme_main

  print(p)
  ggsave(out_file, p, width = 13.5, height = 3.5, units = "in")
  cat(">>> wrote", out_file, "\n")
}

make_cond_cov <- function(df, out_file) {
  df_cond <- df %>%
    filter(pvalue_method == "XGT") %>%
    select(n_ref, method,
           mean_cov_very_rare, lci_cov_very_rare, uci_cov_very_rare,
           mean_cov_rare, lci_cov_rare, uci_cov_rare,
           mean_cov_common, lci_cov_common, uci_cov_common,
           mean_cov_very_common, lci_cov_very_common, uci_cov_very_common) %>%
    pivot_longer(cols = -c(n_ref, method),
                 names_to = c("stat", "frequency_type"),
                 names_pattern = "(mean|lci|uci)_cov_(very_rare|rare|common|very_common)",
                 values_to = "value") %>%
    pivot_wider(names_from = stat, values_from = value) %>%
    mutate(frequency_type = factor(frequency_type,
                                   levels = c("very_rare", "rare", "common", "very_common"),
                                   labels = c("Very Rare", "Rare", "Common", "Very Common")))

  p <- ggplot(df_cond, aes(x = n_ref, y = mean, color = method, shape = method)) +
    geom_line(linewidth = 1) +
    geom_point(size = 2) +
    geom_errorbar(aes(ymin = lci, ymax = uci), width = ERRW, linewidth = 0.7) +
    facet_wrap(~ frequency_type, scales = "fixed", nrow = 1) +
    scale_color_manual(name = "Method", values = custom_colors, guide = guide_legend(order = 1)) +
    scale_shape_manual(name = "Method", values = custom_shapes, guide = guide_legend(order = 1)) +
    geom_hline(yintercept = NOMINAL, linetype = "dashed", color = "black", alpha = 0.5) +
    labs(x = X_LAB, y = "Coverage") +
    theme_main +
    theme(strip.text = element_text(size = 16),
          axis.text = element_text(size = 13),
          legend.text = element_text(size = 14),
          legend.title = element_text(size = 16))

  print(p)
  ggsave(out_file, p, width = 12.5, height = 3.5, units = "in")
  cat(">>> wrote", out_file, "\n")
}

make_prop_joker <- function(df, out_file) {
  df_single <- df %>% filter(method == "CGTC+ (random)")

  # Reference line: true proportion of unseen test labels
  ref_unseen <- df %>%
    group_by(n_ref) %>%
    summarise(mean = mean(mean_prop_unseen, na.rm = TRUE), .groups = "drop")

  p <- ggplot(df_single,
              aes(x = n_ref, y = mean_prop_q,
                  color = pvalue_method, shape = pvalue_method)) +
    geom_line(linewidth = 1.2) +
    geom_point(size = 3) +
    geom_errorbar(aes(ymin = lci_prop_q, ymax = uci_prop_q), width = ERRW, linewidth = 0.7) +
    geom_line(data = ref_unseen, aes(x = n_ref, y = mean),
              inherit.aes = FALSE, linetype = "dashed", color = "grey40") +
    scale_color_manual(name = "P-value Method", values = pvalue_colors,
                       guide = guide_legend(order = 1)) +
    scale_shape_manual(name = "P-value Method", values = pvalue_shapes,
                       guide = guide_legend(order = 1)) +
    labs(x = X_LAB, y = "Proportion of joker") +
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

  print(p)
  ggsave(out_file, p, width = 6.5, height = 3, units = "in")
  cat(">>> wrote", out_file, "\n")
}

make_pvalue_full <- function(df, out_file) {
  df_combined <- bind_rows(
    df %>% select(n_ref, method, pvalue_method, mean = mean_cov_jk, lci = lci_cov_jk, uci = uci_cov_jk) %>%
      mutate(metric = "Coverage"),
    df %>% select(n_ref, method, pvalue_method, mean = mean_size, lci = lci_size, uci = uci_size) %>%
      mutate(metric = "Size"),
    df %>% select(n_ref, method, pvalue_method, mean = mean_prop_q, lci = lci_prop_q, uci = uci_prop_q) %>%
      mutate(metric = "Proportion of joker")
  )

  p <- ggplot(df_combined,
              aes(x = n_ref, y = mean, color = method, shape = method)) +
    geom_line(linewidth = 1.2) +
    geom_point(size = 3) +
    geom_errorbar(aes(ymin = lci, ymax = uci), width = ERRW, linewidth = 0.7) +
    facet_grid(metric ~ pvalue_method, scales = "free_y",
               labeller = labeller(metric = c("Coverage" = "Coverage",
                                              "Size" = "Prediction Size",
                                              "Proportion of joker" = "Joker Prop"))) +
    ggh4x::facetted_pos_scales(y = list(metric == "Size" ~ scale_y_log10())) +
    scale_color_manual(name = "Method", values = custom_colors, guide = guide_legend(order = 1)) +
    scale_shape_manual(name = "Method", values = custom_shapes, guide = guide_legend(order = 1)) +
    geom_hline(data = data.frame(metric = "Coverage", yintercept = NOMINAL),
               aes(yintercept = yintercept), linetype = "dashed", color = "black") +
    labs(x = X_LAB, y = "") +
    theme_main +
    theme(strip.text = element_text(size = 14))

  print(p)
  ggsave(out_file, p, width = 12.5, height = 7, units = "in")
  cat(">>> wrote", out_file, "\n")
}

make_tuned_alphas <- function(df, out_file) {
  df_alpha_values <- df %>%
    filter(pvalue_method == "XGT", method == "CGTC+ (selective)") %>%
    select(n_ref,
           mean_alpha_class, mean_alpha_new, mean_alpha_old,
           lci_alpha_class, lci_alpha_new, lci_alpha_old,
           uci_alpha_class, uci_alpha_new, uci_alpha_old) %>%
    distinct() %>%
    pivot_longer(cols = -n_ref, names_to = "metric_type", values_to = "value") %>%
    separate(metric_type, into = c("stat", "alpha_type"), sep = "_alpha_") %>%
    pivot_wider(names_from = stat, values_from = value) %>%
    mutate(alpha_type = recode(alpha_type,
                               "class" = "class", "new" = "unseen", "old" = "seen"))

  p <- ggplot(df_alpha_values,
              aes(x = n_ref, y = mean, color = alpha_type, shape = alpha_type)) +
    geom_line(linewidth = 1.2) +
    geom_point(size = 3) +
    geom_errorbar(aes(ymin = lci, ymax = uci), width = ERRW, linewidth = 0.7) +
    scale_color_brewer(name = "alpha", palette = "Set2") +
    scale_shape_manual(name = "alpha",
                       values = c("class" = 16, "unseen" = 17, "seen" = 15)) +
    labs(x = X_LAB, y = "alpha value tuning") +
    theme_bw() +
    theme(
      text = element_text(size = 14),
      axis.title = element_text(size = 16),
      axis.text = element_text(size = 14),
      legend.title = element_text(size = 16),
      legend.text = element_text(size = 16),
      panel.grid.major = element_line(linewidth = 0.5),
      panel.grid.minor = element_blank()
    )

  print(p)
  ggsave(out_file, p, width = 6.5, height = 3, units = "in")
  cat(">>> wrote", out_file, "\n")
}

# 3. Generate all figures per split flag
for (split_flag in c(0, 1)) {
  suffix <- if (split_flag == 0) "" else "_split1"
  df_split <- df_mm %>% filter(splitting_method_flag == split_flag)
  if (nrow(df_split) == 0) {
    cat(sprintf("--- no rows for split flag %d; skipping ---\n", split_flag))
    next
  }

  df_sum <- summarize_mm(df_split)
  cat(sprintf("\n--- split flag %d: batches per (n_ref, method, pvalue) ---\n", split_flag))
  df_sum %>%
    filter(pvalue_method == "XGT", method == "CGTC+ (selective)") %>%
    select(n_ref, n_batches) %>% as.data.frame() %>% print()

  make_three_panel(df_sum %>% filter(pvalue_method == "XGT"),
                   sprintf("%s/celeb_mm_four_panel_80_joker_size%s.pdf", fig.dir, suffix))
  make_cond_cov(df_sum,
                sprintf("%s/celeb_mm_cond_cov_four_levels%s.pdf", fig.dir, suffix))
  make_prop_joker(df_sum,
                  sprintf("%s/celeb_mm_pvalue_propjoker%s.pdf", fig.dir, suffix))
  make_pvalue_full(df_sum,
                   sprintf("%s/celeb_mm_pvalue_full%s.pdf", fig.dir, suffix))
  make_tuned_alphas(df_sum,
                    sprintf("%s/celeb_mm_tuned_alphas%s.pdf", fig.dir, suffix))
}

cat("\nAll done.\n")
