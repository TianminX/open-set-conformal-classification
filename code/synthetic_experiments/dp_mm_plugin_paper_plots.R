library(data.table)
library(tidyverse)
library(scales)
library(ggh4x)

# ============================================================
# Main-paper figures for the UPDATED method: CGTC with the
# plug-in missing-mass adjustment and CV-selected beta.
#   data: results_hpc/dp_tuned_mixed_labels_mm_plugin/  (split0)
#
# Drop-in replacements for the original-CGTC main figures
# (dp_original_plot_v2.R), same style and dimensions:
#   dp_mm_four_panel_90_joker_size.pdf   <- dp_four_panel_90_joker_size.pdf
#   dp_mm_cond_cov_four_levels.pdf       <- dp_cond_cov_four_levels.pdf
#   dp_mm_pvalue_propjoker.pdf           <- dp_pvalue_propjoker.pdf
#   dp_mm_pvalue_full.pdf                <- dp_pvalue_full.pdf     (appendix)
#   dp_mm_tuned_alphas.pdf               <- dp_tuned_alphas.pdf    (appendix)
# plus nref-sweep versions (theta fixed, x = n_ref) for the appendix:
#   dp_mm_four_panel_90_joker_size_nref_theta{100,500}.pdf
#   dp_mm_cond_cov_four_levels_nref_theta{100,500}.pdf
#   dp_mm_pvalue_propjoker_nref_theta{100,500}.pdf
# ============================================================

fig.dir <- "."
cond_method <- "fixed"

# 1. Load data
df_mm <- list.files("results_hpc/dp_tuned_mixed_labels_mm_plugin/",
                    pattern = "\\.csv$", full.names = TRUE) %>%
  map_dfr(~ {
    dt <- fread(.x)
    dt[, which(!duplicated(names(dt))), with = FALSE]
  })

# 2. Recode method names (CGTC+ = CGTC with the plug-in missing-mass
#    adjustment; distinguishes these curves from the original-CGTC figures)
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

# 3. Summarize across batches x repetitions
df_mm_summary <- df_mm %>%
  filter(method %in% methods_to_keep,
         splitting_method_flag == 0) %>%
  group_by(theta, alpha_total, method, pvalue_method, calib_num,
           n_ref, n_test, lambda_weight) %>%
  summarise(
    # Tuned (deployed, mm-adjusted) alpha values
    mean_alpha_class = mean(alpha_class, na.rm = TRUE),
    mean_alpha_new = mean(alpha_unseen, na.rm = TRUE),
    mean_alpha_old = mean(alpha_seen, na.rm = TRUE),
    se_alpha_class = sd(alpha_class, na.rm = TRUE)/sqrt(n()),
    se_alpha_new = sd(alpha_unseen, na.rm = TRUE)/sqrt(n()),
    se_alpha_old = sd(alpha_seen, na.rm = TRUE)/sqrt(n()),
    mean_mu_hat = mean(mu_hat, na.rm = TRUE),
    # Main metrics
    mean_cov_jk = mean(`Coverage (?)`, na.rm = TRUE),
    mean_size = mean(`Size`, na.rm = TRUE),
    mean_prop_q = mean(`Prop ?`, na.rm = TRUE),
    mean_prop_unseen = mean(prop_unseen_test, na.rm = TRUE),
    se_cov_jk = sd(`Coverage (?)`, na.rm = TRUE)/sqrt(n()),
    se_size = sd(`Size`, na.rm = TRUE)/sqrt(n()),
    se_prop_q = sd(`Prop ?`, na.rm = TRUE)/sqrt(n()),
    se_prop_unseen = sd(prop_unseen_test, na.rm = TRUE)/sqrt(n()),
    # Conditional coverage by label-frequency bin
    mean_cov_very_rare = mean(.data[[paste0("Coverage (?) (very_rare) ", cond_method)]], na.rm = TRUE),
    se_cov_very_rare = sd(.data[[paste0("Coverage (?) (very_rare) ", cond_method)]], na.rm = TRUE)/sqrt(sum(!is.na(.data[[paste0("Coverage (?) (very_rare) ", cond_method)]]))),
    mean_cov_rare = mean(.data[[paste0("Coverage (?) (rare) ", cond_method)]], na.rm = TRUE),
    se_cov_rare = sd(.data[[paste0("Coverage (?) (rare) ", cond_method)]], na.rm = TRUE)/sqrt(sum(!is.na(.data[[paste0("Coverage (?) (rare) ", cond_method)]]))),
    mean_cov_common = mean(.data[[paste0("Coverage (?) (common) ", cond_method)]], na.rm = TRUE),
    se_cov_common = sd(.data[[paste0("Coverage (?) (common) ", cond_method)]], na.rm = TRUE)/sqrt(sum(!is.na(.data[[paste0("Coverage (?) (common) ", cond_method)]]))),
    mean_cov_very_common = mean(.data[[paste0("Coverage (?) (very_common) ", cond_method)]], na.rm = TRUE),
    se_cov_very_common = sd(.data[[paste0("Coverage (?) (very_common) ", cond_method)]], na.rm = TRUE)/sqrt(sum(!is.na(.data[[paste0("Coverage (?) (very_common) ", cond_method)]]))),
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
    lci_prop_unseen = mean_prop_unseen - 1.96*se_prop_unseen,
    uci_prop_unseen = mean_prop_unseen + 1.96*se_prop_unseen,
    lci_cov_very_rare = mean_cov_very_rare - 1.96*se_cov_very_rare,
    uci_cov_very_rare = mean_cov_very_rare + 1.96*se_cov_very_rare,
    lci_cov_rare = mean_cov_rare - 1.96*se_cov_rare,
    uci_cov_rare = mean_cov_rare + 1.96*se_cov_rare,
    lci_cov_common = mean_cov_common - 1.96*se_cov_common,
    uci_cov_common = mean_cov_common + 1.96*se_cov_common,
    lci_cov_very_common = mean_cov_very_common - 1.96*se_cov_very_common,
    uci_cov_very_common = mean_cov_very_common + 1.96*se_cov_very_common
  )

# Styling shared with the original figures
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
    legend.position = "top",
    legend.direction = "horizontal"
  )

# ============================================================
# Generic builders: x_var is "theta" or "n_ref"
# ============================================================
make_four_panel <- function(df, x_var, x_lab, errw, out_file, log_x = FALSE) {
  metric_levels <- c("Coverage", "Prediction Set Size", "Joker Proportion")
  pieces <- list(
    df %>% select(all_of(x_var), method, mean = mean_cov_jk, lci = lci_cov_jk, uci = uci_cov_jk) %>%
      mutate(metric = "Coverage"),
    df %>% select(all_of(x_var), method, mean = mean_size, lci = lci_size, uci = uci_size) %>%
      mutate(metric = "Prediction Set Size"),
    df %>% select(all_of(x_var), method, mean = mean_prop_q, lci = lci_prop_q, uci = uci_prop_q) %>%
      mutate(metric = "Joker Proportion")
  )
  df_four <- bind_rows(pieces) %>%
    mutate(metric = factor(metric, levels = metric_levels))

  # Reference line in the joker panel: true proportion of unseen test labels
  # (identical across methods, so averaged over them)
  ref_unseen <- df %>%
    group_by(.data[[x_var]]) %>%
    summarise(mean = mean(mean_prop_unseen, na.rm = TRUE), .groups = "drop") %>%
    mutate(metric = factor("Joker Proportion", levels = metric_levels))

  p <- ggplot(df_four, aes(x = .data[[x_var]], y = mean, color = method, shape = method)) +
    geom_line(linewidth = 1.2) +
    geom_point(size = 3) +
    geom_errorbar(aes(ymin = lci, ymax = uci), width = errw, linewidth = 1) +
    geom_line(data = ref_unseen, aes(x = .data[[x_var]], y = mean),
              inherit.aes = FALSE, linetype = "dashed", color = "grey40") +
    facet_wrap(~ metric, scales = "free_y", nrow = 1, ncol = 3) +
    ggh4x::facetted_pos_scales(
      y = list(metric == "Prediction Set Size" ~ scale_y_log10())
    ) +
    scale_color_manual(name = "Method", values = custom_colors, guide = guide_legend(order = 1)) +
    scale_shape_manual(name = "Method", values = custom_shapes, guide = guide_legend(order = 1)) +
    geom_hline(
      data = tibble(metric = factor("Coverage",
                                    levels = levels(df_four$metric)),
                    yintercept = 0.9),
      aes(yintercept = yintercept),
      linetype = "dashed", color = "black"
    ) +
    labs(x = x_lab, y = "") +
    theme_main +
    theme(legend.position = "right", legend.direction = "vertical")
  if (log_x) p <- p + scale_x_log10(labels = label_comma())

  print(p)
  ggsave(out_file, p, width = 13.5, height = 3.5, units = "in")
  cat(">>> wrote", out_file, "\n")
}

make_cond_cov <- function(df, x_var, x_lab, errw, out_file, log_x = FALSE) {
  df_cond <- df %>%
    filter(pvalue_method == "XGT") %>%
    select(all_of(x_var), method,
           mean_cov_very_rare, lci_cov_very_rare, uci_cov_very_rare,
           mean_cov_rare, lci_cov_rare, uci_cov_rare,
           mean_cov_common, lci_cov_common, uci_cov_common,
           mean_cov_very_common, lci_cov_very_common, uci_cov_very_common) %>%
    pivot_longer(cols = -c(all_of(x_var), method),
                 names_to = c("stat", "frequency_type"),
                 names_pattern = "(mean|lci|uci)_cov_(very_rare|rare|common|very_common)",
                 values_to = "value") %>%
    pivot_wider(names_from = stat, values_from = value) %>%
    mutate(frequency_type = factor(frequency_type,
                                   levels = c("very_rare", "rare", "common", "very_common"),
                                   labels = c("Very Rare", "Rare", "Common", "Very Common")))

  p <- ggplot(df_cond, aes(x = .data[[x_var]], y = mean, color = method, shape = method)) +
    geom_line(linewidth = 1) +
    geom_point(size = 2) +
    geom_errorbar(aes(ymin = lci, ymax = uci), width = errw, linewidth = 0.7) +
    facet_wrap(~ frequency_type, scales = "fixed", nrow = 1) +
    scale_color_manual(name = "Method", values = custom_colors, guide = guide_legend(order = 1)) +
    scale_shape_manual(name = "Method", values = custom_shapes, guide = guide_legend(order = 1)) +
    geom_hline(yintercept = 0.9, linetype = "dashed", color = "black", alpha = 0.5) +
    labs(x = x_lab, y = "Coverage") +
    theme_main +
    theme(strip.text = element_text(size = 16),
          axis.text = element_text(size = 14),
          legend.text = element_text(size = 14),
          legend.title = element_text(size = 16))
  if (log_x) p <- p + scale_x_log10(labels = label_comma())

  print(p)
  ggsave(out_file, p, width = 11.5, height = 3.5, units = "in")
  cat(">>> wrote", out_file, "\n")
}

make_prop_joker <- function(df, x_var, x_lab, errw, out_file, log_x = FALSE) {
  df_single <- df %>% filter(method == "CGTC+ (random)")

  # Reference line: true proportion of unseen test labels
  ref_unseen <- df %>%
    group_by(.data[[x_var]]) %>%
    summarise(mean = mean(mean_prop_unseen, na.rm = TRUE), .groups = "drop")

  p <- ggplot(df_single,
              aes(x = .data[[x_var]], y = mean_prop_q,
                  color = pvalue_method, shape = pvalue_method)) +
    geom_line(linewidth = 1.2) +
    geom_point(size = 3) +
    geom_errorbar(aes(ymin = lci_prop_q, ymax = uci_prop_q), width = errw, linewidth = 0.7) +
    geom_line(data = ref_unseen, aes(x = .data[[x_var]], y = mean),
              inherit.aes = FALSE, linetype = "dashed", color = "grey40") +
    scale_color_manual(name = "P-value Method", values = pvalue_colors,
                       guide = guide_legend(order = 1)) +
    scale_shape_manual(name = "P-value Method", values = pvalue_shapes,
                       guide = guide_legend(order = 1)) +
    labs(x = x_lab, y = "Proportion of joker") +
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
  if (log_x) p <- p + scale_x_log10(labels = label_comma())

  print(p)
  ggsave(out_file, p, width = 6.5, height = 3, units = "in")
  cat(">>> wrote", out_file, "\n")
}

# ============================================================
# A. Theta sweep (n_ref = 2000): main-paper figures
# ============================================================
df_theta <- df_mm_summary %>%
  filter(abs(alpha_total - 0.1) < 1e-10,
         calib_num == n_ref * 0.1,
         n_ref == 2000,
         theta != 25)

cat("--- theta sweep: data points per theta (XGT, CGTC (random)) ---\n")
df_mm %>%
  filter(splitting_method_flag == 0, n_ref == 2000,
         pvalue_method == "XGT", method == "CGTC+ (random)") %>%
  count(theta) %>% as_tibble() %>% print(n = 50)

make_four_panel(df_theta %>% filter(pvalue_method == "XGT"),
                "theta", "Dirichlet concentration parameter", 20,
                sprintf("%s/dp_mm_four_panel_90_joker_size.pdf", fig.dir))

make_cond_cov(df_theta, "theta", "Dirichlet concentration parameter", 2,
              sprintf("%s/dp_mm_cond_cov_four_levels.pdf", fig.dir))

make_prop_joker(df_theta, "theta", "Dirichlet concentration parameter", 2,
                sprintf("%s/dp_mm_pvalue_propjoker.pdf", fig.dir))

# Full 3x3 p-value comparison (appendix)
df_combined <- bind_rows(
  df_theta %>% select(theta, method, pvalue_method, mean = mean_cov_jk, lci = lci_cov_jk, uci = uci_cov_jk) %>%
    mutate(metric = "Coverage"),
  df_theta %>% select(theta, method, pvalue_method, mean = mean_size, lci = lci_size, uci = uci_size) %>%
    mutate(metric = "Size"),
  df_theta %>% select(theta, method, pvalue_method, mean = mean_prop_q, lci = lci_prop_q, uci = uci_prop_q) %>%
    mutate(metric = "Proportion of joker")
)

p_full <- ggplot(df_combined,
                 aes(x = theta, y = mean, color = method, shape = method)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = lci, ymax = uci), width = 2, linewidth = 0.7) +
  facet_grid(metric ~ pvalue_method, scales = "free_y",
             labeller = labeller(metric = c("Coverage" = "Coverage",
                                            "Size" = "Prediction Size",
                                            "Proportion of joker" = "Joker Prop"))) +
  ggh4x::facetted_pos_scales(y = list(metric == "Size" ~ scale_y_log10())) +
  scale_color_manual(name = "Method", values = custom_colors, guide = guide_legend(order = 1)) +
  scale_shape_manual(name = "Method", values = custom_shapes, guide = guide_legend(order = 1)) +
  geom_hline(data = data.frame(metric = "Coverage", yintercept = 0.9),
             aes(yintercept = yintercept), linetype = "dashed", color = "black") +
  labs(x = "Dirichlet concentration parameter", y = "") +
  theme_main +
  theme(strip.text = element_text(size = 14))

print(p_full)
ggsave(sprintf("%s/dp_mm_pvalue_full.pdf", fig.dir), p_full, width = 12.5, height = 7, units = "in")
cat(">>> wrote dp_mm_pvalue_full.pdf\n")

# Tuned (deployed, mm-adjusted) alpha allocation vs theta (appendix)
df_alpha_values <- df_theta %>%
  filter(pvalue_method == "XGT", method == "CGTC+ (selective)") %>%
  select(theta,
         mean_alpha_class, mean_alpha_new, mean_alpha_old,
         lci_alpha_class, lci_alpha_new, lci_alpha_old,
         uci_alpha_class, uci_alpha_new, uci_alpha_old) %>%
  distinct() %>%
  pivot_longer(cols = -theta,
               names_to = "metric_type", values_to = "value") %>%
  separate(metric_type, into = c("stat", "alpha_type"), sep = "_alpha_") %>%
  pivot_wider(names_from = stat, values_from = value) %>%
  mutate(alpha_type = recode(alpha_type,
                             "class" = "class", "new" = "unseen", "old" = "seen"))

p_tuned_alphas <- ggplot(df_alpha_values,
                         aes(x = theta, y = mean, color = alpha_type, shape = alpha_type)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = lci, ymax = uci), width = 40, linewidth = 0.7) +
  scale_color_brewer(name = "alpha", palette = "Set2") +
  scale_shape_manual(name = "alpha",
                     values = c("class" = 16, "unseen" = 17, "seen" = 15)) +
  labs(x = "Dirichlet concentration parameter", y = "alpha value tuning") +
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

print(p_tuned_alphas)
ggsave(sprintf("%s/dp_mm_tuned_alphas.pdf", fig.dir), p_tuned_alphas,
       width = 6.5, height = 3, units = "in")
cat(">>> wrote dp_mm_tuned_alphas.pdf\n")

# ============================================================
# B. nref sweep (theta fixed): appendix figures
# ============================================================
for (th in c(100, 500)) {
  df_nref <- df_mm_summary %>%
    filter(abs(alpha_total - 0.1) < 1e-10,
           calib_num == n_ref * 0.1,
           theta == th) %>%
    { if (n_distinct(.$n_ref) > 1) filter(., TRUE) else . }

  if (n_distinct(df_nref$n_ref) < 2) {
    cat(sprintf("--- skipping nref sweep for theta = %d (only one n_ref) ---\n", th))
    next
  }

  cat(sprintf("--- nref sweep, theta = %d: n_ref values: %s ---\n",
              th, paste(sort(unique(df_nref$n_ref)), collapse = ", ")))

  make_four_panel(df_nref %>% filter(pvalue_method == "XGT"),
                  "n_ref", "Number of reference observations", 0.05,
                  sprintf("%s/dp_mm_four_panel_90_joker_size_nref_theta%d.pdf", fig.dir, th),
                  log_x = TRUE)

  make_cond_cov(df_nref, "n_ref", "Number of reference observations", 0.02,
                sprintf("%s/dp_mm_cond_cov_four_levels_nref_theta%d.pdf", fig.dir, th),
                log_x = TRUE)

  make_prop_joker(df_nref, "n_ref", "Number of reference observations", 0.02,
                  sprintf("%s/dp_mm_pvalue_propjoker_nref_theta%d.pdf", fig.dir, th),
                  log_x = TRUE)
}

cat("\nAll done.\n")
