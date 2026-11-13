library(data.table)
library(tidyverse)
library(scales)
library(ggh4x)

# ============================================================
# Appendix figures for the ORIGINAL CGTC method (no missing-mass
# adjustment), regenerated in the revised layout used by the
# CGTC+ main-paper figures (dp_mm_plugin_paper_plots.R):
#   - the "Unseen Test Label Proportion" panel is dropped; the true
#     unseen proportion appears instead as a dashed grey reference
#     line inside the Joker Proportion panel;
#   - the p-value comparison figure gains the same reference line.
#   data: results_hpc/dp_tuned_mixed_labels/rebuttal/ (CV-selected beta),
#   restricted to the adaptively tuned runs (tuning_method_flag == 0); the
#   folder also holds fixed-alpha reruns (tune-1) which are excluded here.
# Outputs (same filenames as before):
#   dp_four_panel_90_joker_size.pdf   (now 3 panels)
#   dp_pvalue_propjoker.pdf           (reference line added)
#   dp_cond_cov_four_levels.pdf       (refreshed from the rebuttal data)
#   dp_pvalue_full.pdf                (refreshed from the rebuttal data)
#   dp_tuned_alphas.pdf               (refreshed from the rebuttal data)
# ============================================================

fig.dir <- "."

# 1. Load data
df_orig <- list.files("results_hpc/dp_tuned_mixed_labels/rebuttal/",
                      pattern = "\\.csv$", full.names = TRUE) %>%
  map_dfr(~ {
    dt <- fread(.x)
    dt[, which(!duplicated(names(dt))), with = FALSE]
  })

# Keep only the adaptively tuned runs; the rebuttal folder also contains
# fixed-alpha (tuning_method_flag == -1) reruns that must not be pooled in.
df_orig <- df_orig %>%
  filter(tuning_method_flag == 0)

# 2. Recode method names (original CGTC keeps the plain "CGTC" label)
df_orig <- df_orig %>%
  mutate(method = recode(method,
                         "Method (random splitting)" = "CGTC (random)",
                         "Method (benchmark)" = "standard (random)",
                         "Method (Bernoulli)" = "CGTC (selective)",
                         "Method (Bernoulli benchmark)" = "standard (selective)"))

methods_to_keep <- c("CGTC (random)",
                     "CGTC (selective)",
                     "standard (random)",
                     "standard (selective)")

# 3. Summarize across batches
cond_method <- "fixed"

df_summary <- df_orig %>%
  filter(method %in% methods_to_keep) %>%
  group_by(theta, alpha_total, method, pvalue_method, calib_num, n_ref) %>%
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

df_theta <- df_summary %>%
  filter(abs(alpha_total - 0.1) < 1e-10,
         calib_num == n_ref * 0.1,
         n_ref == 2000,
         theta != 25)

cat("--- batches per theta (XGT, CGTC (random)) ---\n")
df_theta %>%
  filter(pvalue_method == "XGT", method == "CGTC (random)") %>%
  select(theta, n_batches) %>% as.data.frame() %>% print()

# Styling shared with the CGTC+ figures
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
ggsave(sprintf("%s/dp_four_panel_90_joker_size.pdf", fig.dir), p_three,
       width = 13.5, height = 3.5, units = "in")
cat(">>> wrote dp_four_panel_90_joker_size.pdf (3-panel layout)\n")

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
    legend.position = "right",
    legend.direction = "vertical",
    panel.grid.major = element_line(linewidth = 0.5),
    panel.grid.minor = element_blank()
  )

print(p_pj)
ggsave(sprintf("%s/dp_pvalue_propjoker.pdf", fig.dir), p_pj,
       width = 6.5, height = 3, units = "in")
cat(">>> wrote dp_pvalue_propjoker.pdf (reference line added)\n")

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
        legend.title = element_text(size = 16))

print(p_cond)
ggsave(sprintf("%s/dp_cond_cov_four_levels.pdf", fig.dir), p_cond,
       width = 11.5, height = 3.5, units = "in")
cat(">>> wrote dp_cond_cov_four_levels.pdf\n")

# 7. Full 3x3 p-value comparison (appendix)
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
  theme(legend.position = "right", legend.direction = "vertical",
        strip.text = element_text(size = 14),
        panel.spacing.x = grid::unit(2, "lines"))

print(p_full)
ggsave(sprintf("%s/dp_pvalue_full.pdf", fig.dir), p_full, width = 14.5, height = 7, units = "in")
cat(">>> wrote dp_pvalue_full.pdf\n")

# 8. Tuned alpha allocation vs theta (appendix)
df_alpha_values <- df_theta %>%
  filter(pvalue_method == "XGT", method == "CGTC (selective)") %>%
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
    legend.position = "right",
    legend.direction = "vertical",
    panel.grid.major = element_line(linewidth = 0.5),
    panel.grid.minor = element_blank()
  )

print(p_tuned_alphas)
ggsave(sprintf("%s/dp_tuned_alphas.pdf", fig.dir), p_tuned_alphas,
       width = 6.5, height = 3, units = "in")
cat(">>> wrote dp_tuned_alphas.pdf\n")

cat("\nAll done.\n")
