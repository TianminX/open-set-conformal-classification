library(data.table)
library(tidyverse)
library(scales)
library(ggh4x)

# ============================================================
# Appendix CelebA figures for CGTC, i.e., our method without the
# missing-mass adjustment (real-data counterpart of
# ../synthetic_experiments/dp_original_appendix_plots.R).
#   data: results_hpc/celeb/ (real_experiment_celeb.py), restricted to the
#   runs whose allocation is tuned by cross-validation with random
#   splitting within folds (tuning_method_flag == 0), the same convention
#   as the CGTC+ figures of celeb_mm_plugin_paper_plots.R.
# Outputs, in the style and dimensions of the CGTC+ figures:
#   celeb_four_panel_80_joker_size.pdf   (Coverage / Prediction Set Size /
#                                         Joker Proportion; dashed grey line
#                                         = true unseen proportion)
#   celeb_cond_cov_four_levels.pdf       (coverage stratified by the
#                                         frequency of the test label, with
#                                         the same bins as the CGTC+ figure)
# Run from code/real_experiment/; outputs are written to this folder.
# ============================================================

RESULTS_DIR <- "results_hpc/celeb/"
fig.dir <- "."
cond_method <- "fixed"
ALPHA_TOT <- 0.20
NLABEL <- 2000
KTOP <- 0
KBOT <- 0
TUNE_FLAG <- 0

# 1. Load
files <- list.files(RESULTS_DIR, pattern = "\\.csv$", full.names = TRUE)
if (length(files) == 0)
  stop("No CSV files found in ", RESULTS_DIR, ". Run the experiment / download results first.")

df_orig <- map_dfr(files, ~ {
  dt <- fread(.x)
  dt[, which(!duplicated(names(dt))), with = FALSE]
})

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

# 2. Summarize across batches
df_sum <- df_orig %>%
  filter(method %in% methods_to_keep,
         abs(alpha_total - ALPHA_TOT) < 1e-10,
         tuning_method_flag == TUNE_FLAG,
         n_label_total == NLABEL, k_top == KTOP, k_bot == KBOT,
         calib_num == n_ref * 0.1,
         pvalue_method == "XGT") %>%
  group_by(n_ref, method, pvalue_method) %>%
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

cat("--- batches per n_ref (XGT, CGTC (selective)) ---\n")
df_sum %>%
  filter(method == "CGTC (selective)") %>%
  select(n_ref, n_batches) %>% as.data.frame() %>% print()

# Styling shared with the CGTC+ figures
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

NOMINAL <- 1 - ALPHA_TOT
ERRW <- 150
X_LAB <- "Number of reference observations"

# 3. Three-panel performance figure (Coverage / Size / Joker Proportion)
metric_levels <- c("Coverage", "Prediction Set Size", "Joker Proportion")
df_three <- bind_rows(
  df_sum %>% select(n_ref, method, mean = mean_cov_jk, lci = lci_cov_jk, uci = uci_cov_jk) %>%
    mutate(metric = "Coverage"),
  df_sum %>% select(n_ref, method, mean = mean_size, lci = lci_size, uci = uci_size) %>%
    mutate(metric = "Prediction Set Size"),
  df_sum %>% select(n_ref, method, mean = mean_prop_q, lci = lci_prop_q, uci = uci_prop_q) %>%
    mutate(metric = "Joker Proportion")
) %>%
  mutate(metric = factor(metric, levels = metric_levels))

ref_unseen <- df_sum %>%
  group_by(n_ref) %>%
  summarise(mean = mean(mean_prop_unseen, na.rm = TRUE), .groups = "drop") %>%
  mutate(metric = factor("Joker Proportion", levels = metric_levels))

p_three <- ggplot(df_three, aes(x = n_ref, y = mean, color = method, shape = method)) +
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

print(p_three)
ggsave(sprintf("%s/celeb_four_panel_80_joker_size.pdf", fig.dir), p_three,
       width = 13.5, height = 3.5, units = "in")
cat(">>> wrote celeb_four_panel_80_joker_size.pdf\n")

# 4. Conditional coverage stratified by label frequency
df_cond <- df_sum %>%
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

p_cond <- ggplot(df_cond, aes(x = n_ref, y = mean, color = method, shape = method)) +
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

print(p_cond)
ggsave(sprintf("%s/celeb_cond_cov_four_levels.pdf", fig.dir), p_cond,
       width = 12.5, height = 3.5, units = "in")
cat(">>> wrote celeb_cond_cov_four_levels.pdf\n")

cat("\nAll done.\n")
