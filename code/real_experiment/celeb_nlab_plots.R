library(data.table)
library(tidyverse)
library(scales)
library(ggh4x)

# ============================================================
# CelebA figures for the varying-number-of-labels appendix setting
# (x = total number of possible labels, n_ref fixed at 2000):
#   celeb_four_panel_80_joker_size_nlab.pdf   (3-panel performance)
#   celeb_pvalue_full_nlab.pdf                (metric x p-value grid)
#   celeb_tuned_alphas_nlab.pdf               (tuned alpha allocation)
# Data: the archived celeb_tuned_mixed_labels snapshot (tune1 runs of
# real_experiment_celeb.py with the n_label_total sweep), collected
# under results_hpc/celeb_tuned_mixed_labels/; the revision-era rerun
# did not repeat the varying-number-of-labels setting.
# Run from code/real_experiment/; outputs are written to figures/.
# ============================================================

idir <- "results_hpc/celeb_tuned_mixed_labels"
fig.dir <- "figures"
dir.create(fig.dir, showWarnings = FALSE)

# 1. Load data
df_real_mixed_labels <- list.files(idir, pattern = "\\.csv$", full.names = TRUE) %>%
  map_dfr(~ {
    dt <- fread(.x)
    dt[, which(!duplicated(names(dt))), with = FALSE]
  })

# 2. Recode method names
df_real_mixed_labels <- df_real_mixed_labels %>%
  mutate(method = recode(method,
                         "Method (random splitting)" = "CGTC (random)",
                         "Method (benchmark)" = "standard (random)",
                         "Method (Bernoulli)" = "CGTC (selective)",
                         "Method (Bernoulli benchmark)" = "standard (selective)"))

methods_to_keep <- c("CGTC (random)",
                     "CGTC (selective)",
                     "standard (random)",
                     "standard (selective)")

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

# 3. Summarize across batches, grouped by the number of possible labels
df_real_analysis <- df_real_mixed_labels %>%
  filter(method %in% methods_to_keep,
         abs(alpha_total - 0.2) < 1e-10,
         tuning_method_flag == 1,
         calib_num == n_ref * 0.1,
         n_ref == 2000,
         n_label_total <= 3000,
         k_top == 0, k_bot == 0) %>%
  group_by(n_label_total, method, pvalue_method) %>%
  summarise(
    mean_alpha_class = mean(alpha_class, na.rm = TRUE),
    mean_alpha_new = mean(alpha_new, na.rm = TRUE),
    mean_alpha_old = mean(alpha_old, na.rm = TRUE),
    se_alpha_class = sd(alpha_class, na.rm = TRUE)/sqrt(n()),
    se_alpha_new = sd(alpha_new, na.rm = TRUE)/sqrt(n()),
    se_alpha_old = sd(alpha_old, na.rm = TRUE)/sqrt(n()),
    mean_cov_jk = mean(`Coverage (?)`, na.rm = TRUE),
    mean_size = mean(`Size`, na.rm = TRUE),
    mean_prop_q = mean(`Prop ?`, na.rm = TRUE),
    mean_prop_unseen = mean(prop_unseen_test, na.rm = TRUE),
    se_cov_jk = sd(`Coverage (?)`, na.rm = TRUE)/sqrt(n()),
    se_size = sd(`Size`, na.rm = TRUE)/sqrt(n()),
    se_prop_q = sd(`Prop ?`, na.rm = TRUE)/sqrt(n()),
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
    uci_prop_q = mean_prop_q + 1.96*se_prop_q
  )

cat("--- batches per n_label_total (XGT, CGTC (random)) ---\n")
df_real_analysis %>%
  filter(pvalue_method == "XGT", method == "CGTC (random)") %>%
  select(n_label_total, n_batches) %>% as.data.frame() %>% print()

nlab_breaks <- c(500, 1000, 2000, 3000)

# 4. Three-panel performance figure (Coverage / Size / Joker Proportion, XGT)
df_xgt <- df_real_analysis %>% filter(pvalue_method == "XGT")
metric_levels <- c("Coverage", "Prediction Set Size", "Joker Proportion")

df_three <- bind_rows(
  df_xgt %>% select(n_label_total, method, mean = mean_cov_jk, lci = lci_cov_jk, uci = uci_cov_jk) %>%
    mutate(metric = "Coverage"),
  df_xgt %>% select(n_label_total, method, mean = mean_size, lci = lci_size, uci = uci_size) %>%
    mutate(metric = "Prediction Set Size"),
  df_xgt %>% select(n_label_total, method, mean = mean_prop_q, lci = lci_prop_q, uci = uci_prop_q) %>%
    mutate(metric = "Joker Proportion")
) %>%
  mutate(metric = factor(metric, levels = metric_levels))

ref_unseen <- df_xgt %>%
  group_by(n_label_total) %>%
  summarise(mean = mean(mean_prop_unseen, na.rm = TRUE), .groups = "drop") %>%
  mutate(metric = factor("Joker Proportion", levels = metric_levels))

p_three <- ggplot(df_three, aes(x = n_label_total, y = mean, color = method, shape = method)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = lci, ymax = uci), width = 50, linewidth = 1) +
  geom_line(data = ref_unseen, aes(x = n_label_total, y = mean),
            inherit.aes = FALSE, linetype = "dashed", color = "grey40") +
  facet_wrap(~ metric, scales = "free_y", nrow = 1, ncol = 3) +
  ggh4x::facetted_pos_scales(
    y = list(metric == "Prediction Set Size" ~ scale_y_log10())
  ) +
  scale_color_manual(name = "Method", values = custom_colors, guide = guide_legend(order = 1)) +
  scale_shape_manual(name = "Method", values = custom_shapes, guide = guide_legend(order = 1)) +
  geom_hline(
    data = tibble(metric = factor("Coverage", levels = metric_levels),
                  yintercept = 0.8),
    aes(yintercept = yintercept),
    linetype = "dashed", color = "black"
  ) +
  scale_x_continuous(breaks = nlab_breaks) +
  labs(x = "Number of total possible labels", y = "") +
  theme_main

print(p_three)
ggsave(sprintf("%s/celeb_four_panel_80_joker_size_nlab.pdf", fig.dir), p_three,
       width = 13.5, height = 3.5, units = "in")

# 5. Metric x p-value grid
df_combined <- bind_rows(
  df_real_analysis %>%
    select(n_label_total, method, pvalue_method,
           mean = mean_cov_jk, lci = lci_cov_jk, uci = uci_cov_jk) %>%
    mutate(metric = "Coverage"),
  df_real_analysis %>%
    select(n_label_total, method, pvalue_method,
           mean = mean_size, lci = lci_size, uci = uci_size) %>%
    mutate(metric = "Size"),
  df_real_analysis %>%
    select(n_label_total, method, pvalue_method,
           mean = mean_prop_q, lci = lci_prop_q, uci = uci_prop_q) %>%
    mutate(metric = "Proportion of joker")
)

p_real_combined <- ggplot(df_combined,
                          aes(x = n_label_total, y = mean,
                              color = method, shape = method)) +
  geom_line(size = 1.2) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = lci, ymax = uci), width = 100, size = 0.7) +
  facet_grid(metric ~ pvalue_method, scales = "free_y",
             labeller = labeller(metric = c("Coverage" = "Avg. Coverage",
                                            "Size" = "Avg. Prediction Size",
                                            "Proportion of joker" = "Avg. Joker Prop"))) +
  scale_color_manual(name = "Method", values = custom_colors,
                     guide = guide_legend(order = 1)) +
  scale_shape_manual(name = "Method", values = custom_shapes,
                     guide = guide_legend(order = 1)) +
  geom_hline(data = data.frame(metric = "Coverage", yintercept = 0.8),
             aes(yintercept = yintercept),
             linetype = "dashed", color = "black") +
  scale_x_continuous(breaks = nlab_breaks) +
  labs(x = "Number of total possible labels", y = "") +
  theme_bw() +
  theme(
    panel.spacing.x = grid::unit(1.2, "lines"),
    text         = element_text(size = 14),
    axis.title   = element_text(size = 18),
    axis.text    = element_text(size = 14),
    legend.title = element_text(size = 18),
    legend.text  = element_text(size = 16),
    panel.grid.major = element_line(size = 0.5),
    panel.grid.minor = element_blank(),
    strip.text = element_text(size = 14),
    legend.position = "top",
    legend.direction = "horizontal"
  )

print(p_real_combined)
ggsave(sprintf("%s/celeb_pvalue_full_nlab.pdf", fig.dir), p_real_combined,
       width = 11.5, height = 7, units = "in")

# 6. Tuned alpha allocation
df_alpha_values <- df_real_analysis %>%
  select(n_label_total,
         mean_alpha_class, mean_alpha_new, mean_alpha_old,
         lci_alpha_class, lci_alpha_new, lci_alpha_old,
         uci_alpha_class, uci_alpha_new, uci_alpha_old) %>%
  distinct() %>%
  pivot_longer(cols = c(mean_alpha_class, mean_alpha_new, mean_alpha_old,
                        lci_alpha_class, lci_alpha_new, lci_alpha_old,
                        uci_alpha_class, uci_alpha_new, uci_alpha_old),
               names_to = "metric_type",
               values_to = "value") %>%
  separate(metric_type, into = c("stat", "alpha_type"), sep = "_alpha_", remove = FALSE) %>%
  select(-metric_type) %>%
  pivot_wider(names_from = stat, values_from = value) %>%
  mutate(alpha_type = recode(alpha_type,
                             "class" = "class",
                             "new" = "unseen",
                             "old" = "seen"))

p_tuned_alphas <- ggplot(df_alpha_values,
                         aes(x = n_label_total, y = mean,
                             color = alpha_type, shape = alpha_type)) +
  geom_line(size = 1.2) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = lci, ymax = uci), width = 400, size = 0.7) +
  scale_color_brewer(name = "alpha", palette = "Set2") +
  scale_shape_manual(name = "alpha",
                     values = c("class" = 16,
                                "unseen" = 17,
                                "seen" = 15)) +
  scale_x_continuous(breaks = sort(unique(df_alpha_values$n_label_total))) +
  labs(x = "Number of total possible labels",
       y = "alpha value tuning") +
  theme_bw() +
  theme(
    text         = element_text(size = 14),
    axis.title   = element_text(size = 18),
    axis.text    = element_text(size = 14),
    legend.title = element_text(size = 18),
    legend.text  = element_text(size = 16),
    panel.grid.major = element_line(size = 0.5),
    panel.grid.minor = element_blank()
  )

print(p_tuned_alphas)
ggsave(sprintf("%s/celeb_tuned_alphas_nlab.pdf", fig.dir), p_tuned_alphas,
       width = 6.5, height = 3, units = "in")

cat("\nAll done.\n")
