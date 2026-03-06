library(data.table)
library(tidyverse)
library(scales)
library(ggh4x)

# 1. Load data from OpenMax experiment results
df_openmax <- list.files("results_hpc/dp_openmax/",
                         pattern = "\\.csv$", full.names = TRUE) %>%
  map_dfr(~ fread(.x))

# 2. Recode method names for cleaner labels
df_openmax <- df_openmax %>%
  mutate(method = recode(method,
                         "Method (OpenMax-KNN)" = "OpenMax-KNN",
                         "Method (OpenMax-MLP)" = "OpenMax-MLP"))

# Check data points per theta value
df_openmax %>%
  filter(abs(alpha_total - 0.1) < 1e-10,
         n_ref == 2000,
         method == "OpenMax-KNN") %>%
  group_by(theta) %>%
  summarise(n_points = n()) %>%
  {cat("Data points per theta:", paste(.$theta, "=", .$n_points, collapse=", "), "\n")}

methods_to_keep <- c("OpenMax-KNN", "OpenMax-MLP")

# 3. Summarize data - group by theta
df_openmax_summary <- df_openmax %>%
  filter(method %in% methods_to_keep) %>%
  group_by(theta, alpha_total, method, n_ref, n_test, calib_num) %>%
  summarise(
    # Coverage (joker_train) - the correct metric for OpenMax
    mean_cov_jk_train = mean(`Coverage (joker_train)`, na.rm = TRUE),
    se_cov_jk_train = sd(`Coverage (joker_train)`, na.rm = TRUE)/sqrt(n()),
    # Seen/unseen coverage (joker_train)
    mean_seen_cov_train = mean(`Seen Coverage (joker_train)`, na.rm = TRUE),
    se_seen_cov_train = sd(`Seen Coverage (joker_train)`, na.rm = TRUE)/sqrt(sum(!is.na(`Seen Coverage (joker_train)`))),
    mean_unseen_cov_train = mean(`Unseen Coverage (joker_train)`, na.rm = TRUE),
    se_unseen_cov_train = sd(`Unseen Coverage (joker_train)`, na.rm = TRUE)/sqrt(sum(!is.na(`Unseen Coverage (joker_train)`))),
    # Naive coverage (without joker)
    mean_cov_wo = mean(Coverage, na.rm = TRUE),
    se_cov_wo = sd(Coverage, na.rm = TRUE)/sqrt(n()),
    # Size (excluding '?')
    mean_size = mean(Size, na.rm = TRUE),
    se_size = sd(Size, na.rm = TRUE)/sqrt(n()),
    mean_size_ratio = mean(Size/num_unique_labels_train, na.rm = TRUE),
    se_size_ratio = sd(Size/num_unique_labels_train, na.rm = TRUE)/sqrt(n()),
    # Joker proportion
    mean_prop_q = mean(`Prop ?`, na.rm = TRUE),
    se_prop_q = sd(`Prop ?`, na.rm = TRUE)/sqrt(n()),
    # Empty proportion
    mean_prop_emp = mean(`Prop empty`, na.rm = TRUE),
    se_prop_emp = sd(`Prop empty`, na.rm = TRUE)/sqrt(n()),
    # Proportion unseen (relative to ref = train + calib)
    mean_prop_unseen = mean(prop_unseen_test, na.rm = TRUE),
    se_prop_unseen = sd(prop_unseen_test, na.rm = TRUE)/sqrt(n()),
    # Proportion unseen in train only
    mean_prop_unseen_train = mean(prop_unseen_train, na.rm = TRUE),
    se_prop_unseen_train = sd(prop_unseen_train, na.rm = TRUE)/sqrt(n()),
    # Proportion in calib but not in train (derived per row, then summarized)
    mean_prop_calib_not_train = mean(prop_unseen_train - prop_unseen_test, na.rm = TRUE),
    se_prop_calib_not_train = sd(prop_unseen_train - prop_unseen_test, na.rm = TRUE)/sqrt(n()),
    .groups = "drop"
  ) %>%
  mutate(
    # Confidence intervals
    lci_cov_jk_train = mean_cov_jk_train - 1.96*se_cov_jk_train,
    uci_cov_jk_train = mean_cov_jk_train + 1.96*se_cov_jk_train,
    lci_seen_cov_train = mean_seen_cov_train - 1.96*se_seen_cov_train,
    uci_seen_cov_train = mean_seen_cov_train + 1.96*se_seen_cov_train,
    lci_unseen_cov_train = mean_unseen_cov_train - 1.96*se_unseen_cov_train,
    uci_unseen_cov_train = mean_unseen_cov_train + 1.96*se_unseen_cov_train,
    lci_cov_wo = mean_cov_wo - 1.96*se_cov_wo,
    uci_cov_wo = mean_cov_wo + 1.96*se_cov_wo,
    lci_size = mean_size - 1.96*se_size,
    uci_size = mean_size + 1.96*se_size,
    lci_size_ratio = mean_size_ratio - 1.96*se_size_ratio,
    uci_size_ratio = mean_size_ratio + 1.96*se_size_ratio,
    lci_prop_q = mean_prop_q - 1.96*se_prop_q,
    uci_prop_q = mean_prop_q + 1.96*se_prop_q,
    lci_prop_emp = mean_prop_emp - 1.96*se_prop_emp,
    uci_prop_emp = mean_prop_emp + 1.96*se_prop_emp,
    lci_prop_unseen = mean_prop_unseen - 1.96*se_prop_unseen,
    uci_prop_unseen = mean_prop_unseen + 1.96*se_prop_unseen,
    lci_prop_unseen_train = mean_prop_unseen_train - 1.96*se_prop_unseen_train,
    uci_prop_unseen_train = mean_prop_unseen_train + 1.96*se_prop_unseen_train,
    lci_prop_calib_not_train = mean_prop_calib_not_train - 1.96*se_prop_calib_not_train,
    uci_prop_calib_not_train = mean_prop_calib_not_train + 1.96*se_prop_calib_not_train
  )

# Filter data for analysis
df_openmax_analysis <- df_openmax_summary %>%
  filter(
    abs(alpha_total - 0.1) < 1e-10,
    calib_num == n_ref * 0.1,
    n_ref == 2000
  )

# Define custom shapes and colors for methods
custom_shapes <- c("OpenMax-KNN" = 16,
                   "OpenMax-MLP" = 17)

custom_colors <- c("OpenMax-KNN" = "#E41A1C",
                   "OpenMax-MLP" = "#377EB8")

# ============================================================
# Four-panel plot (similar to p_dp_four_panel)
# ============================================================

df_coverage <- df_openmax_analysis %>%
  select(theta, method, mean_cov_jk_train, lci_cov_jk_train, uci_cov_jk_train) %>%
  rename(mean = mean_cov_jk_train, lci = lci_cov_jk_train, uci = uci_cov_jk_train) %>%
  mutate(metric = "Coverage")

df_size <- df_openmax_analysis %>%
  select(theta, method, mean_size, lci_size, uci_size) %>%
  rename(mean = mean_size, lci = lci_size, uci = uci_size) %>%
  mutate(metric = "Prediction Set Size")

df_joker <- df_openmax_analysis %>%
  select(theta, method, mean_prop_q, lci_prop_q, uci_prop_q) %>%
  rename(mean = mean_prop_q, lci = lci_prop_q, uci = uci_prop_q) %>%
  mutate(metric = "Joker Proportion")

df_unseen <- df_openmax_analysis %>%
  select(theta, method, mean_prop_unseen, lci_prop_unseen, uci_prop_unseen) %>%
  rename(mean = mean_prop_unseen, lci = lci_prop_unseen, uci = uci_prop_unseen) %>%
  mutate(metric = "Unseen in Ref (Train+Calib)")

df_calib_not_train <- df_openmax_analysis %>%
  select(theta, method, mean_prop_calib_not_train, lci_prop_calib_not_train, uci_prop_calib_not_train) %>%
  rename(mean = mean_prop_calib_not_train, lci = lci_prop_calib_not_train, uci = uci_prop_calib_not_train) %>%
  mutate(metric = "In Calib but not Train")

df_combined_four <- bind_rows(df_coverage, df_size, df_joker, df_unseen, df_calib_not_train) %>%
  mutate(metric = factor(
    metric,
    levels = c("Coverage", "Prediction Set Size",
               "Joker Proportion", "Unseen in Ref (Train+Calib)",
               "In Calib but not Train")
  ))

p_openmax_four_panel <- ggplot(df_combined_four,
                               aes(x = theta, y = mean, color = method, shape = method)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = lci, ymax = uci), width = 20, linewidth = 1) +
  facet_wrap(~ metric, scales = "free_y", nrow = 2, ncol = 3) +
  ggh4x::facetted_pos_scales(
    y = list(
      metric == "Prediction Set Size" ~ scale_y_log10()
    )
  ) +
  scale_color_manual(name = "Method", values = custom_colors, guide = guide_legend(order = 1)) +
  scale_shape_manual(name = "Method", values = custom_shapes, guide = guide_legend(order = 1)) +
  geom_hline(
    data = tibble(metric = factor("Coverage", levels = levels(df_combined_four$metric)),
                  yintercept = 0.9),
    aes(yintercept = yintercept),
    linetype = "dashed", color = "black"
  ) +
  labs(x = "Dirichlet concentration parameter", y = "") +
  theme_bw() +
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
    legend.position = "right"
  )

print(p_openmax_four_panel)

ggsave("dp_openmax_five_panel.pdf", p_openmax_four_panel, width = 18, height = 8, units = "in")

# ============================================================
# Conditional coverage: Seen vs Unseen (faceted)
# ============================================================

df_conditional_cov <- df_openmax_analysis %>%
  select(theta, method,
         mean_seen_cov_train, lci_seen_cov_train, uci_seen_cov_train,
         mean_unseen_cov_train, lci_unseen_cov_train, uci_unseen_cov_train) %>%
  pivot_longer(cols = c(mean_seen_cov_train, lci_seen_cov_train, uci_seen_cov_train,
                        mean_unseen_cov_train, lci_unseen_cov_train, uci_unseen_cov_train),
               names_to = c("stat", "label_type"),
               names_pattern = "(mean|lci|uci)_(seen|unseen)_cov_train",
               values_to = "value") %>%
  pivot_wider(names_from = stat, values_from = value) %>%
  mutate(label_type = factor(label_type,
                             levels = c("seen", "unseen"),
                             labels = c("Seen Labels", "Unseen Labels")))

p_openmax_conditional <- ggplot(df_conditional_cov,
                                aes(x = theta, y = mean,
                                    color = method, shape = method)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = lci, ymax = uci),
                width = 20, linewidth = 0.7) +
  facet_wrap(~ label_type, scales = "fixed", nrow = 1) +
  scale_color_manual(name = "Method",
                     values = custom_colors,
                     guide = guide_legend(order = 1)) +
  scale_shape_manual(name = "Method",
                     values = custom_shapes,
                     guide = guide_legend(order = 1)) +
  geom_hline(yintercept = 0.9, linetype = "dashed", color = "black", alpha = 0.5) +
  labs(x = "Dirichlet concentration parameter",
       y = "Coverage") +
  theme_bw() +
  theme(
    text = element_text(size = 14),
    axis.title = element_text(size = 18),
    axis.text = element_text(size = 14),
    legend.title = element_text(size = 16),
    legend.text = element_text(size = 14),
    panel.grid.major = element_line(linewidth = 0.5),
    panel.grid.minor = element_blank(),
    strip.text = element_text(size = 16)
  )

print(p_openmax_conditional)

ggsave("dp_openmax_conditional_coverage.pdf", p_openmax_conditional, width = 10, height = 4, units = "in")