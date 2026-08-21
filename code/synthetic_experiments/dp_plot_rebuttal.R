rm(list = ls())
library(data.table)
library(tidyverse)
library(scales)
library(ggh4x)

# # 1. Load data from synthetic experiments folder
df_dp_mixed_labels <- list.files("/Users/yanfeizhou/Desktop/Tianmin&Cora/yanfei_results/dp_tuned_mixed_labels/",
                                 pattern = "\\.csv$", full.names = TRUE) %>%
  map_dfr(~ fread(.x))

idir <- "/Users/yanfeizhou/Desktop/Tianmin&Cora/yanfei_results/dp_tuned_mixed_labels/"
ifile.list <- list.files(idir)
df_dp_mixed_labels <- do.call("rbind", lapply(ifile.list, function(ifile) {
  df <- read_delim(sprintf("%s/%s", idir, ifile), delim=",", col_types=cols())
}))

fig.dir <- "/Users/yanfeizhou/Desktop/Tianmin&Cora/yanfei_results/figures"


# 2. Recode method names (same as real data)
recode_methods <- function(df) {
  df %>%
    mutate(method = recode(method,
                           "Method (random splitting)" = "CGTC (random)",
                           "Method (benchmark)" = "standard (random)",
                           "Method (Bernoulli)" = "CGTC (selective)",
                           "Method (Bernoulli benchmark)" = "standard (selective)",
                           "Method (Bernoulli uniform)" = "CGTC (Bernoulli uniform)"))
}

df_dp_mixed_labels <- recode_methods(df_dp_mixed_labels)

# Check data points per theta value
df_dp_mixed_labels %>%
  filter(abs(alpha_total - 0.1) < 1e-10,
         tuning_method_flag == 0,
         calib_num == n_ref * 0.1,
         # theta == 1000,
         n_ref == 2000,
         pvalue_method == "GT",
         method ==  "CGTC (random)" ) %>%
  group_by(theta) %>%
  summarise(n_points = n()) %>%
  {cat("Data points per theta:", paste(.$theta, "=", .$n_points, collapse=", "), "\n")}

methods_to_keep <- c("CGTC (random)",
                     "CGTC (selective)",
                     "standard (random)",
                     "standard (selective)")


# Set the conditional method variable
cond_method <- "fixed1"

# 3. Summarize data - group by theta instead of n_ref
df_dp_summary <- df_dp_mixed_labels %>%
  filter(method %in% methods_to_keep) %>%
  group_by(theta, alpha_total, method, pvalue_method, calib_num, 
           n_ref, n_test, lambda_weight, tuning_method_flag) %>%
  summarise(
    # Include average tuned alpha values
    mean_alpha_class = mean(alpha_class, na.rm = TRUE),
    mean_alpha_new = mean(alpha_new, na.rm = TRUE),
    mean_alpha_old = mean(alpha_old, na.rm = TRUE),
    se_alpha_class = sd(alpha_class, na.rm = TRUE)/sqrt(n()),
    se_alpha_new = sd(alpha_new, na.rm = TRUE)/sqrt(n()),
    se_alpha_old = sd(alpha_old, na.rm = TRUE)/sqrt(n()),
    # Original metrics
    mean_cov_wo = mean(Coverage, na.rm = TRUE),
    mean_cov_jk = mean(`Coverage (?)`, na.rm = TRUE),
    mean_size = mean(`Size (?)`, na.rm = TRUE),
    mean_size_ratio = mean(`Size (?)`/num_unique_labels, na.rm = TRUE),
    mean_prop_q = mean(`Prop ?`, na.rm = TRUE),
    mean_prop_emp = mean(`Prop empty`, na.rm = TRUE),
    mean_prop_unseen = mean(prop_unseen_test, na.rm = TRUE),
    se_cov_wo = sd(Coverage, na.rm = TRUE)/sqrt(n()),
    se_cov_jk = sd(`Coverage (?)`, na.rm = TRUE)/sqrt(n()),
    se_size = sd(`Size (?)`, na.rm = TRUE)/sqrt(n()),
    se_size_ratio = sd(`Size (?)`/num_unique_labels, na.rm = TRUE)/sqrt(n()),
    se_prop_q = sd(`Prop ?`, na.rm = TRUE)/sqrt(n()),
    se_prop_emp = sd(`Prop empty`, na.rm = TRUE)/sqrt(n()),
    se_prop_unseen = sd(prop_unseen_test, na.rm = TRUE)/sqrt(n()),
    # NEW: Conditional coverage metrics
    mean_seen_cov = mean(`Seen Coverage`, na.rm = TRUE),
    mean_seen_cov_jk = mean(`Seen Coverage (?)`, na.rm = TRUE),
    mean_unseen_cov = mean(`Unseen Coverage`, na.rm = TRUE),
    mean_unseen_cov_jk = mean(`Unseen Coverage (?)`, na.rm = TRUE),
    se_seen_cov = sd(`Seen Coverage`, na.rm = TRUE)/sqrt(sum(!is.na(`Seen Coverage`))),
    se_seen_cov_jk = sd(`Seen Coverage (?)`, na.rm = TRUE)/sqrt(sum(!is.na(`Seen Coverage (?)`))),
    se_unseen_cov = sd(`Unseen Coverage`, na.rm = TRUE)/sqrt(sum(!is.na(`Unseen Coverage`))),
    se_unseen_cov_jk = sd(`Unseen Coverage (?)`, na.rm = TRUE)/sqrt(sum(!is.na(`Unseen Coverage (?)`))),
    # Count valid observations for conditional coverage
    n_seen_valid = sum(!is.na(`Seen Coverage`)),
    n_unseen_valid = sum(!is.na(`Unseen Coverage`)),
    # Tuning metrics
    mean_tuning_loss = mean(tuning_loss, na.rm = TRUE),
    mean_tuning_normalized_size = mean(tuning_normalized_size, na.rm = TRUE),
    mean_tuning_joker_waste = mean(tuning_joker_waste, na.rm = TRUE),
    se_tuning_loss = sd(tuning_loss, na.rm = TRUE)/sqrt(n()),
    se_tuning_normalized_size = sd(tuning_normalized_size, na.rm = TRUE)/sqrt(n()),
    se_tuning_joker_waste = sd(tuning_joker_waste, na.rm = TRUE)/sqrt(n()),
    # Conditional coverage metrics
    # Very rare
    mean_cov_very_rare = mean(.data[[paste0("Coverage (?) (very_rare) ", cond_method)]], na.rm = TRUE),
    se_cov_very_rare = sd(.data[[paste0("Coverage (?) (very_rare) ", cond_method)]], na.rm = TRUE)/sqrt(sum(!is.na(.data[[paste0("Coverage (?) (very_rare) ", cond_method)]]))),
    n_very_rare_valid = sum(!is.na(.data[[paste0("Coverage (?) (very_rare) ", cond_method)]])),
    # Rare
    mean_cov_rare = mean(.data[[paste0("Coverage (?) (rare) ", cond_method)]], na.rm = TRUE),
    se_cov_rare = sd(.data[[paste0("Coverage (?) (rare) ", cond_method)]], na.rm = TRUE)/sqrt(sum(!is.na(.data[[paste0("Coverage (?) (rare) ", cond_method)]]))),
    n_rare_valid = sum(!is.na(.data[[paste0("Coverage (?) (rare) ", cond_method)]])),
    # Common
    mean_cov_common = mean(.data[[paste0("Coverage (?) (common) ", cond_method)]], na.rm = TRUE),
    se_cov_common = sd(.data[[paste0("Coverage (?) (common) ", cond_method)]], na.rm = TRUE)/sqrt(sum(!is.na(.data[[paste0("Coverage (?) (common) ", cond_method)]]))),
    n_common_valid = sum(!is.na(.data[[paste0("Coverage (?) (common) ", cond_method)]])),
    # Very common
    mean_cov_very_common = mean(.data[[paste0("Coverage (?) (very_common) ", cond_method)]], na.rm = TRUE),
    se_cov_very_common = sd(.data[[paste0("Coverage (?) (very_common) ", cond_method)]], na.rm = TRUE)/sqrt(sum(!is.na(.data[[paste0("Coverage (?) (very_common) ", cond_method)]]))),
    n_very_common_valid = sum(!is.na(.data[[paste0("Coverage (?) (very_common) ", cond_method)]])),
    # Conditional size metrics 
    # Very rare
    mean_size_very_rare = mean(.data[[paste0("Size (very_rare) ", cond_method)]], na.rm = TRUE),
    se_size_very_rare = sd(.data[[paste0("Size (very_rare) ", cond_method)]], na.rm = TRUE)/sqrt(sum(!is.na(.data[[paste0("Size (very_rare) ", cond_method)]]))),
    # Rare
    mean_size_rare = mean(.data[[paste0("Size (rare) ", cond_method)]], na.rm = TRUE),
    se_size_rare = sd(.data[[paste0("Size (rare) ", cond_method)]], na.rm = TRUE)/sqrt(sum(!is.na(.data[[paste0("Size (rare) ", cond_method)]]))),
    # Common
    mean_size_common = mean(.data[[paste0("Size (common) ", cond_method)]], na.rm = TRUE),
    se_size_common = sd(.data[[paste0("Size (common) ", cond_method)]], na.rm = TRUE)/sqrt(sum(!is.na(.data[[paste0("Size (common) ", cond_method)]]))),
    # Very common
    mean_size_very_common = mean(.data[[paste0("Size (very_common) ", cond_method)]], na.rm = TRUE),
    se_size_very_common = sd(.data[[paste0("Size (very_common) ", cond_method)]], na.rm = TRUE)/sqrt(sum(!is.na(.data[[paste0("Size (very_common) ", cond_method)]]))),
    
    .groups = "drop"
  ) %>%
  mutate(
    # Confidence intervals for alpha values
    lci_alpha_class = mean_alpha_class - 1.96*se_alpha_class,
    uci_alpha_class = mean_alpha_class + 1.96*se_alpha_class,
    lci_alpha_new = mean_alpha_new - 1.96*se_alpha_new,
    uci_alpha_new = mean_alpha_new + 1.96*se_alpha_new,
    lci_alpha_old = mean_alpha_old - 1.96*se_alpha_old,
    uci_alpha_old = mean_alpha_old + 1.96*se_alpha_old,
    # Original confidence intervals
    lci_cov_wo = mean_cov_wo - 1.96*se_cov_wo,
    uci_cov_wo = mean_cov_wo + 1.96*se_cov_wo,
    lci_cov_jk = mean_cov_jk - 1.96*se_cov_jk,
    uci_cov_jk = mean_cov_jk + 1.96*se_cov_jk,
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
    # NEW: Confidence intervals for conditional coverage
    lci_seen_cov = mean_seen_cov - 1.96*se_seen_cov,
    uci_seen_cov = mean_seen_cov + 1.96*se_seen_cov,
    lci_seen_cov_jk = mean_seen_cov_jk - 1.96*se_seen_cov_jk,
    uci_seen_cov_jk = mean_seen_cov_jk + 1.96*se_seen_cov_jk,
    lci_unseen_cov = mean_unseen_cov - 1.96*se_unseen_cov,
    uci_unseen_cov = mean_unseen_cov + 1.96*se_unseen_cov,
    lci_unseen_cov_jk = mean_unseen_cov_jk - 1.96*se_unseen_cov_jk,
    uci_unseen_cov_jk = mean_unseen_cov_jk + 1.96*se_unseen_cov_jk,
    # Confidence intervals for tuning metrics
    lci_tuning_loss = mean_tuning_loss - 1.96*se_tuning_loss,
    uci_tuning_loss = mean_tuning_loss + 1.96*se_tuning_loss,
    lci_tuning_normalized_size = mean_tuning_normalized_size - 1.96*se_tuning_normalized_size,
    uci_tuning_normalized_size = mean_tuning_normalized_size + 1.96*se_tuning_normalized_size,
    lci_tuning_joker_waste = mean_tuning_joker_waste - 1.96*se_tuning_joker_waste,
    uci_tuning_joker_waste = mean_tuning_joker_waste + 1.96*se_tuning_joker_waste,
    # Conditional coverage more levels
    # Confidence intervals for conditional coverage
    lci_cov_very_rare = mean_cov_very_rare - 1.96*se_cov_very_rare,
    uci_cov_very_rare = mean_cov_very_rare + 1.96*se_cov_very_rare,
    lci_cov_rare = mean_cov_rare - 1.96*se_cov_rare,
    uci_cov_rare = mean_cov_rare + 1.96*se_cov_rare,
    lci_cov_common = mean_cov_common - 1.96*se_cov_common,
    uci_cov_common = mean_cov_common + 1.96*se_cov_common,
    lci_cov_very_common = mean_cov_very_common - 1.96*se_cov_very_common,
    uci_cov_very_common = mean_cov_very_common + 1.96*se_cov_very_common,
    # Conditional size more levels
    # Confidence intervals for conditional size
    lci_size_very_rare = mean_size_very_rare - 1.96*se_size_very_rare,
    uci_size_very_rare = mean_size_very_rare + 1.96*se_size_very_rare,
    lci_size_rare = mean_size_rare - 1.96*se_size_rare,
    uci_size_rare = mean_size_rare + 1.96*se_size_rare,
    lci_size_common = mean_size_common - 1.96*se_size_common,
    uci_size_common = mean_size_common + 1.96*se_size_common,
    lci_size_very_common = mean_size_very_common - 1.96*se_size_very_common,
    uci_size_very_common = mean_size_very_common + 1.96*se_size_very_common
  )

# Filter data for analysis - adjust filtering as needed
df_dp_analysis <- df_dp_summary %>%
  filter(
    abs(alpha_total - 0.1) < 1e-10,  # Filter by total alpha
    tuning_method_flag == 0,          # Filter for random tuning method
    calib_num == n_ref * 0.1,        # 10% calibration
    n_ref == 2000,                     # Fixed n_ref value
    # theta == 1000,
    theta != 25
    )

# Print summary of tuned alpha values
print("Average tuned alpha values for alpha_total = 0.2, tuning_method = random:")
df_dp_analysis %>%
  select(n_ref, mean_alpha_class, mean_alpha_new, mean_alpha_old) %>%
  distinct() %>%
  print()

# Define custom shapes and colors for methods
custom_shapes <- c("CGTC (random)" = 16,
                   "standard (random)" = 15,
                   "CGTC (selective)" = 18,
                   "standard (selective)" = 8,
                   "CGTC (Bernoulli uniform)" = 4)

custom_colors <- c("CGTC (random)" = "#E41A1C",
                   "standard (random)" = "#4DAF4A",
                   "CGTC (selective)" = "#377EB8",
                   "standard (selective)" = "#FF7F00",
                   "CGTC (Bernoulli uniform)" = "#A65628")

# # Plot 1: Coverage without joker (x-axis is now theta)
# p_dp_coverage_wo <- ggplot(df_dp_analysis, 
#                            aes(x = n_ref, #theta, 
#                                y = mean_cov_wo, 
#                                color = method,
#                                shape = method)) +
#   geom_line(linewidth = 1.2) +
#   geom_point(size = 3) +
#   geom_errorbar(aes(ymin = lci_cov_wo, ymax = uci_cov_wo),
#                 width = 2, size = 0.7) +
#   scale_color_manual(name = "Method", 
#                      values = custom_colors,
#                      guide = guide_legend(order = 1)) +
#   scale_shape_manual(name = "Method", 
#                      values = custom_shapes,
#                      guide = guide_legend(order = 1)) +
#   geom_hline(yintercept = 0.9, linetype = "dashed", color = "black") +
#   labs(x = "Dirichlet concentration parameter",
#        y = "Average Coverage (without joker)",
#        title = paste0("Alpha Total = 0.2, Tuning Method = Random")) +
#   theme_bw() +
#   theme(
#     text = element_text(size = 14),
#     axis.title = element_text(size = 18),
#     axis.text = element_text(size = 14),
#     legend.title = element_text(size = 18),
#     legend.text = element_text(size = 16),
#     panel.grid.major = element_line(size = 0.5),
#     panel.grid.minor = element_blank()
#   )
# 
# print(p_dp_coverage_wo)

# # Plot 2: Coverage with facets for p-value methods
# p_dp_coverage_facet <- ggplot(df_dp_analysis, 
#                               aes(x = n_ref, #theta, 
#                                   y = mean_cov_jk, 
#                                   color = method,
#                                   shape = method)) +
#   geom_line(size = 1.2) +
#   geom_point(size = 3) +
#   geom_errorbar(aes(ymin = lci_cov_jk, ymax = uci_cov_jk), 
#                 width = 2, size = 0.7) +
#   facet_wrap(~ pvalue_method) +
#   scale_color_manual(name = "Method", 
#                      values = custom_colors,
#                      guide = guide_legend(order = 1)) +
#   scale_shape_manual(name = "Method", 
#                      values = custom_shapes,
#                      guide = guide_legend(order = 1)) +
#   geom_hline(yintercept = 0.9, linetype = "dashed", color = "black") +
#   labs(x = "Dirichlet concentration parameter",
#        y = "Average Coverage",
#        title = paste0("Alpha Total = 0.2, Tuning Method = Random")) +
#   theme_bw() +
#   theme(
#     text = element_text(size = 14),
#     axis.title = element_text(size = 18),
#     axis.text = element_text(size = 14),
#     legend.title = element_text(size = 18),
#     legend.text = element_text(size = 16),
#     panel.grid.major = element_line(size = 0.5),
#     panel.grid.minor = element_blank()
#   )
# 
# print(p_dp_coverage_facet)

# Plot 3: Tuned alpha values over theta
df_alpha_values <- df_dp_analysis %>%
  select(theta, # n_ref, # theta, 
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
                         aes(x = theta, #n_ref, # theta, #n_ref, 
                             y = mean, color = alpha_type, shape = alpha_type)) +
  geom_line(size = 1.2) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = lci, ymax = uci),
                width = 40, size = 0.7) +
  scale_color_brewer(name = "alpha",
                     palette = "Set2") + 
  scale_shape_manual(name = "alpha",
                     values = c("class" = 16,
                                "unseen" = 17,
                                "seen" = 15)) +
  scale_x_continuous(breaks = sort(unique(df_alpha_values$n_ref))) +  # Use all unique values
  labs(x = "Number of reference observations", # "Dirichlet concentration parameter", #  
       y = "alpha value tuning") +
  theme_bw() +
  theme(
    text = element_text(size = 14),
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 14),
    legend.title = element_text(size = 16),
    legend.text = element_text(size = 16),
    panel.grid.major = element_line(size = 0.5),
    panel.grid.minor = element_blank()
  )

print(p_tuned_alphas)
ggsave(sprintf("%s/dp_tuned_alphas_nref.pdf", fig.dir), p_tuned_alphas, width = 6.5, height = 3, units = "in")


# # Filter for single pvalue_method for subsequent plots
# df_dp_single_pvalue <- df_dp_analysis %>% filter(pvalue_method == "RGT")
# 
# # Plot 4: Prediction set size
# p_dp_size <- ggplot(df_dp_single_pvalue, 
#                     aes(x = n_ref,# theta, 
#                         y = mean_size, 
#                         color = method,
#                         shape = method)) +
#   scale_y_log10() +
#   geom_line(size = 1.2) +
#   geom_point(size = 3) +
#   geom_errorbar(aes(ymin = lci_size, ymax = uci_size), 
#                 width = 2, size = 0.7) +
#   scale_color_manual(name = "Method", 
#                      values = custom_colors,
#                      guide = guide_legend(order = 1)) +
#   scale_shape_manual(name = "Method", 
#                      values = custom_shapes,
#                      guide = guide_legend(order = 1)) +
#   scale_x_continuous(breaks = sort(unique(df_alpha_values$n_ref))) +
#   labs(x = "Number of reference observations", #"Dirichlet concentration parameter",
#        y = "Average Prediction Set Size") +
#   theme_bw() +
#   theme(
#     text = element_text(size = 14),
#     axis.title = element_text(size = 18),
#     axis.text = element_text(size = 14),
#     legend.title = element_text(size = 18),
#     legend.text = element_text(size = 16),
#     panel.grid.major = element_line(size = 0.5),
#     panel.grid.minor = element_blank()
#   )
# 
# print(p_dp_size)

# # Plot 5: Normalized prediction set size
# p_dp_size_ratio <- ggplot(df_dp_single_pvalue, 
#                           aes(x = theta, 
#                               y = mean_size_ratio, 
#                               color = method,
#                               shape = method)) +
#   geom_line(size = 1.2) +
#   geom_point(size = 3) +
#   geom_errorbar(aes(ymin = lci_size_ratio, ymax = uci_size_ratio), 
#                 width = 2, size = 0.7) +
#   scale_color_manual(name = "Method", 
#                      values = custom_colors,
#                      guide = guide_legend(order = 1)) +
#   scale_shape_manual(name = "Method", 
#                      values = custom_shapes,
#                      guide = guide_legend(order = 1)) +
#   labs(x = "Dirichlet concentration parameter",
#        y = "Normalized Prediction Set Size (Size / # Unique Labels)") +
#   theme_bw() +
#   theme(
#     text = element_text(size = 14),
#     axis.title = element_text(size = 18),
#     axis.text = element_text(size = 14),
#     legend.title = element_text(size = 18),
#     legend.text = element_text(size = 16),
#     panel.grid.major = element_line(size = 0.5),
#     panel.grid.minor = element_blank()
#   )
# 
# print(p_dp_size_ratio)

# Define colors for p-value methods
pvalue_colors <- c("GT" = "#1b9e77",
                   "RGT" = "#d95f02",
                   "XGT" = "#7570b3")

pvalue_shapes <- c("GT" = 16,
                   "RGT" = 17,
                   "XGT" = 15)

# Plot 6: Proportion of joker
df_dp_single_method <- df_dp_analysis %>% filter(method == "CGTC (random)")

p_dp_prop_joker <- ggplot(df_dp_single_method, 
                          aes(x = theta, 
                              y = mean_prop_q, 
                              color = pvalue_method,
                              shape = pvalue_method)) +
  geom_line(size = 1.2) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = lci_prop_q, ymax = uci_prop_q), 
                width = 2, size = 0.7) +
  scale_color_manual(name = "P-value Method", 
                     values = pvalue_colors,
                     guide = guide_legend(order = 1)) +
  scale_shape_manual(name = "P-value Method", 
                     values = pvalue_shapes,
                     guide = guide_legend(order = 1)) +
  labs(x = "Dirichlet concentration parameter",
       y = "Proportion of joker") +
  theme_bw() +
  theme(
    text = element_text(size = 14),
    axis.title = element_text(size = 15),
    axis.text = element_text(size = 14),
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 14),
    panel.grid.major = element_line(size = 0.5),
    panel.grid.minor = element_blank()
  )

print(p_dp_prop_joker)
ggsave(sprintf("%s/dp_pvalue_propjoker.pdf", fig.dir), p_dp_prop_joker, width = 6.5, height = 3, units = "in")


# # Plot 7: Proportion of empty sets
# df_dp_empty_plot <- df_dp_analysis %>%
#   filter(method == "CGTC (random)", 
#          pvalue_method == "GT")
# 
# p_dp_empty_prop <- ggplot(df_dp_empty_plot, aes(x = theta, y = mean_prop_emp)) +
#   geom_line(size = 1.2) +
#   geom_point(size = 3) +
#   geom_errorbar(aes(ymin = lci_prop_emp, ymax = uci_prop_emp), 
#                 width = 2, size = 0.7) +
#   labs(x = "Dirichlet concentration parameter",
#        y = "Proportion of Empty Sets") +
#   theme_bw() +
#   theme(
#     text = element_text(size = 14),
#     axis.title = element_text(size = 18),
#     axis.text = element_text(size = 14),
#     legend.title = element_text(size = 18),
#     legend.text = element_text(size = 16),
#     panel.grid.major = element_line(size = 0.5),
#     panel.grid.minor = element_blank()
#   )
# 
# print(p_dp_empty_prop)

# Combined plot with facet grid
df_coverage <- df_dp_analysis %>%
  select(theta, # n_ref, #theta, 
         method, pvalue_method, mean_cov_jk, lci_cov_jk, uci_cov_jk) %>%
  rename(mean = mean_cov_jk, lci = lci_cov_jk, uci = uci_cov_jk) %>%
  mutate(metric = "Coverage")

df_size <- df_dp_analysis %>%
  select(theta, #n_ref, #theta, 
         method, pvalue_method, mean_size, lci_size, uci_size) %>%
  rename(mean = mean_size, lci = lci_size, uci = uci_size) %>%
  mutate(metric = "Size")

df_prop_joker <- df_dp_analysis %>%
  select(theta, # n_ref, # theta,
         method, pvalue_method, mean_prop_q, lci_prop_q, uci_prop_q) %>%
  rename(mean = mean_prop_q, lci = lci_prop_q, uci = uci_prop_q) %>%
  mutate(metric = "Proportion of joker")

df_combined <- bind_rows(df_coverage, df_size, df_prop_joker)

# Combined facet grid plot
p_dp_combined <- ggplot(df_combined, 
                        aes(x = theta, #n_ref, #theta, 
                            y = mean, 
                            color = method,
                            shape = method)) +
  geom_line(size = 1.2) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = lci, ymax = uci), 
                width = 2, size = 0.7) +
  facet_grid(metric ~ pvalue_method, scales = "free_y",
             labeller = labeller(metric = c("Coverage" = "Coverage",
                                            "Size" = "Prediction Size",
                                            "Proportion of joker" = "Joker Prop"))) +
  ggh4x::facetted_pos_scales(
    y = list(
      metric == "Size" ~ scale_y_log10()
    )
  ) +
  scale_color_manual(name = "Method", 
                     values = custom_colors,
                     guide = guide_legend(order = 1)) +
  scale_shape_manual(name = "Method", 
                     values = custom_shapes,
                     guide = guide_legend(order = 1)) +
  geom_hline(data = data.frame(metric = "Coverage", yintercept = 0.9),
             aes(yintercept = yintercept), 
             linetype = "dashed", 
             color = "black") +
  scale_x_continuous(breaks = sort(unique(df_alpha_values$n_ref))) +
  labs(x = "Number of reference observations", #"Dirichlet concentration parameter",
  # labs(x = "Dirichlet concentration parameter",
       y = "") +
  theme_bw() +
  theme(
    text = element_text(size = 14),
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 14),
    legend.title = element_text(size = 16),
    legend.text = element_text(size = 16),
    panel.grid.major = element_line(size = 0.5),
    panel.grid.minor = element_blank(),
    strip.text = element_text(size = 14),
    legend.position = "top",
    legend.direction = "horizontal"
  )

print(p_dp_combined)
ggsave(sprintf("%s/dp_pvalue_full_nref.pdf", fig.dir), p_dp_combined, width = 12.5, height = 7, units = "in")


# # Proportion of unseen test labels
# df_dp_unseen_simple <- df_dp_analysis %>%
#   filter(method == "CGTC (random)", pvalue_method == "GT") %>%
#   select(n_ref, #theta, 
#          mean_prop_unseen, lci_prop_unseen, uci_prop_unseen)
# 
# p_dp_prop_unseen_simple <- ggplot(df_dp_unseen_simple, 
#                                   aes(x = n_ref, #theta, 
#                                       y = mean_prop_unseen)) +
#   geom_line(size = 1.2, color = "#2166AC") +
#   geom_point(size = 3, color = "#2166AC") +
#   geom_errorbar(aes(ymin = lci_prop_unseen, ymax = uci_prop_unseen), 
#                 width = 2, size = 0.7, color = "#2166AC") +
#   labs(x = "Dirichlet concentration parameter",
#        y = "Proportion of Unseen Test Labels") +
#   theme_bw() +
#   theme(
#     text = element_text(size = 14),
#     axis.title = element_text(size = 18),
#     axis.text = element_text(size = 14),
#     panel.grid.major = element_line(size = 0.5),
#     panel.grid.minor = element_blank()
#   )
# 
# print(p_dp_prop_unseen_simple)

# Four-panel plot (for XGT)
df_dp_xgt_filtered <- df_dp_analysis %>% 
  filter(pvalue_method == "XGT")

df_dp_coverage_xgt <- df_dp_xgt_filtered %>%
  select(theta,  # n_ref, #
         method, mean_cov_jk, lci_cov_jk, uci_cov_jk) %>%
  rename(mean = mean_cov_jk, lci = lci_cov_jk, uci = uci_cov_jk) %>%
  mutate(metric = "Coverage")

df_dp_size_xgt <- df_dp_xgt_filtered %>%
  select(theta, # n_ref, #
         method, mean_size, lci_size, uci_size) %>%
  rename(mean = mean_size, lci = lci_size, uci = uci_size) %>%
  mutate(metric = "Prediction Set Size")

df_dp_joker_xgt <- df_dp_xgt_filtered %>%
  select(theta,  #n_ref, #
         method, mean_prop_q, lci_prop_q, uci_prop_q) %>%
  rename(mean = mean_prop_q, lci = lci_prop_q, uci = uci_prop_q) %>%
  mutate(metric = "Joker Proportion")

df_dp_unseen_xgt <- df_dp_xgt_filtered %>%
  select(theta,  # n_ref, #
         method, mean_prop_unseen, lci_prop_unseen, uci_prop_unseen) %>%
  rename(mean = mean_prop_unseen, lci = lci_prop_unseen, uci = uci_prop_unseen) %>%
  mutate(metric = "Unseen Test Label Proportion")

df_dp_combined_four <- bind_rows(
  df_dp_coverage_xgt, df_dp_size_xgt,
  df_dp_joker_xgt, df_dp_unseen_xgt
) %>%
  mutate(metric = factor(
    metric,
    levels = c("Coverage", "Prediction Set Size",
               "Joker Proportion", "Unseen Test Label Proportion")
  ))

# four-panel main plot
p_dp_four_panel <- ggplot(df_dp_combined_four, 
                          aes(x = theta, # n_ref, #
                              y = mean, color = method, shape = method)) +
  geom_line(size = 1.2) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = lci, ymax = uci), width = 20, size = 1) +
  facet_wrap(~ metric, scales = "free_y", nrow = 1, ncol = 4) +
  ggh4x::facetted_pos_scales(
    y = list(
      metric == "Prediction Set Size" ~ scale_y_log10()
    )
  ) +
  scale_color_manual(name = "Method", values = custom_colors, guide = guide_legend(order = 1)) +
  scale_shape_manual(name = "Method", values = custom_shapes, guide = guide_legend(order = 1)) +
  geom_hline(
    data = tibble(metric = factor("Coverage", levels = levels(df_dp_combined_four$metric)),
                  yintercept = 0.9),
    aes(yintercept = yintercept),
    linetype = "dashed", color = "black"
  ) +
  # scale_x_continuous(breaks = sort(unique(df_alpha_values$n_ref))) +
  # labs(x = "Number of reference observations", y = "") +  #"Dirichlet concentration parameter",
  labs(x = "Dirichlet concentration parameter", y = "") +
  # theme_bw() +
  # theme(
  #   text = element_text(size = 16),
  #   axis.title = element_text(size = 25),
  #   axis.text = element_text(size = 22),
  #   legend.title = element_text(size = 24),
  #   legend.text = element_text(size = 24),
  #   panel.grid.major = element_line(size = 0.5),
  #   panel.grid.minor = element_blank(),
  #   strip.text = element_text(size = 20, face = "plain"),
  #   strip.background = element_rect(fill = "grey90", color = "black"),
  #   # legend.position = "right",
  #   legend.position = "top",
  #   legend.direction = "horizontal"
  theme_bw() +
  theme(
    text = element_text(size = 15),
    axis.title = element_text(size = 18),
    axis.text = element_text(size = 15),
    legend.title = element_text(size = 18),
    legend.text = element_text(size = 18),
    panel.grid.major = element_line(size = 0.5),
    panel.grid.minor = element_blank(),
    strip.text = element_text(size = 15),
    legend.position = "top",
    legend.direction = "horizontal"
  )

# four-panel main plot
p_dp_four_panel <- ggplot(df_dp_combined_four, 
                          aes(x = theta, # n_ref, #
                              y = mean, color = method, shape = method)) +
  geom_line(size = 1.2) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = lci, ymax = uci), width = 20, size = 1) +
  facet_wrap(~ metric, scales = "free_y", nrow = 2, ncol = 2) +
  ggh4x::facetted_pos_scales(
    y = list(
      metric == "Prediction Set Size" ~ scale_y_log10()
    )
  ) +
  scale_color_manual(name = "Method", values = custom_colors, guide = guide_legend(order = 1)) +
  scale_shape_manual(name = "Method", values = custom_shapes, guide = guide_legend(order = 1)) +
  geom_hline(
    data = tibble(metric = factor("Coverage", levels = levels(df_dp_combined_four$metric)),
                  yintercept = 0.9),
    aes(yintercept = yintercept),
    linetype = "dashed", color = "black"
  ) +
  # scale_x_continuous(breaks = sort(unique(df_alpha_values$n_ref))) +
  # labs(x = "Number of reference observations", y = "") +  #"Dirichlet concentration parameter",
  labs(x = "Dirichlet concentration parameter", y = "") +
  theme_bw() +
  theme(
    text = element_text(size = 16),
    axis.title = element_text(size = 25),
    axis.text = element_text(size = 22),
    legend.title = element_text(size = 24),
    legend.text = element_text(size = 24),
    panel.grid.major = element_line(size = 0.5),
    panel.grid.minor = element_blank(),
    strip.text = element_text(size = 20, face = "plain"),
    strip.background = element_rect(fill = "grey90", color = "black"),
    legend.position = "top",
    legend.direction = "horizontal"
  # theme_bw() +
  # theme(
  #   text = element_text(size = 15),
  #   axis.title = element_text(size = 18),
  #   axis.text = element_text(size = 15),
  #   legend.title = element_text(size = 18),
  #   legend.text = element_text(size = 18),
  #   panel.grid.major = element_line(size = 0.5),
  #   panel.grid.minor = element_blank(),
  #   strip.text = element_text(size = 15),
  #   legend.position = "top",
  #   legend.direction = "horizontal"
  )
print(p_dp_four_panel)
ggsave(sprintf("%s/dp_four_panel_90_joker_size_V4.pdf", fig.dir), p_dp_four_panel, width = 12.5, height = 7.5, units = "in")


print(p_dp_four_panel)
ggsave(sprintf("%s/dp_four_panel_90_joker_size_nref.pdf", fig.dir), p_dp_four_panel, width = 20, height = 5, units = "in")



# df_conditional_cov <- df_dp_analysis %>%
#   filter(pvalue_method == "XGT") %>%  # Choose your preferred p-value method
#   select(theta, method, 
#          mean_seen_cov_jk, lci_seen_cov_jk, uci_seen_cov_jk,
#          mean_unseen_cov_jk, lci_unseen_cov_jk, uci_unseen_cov_jk) %>%
#   pivot_longer(cols = c(mean_seen_cov_jk, lci_seen_cov_jk, uci_seen_cov_jk,
#                         mean_unseen_cov_jk, lci_unseen_cov_jk, uci_unseen_cov_jk),
#                names_to = c("stat", "label_type", "joker"),
#                names_pattern = "(mean|lci|uci)_(seen|unseen)_cov_(jk)",
#                values_to = "value") %>%
#   select(-joker) %>%
#   pivot_wider(names_from = stat, values_from = value) %>%
#   mutate(label_type = factor(label_type, 
#                              levels = c("seen", "unseen"),
#                              labels = c("Seen Labels", "Unseen Labels")))
# 
# p_conditional_coverage <- ggplot(df_conditional_cov, 
#                                  aes(x = theta, y = mean, 
#                                      color = method, shape = method,
#                                      linetype = label_type)) +
  # geom_line(size = 1.2) +
  # geom_point(size = 3) +
  # geom_errorbar(aes(ymin = lci, ymax = uci), 
  #               width = 2, size = 0.7) +
  # scale_color_manual(name = "Method", 
  #                    values = custom_colors,
  #                    guide = guide_legend(order = 1)) +
  # scale_shape_manual(name = "Method", 
  #                    values = custom_shapes,
  #                    guide = guide_legend(order = 1)) +
  # scale_linetype_manual(name = "Label Type",
  #                       values = c("Seen Labels" = "solid", 
  #                                  "Unseen Labels" = "dashed")) +
  # geom_hline(yintercept = 0.9, linetype = "dotted", color = "black", alpha = 0.5) +
  # labs(x = "Dirichlet concentration parameter",
  #      y = "Average Conditional Coverage (with joker)",
  #      title = "Conditional Coverage: Seen vs Unseen Test Labels") +
  # theme_bw() +
  # theme(
  #   text = element_text(size = 14),
  #   axis.title = element_text(size = 18),
  #   axis.text = element_text(size = 14),
  #   legend.title = element_text(size = 16),
  #   legend.text = element_text(size = 14),
  #   panel.grid.major = element_line(size = 0.5),
  #   panel.grid.minor = element_blank()
  # )

# print(p_conditional_coverage)
# 
# # Plot: Faceted Conditional Coverage by P-value Method
# df_conditional_cov_facet <- df_dp_analysis %>%
#   select(theta, method, pvalue_method,
#          mean_seen_cov_jk, lci_seen_cov_jk, uci_seen_cov_jk,
#          mean_unseen_cov_jk, lci_unseen_cov_jk, uci_unseen_cov_jk) %>%
#   pivot_longer(cols = c(mean_seen_cov_jk, lci_seen_cov_jk, uci_seen_cov_jk,
#                         mean_unseen_cov_jk, lci_unseen_cov_jk, uci_unseen_cov_jk),
#                names_to = c("stat", "label_type", "joker"),
#                names_pattern = "(mean|lci|uci)_(seen|unseen)_cov_(jk)",
#                values_to = "value") %>%
#   select(-joker) %>%
#   pivot_wider(names_from = stat, values_from = value) %>%
#   mutate(label_type = factor(label_type, 
#                              levels = c("seen", "unseen"),
#                              labels = c("Seen", "Unseen")))
# 
# p_conditional_coverage_facet <- ggplot(df_conditional_cov_facet, 
#                                        aes(x = theta, y = mean, 
#                                            color = method, shape = method)) +
#   geom_line(size = 1.2) +
#   geom_point(size = 3) +
#   geom_errorbar(aes(ymin = lci, ymax = uci), 
#                 width = 2, size = 0.7) +
#   facet_grid(label_type ~ pvalue_method) +
#   scale_color_manual(name = "Method", 
#                      values = custom_colors,
#                      guide = guide_legend(order = 1)) +
#   scale_shape_manual(name = "Method", 
#                      values = custom_shapes,
#                      guide = guide_legend(order = 1)) +
#   geom_hline(yintercept = 0.9, linetype = "dashed", color = "black", alpha = 0.5) +
#   labs(x = "Dirichlet concentration parameter",
#        y = "Average Coverage (with joker)",
#        title = "Conditional Coverage by Label Type and P-value Method") +
#   theme_bw() +
#   theme(
#     text = element_text(size = 14),
#     axis.title = element_text(size = 18),
#     axis.text = element_text(size = 14),
#     legend.title = element_text(size = 16),
#     legend.text = element_text(size = 14),
#     panel.grid.major = element_line(size = 0.5),
#     panel.grid.minor = element_blank(),
#     strip.text = element_text(size = 14),
#     legend.position = "top",
#     legend.direction = "horizontal"
#   )
# 
# print(p_conditional_coverage_facet)
# 
# # Plot: Comparison of Coverage With and Without Joker for Unseen Labels
# df_unseen_joker_comparison <- df_dp_analysis %>%
#   filter(pvalue_method == "XGT") %>%
#   select(theta, method,
#          mean_unseen_cov, lci_unseen_cov, uci_unseen_cov,
#          mean_unseen_cov_jk, lci_unseen_cov_jk, uci_unseen_cov_jk)
# 
# # Reshape for without joker
# df_without_joker <- df_unseen_joker_comparison %>%
#   select(theta, method, mean_unseen_cov, lci_unseen_cov, uci_unseen_cov) %>%
#   rename(mean = mean_unseen_cov, lci = lci_unseen_cov, uci = uci_unseen_cov) %>%
#   mutate(cov_type = "Without Joker")
# 
# # Reshape for with joker  
# df_with_joker <- df_unseen_joker_comparison %>%
#   select(theta, method, mean_unseen_cov_jk, lci_unseen_cov_jk, uci_unseen_cov_jk) %>%
#   rename(mean = mean_unseen_cov_jk, lci = lci_unseen_cov_jk, uci = uci_unseen_cov_jk) %>%
#   mutate(cov_type = "With Joker")
# 
# # Combine
# df_unseen_joker_comparison <- bind_rows(df_without_joker, df_with_joker) %>%
#   mutate(cov_type = factor(cov_type, levels = c("Without Joker", "With Joker")))
# 
# p_unseen_joker_effect <- ggplot(df_unseen_joker_comparison,
#                                 aes(x = theta, y = mean,
#                                     color = method, shape = method,
#                                     linetype = cov_type)) +
#   geom_line(size = 1.2) +
#   geom_point(size = 3) +
#   geom_errorbar(aes(ymin = lci, ymax = uci),
#                 width = 2, size = 0.7) +
#   scale_color_manual(name = "Method",
#                      values = custom_colors,
#                      guide = guide_legend(order = 1)) +
#   scale_shape_manual(name = "Method",
#                      values = custom_shapes,
#                      guide = guide_legend(order = 1)) +
#   scale_linetype_manual(name = "Coverage Type",
#                         values = c("Without Joker" = "dotted",
#                                    "With Joker" = "solid")) +
#   geom_hline(yintercept = 0.9, linetype = "dashed", color = "black", alpha = 0.5) +
#   labs(x = "Dirichlet concentration parameter",
#        y = "Average Coverage for Unseen Labels",
#        title = "Impact of Joker on Unseen Label Coverage") +
#   theme_bw() +
#   theme(
#     text = element_text(size = 14),
#     axis.title = element_text(size = 18),
#     axis.text = element_text(size = 14),
#     legend.title = element_text(size = 16),
#     legend.text = element_text(size = 14),
#     panel.grid.major = element_line(size = 0.5),
#     panel.grid.minor = element_blank()
#   )
# 
# print(p_unseen_joker_effect)



# 
# # Plot: Faceted Conditional Coverage for XGT only (Seen vs Unseen)
# df_conditional_cov_facet_xgt <- df_dp_analysis %>%
#   filter(pvalue_method == "XGT") %>%  # Filter for XGT only
#   select(theta, method,
#          mean_seen_cov_jk, lci_seen_cov_jk, uci_seen_cov_jk,
#          mean_unseen_cov_jk, lci_unseen_cov_jk, uci_unseen_cov_jk) %>%
#   pivot_longer(cols = c(mean_seen_cov_jk, lci_seen_cov_jk, uci_seen_cov_jk,
#                         mean_unseen_cov_jk, lci_unseen_cov_jk, uci_unseen_cov_jk),
#                names_to = c("stat", "label_type", "joker"),
#                names_pattern = "(mean|lci|uci)_(seen|unseen)_cov_(jk)",
#                values_to = "value") %>%
#   select(-joker) %>%
#   pivot_wider(names_from = stat, values_from = value) %>%
#   mutate(label_type = factor(label_type, 
#                              levels = c("seen", "unseen"),
#                              labels = c("Seen Labels", "Unseen Labels")))
# 
# p_conditional_coverage_facet_xgt <- ggplot(df_conditional_cov_facet_xgt, 
#                                        aes(x = theta, y = mean, 
#                                            color = method, shape = method)) +
#   geom_line(size = 1) +
#   geom_point(size = 2) +
#   # geom_errorbar(aes(ymin = lci, ymax = uci), 
#   #               width = 2, size = 0.7) +
#   facet_wrap(~ label_type, scales = "fixed", nrow = 1) +  # Changed to facet_wrap for side-by-side
#   scale_color_manual(name = "Method", 
#                      values = custom_colors,
#                      guide = guide_legend(order = 1)) +
#   scale_shape_manual(name = "Method", 
#                      values = custom_shapes,
#                      guide = guide_legend(order = 1)) +
#   geom_hline(yintercept = 0.9, linetype = "dashed", color = "black", alpha = 0.5) +
#   labs(x = "Dirichlet concentration parameter",
#        y = "Coverage") +
#   theme_bw() +
#   theme(
#     text = element_text(size = 14),
#     axis.title = element_text(size = 18),
#     axis.text = element_text(size = 14),
#     legend.title = element_text(size = 16),
#     legend.text = element_text(size = 14),
#     panel.grid.major = element_line(size = 0.5),
#     panel.grid.minor = element_blank(),
#     strip.text = element_text(size = 16),
#     legend.position = "top",
#     legend.direction = "horizontal"
#   )
# 
# print(p_conditional_coverage_facet_xgt)
# 
# ggsave("dp_conditional_coverage_facet_xgt_90.pdf", p_conditional_coverage_facet_xgt, 
#        width = 10, height = 4, units = "in")

# notemark
# Conditional coverage at more levels
# Create four-panel conditional coverage plot for frequency-based labels
df_conditional_cov_facet_xgt <- df_dp_analysis %>%
  filter(pvalue_method == "XGT") %>%  # Filter for XGT only
  select(theta,  # n_ref,
         method,
         `mean_cov_very_rare`, `lci_cov_very_rare`, `uci_cov_very_rare`,
         `mean_cov_rare`, `lci_cov_rare`, `uci_cov_rare`,
         `mean_cov_common`, `lci_cov_common`, `uci_cov_common`,
         `mean_cov_very_common`, `lci_cov_very_common`, `uci_cov_very_common`) %>%
  pivot_longer(cols = c(`mean_cov_very_rare`, `lci_cov_very_rare`, `uci_cov_very_rare`,
                        `mean_cov_rare`, `lci_cov_rare`, `uci_cov_rare`,
                        `mean_cov_common`, `lci_cov_common`, `uci_cov_common`,
                        `mean_cov_very_common`, `lci_cov_very_common`, `uci_cov_very_common`),
               names_to = c("stat", "frequency_type"),
               names_pattern = "(mean|lci|uci)_cov_(very_rare|rare|common|very_common)",
               values_to = "value") %>%
  pivot_wider(names_from = stat, values_from = value) %>%
  mutate(frequency_type = factor(frequency_type, 
                                 levels = c("very_rare", "rare", "common", "very_common"),
                                 labels = c("Very Rare", "Rare", "Common", "Very Common")))

p_conditional_coverage_facet_xgt <- ggplot(df_conditional_cov_facet_xgt, 
                                           aes(x =theta, #  n_ref, #
                                               y = mean, 
                                               color = method, shape = method)) +
  geom_line(size = 1) +
  geom_point(size = 2) +
  geom_errorbar(aes(ymin = lci, ymax = uci), 
                width = 2, size = 0.7) +
  facet_wrap(~ frequency_type, scales = "fixed", nrow = 1) +  # Four panels side-by-side
  scale_color_manual(name = "Method", 
                     values = custom_colors,
                     guide = guide_legend(order = 1)) +
  scale_shape_manual(name = "Method", 
                     values = custom_shapes,
                     guide = guide_legend(order = 1)) +
  geom_hline(yintercept = 0.9, linetype = "dashed", color = "black", alpha = 0.5) +
  #scale_x_continuous(breaks = sort(unique(df_alpha_values$n_ref))) +
  #labs(x = "Number of reference observations", #"Dirichlet concentration parameter",
  labs(x = "Dirichlet concentration parameter",
       y = "Coverage") +
  theme_bw() +
  theme(
    text = element_text(size = 14),
    axis.title = element_text(size = 18),
    axis.text = element_text(size = 14),
    legend.title = element_text(size = 16),
    legend.text = element_text(size = 14),
    panel.grid.major = element_line(size = 0.5),
    panel.grid.minor = element_blank(),
    strip.text = element_text(size = 16),
    legend.position = "top",
    legend.direction = "horizontal"
  )

print(p_conditional_coverage_facet_xgt)
ggsave(sprintf("%s/dp_cond_cov_four_levels.pdf", fig.dir), p_conditional_coverage_facet_xgt, width = 11.5, height = 3.5, units = "in")



# Create 2x4 panel plot: Coverage (top row) and Size (bottom row) for frequency-based labels
df_conditional_cov_size_facet_xgt <- df_dp_analysis %>%
  filter(pvalue_method == "XGT") %>%  # Filter for XGT only
  select(theta, method,
         # Coverage metrics
         `mean_cov_very_rare`, `lci_cov_very_rare`, `uci_cov_very_rare`,
         `mean_cov_rare`, `lci_cov_rare`, `uci_cov_rare`,
         `mean_cov_common`, `lci_cov_common`, `uci_cov_common`,
         `mean_cov_very_common`, `lci_cov_very_common`, `uci_cov_very_common`,
         # Size metrics
         `mean_size_very_rare`, `lci_size_very_rare`, `uci_size_very_rare`,
         `mean_size_rare`, `lci_size_rare`, `uci_size_rare`,
         `mean_size_common`, `lci_size_common`, `uci_size_common`,
         `mean_size_very_common`, `lci_size_very_common`, `uci_size_very_common`) %>%
  pivot_longer(cols = c(`mean_cov_very_rare`, `lci_cov_very_rare`, `uci_cov_very_rare`,
                        `mean_cov_rare`, `lci_cov_rare`, `uci_cov_rare`,
                        `mean_cov_common`, `lci_cov_common`, `uci_cov_common`,
                        `mean_cov_very_common`, `lci_cov_very_common`, `uci_cov_very_common`,
                        `mean_size_very_rare`, `lci_size_very_rare`, `uci_size_very_rare`,
                        `mean_size_rare`, `lci_size_rare`, `uci_size_rare`,
                        `mean_size_common`, `lci_size_common`, `uci_size_common`,
                        `mean_size_very_common`, `lci_size_very_common`, `uci_size_very_common`),
               names_to = c("stat", "metric_type", "frequency_type"),
               names_pattern = "(mean|lci|uci)_(cov|size)_(very_rare|rare|common|very_common)",
               values_to = "value") %>%
  pivot_wider(names_from = stat, values_from = value) %>%
  mutate(
    frequency_type = factor(frequency_type, 
                           levels = c("very_rare", "rare", "common", "very_common"),
                           labels = c("Very Rare", "Rare", "Common", "Very Common")),
    metric_type = factor(metric_type,
                        levels = c("cov", "size"),
                        labels = c("Coverage", "Size"))
  )

p_conditional_coverage_size_facet_xgt <- ggplot(df_conditional_cov_size_facet_xgt, 
                                               aes(x = theta, y = mean, 
                                                   color = method, shape = method)) +
  geom_line(size = 1) +
  geom_point(size = 2) +
  geom_errorbar(aes(ymin = lci, ymax = uci), 
                width = 2, size = 0.7) +
  facet_grid(metric_type ~ frequency_type, scales = "free_y") +  # 2 rows x 4 columns
  ggh4x::facetted_pos_scales(
    y = list(
      metric_type == "Size" ~ scale_y_log10()
    )
  ) +
  scale_color_manual(name = "Method", 
                     values = custom_colors,
                     guide = guide_legend(order = 1)) +
  scale_shape_manual(name = "Method", 
                     values = custom_shapes,
                     guide = guide_legend(order = 1)) +
  # Add reference lines - coverage = 0.9 for Coverage panels only
  geom_hline(data = subset(df_conditional_cov_size_facet_xgt, metric_type == "Coverage"),
             yintercept = 0.9, linetype = "dashed", color = "black", alpha = 0.5) +
  labs(x = "Dirichlet concentration parameter",
       y = "") +  # Remove y-label since we have different metrics
  theme_bw() +
  theme(
    text = element_text(size = 14),
    axis.title = element_text(size = 18),
    axis.text = element_text(size = 14),
    legend.title = element_text(size = 16),
    legend.text = element_text(size = 14),
    panel.grid.major = element_line(size = 0.5),
    panel.grid.minor = element_blank(),
    strip.text = element_text(size = 16),
    legend.position = "top",
    legend.direction = "horizontal"
  )

print(p_conditional_coverage_size_facet_xgt)


# Save plots with appropriate names
# ggsave("dp_coverage_wo_mixed_labels.pdf", p_dp_coverage_wo, width = 9, height = 4, units = "in")
# ggsave("dp_coverage_facet_mixed_labels.pdf", p_dp_coverage_facet, width = 14, height = 4, units = "in")
# ggsave("dp_tuned_alphas_mixed_labels.pdf", p_tuned_alphas, width = 9, height = 5, units = "in")
# ggsave("dp_size_mixed_labels.pdf", p_dp_size, width = 9, height = 4, units = "in")
# ggsave("dp_size_ratio_mixed_labels.pdf", p_dp_size_ratio, width = 9, height = 4, units = "in")
# ggsave("dp_prop_joker_mixed_labels.pdf", p_dp_prop_joker, width = 9, height = 4, units = "in")
# ggsave("dp_empty_prop_mixed_labels.pdf", p_dp_empty_prop, width = 6.5, height = 4, units = "in")
# ggsave("dp_combined_mixed_labels.pdf", p_dp_combined, width = 14, height = 8, units = "in")
# ggsave("dp_prop_unseen_simple.pdf", p_dp_prop_unseen_simple, width = 6.5, height = 4, units = "in")
# ggsave("dp_four_panel_XGT_mixed_labels_random_tuning.pdf", p_dp_four_panel, width = 14, height = 8, units = "in")
ggsave(sprintf("%s/dp_four_panel_90_joker_size.pdf", fig.dir), p_dp_four_panel, width = 16, height = 4.5, units = "in")