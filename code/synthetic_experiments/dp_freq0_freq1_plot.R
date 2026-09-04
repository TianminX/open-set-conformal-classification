library(data.table)
library(tidyverse)

# ============================================================
# Ratio of unseen test labels (reference frequency 0) to singleton
# labels (reference frequency 1) as a function of the Dirichlet
# concentration parameter, for the manuscript appendix
# (dp_freq0_freq1_ratio.pdf).
# Data: fixed-beta (1.6) theta-sweep runs of synthetic_experiment_dp.py,
# collected under results_hpc/dp_tuned_mixed_labels/beta1.6/.
# Run from code/synthetic_experiments/; output is written to figures/.
# ============================================================

idir <- "results_hpc/dp_tuned_mixed_labels/beta1.6"
fig.dir <- "figures"
dir.create(fig.dir, showWarnings = FALSE)

# 1. Load data
ifile.list <- list.files(idir, pattern = "\\.csv$", full.names = TRUE)
df_dp_mixed_labels <- map_dfr(ifile.list, ~ fread(.x))

# 2. Recode method names
df_dp_mixed_labels <- df_dp_mixed_labels %>%
  mutate(method = recode(method,
                         "Method (random splitting)" = "CGTC (random)",
                         "Method (benchmark)" = "standard (random)",
                         "Method (Bernoulli)" = "CGTC (selective)",
                         "Method (Bernoulli benchmark)" = "standard (selective)"))

# Frequency counts in these runs use the "fixed" conditional binning
cond_method <- "fixed"

# 3. Compute freq0/freq1 ratio per replicate, then average across replicates
df_freq_ratio_plot <- df_dp_mixed_labels %>%
  filter(method == "CGTC (random)",
         pvalue_method == "XGT",
         abs(alpha_total - 0.1) < 1e-10,
         tuning_method_flag == 0,
         calib_num == n_ref * 0.1,
         n_ref == 2000,
         theta != 25) %>%
  mutate(n_freq0 = num_unseen_test,
         n_freq1 = .data[[paste0("Count (very_rare) ", cond_method)]] - num_unseen_test,
         ratio   = n_freq0 / n_freq1) %>%
  filter(is.finite(ratio)) %>%
  group_by(theta) %>%
  summarise(mean_ratio = mean(ratio, na.rm = TRUE),
            se_ratio   = sd(ratio,  na.rm = TRUE) / sqrt(n()),
            lci_ratio  = mean_ratio - 1.96 * se_ratio,
            uci_ratio  = mean_ratio + 1.96 * se_ratio,
            .groups = "drop")

p_freq_ratio <- ggplot(df_freq_ratio_plot,
                       aes(x = theta, y = mean_ratio)) +
  geom_line(size = 1, color = "#2166AC") +
  geom_point(size = 2.5, color = "#2166AC") +
  geom_errorbar(aes(ymin = lci_ratio, ymax = uci_ratio),
                width = 20, size = 0.7, color = "#2166AC") +
  labs(x = "Dirichlet concentration parameter",
       y = expression("Ratio  " * n[freq~0] / n[freq~1])) +
  theme_bw() +
  theme(
    text = element_text(size = 14),
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 13),
    panel.grid.major = element_line(size = 0.5),
    panel.grid.minor = element_blank()
  )

print(p_freq_ratio)
ofile <- "dp_freq0_freq1_ratio.pdf"
ggsave(file.path(fig.dir, ofile), p_freq_ratio, width = 6, height = 3.5, units = "in")
cat(sprintf(">>> wrote %s\n", file.path(fig.dir, ofile)))
