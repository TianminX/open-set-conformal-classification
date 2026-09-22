library(data.table)
library(tidyverse)

# ============================================================
# Probability mass of the four test-label frequency bins used in the
# stratified-coverage figure (dp_mm_cond_cov_four_levels.pdf), as a
# function of the Dirichlet concentration parameter theta
# (dp_bin_mass.pdf, manuscript appendix and response letter).
# Bins follow the "fixed" binning of cgtc/conformal_methods.py, based on
# the frequency of the test label among the n_ref reference observations:
#   very rare   = frequency 0 or 1 (unseen or singleton)
#   rare        = frequency 2
#   common      = frequency 3 to 5
#   very common = frequency 6 or more
# The four masses sum to one at every theta.
# Data: results_hpc/dp_tuned_mixed_labels_mm_plugin/ (the CGTC+ runs
# behind the main-paper figures), with the same filters as the theta
# sweep of dp_mm_plugin_paper_plots.R. Bin counts depend only on the
# sampled reference and test labels, so they are identical across
# methods and p-values within a replicate; one combination is kept to
# avoid triplication (50 replicates per theta).
# Run from code/synthetic_experiments/; output is written to figures/.
# ============================================================

idir <- "results_hpc/dp_tuned_mixed_labels_mm_plugin"
fig.dir <- "figures"
dir.create(fig.dir, showWarnings = FALSE)
cond_method <- "fixed"

# 1. Load data
df_mm <- list.files(idir, pattern = "\\.csv$", full.names = TRUE) %>%
  map_dfr(~ {
    dt <- fread(.x)
    dt[, which(!duplicated(names(dt))), with = FALSE]
  })

# 2. One row per replicate (filters as in dp_mm_plugin_paper_plots.R)
df_rep <- df_mm %>%
  filter(method == "Method (random splitting)",
         pvalue_method == "XGT",
         splitting_method_flag == 0,
         abs(alpha_total - 0.1) < 1e-10,
         calib_num == n_ref * 0.1,
         n_ref == 2000,
         theta != 25)

bin_levels <- c("very_rare", "rare", "common", "very_common")
bin_labels <- c("Very Rare", "Rare", "Common", "Very Common")

# 3. Bin masses per replicate, then mean and standard error across replicates
df_mass <- df_rep %>%
  transmute(theta, n_test,
            very_rare   = .data[[paste0("Count (very_rare) ", cond_method)]],
            rare        = .data[[paste0("Count (rare) ", cond_method)]],
            common      = .data[[paste0("Count (common) ", cond_method)]],
            very_common = .data[[paste0("Count (very_common) ", cond_method)]]) %>%
  pivot_longer(cols = all_of(bin_levels), names_to = "bin", values_to = "count") %>%
  mutate(mass = count / n_test) %>%
  group_by(theta, bin) %>%
  summarise(n_rep = n(),
            mean_mass = mean(mass),
            se_mass = sd(mass) / sqrt(n()),
            lci_mass = mean_mass - 1.96 * se_mass,
            uci_mass = mean_mass + 1.96 * se_mass,
            .groups = "drop") %>%
  mutate(bin = factor(bin, levels = bin_levels, labels = bin_labels))

# Sanity check: the four masses sum to one at every theta
mass_sums <- df_mass %>% group_by(theta) %>% summarise(s = sum(mean_mass), .groups = "drop")
stopifnot(all(abs(mass_sums$s - 1) < 1e-12))

cat("--- mean bin mass by theta ---\n")
df_mass %>%
  select(theta, bin, n_rep, mean_mass) %>%
  pivot_wider(names_from = bin, values_from = mean_mass) %>%
  print(n = 50)

# Split of the very rare bin into unseen (frequency 0) and singleton
# (frequency 1) test points, for reference in the text (not plotted)
cat("--- very rare bin: unseen and singleton shares of the test set ---\n")
df_rep %>%
  transmute(theta,
            freq0 = num_unseen_test / n_test,
            freq1 = (.data[[paste0("Count (very_rare) ", cond_method)]] - num_unseen_test) / n_test) %>%
  group_by(theta) %>%
  summarise(mean_freq0 = mean(freq0), mean_freq1 = mean(freq1),
            missing_mass = first(theta) / (first(theta) + 2000), .groups = "drop") %>%
  print(n = 50)

# 4. Plot: one line per bin, masses sum to one at every theta
#    (Okabe-Ito colorblind-safe palette, ordered warm to cool from the
#    rarest to the most common bin; shapes give a second encoding)
bin_colors <- c("Very Rare" = "#D55E00", "Rare" = "#E69F00",
                "Common" = "#009E73", "Very Common" = "#0072B2")
bin_shapes <- c("Very Rare" = 16, "Rare" = 17, "Common" = 15, "Very Common" = 18)

p_mass <- ggplot(df_mass, aes(x = theta, y = mean_mass, color = bin, shape = bin)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2.5) +
  geom_errorbar(aes(ymin = lci_mass, ymax = uci_mass), width = 20, linewidth = 0.7) +
  scale_color_manual(name = "Frequency bin", values = bin_colors) +
  scale_shape_manual(name = "Frequency bin", values = bin_shapes) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.2)) +
  labs(x = "Dirichlet concentration parameter",
       y = "Proportion of test points") +
  theme_bw() +
  theme(
    text = element_text(size = 14),
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 13),
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 13),
    panel.grid.major = element_line(linewidth = 0.5),
    panel.grid.minor = element_blank()
  )

print(p_mass)
ofile <- "dp_bin_mass.pdf"
ggsave(file.path(fig.dir, ofile), p_mass, width = 6.5, height = 3.5, units = "in")
cat(sprintf(">>> wrote %s\n", file.path(fig.dir, ofile)))

# Optional PNG preview (set DP_BIN_MASS_PNG to a file path); not used in the paper
png_out <- Sys.getenv("DP_BIN_MASS_PNG")
if (nzchar(png_out)) {
  ggsave(png_out, p_mass, width = 6.5, height = 3.5, units = "in", dpi = 150)
  cat(sprintf(">>> wrote %s\n", png_out))
}
