library(data.table)
library(tidyverse)

# ============================================================
# Coverage boxplots as a function of the calibration sample size,
# in the spirit of Figure 1 (right) of Sesia and Candes (2020),
# "A comparison of some conformal quantile regression methods".
#
# Two variants, selected by the environment variable VARY_CALPROP_VARIANT:
#   cgtc (default): original-CGTC sweep produced by synthetic_experiment_dp.py
#                   (results_hpc/dp_tuned_mixed_labels/vary_calprop),
#                   output figures/dp_coverage_boxplot_varyCalib.pdf
#   mm:             CGTC+ sweep produced by synthetic_experiment_dp_mm_plugin.py
#                   via submit_synthetic_experiment_dp_mm_plugin_vary_calprop.sh
#                   (results_hpc/dp_tuned_mixed_labels_mm_plugin_vary_calprop),
#                   output figures/dp_coverage_boxplot_varyCalib_mm.pdf
# Both sweeps: theta = 1000, n_ref = 2000, LOF one-class classifier,
# calib_num in {100, 200, 400, 1000}, 10 batches x 5 replicates.  One box =
# the 50 replicate values of the marginal coverage (joker counted as
# covering unseen labels), XGT p-values.
# Run from code/synthetic_experiments/; output written to figures/.
# The script also prints the across-replicate standard deviation of the
# coverage for each method and calibration size, next to the reference
# value sqrt(alpha (1 - alpha) / n_cal) of a one-dimensional quantile and
# the test-set evaluation noise sqrt(alpha (1 - alpha) / n_test).
# ============================================================

variant <- Sys.getenv("VARY_CALPROP_VARIANT", "cgtc")
if (variant == "mm") {
  idir <- "results_hpc/dp_tuned_mixed_labels_mm_plugin_vary_calprop"
  method_prefix <- "CGTC+"
  flag_col <- "splitting_method_flag"
  out_suffix <- "_mm"
} else {
  idir <- "results_hpc/dp_tuned_mixed_labels/vary_calprop"
  method_prefix <- "CGTC"
  flag_col <- "tuning_method_flag"
  out_suffix <- ""
}
fig.dir <- "figures"
dir.create(fig.dir, showWarnings = FALSE)

df_all <- list.files(idir, pattern = "\\.csv$", full.names = TRUE) %>%
  map_dfr(~ {
    dt <- fread(.x)
    dt[, which(!duplicated(names(dt))), with = FALSE]
  })

m_random <- paste0(method_prefix, " (random)")
m_selective <- paste0(method_prefix, " (selective)")
recode_map <- c("Method (random splitting)" = m_random,
                "Method (benchmark)" = "standard (random)",
                "Method (Bernoulli)" = m_selective,
                "Method (Bernoulli benchmark)" = "standard (selective)")
df_all <- df_all %>% mutate(method = recode(method, !!!recode_map))

methods_to_keep <- c(m_random, m_selective, "standard (random)", "standard (selective)")

custom_colors <- setNames(c("#E41A1C", "#377EB8", "#4DAF4A", "#FF7F00"), methods_to_keep)

theme_main <- theme_bw() +
  theme(
    text = element_text(size = 15),
    axis.title = element_text(size = 18),
    axis.text = element_text(size = 15),
    legend.title = element_text(size = 18),
    legend.text = element_text(size = 18),
    panel.grid.major = element_line(linewidth = 0.5),
    panel.grid.minor = element_blank(),
    legend.position = "right",
    legend.direction = "vertical"
  )

df_box <- df_all %>%
  filter(method %in% methods_to_keep,
         .data[[flag_col]] == 0,
         abs(alpha_total - 0.1) < 1e-10,
         theta == 1000,
         n_ref == 2000,
         pvalue_method == "XGT") %>%
  mutate(method = factor(method, levels = methods_to_keep),
         calib_num = factor(calib_num, levels = sort(unique(calib_num))),
         coverage = `Coverage (?)`)

cat("--- replicates per (calib_num, method) ---\n")
df_box %>% count(calib_num, method) %>% as.data.frame() %>% print()

alpha <- 0.1
n_test <- unique(df_box$n_test)[1]
sd_table <- df_box %>%
  group_by(calib_num, method) %>%
  summarise(mean_cov = mean(coverage), sd_cov = sd(coverage), n = n(), .groups = "drop") %>%
  mutate(n_cal = as.numeric(as.character(calib_num)),
         sd_quantile = sqrt(alpha * (1 - alpha) / (n_cal + 2)),
         sd_test = sqrt(alpha * (1 - alpha) / n_test),
         sd_quantile_plus_test = sqrt(sd_quantile^2 + sd_test^2))
cat("--- coverage across replicates: mean, sd, and reference values ---\n")
sd_table %>% as.data.frame() %>% print(digits = 3)

p <- ggplot(df_box, aes(x = calib_num, y = coverage, fill = method)) +
  geom_hline(yintercept = 1 - alpha, linetype = "dashed", color = "black") +
  geom_boxplot(position = position_dodge(width = 0.8), width = 0.7,
               outlier.size = 1, alpha = 0.85) +
  scale_fill_manual(name = "Method", values = custom_colors) +
  labs(x = "Calibration sample size", y = "Coverage") +
  theme_main

print(p)
ofile <- paste0("dp_coverage_boxplot_varyCalib", out_suffix, ".pdf")
ggsave(file.path(fig.dir, ofile), p, width = 9, height = 3.8, units = "in")
cat(sprintf(">>> wrote %s\n", file.path(fig.dir, ofile)))
write.csv(sd_table, file.path(fig.dir, paste0("dp_coverage_boxplot_varyCalib", out_suffix, "_sd.csv")),
          row.names = FALSE)
