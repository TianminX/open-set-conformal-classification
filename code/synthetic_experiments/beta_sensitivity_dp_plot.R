library(data.table)
library(tidyverse)

# ============================================================
# DP-focused sensitivity of psi_seen to beta, fine theta grid.
# Reads the CSV produced by:
#   python beta_sensitivity_experiment.py 100 2026 dp
#
# Figure: E[min(psi_seen, 1)] vs beta, one facet per n, colour
# by theta (viridis, ordered). Dots mark per-curve minimizers,
# dashed vertical line marks the paper default beta = 1.6.
# ============================================================

csv_file <- "results/beta_sensitivity/beta_sensitivity_dp_reps100_seed2026.csv"
df <- fread(csv_file)

df <- df %>%
  mutate(
    theta = factor(param, levels = sort(unique(param))),
    n = factor(n, levels = sort(unique(n)),
               labels = paste0("n = ", sort(unique(n))))
  )

agg <- df %>%
  group_by(theta, n, beta) %>%
  summarise(
    psi_mean = mean(psi_seen),
    psi_trunc_mean = mean(pmin(psi_seen, 1)),
    psi_rgt_trunc_mean = mean(psi_rgt_trunc),
    .groups = "drop"
  )

argmin_trunc <- agg %>%
  group_by(theta, n) %>%
  filter(min(psi_trunc_mean) < 1) %>%   # only mark informative curves
  slice_min(psi_trunc_mean, n = 1, with_ties = FALSE) %>%
  ungroup()

theme_paper <- theme_bw() +
  theme(
    text = element_text(size = 14),
    axis.title = element_text(size = 18),
    axis.text = element_text(size = 12),
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 12),
    strip.text = element_text(size = 13),
    panel.grid.major = element_line(linewidth = 0.5),
    panel.grid.minor = element_blank()
  )

p1 <- ggplot(agg,
             aes(x = beta, y = psi_trunc_mean, colour = theta)) +
  geom_vline(xintercept = 1.6, colour = "grey40", linetype = "dashed") +
  geom_hline(yintercept = 1, colour = "grey70", linetype = "dotted") +
  geom_line(linewidth = 0.9) +
  geom_point(data = argmin_trunc, size = 2.4) +
  facet_wrap(~n, nrow = 1) +
  scale_colour_viridis_d(name = expression(theta), option = "plasma", end = 0.9) +
  labs(x = expression(beta),
       y = expression(E * "[min(" * psi[seen]^{GT} * ", 1)]")) +
  theme_paper

ggsave("beta_sensitivity_dp_truncated.pdf", p1, width = 10, height = 3)

# Raw CV criterion, same layout
argmin_raw <- agg %>%
  group_by(theta, n) %>%
  slice_min(psi_mean, n = 1, with_ties = FALSE) %>%
  ungroup()

p2 <- ggplot(agg,
             aes(x = beta, y = psi_mean, colour = theta)) +
  geom_vline(xintercept = 1.6, colour = "grey40", linetype = "dashed") +
  geom_hline(yintercept = 1, colour = "grey70", linetype = "dotted") +
  geom_line(linewidth = 0.9) +
  geom_point(data = argmin_raw, size = 2.2) +
  facet_wrap(~n, nrow = 1) +
  scale_y_log10() +
  scale_colour_viridis_d(name = expression(theta), option = "plasma", end = 0.9) +
  labs(x = expression(beta),
       y = expression(E * "[" * psi[seen] * "]  (log scale)")) +
  theme_paper

ggsave("beta_sensitivity_dp_raw.pdf", p2, width = 13, height = 3.8)

# ============================================================
# RGT variant: same layout, truncated scale (randomized p-values,
# averaged over 20 randomization draws per label sample)
# ============================================================
argmin_rgt <- agg %>%
  group_by(theta, n) %>%
  filter(min(psi_rgt_trunc_mean) < 1) %>%
  slice_min(psi_rgt_trunc_mean, n = 1, with_ties = FALSE) %>%
  ungroup()

p3 <- ggplot(agg,
             aes(x = beta, y = psi_rgt_trunc_mean, colour = theta)) +
  geom_vline(xintercept = 1.6, colour = "grey40", linetype = "dashed") +
  geom_hline(yintercept = 1, colour = "grey70", linetype = "dotted") +
  geom_line(linewidth = 0.9) +
  geom_point(data = argmin_rgt, size = 2.4) +
  facet_wrap(~n, nrow = 1) +
  scale_colour_viridis_d(name = expression(theta), option = "plasma", end = 0.9) +
  labs(x = expression(beta),
       y = expression(E * "[min(" * psi[seen]^{RGT} * ", 1)]")) +
  theme_paper

ggsave("beta_sensitivity_dp_rgt_truncated.pdf", p3, width = 10, height = 3)

# ============================================================
# Console summary: minimizer and where beta = 1.6 stands
# ============================================================
summary_tbl <- agg %>%
  group_by(theta, n) %>%
  summarise(
    beta_star_gt = beta[which.min(psi_trunc_mean)],
    beta_star_rgt = beta[which.min(psi_rgt_trunc_mean)],
    psi_min = min(psi_trunc_mean),
    psi_16 = psi_trunc_mean[beta == 1.6],
    informative_gt = min(psi_trunc_mean) < 1,
    informative_rgt = min(psi_rgt_trunc_mean) < 1,
    .groups = "drop"
  ) %>%
  mutate(rel_loss_16 = psi_16 / psi_min - 1)

print(summary_tbl, n = Inf)
