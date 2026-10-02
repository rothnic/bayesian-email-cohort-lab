# Controlled component validation

Known-shape Gamma-Poisson rate component with fixed value/cost/exposure only; not full modular pipeline validation.

All 2,000 prespecified worlds retained; seed 2026100201; origins [3, 7, 14, 30, 60]; horizon 180.

Exact predictive intervals include future Poisson variation and posterior amplitude uncertainty. Their monetary transform uses a fixed positive value and known costs.

| Origin | 80% coverage | 95% coverage | Amplitude 95% coverage | Mean 95% monetary width | Brier |
|---:|---:|---:|---:|---:|---:|
| 3 | 0.824 | 0.953 | 0.947 | 12.26 | 0.106 |
| 7 | 0.808 | 0.942 | 0.944 | 10.44 | 0.088 |
| 14 | 0.827 | 0.948 | 0.942 | 8.49 | 0.073 |
| 30 | 0.830 | 0.955 | 0.948 | 6.05 | 0.050 |
| 60 | 0.841 | 0.953 | 0.954 | 3.83 | 0.031 |

Main-package posterior-sampler cross-check: mean error 0.52 standard errors; variance relative error 1.272%.

Posterior mean/variance agree with the identical analytic Gamma update on a Snapshot fixture. The validation is component-level; fixed marks, no reporting delays, no revisions, and known exposure remove the main pipeline's additional modeling risks.

## Unchanged full-world stress comparison

- calendar_attribution_shock: 80% coverage 0.467; 95% coverage 0.667, at the same horizon, across 45 correlated cases in 3 worlds
- collapse_tail: 80% coverage 0.467; 95% coverage 0.689, at the same horizon, across 45 correlated cases in 3 worlds
- late_reactivation: 80% coverage 0.022; 95% coverage 0.089, at the same horizon, across 45 correlated cases in 3 worlds
- stationary: 80% coverage 0.733; 95% coverage 0.933, at the same horizon, across 45 correlated cases in 3 worlds

The contrast establishes that conjugate updating can work under matching assumptions while the full modular model remains undercovered on the richer synthetic worlds. It is not evidence that full-pipeline profitability probabilities are trustworthy.

PNG and SVG charts: controlled_vs_full_world_coverage and controlled_updating_and_ranks. Exact forecast and world files retain every prespecified case.
