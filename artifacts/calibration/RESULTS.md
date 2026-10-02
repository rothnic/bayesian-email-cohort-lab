# Synthetic Bayesian paid-email cohort lab: results

All sources, leads, observations, monetary amounts and outcomes are fictional. This report is generated from the CSV artifacts in this directory, not from hand-entered example numbers.

## What this run establishes

- Mode: `calibration`; forecast target shown in headline figures: day 180
- Origins present: 3, 7, 14, 30, 60 days
- Worlds in these CSVs: 12; sources: low, middle, high
- Predictive draws per forecast: 250
- Measured runner wall time before rendering: 27.9 seconds

Probabilities are finite-draw Monte Carlo estimates. Observed 0% or 100% is not certainty. Under independent predictive sampling, worst-case probability Monte Carlo standard error is approximately 250 draws: 0.032 probability units (3.2 percentage points). Quantile estimates also have Monte Carlo noise; these errors do not include model misspecification.

The point forecast is a posterior-predictive mean. The 80% and 95% intervals describe modeled realized economic contribution, not a confidence interval for a population-average effect or a parameter-only expected contribution. Early economic history can remain uncertain because responses, accounting postings and later revisions arrive on different clocks.

## Coverage and distribution scores

Coverage and CRPS below use day-180 prospective forecasts. Coverage uncertainty resamples whole worlds, not repeated sources/origins as independent observations. The scenarios deliberately differ from model assumptions. This is misspecification stress validation, not prior-generated Bayesian simulation-based calibration.

| Origin | 80% coverage [bootstrap 95%] | 95% coverage [bootstrap 95%] | CRPS / lead [bootstrap 95%] | Worlds | Forecasts |
| --- | --- | --- | --- | --- | --- |
| 3 | 44% [28%, 61%] | 64% [44%, 81%] | $0.1626 [$0.0984, $0.2301] | 12 | 36 |
| 7 | 39% [19%, 58%] | 58% [39%, 78%] | $0.1621 [$0.0971, $0.2343] | 12 | 36 |
| 14 | 39% [19%, 61%] | 53% [31%, 72%] | $0.1164 [$0.0654, $0.1706] | 12 | 36 |
| 30 | 47% [28%, 67%] | 61% [39%, 83%] | $0.0807 [$0.0353, $0.1299] | 12 | 36 |
| 60 | 42% [22%, 61%] | 61% [39%, 81%] | $0.0648 [$0.0195, $0.1163] | 12 | 36 |

## Scenario failures at origin 60, horizon 180

These numbers average within each world first, then across worlds. Positive signed error is optimistic. Averages can hide cancellation, so individual forecast intervals and the largest misses are retained.

| Scenario | CRPS / lead | 95% interval miss fraction | Signed mean error / lead |
| --- | --- | --- | --- |
| calendar_attribution_shock | $0.0154 | 22% | $0.0021 |
| collapse_tail | $0.0260 | 44% | −$0.0240 |
| late_reactivation | $0.2064 | 89% | −$0.2194 |
| stationary | $0.0114 | 0% | $0.0034 |

Largest individual day-target predictive-mean errors (no adverse seeds discarded):

| Scenario | Seed | Source | Origin | Mean / lead | Truth / lead | Absolute error / lead | 95% interval |
| --- | --- | --- | --- | --- | --- | --- | --- |
| collapse_tail | 10202 | high | 3 | $1.481 | $0.475 | $1.006 | missed |
| collapse_tail | 10202 | high | 7 | $1.342 | $0.475 | $0.867 | missed |
| late_reactivation | 30202 | high | 3 | $0.226 | $1.077 | $0.850 | missed |
| calendar_attribution_shock | 20303 | high | 7 | $1.487 | $0.649 | $0.838 | missed |
| late_reactivation | 30202 | high | 7 | $0.269 | $1.077 | $0.808 | missed |
| calendar_attribution_shock | 20303 | high | 14 | $1.440 | $0.649 | $0.791 | missed |
| calendar_attribution_shock | 20303 | high | 3 | $1.418 | $0.649 | $0.769 | missed |
| late_reactivation | 30303 | high | 7 | $0.173 | $0.878 | $0.705 | missed |

## Descriptive reliability

Counts below are correlated forecast rows, not independent worlds; empty bins remain empty. No IID binomial uncertainty is asserted.

| Probability bin | Mean predicted profitability | Realized profitability | Forecast count |
| --- | --- | --- | --- |
| 0–.2 | 2% | 9% | 81 |
| .2–.4 | 27% | 80% | 5 |
| .4–.6 | 58% | 100% | 1 |
| .6–.8 | 71% | 100% | 8 |
| .8–1 | 99.4% | 100% | 85 |

## Point accuracy comparison at day 180

Mean absolute point error per lead, equal-weighting world means. Bayesian uses the predictive mean; the two baselines are point forecasts. This does not establish superiority, and point baselines do not have predictive-interval or CRPS scores.

| Model | Origin | Mean absolute error / lead |
| --- | --- | --- |
| bayesian_hierarchical | 3 | $0.2023 |
| bayesian_hierarchical | 7 | $0.1919 |
| bayesian_hierarchical | 14 | $0.1392 |
| bayesian_hierarchical | 30 | $0.0966 |
| bayesian_hierarchical | 60 | $0.0714 |
| fixed_aggregate | 3 | $0.2529 |
| fixed_aggregate | 7 | $0.2369 |
| fixed_aggregate | 14 | $0.2140 |
| fixed_aggregate | 30 | $0.1650 |
| fixed_aggregate | 60 | $0.1157 |
| observable_survival | 3 | $0.1210 |
| observable_survival | 7 | $0.1303 |
| observable_survival | 14 | $0.1205 |
| observable_survival | 30 | $0.1089 |
| observable_survival | 60 | $0.0503 |

## Reading and limitations

- The terminal policy is finite: sends stop at day 365; delayed responses/accounting can settle through day 425. “Never” means no first-payback crossing within that policy, not an infinite-lifetime assertion
- An unconditional payback CDF ends at one minus the never-under-policy mass. Do not normalize away unsuccessful draws or report a payback-age percentile that the CDF never reaches
- Posterior-predictive bands account for modeled uncertainty only. Unmodeled calendar shocks, tail changes, attribution revisions, selection and value-dependent reporting can break them
- Sources are fictional labels, not randomized treatment arms. These forecasts provide no causal A/B evidence and are not proof that acquisition or send-stopping decisions improve outcomes
- The synthetic suite has few independent worlds. Repeated origins, horizons and source cohorts within a world are correlated; large row counts do not make a large independent sample
- Truth enters scoring and these evaluation plots only after forecasting. The as-of reported ledger is kept distinct from complete economic truth

## Figures

- [empirical interval coverage](figures/empirical_interval_coverage.png) · [editable SVG](figures/empirical_interval_coverage.svg)
- [profitability reliability](figures/profitability_reliability.png) · [editable SVG](figures/profitability_reliability.svg)
- [scenario stress scores](figures/scenario_stress_scores.png) · [editable SVG](figures/scenario_stress_scores.svg)
- [scenario margin intervals](figures/scenario_margin_intervals.png) · [editable SVG](figures/scenario_margin_intervals.svg)

All plotted figures have an explicit synthetic/fictional label. CSVs remain the numerical source of record; this report can be rebuilt without refitting.
