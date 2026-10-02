# Synthetic Bayesian paid-email cohort lab: results

All sources, leads, observations, monetary amounts and outcomes are fictional. This report is generated from the CSV artifacts in this directory, not from hand-entered example numbers.

## What this run establishes

- Mode: `demo`; forecast target shown in headline figures: day 180
- Origins present: 3, 7, 14, 30, 60 days
- Worlds in these CSVs: 1; sources: low, middle, high
- Predictive draws per forecast: 500
- Measured runner wall time before rendering: 6.1 seconds

Probabilities are finite-draw Monte Carlo estimates. Observed 0% or 100% is not certainty. Under independent predictive sampling, worst-case probability Monte Carlo standard error is approximately 500 draws: 0.022 probability units (2.2 percentage points). Quantile estimates also have Monte Carlo noise; these errors do not include model misspecification.

The point forecast is a posterior-predictive mean. The 80% and 95% intervals describe modeled realized economic contribution, not a confidence interval for a population-average effect or a parameter-only expected contribution. Early economic history can remain uncertain because responses, accounting postings and later revisions arrive on different clocks.

## Day-180 forecasts (dollars per acquired lead)

Truth is available only to the evaluator after forecasting. It is printed here to make the synthetic illustration inspectable.

| Source | Origin | Mean | Median | 80% predictive interval | 95% predictive interval | Truth | P(positive at 180) | P(payback by 180) | P(never under policy) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| high | 3 | $9.841 | $0.076 | −$0.139 to $1.489 | −$0.166 to $4.500 | $0.696 | 59% | 61% | 38% |
| high | 7 | $0.714 | $0.099 | −$0.092 to $1.041 | −$0.114 to $3.565 | $0.696 | 64% | 67% | 32% |
| high | 14 | $0.429 | $0.134 | −$0.007 to $0.755 | −$0.029 to $1.602 | $0.696 | 88% | 99.8% | 0.2% |
| high | 30 | $1.478 | $1.469 | $1.156 to $1.852 | $0.767 to $2.130 | $0.696 | 100% | 100% | 0% |
| high | 60 | $0.674 | $0.673 | $0.637 to $0.708 | $0.618 to $0.738 | $0.696 | 100% | 100% | 0% |
| low | 3 | $0.804 | −$0.119 | −$0.212 to $0.465 | −$0.227 to $1.044 | −$0.132 | 31% | 31% | 65% |
| low | 7 | −$0.089 | −$0.166 | −$0.215 to $0.088 | −$0.225 to $0.453 | −$0.132 | 17% | 17% | 80% |
| low | 14 | $0.099 | −$0.094 | −$0.186 to $0.374 | −$0.201 to $1.330 | −$0.132 | 32% | 32% | 62% |
| low | 30 | −$0.160 | −$0.164 | −$0.176 to −$0.143 | −$0.180 to −$0.112 | −$0.132 | 0% | 0% | 100% |
| low | 60 | −$0.141 | −$0.141 | −$0.147 to −$0.134 | −$0.149 to −$0.130 | −$0.132 | 0% | 0% | 100% |
| middle | 3 | $0.476 | −$0.087 | −$0.201 to $0.523 | −$0.214 to $1.847 | $0.086 | 36% | 37% | 59% |
| middle | 7 | $0.362 | $0.001 | −$0.173 to $0.758 | −$0.194 to $2.588 | $0.086 | 50% | 50% | 46% |
| middle | 14 | −$0.046 | −$0.134 | −$0.172 to $0.134 | −$0.180 to $0.645 | $0.086 | 18% | 18% | 81% |
| middle | 30 | $0.177 | $0.183 | $0.035 to $0.303 | $0.009 to $0.393 | $0.086 | 99.4% | 99.8% | 0.2% |
| middle | 60 | $0.073 | $0.073 | $0.057 to $0.088 | $0.049 to $0.096 | $0.086 | 100% | 100% | 0% |

**Tail-sensitive mean warning:** 1 forecast(s) have a predictive mean outside their central 95% predictive interval. This is possible for a skewed/heavy-tailed draw distribution. These means are retained in the chart scale and numeric table without trimming; a few extreme draws can dominate the average. Finite bounded model moments do not imply that a 500-draw mean is stable. The table also retains medians; the fan uses the median instead. Use medians/quantiles and inspect prior/variance sensitivity before interpreting a sparse-data mean as a stable expected-value estimate.

At the latest observed origin, interval widths are model-conditional. They may be smaller, larger or shifted relative to earlier fits; this renderer does not impose monotonically shrinking uncertainty.

The probabilities by day 180, later through the policy endpoint, and never under policy form mutually exclusive first-payback states. P(positive margin at the horizon) is a different event: a first crossing can be reversed by later costs or revisions.

Latest-origin first-payback state probabilities:

| Source | By day 180 | Later through endpoint | Never under policy |
| --- | --- | --- | --- |
| low | 0% | 0% | 100% |
| middle | 100% | 0% | 0% |
| high | 100% | 0% | 0% |

## Point accuracy comparison at day 180

Mean absolute point error per lead, equal-weighting world means. Bayesian uses the predictive mean; the two baselines are point forecasts. This does not establish superiority, and point baselines do not have predictive-interval or CRPS scores.

| Model | Origin | Mean absolute error / lead |
| --- | --- | --- |
| bayesian_hierarchical | 3 | $3.4899 |
| bayesian_hierarchical | 7 | $0.1124 |
| bayesian_hierarchical | 14 | $0.2103 |
| bayesian_hierarchical | 30 | $0.3003 |
| bayesian_hierarchical | 60 | $0.0147 |
| fixed_aggregate | 3 | $0.4007 |
| fixed_aggregate | 7 | $0.2510 |
| fixed_aggregate | 14 | $0.2374 |
| fixed_aggregate | 30 | $0.1790 |
| fixed_aggregate | 60 | $0.1118 |
| observable_survival | 3 | $0.2711 |
| observable_survival | 7 | $0.1823 |
| observable_survival | 14 | $0.1511 |
| observable_survival | 30 | $0.1941 |
| observable_survival | 60 | $0.0706 |

## Reading and limitations

- The terminal policy is finite: sends stop at day 365; delayed responses/accounting can settle through day 425. “Never” means no first-payback crossing within that policy, not an infinite-lifetime assertion
- An unconditional payback CDF ends at one minus the never-under-policy mass. Do not normalize away unsuccessful draws or report a payback-age percentile that the CDF never reaches
- Posterior-predictive bands account for modeled uncertainty only. Unmodeled calendar shocks, tail changes, attribution revisions, selection and value-dependent reporting can break them
- Sources are fictional labels, not randomized treatment arms. These forecasts provide no causal A/B evidence and are not proof that acquisition or send-stopping decisions improve outcomes
- The synthetic suite has few independent worlds. Repeated origins, horizons and source cohorts within a world are correlated; large row counts do not make a large independent sample
- Truth enters scoring and these evaluation plots only after forecasting. The as-of reported ledger is kept distinct from complete economic truth

## Figures

- [uncertainty by origin](figures/uncertainty_by_origin.png) · [editable SVG](figures/uncertainty_by_origin.svg)
- [posterior predictive fan](figures/posterior_predictive_fan.png) · [editable SVG](figures/posterior_predictive_fan.svg)
- [unconditional payback cdf](figures/unconditional_payback_cdf.png) · [editable SVG](figures/unconditional_payback_cdf.svg)
- [payback state probabilities](figures/payback_state_probabilities.png) · [editable SVG](figures/payback_state_probabilities.svg)

All plotted figures have an explicit synthetic/fictional label. CSVs remain the numerical source of record; this report can be rebuilt without refitting.
