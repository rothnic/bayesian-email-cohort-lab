# Synthetic Bayesian paid-email cohort lab: results

All sources, leads, observations, monetary amounts and outcomes are fictional. This report is generated from the CSV artifacts in this directory, not from hand-entered example numbers.

## What this run establishes

- Mode: `demo`; forecast target shown in headline figures: day 180
- Origins present: 3, 7, 14, 30, 60 days
- Worlds in these CSVs: 1; sources: low, middle, high
- Predictive draws per forecast: 500
- Measured runner wall time before rendering: 6.5 seconds

Probabilities are finite-draw Monte Carlo estimates. Observed 0% or 100% is not certainty. Under independent predictive sampling, worst-case probability Monte Carlo standard error is approximately 500 draws: 0.022 probability units (2.2 percentage points). Quantile estimates also have Monte Carlo noise; these errors do not include model misspecification.

The point forecast is a posterior-predictive mean. The 80% and 95% intervals describe modeled realized economic contribution, not a confidence interval for a population-average effect or a parameter-only expected contribution. Early economic history can remain uncertain because responses, accounting postings and later revisions arrive on different clocks.

## Day-180 forecasts (dollars per acquired lead)

Truth is available only to the evaluator after forecasting. It is printed here to make the synthetic illustration inspectable.

| Source | Origin | Mean | Median | 80% predictive interval | 95% predictive interval | Truth | P(positive at 180) | P(payback by 180) | P(never under policy) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| high | 3 | $1.463 | $1.456 | $1.319 to $1.616 | $1.224 to $1.711 | $1.713 | 100% | 100% | 0% |
| high | 7 | $1.371 | $1.364 | $1.249 to $1.500 | $1.191 to $1.554 | $1.713 | 100% | 100% | 0% |
| high | 14 | $1.465 | $1.466 | $1.357 to $1.574 | $1.312 to $1.629 | $1.713 | 100% | 100% | 0% |
| high | 30 | $1.440 | $1.438 | $1.372 to $1.508 | $1.338 to $1.539 | $1.713 | 100% | 100% | 0% |
| high | 60 | $1.570 | $1.570 | $1.523 to $1.619 | $1.495 to $1.647 | $1.713 | 100% | 100% | 0% |
| low | 3 | −$0.151 | −$0.151 | −$0.168 to −$0.131 | −$0.175 to −$0.122 | −$0.130 | 0% | 0% | 100% |
| low | 7 | −$0.154 | −$0.155 | −$0.168 to −$0.139 | −$0.176 to −$0.129 | −$0.130 | 0% | 0% | 100% |
| low | 14 | −$0.163 | −$0.163 | −$0.175 to −$0.151 | −$0.178 to −$0.145 | −$0.130 | 0% | 0% | 100% |
| low | 30 | −$0.155 | −$0.155 | −$0.163 to −$0.146 | −$0.166 to −$0.141 | −$0.130 | 0% | 0% | 100% |
| low | 60 | −$0.143 | −$0.143 | −$0.149 to −$0.137 | −$0.151 to −$0.134 | −$0.130 | 0% | 0% | 100% |
| middle | 3 | $0.135 | $0.132 | $0.087 to $0.184 | $0.067 to $0.213 | $0.210 | 100% | 100% | 0% |
| middle | 7 | $0.135 | $0.133 | $0.095 to $0.178 | $0.075 to $0.204 | $0.210 | 100% | 100% | 0% |
| middle | 14 | $0.145 | $0.143 | $0.108 to $0.181 | $0.093 to $0.202 | $0.210 | 100% | 100% | 0% |
| middle | 30 | $0.149 | $0.150 | $0.125 to $0.173 | $0.114 to $0.190 | $0.210 | 100% | 100% | 0% |
| middle | 60 | $0.175 | $0.175 | $0.157 to $0.193 | $0.149 to $0.202 | $0.210 | 100% | 100% | 0% |

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
| bayesian_hierarchical | 3 | $0.1155 |
| bayesian_hierarchical | 7 | $0.1472 |
| bayesian_hierarchical | 14 | $0.1155 |
| bayesian_hierarchical | 30 | $0.1197 |
| bayesian_hierarchical | 60 | $0.0637 |
| fixed_aggregate | 3 | $0.5622 |
| fixed_aggregate | 7 | $0.5326 |
| fixed_aggregate | 14 | $0.4729 |
| fixed_aggregate | 30 | $0.3719 |
| fixed_aggregate | 60 | $0.2106 |
| observable_survival | 3 | $0.2057 |
| observable_survival | 7 | $0.1946 |
| observable_survival | 14 | $0.1773 |
| observable_survival | 30 | $0.1583 |
| observable_survival | 60 | $0.0883 |

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
