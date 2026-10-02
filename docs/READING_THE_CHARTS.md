# Reading the Bayesian cohort lab charts

Every chart uses synthetic, fictional data. The low, middle and high sources are generator labels, not real businesses or randomized experimental arms. Dollar amounts are illustrative.

## What is forecast

The forecast target is a cohort's **economic contribution**, after acquisition and send costs and signed revenue/reversal entries, divided by its number of acquired leads where the axis says “per acquired lead.” It is not gross revenue, cash actually reported to date, or contribution per click. The denominator stays the acquired cohort size, including inactive and operationally exited leads.

The finite sending policy sends through cohort age 365. Responses can arrive later and accounting can settle later still; the policy evaluation endpoint is day 425. Horizon 180 is the main comparison target, while full economic paths continue to the endpoint.

Forecasts are replayed at the origins actually present in the CSVs, normally days 3, 7, 14, 30 and 60. Each origin is a new as-of view of the same cohort, not a separate cohort or extra acquisition spend. The `--quick` wiring run has only days 3 and 60; missing origins are not invented.

## Predictive intervals, not an interval for a mean effect

The mean point in the uncertainty chart is the average of posterior-predictive draws. The 80% interval uses their 10th and 90th percentiles, and the 95% interval uses their 2.5th and 97.5th percentiles. These intervals describe the modeled distribution of **realized cohort contribution**. They include modeled uncertainty about parameters/latent economics and event-level future variability. They are not parameter-only credible intervals for an expected contribution, nor confidence intervals for an A/B treatment effect.

An interval's stated probability is conditional on the observation model, response/value model, prior, tail assumptions and finite policy. It does not protect against every misspecification or future calendar change. New observations can move a forecast, expose a different tail or widen an interval. Narrower later intervals are neither imposed by the renderer nor guaranteed.

The predictive mean is tail-sensitive. In a highly skewed/heavy-tailed draw distribution it can fall outside a central 95% interval, because a few extreme draws dominate the average. The chart scale includes that mean rather than clipping it, and the generated report flags such cases. The fan displays a median instead. Inspect prior/variance sensitivity before treating a sparse-data mean as a stable expected-value estimate.

The light interval is 95%; the darker interval is 80%. The zero line is the break-even economic contribution threshold. All source panels within a chart share their monetary scale. The point and interval answer different questions: a slightly positive mean with substantial negative probability is not a confident profitable verdict.

## Economic history and the reported ledger

The fan chart intentionally keeps three objects separate:

- Blue median and bands: the posterior-predictive economic path, including uncertain past economics when reporting is delayed
- Gold observed line: the signed ledger visible at that forecast origin, shown only through that origin
- Dashed dark line: complete evaluator-only economic truth, available only after forecasting for synthetic scoring

The forecast origin is a dotted vertical line; the area to its left is the as-of past. An economic event can occur before a response/report is visible, and an accounting entry can be revised after its first posting. Therefore the gold reported path need not equal the true economic path at the origin, and the blue band need not collapse in the past. The gold line stops at the origin; extending its known postings across the future would misleadingly make it look like a forecast.

The truth overlays make this synthetic exercise inspectable. The inference call sees only its as-of snapshot; the runner accesses complete truth afterwards for scoring and plotting. Never use these truth fields as production features.

## First payback, horizon positivity and never-under-policy

First payback is the first cohort age where cumulative economic contribution is at least zero. Later costs or revisions can make contribution negative again. Consequently these are distinct:

- Probability of first payback by day 180
- Probability of positive contribution at day 180
- Probability of remaining nonnegative after the first crossing through a given horizon

The unconditional CDF plots P(first payback at or before a given age), **including** unsuccessful draws in the denominator. Its terminal height is one minus P(never under this policy). It must not be renormalized to reach 100% among the successes. For example, if it ends at 65%, the unconditional 80th-percentile first-payback age is unavailable under this policy, rather than an extrapolated finite age.

The stacked columns show mutually exclusive first-payback states: by day 180, later during days 181–425, and never through day 425. Each column sums to 100%. “Never” is a finite-policy label: it is not a claim about unrestricted lifetime, continuing sends indefinitely or a future policy change. Zero never mass is possible in a finite Monte Carlo run and does not prove that risk is impossible.

Probabilities and quantiles are estimated from finitely many predictive draws. An estimate of 0% or 100% is not certainty. With independent predictive draws, a probability's worst-case Monte Carlo standard error is 0.5/√B, approximately 0.022 (2.2 percentage points) for 500 draws or 0.032 (3.2 points) for 250 draws. This sampling noise is separate from model failure and limited independent-world validation. The generated report records the draw count actually used.

## Calibration and stress validation

The calibration mode evaluates prospective forecasts only. The headline coverage and reliability figures use day-180 outcomes, so all origins in this configured run precede the target.

### Empirical coverage

Coverage is the fraction of complete synthetic economic outcomes inside the corresponding 80% or 95% predictive interval. Horizontal references show nominal target coverage. Error bars are 95% percentile-bootstrap intervals from resampling **whole worlds**. Sources, origins and horizons within a world share calendar effects and observations, so treating all rows as independent would exaggerate precision.

The default suite has a modest number of independent worlds and only three fixed seeds per scenario. Its generators deliberately depart from model assumptions. This is an out-of-model stress check, not simulation-based calibration using parameters drawn from the Bayesian prior, and not proof of calibration in a real deployment. A one-world quick run has degenerate bootstrap bars and is only a wiring test.

### Descriptive probability reliability

The reliability plot compares mean predicted probability of positive contribution against the realized profitable fraction in five fixed probability bins. Counts beside points are forecast-row counts, **not independent-world counts**. Empty bins have no point; there is no fabricated outcome there. The diagonal is a perfect-reliability reference. No IID binomial confidence intervals are asserted.

Different origins and sources are pooled in this descriptive display and are correlated within worlds. A small count, a clustered world or a change in mixture can move a bin. Inspect coverage, stress scenarios and sample counts together rather than interpreting an isolated point as decisive evidence.

### Distribution errors and individual misses

CRPS evaluates the whole predictive distribution in economic-contribution units; a smaller value is better. The scenario-score figure uses CRPS per acquired lead, 95% interval miss fraction and signed predictive-mean error. Positive signed error means an optimistic forecast; negative error means pessimistic. Averages first summarize each world, then give worlds equal weight. These descriptive lines do not establish a statistically significant ranking.

The individual scenario-margin figure keeps each forecast's actual 80%/95% interval and evaluator truth at its source/cohort/world grain. It never averages interval bounds and presents that average as uncertainty about a scenario mean. Its common monetary scale preserves visible bad misses. The generated report also includes the largest individual point errors, including adverse seeds.

Point-baseline comparison, when present in the generated report, uses the same horizon and absolute point error per acquired lead. Point baselines have no probability distribution; the report does not fabricate CRPS, predictive coverage or Brier scores for them. Ranking by point accuracy alone cannot establish better uncertainty, better decisions or causal value.

## What these figures do not establish

- No causal A/B result: source labels are not randomized interventions, and forecasts are not estimated treatment effects
- No proof of profitable stopping or allocation decisions: a policy-specific forecast is an input to a decision, not evidence of the decision's causal benefit
- No real-world performance claim: all leads, amounts and outcomes are fictional
- No guaranteed tail accuracy: extrapolation beyond observed support remains assumption-driven; late reactivation, future shocks and attribution changes can invalidate predictions
- No large-sample validation from many CSV rows: independent world counts, rather than correlated row counts, govern the stress evidence

## Re-render and inspect

The runner invokes the renderer automatically. It can also rebuild plots and the numeric report from existing CSVs without refitting:

```sh
python scripts/plot_results.py artifacts/demo --mode demo
python scripts/plot_results.py artifacts/calibration --mode calibration
```

Each figure is saved as a readable PNG and an editable SVG under the output directory's `figures/`. That output directory's `RESULTS.md` is generated directly from its current CSVs and records the origins, draw counts when metadata exists, numeric forecasts or coverage scores, failures and limitations. CSVs are the numerical source of record; the renderer does not alter them.
