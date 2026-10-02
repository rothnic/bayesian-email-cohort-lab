# Controlled calibration of the conjugate rate component

The main full-world stress suite shows substantial undercoverage. That suite is
retained unchanged. This separate, prespecified controlled scenario checks a
narrower question: **does the known-shape Gamma-Poisson response component update
correctly and produce calibrated posterior predictive uncertainty when its
assumptions actually generate the data?**

This does not validate the full modular mark, operational, observation-delay,
revision, acquisition-invoice, or source-grid model.

## Frozen configuration and retained worlds

`scripts/validate_conjugate_scenario.py` declares a distinct seed `2026100201`,
2,000 worlds, original/fixed accepted exposure of 100 each day, observations at
ages 3, 7, 14, 30, and 60, and a prediction horizon of 180. Every prespecified
world is retained. Repeated origins share the same world and are correlated;
there are 2,000 independent worlds, not 10,000 independent experiments.

The curve is known and equals one allowed main-model shape: fast weight 0.8,
fast timescale 1 day, tail timescale 60 days. Source mean 0.12 and cohort Gamma
shape 8 also match the main package's rate parameterization. Each world's
amplitude is independently drawn from `Gamma(8, rate=8/0.12)`. Conditional daily
counts are independently Poisson with mean `100 × theta × f(age)`.

Monetary value is a known $0.125 per response. Acquisition costs $0.15 per
original lead; each accepted exposure incurs a known $0.00025 sending charge.
There are no response/reporting delays, revisions, missing observations,
operational exits, unknown acceptance probabilities, or mark uncertainty in this
controlled scenario. These simplifications are central to its scope.

## Exact sequential posterior and predictive distribution

At each origin, let `N` be observed cumulative responses and `L_obs` the sum of
known exposure-weighted age intensities. The exact posterior is:

`theta | data ~ Gamma(8 + N, rate=8/0.12 + L_obs)`

This is the same conditional update used in the implemented main response
module. If `L_future` is the remaining weighted exposure through age 180,
future response count has the exact negative-binomial mixture:

`N_future | data ~ NB(shape=8+N, p=rate/(rate+L_future))`

Central 80% and 95% predictive intervals use the exact negative-binomial
quantiles, then apply the affine transform
`margin = 0.125 × (N + N_future) - known_total_cost`. No Monte Carlo draw
approximation is used for predictive coverage. Integer count quantiles can make
central intervals slightly conservative. Both posterior amplitude uncertainty
and future Poisson variation are included.

The script also computes the actual exact probability mass inside each integer
interval. Realized coverage is checked against that mass, using its conditional
Bernoulli sampling standard error, rather than assuming discrete intervals
attain precisely their nominal labels.

Profitable-at-horizon probabilities use the exact negative-binomial survival
function at the integer monetary threshold. Brier scores and reliability bins
compare those probabilities with each retained world's realized margin.
Parameter credible interval coverage, posterior CDF ranks of the true
amplitude, and randomized predictive probability-integral-transform values are
reported separately by origin. Rank uniformity is expected for this
prior-predictive check; it is not a guarantee of conditional frequentist
coverage for every fixed amplitude or parameter region.

## Connection to the actual implemented model

The script constructs a public `Snapshot` fixture from one prespecified world's
observed counts and fixed measured exposure. It calls the actual
`bayes_cohort.forecast_cohort` API with a single known-shape/source-mean grid
point and immediate observation clocks. The returned posterior amplitude draws
are checked against the exact Gamma posterior mean and variance. Thus the
controlled check is tied to the implemented rate component rather than being
an unrelated toy model. It deliberately does not score the other modules invoked
by that API call.

The main model source remains frozen. This script neither estimates nor tunes
new full-world priors and does not remove, revise, or relabel any stress worlds.

## Running and artifacts

```bash
python scripts/validate_conjugate_scenario.py
python -m unittest discover -s tests -p test_conjugate_scenario.py -v
```

Default output is `artifacts/controlled_scenario/`:

- `metrics_by_origin.csv`: exact predictive coverage, Wilson intervals,
  parameter coverage/width, monetary width, Brier and rank/PIT summaries
- `exact_forecasts.csv.gz` and `all_prespecified_worlds.csv.gz`: every world and
  every origin, without selection or reweighting
- `profitability_reliability.csv`: probability reliability by origin
- `run_metadata.json`: frozen configuration, distinct seed, versions, source
  hashes and actual-package posterior-sampler cross-check
- `paired_full_world_stress.csv`: read-only aggregation of the existing stress
  scores at the same 180-day horizon, with independent-world counts
- PNG/SVG `controlled_vs_full_world_coverage` and
  `controlled_updating_and_ranks`, plus an artifact `RESULTS.md`

The paired chart keeps the main stress suite's undercoverage visible. Successful
component calibration establishes a useful mathematical/software positive
control. It does not establish that the full modular pipeline's profitability
probabilities are calibrated or suitable for actual spending decisions.
