# Independent implementation review

Reviewed on 2026-10-02. This review concerns the declared synthetic model and
its public examples. It does not validate a production spending policy,
unrestricted lifetime value, causal treatment effects, or calibrated real-world
probabilities.

## Finding

No unresolved correctness blocker was found in the supported default synthetic
workflow. The response-rate module implements the stated finite-grid posterior.
The observation/accounting modules remain explicitly modular and approximate.
The adverse-scenario results show substantial undercoverage; they must remain
visible rather than being described as successful production calibration.

Two substantive findings were addressed during review:

1. A positive observed count with zero accepted exposure or zero visibility
   probability has zero likelihood under the declared Poisson observation
   model. Such inputs now raise `ValueError`; they no longer silently produce a
   posterior after dropping likelihood constants
2. Prior-dominated cold-start monetary means can be dominated by a few extreme
   draws. A finite log-variance bound ensures finite moments, but does not make
   a 500-draw mean precise. Keep the original outcomes, show medians and
   intervals, disclose Monte Carlo error, and inspect the variance-cap
   sensitivity alongside the tail-prior sensitivity

For example, the first-cohort high-source day-180 forecast made at age 3 has a
predictive mean of approximately $9.84 per lead, a median of $0.076, and a 95%
interval of approximately [-$0.166, $4.500]. These are different summaries of the
same heavy-tailed simulation, not evidence that the mean is a typical outcome.
Values are fictional and conditional on the declared prior.

## Independent checks

The 17 tests in [`test_integration_review.py`](../tests/test_integration_review.py)
verify:

- Source-grid weights against direct numerical integration of a Gamma prior
  times the Poisson probabilities for two cohorts and two observed ages
- The sorted empirical CRPS implementation against the full quadratic pairwise
  definition, including one draw, ties, negative outcomes, translation, and
  monetary scaling
- First crossing at exactly zero, strict positive-profit semantics, reversals,
  late first payback, terminal first payback, and the unconditional never atom
- Inclusive predictive-interval coverage, Brier scores, and interval width
- Whole-world bootstrap accounting, distinct stress-world random streams, and
  strictly prospective score horizons
- Mature missing revenue receipts versus explicitly observed zero-value marks;
  missing exposure cells versus measured zero-response cells
- No generator-parameter file reads during forecasting, and isolation from
  future acquisitions, unobserved corrections, and unavailable receipts
- Exact preservation of visible signed ledger entries and separately appended
  revisions; economic-date placement of unreported receipts
- The separation of sunk acquisition costs, pending old-send outcomes, and
  incremental future-send value
- Agreement between serialized draws, summary scores, and unconditional payback
  CDFs in both the historical and cold-start public examples
- Fictional observation identifiers and schemas, with latent generator truth
  excluded from the public observation tables

The existing suite separately covers deterministic seed replay, no mutation of
input snapshots, delayed-observation conjugacy, partially missing arm telemetry,
within-source pooling, bounded monetary variance, complete terminal
reconciliation, pending settlements after sending ends, and baseline replay
accounting.

Commands:

```sh
python -m unittest discover -s tests -p test_integration_review.py -v
python -m unittest discover -s tests -v
python scripts/validate_public_artifacts.py
```

Final verification passed all 60 tests, including the 17 independent checks.
The complete documented reproduction command finished successfully. An
independent rerun of the public-artifact validator passed after regeneration;
the demonstration, cold-start, stress suite, and sensitivity metadata match the
current source hashes. The validator also checks finite values, probability
semantics, figures, and accidental private-content leakage.

Evidence retained and checked:

- Historical-cohort illustration: 15 forecasts, 500 draws per forecast
- First-cohort cold start: 15 forecasts, 500 draws per forecast
- Adverse-scenario stress suite: 180 forecasts, 250 draws per forecast, all 12
  distinct seeded worlds retained
- Controlled rate-component experiment: all 2,000 declared worlds and 10,000
  correlated origin cases retained, with 95% predictive coverage from 94.2% to
  95.5% across origins; this does not validate the full modular pipeline
- The controlled package sampler's mean differs from the analytic Gamma mean
  by 0.52 Monte Carlo standard errors; sampled variance differs by 1.27%
- Serialized monetary-mean MCSE values equal sample standard deviation divided
  by the square root of the draw count. These empirical values do not bound
  unseen heavy-tail outcomes

A temporary local wheel build and imports of both library modules also passed
using the already installed scientific dependencies. This is a packaging smoke
check, not an empty-machine installation test. The supported reproduction path
is the README's source checkout plus pinned requirements and scripts;
standalone installed-package generation still expects the source-root parameter
file.

## Interpretation and remaining limits

- The finite-grid likelihood is exact for its declared rate model. Missing
  exposure prediction, ledger-only occurrence imputation, and incomplete
  telemetry are not a claim to an exact joint posterior for every record
- Independent finite-mixture/conjugate draws do not require MCMC R-hat or chain
  ESS. Effective grid size describes posterior concentration
- Twelve stress worlds, with three seeds per deliberately different scenario,
  are limited evidence. Repeated sources, origins, and horizons within a world
  are correlated; whole-world resampling respects that dependence
- Known synthetic response, reporting, and accounting clocks make this easier
  than estimating real-world unknown or informative delays
- The log-variance cap is part of the prior. It guarantees existence of
  monetary moments, not stable finite-draw tail estimates; sensitivity remains
  decision-relevant, especially without mature revenue history
- First payback is an economic first crossing under a finite policy. It is
  different from positive terminal margin and from staying positive afterward
- Forecasts are marginal per cohort. Independently produced paths do not
  provide a joint portfolio-risk distribution
- Operational contactability is distinct from human engagement. No response
  does not establish permanent inactivity, and response-event intensity is
  not unique-person retention
- Neither source comparisons nor model-based continuation values estimate a
  randomized causal A/B treatment effect

These limits should remain prominent in the README, generated reports, and
chart explanations. Synthetic observations and evaluator-only truth may both
be published for reproducibility, provided the inference boundary remains
clear.
