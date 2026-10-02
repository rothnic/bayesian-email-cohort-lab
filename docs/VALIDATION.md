# Validation, calibration and reproducibility

## Three different questions

1. **Does the implementation calculate its declared model correctly?** Focused tests check conjugate updates, delay thinning, partial pooling, economic accounting and reproducibility. Independent posterior draws come from a finite mixture and conditional conjugate distributions, so MCMC R-hat, chain ESS and divergences are not relevant diagnostics. Effective grid size is posterior concentration, not sampler effective sample size.
2. **Does the assumed model describe these fictional worlds well?** The replay compares future economic truth with forecasts made from actual as-of snapshots. Calibration and scoring results are in [the stress report](../artifacts/calibration/RESULTS.md). Failures remain in the report.
3. **Would it make good real decisions?** This repository does not establish that. Real measurement validation, mature holdouts, joint source risk, decision loss, budget constraints and causal experiment checks are still needed.

## Reproducible units

- The fixed illustrations use seed 211, eight weekly cohorts/source and 180 leads/cohort; the main demo targets the last cohort, while the cold-start companion targets the first cohort without older history
- The stress suite uses four scenarios and three base seeds, with scenario-index offsets of 10,000 to give 12 distinct random streams; each has six cohorts/source and 100 leads/cohort
- Each world forecasts the final cohort of each source at ages 3, 7, 14, 30 and 60
- The illustration uses 500 predictive draws/forecast; stress validation uses 250
- Forecasting accepts only a `Snapshot`, a public sending policy, cohort ID and frozen model configuration. The generator's complete truth is passed only to evaluation, after forecasting
- No retrospective or at-origin horizon enters prospective scoring. For example, day-30 economics inferred at age 60 is not scored as a future forecast

## Scores and coverage

For predictive draws x₁,…,xᴮ and realized y, empirical CRPS is mean |xᵇ−y| minus half the average pairwise |xᵇ−xᵈ|. The implementation calculates the pairwise term by sorting, avoiding a quadratic matrix. Lower is better. Monetary errors and CRPS are available both per cohort and per original acquired lead.

Profitability Brier score is (p−1[y>0])². Payback Brier score uses the unconditional probability that first payback occurs by a horizon, including nonpaying draws. Coverage is the fraction of complete economic truths falling inside a stated predictive interval. Interval width is reported alongside coverage; simply making every interval huge is not success.

Repeated origins, sources and horizons within a world are correlated. Confidence bands for diagnostic averages resample complete worlds, not individual rows. The worlds come from four intentionally different scenarios; pooled coverage describes that equal-weight scenario mixture, not a naturally sampled production population. Twelve independent worlds, only three per scenario, produce limited evidence and potentially unstable bootstrap intervals. Reliability bins include counts; they are descriptive and have no fabricated iid error bars.

With B independent predictive draws, Monte Carlo standard error of a probability estimate is approximately sqrt(p(1−p)/B), at most 0.022 for B=500 and 0.032 for B=250. Reported 0% or 100% therefore does not mean certainty. Tail quantiles and monetary sample means can have appreciable Monte Carlo noise. The bounded log-variance prior ensures monetary moments exist; the bound remains a substantive sensitivity assumption.

## Controlled positive check

A separate [2,000-world controlled scenario](CONTROLLED_SCENARIO.md) uses the same Gamma–Poisson rate update with known shape, exposure, costs and fixed click values. Its exact negative-binomial predictive intervals achieve approximately nominal coverage, allowing for discrete quantiles. The module sampler is cross-checked against the analytic posterior. This checks the simplified rate component, not the full unknown-value, delayed-ledger pipeline. The richer stress failures remain unchanged.

## What has not been established

The held-out generator is deliberately different from the inference model. This is **predictive stress validation**, not formal prior-generated simulation-based calibration (SBC). There is no universal-calibration or Bayesian-superiority claim. The known synthetic measurement clocks are favorable assumptions. Permanently missing and temporarily misclassified events, shared shocks, late reactivation and multiple correlated attribution corrections can violate those assumptions.

The code currently produces marginal per-cohort simulations. Because it does not generate one joint posterior future calendar process across all forecasts, combining independently generated cohort draws would underrepresent portfolio dependence. No such portfolio uncertainty claim is made.

## Check commands

```sh
python -m unittest discover -s tests -v
python run_bayesian_demo.py demo
python run_bayesian_demo.py cold-start
python run_bayesian_demo.py calibration
python run_sensitivity.py
python scripts/validate_conjugate_scenario.py
python scripts/validate_public_artifacts.py
```

Run metadata includes exact dependency versions, configuration, elapsed time and SHA-256 hashes of the source used to produce the evidence. Generated gzip timestamps are fixed where practical; floating-point behavior can still differ slightly across library/platform versions. Runtime and resource metadata naturally differ between runs. Package-installation from an empty machine is not implied by a passing local scientific test.
