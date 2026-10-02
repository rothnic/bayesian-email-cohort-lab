# Implemented experiment and evaluation boundary

This is a fitted extension, separate from the original v1. `MODEL.md` specifies the implemented Gaussian approximation and finite hyperprior. `DESIGN.md` in the parent folder contains the broader staged design; mechanisms proposed there are not automatically claimed as implemented.

## Nested fictional worlds

The generator starts100-lead cohorts weekly. Two established sources arrive from calendar0 through126. A new low-cost source, when present, starts on70. The public age curve is `.75 exp(-age/3)+.25 exp(-age/90)` and is the same fixed curve used by inference. Accepted exposure is100 each day through age365, with no dropout or reactivation in this extension.

1. `fixed_price`: stable underlying human response and net CPC, plus modest shared weekly noise
2. `cpc_squeeze`: net human CPC declines after calendar42 through126, then stabilizes
3. `rising_cpl`: the same squeeze, plus higher quotes for later purchases
4. `human_decline_bot_rise`: also lower human response intensity and more bot response events
5. `cheap_good`: the preceding mechanisms plus a productive cheap source arriving on70
6. `cheap_bad`: the same observable prelaunch offer/features, but an unproductive source

Commercial net payments are nonnegative Gamma marks; count cells are NB2. Raw human/bot events are sampled by a50% audit, with known label sensitivity95% and specificity98%. Response cells arrive after2days, audit labels after7, and commercially verified human count/net-payment summaries after14. Inference never receives latent simulator rate, true bot counts or true calendar state. Known exposure, known instrument quality and generator-matched age shape are favorable assumptions, not production evidence.

The prior and generator are different. This is not a simulation-based-calibration experiment under the exact fitted model. The new model uses approximate Gaussian log measurements, broader net-mark variation and an unknown population/calendar hierarchy. The original exact Gamma–Poisson component check remains separately available.

## Fixed economics and changing futures

Each forecast fits the observations available at its calendar cutoff. It then conditions on one future:

- `hold_current`: no deterministic future log-state drift, with sampled weekly innovations and consistent interpolation; this is a scenario, not learned stabilization
- `continue_recent_trend`: continue each posterior draw's recent response AND payout state slope, preserving uncertainty in those slopes
- `fixed_price`: freeze future payout at the learned calendar0 reference; observed mature historical receipts stay fixed and future response uncertainty remains

All three reuse one response path and one payout path per draw across targets. The two processes' covariance is not learned. Future price quotes cannot rewrite the cost of an existing cohort. Next-purchase evaluation explicitly locks the current observed quote for a planned later batch. That is a conditional purchase assumption, not a forecast of future quote acceptance.

## Comparisons

`run_config.json` lists18 independent held-out worlds, three per scenario, with distinct seed streams. These differ from the demonstration seed884000. The original schedule uses cutoffs63 and119. Cheap-source cases also use69 and91 so the new source is directly scored before launch and after evidence arrives. That evaluation-scope correction is recorded in `review_history`; no world was selected by outcome.

Every model sees the same records/clocks and uses the same costs and targets. Target-only and source-pooling modes have independent local dispersion hyperposteriors. Global pooling, static hierarchy and dynamic hierarchy progressively change information sharing. All are versions of this new approximation, not relabeled original-v1 models. The original v1 replay and failures remain in their original artifact paths and the separate `prototype/v1_replay` checkpoint.

Coverage, CRPS, probability Brier scores and mean absolute error are evaluated for day180 contribution. Conditional acquisition loss is `max(actual_margin,0) - buy_decision*actual_margin`, divided by100 leads, with buy when sampled expected contribution is positive. It is realized regret against a hindsight-perfect buy/skip oracle. Only future-purchase targets enter that loss. It does not model adaptive bidding, causal acquisition effects, sending cessation or an optimal production policy.

Repeated forecasts, sources and cohorts within a world are correlated. Aggregates first average within each world and then weight worlds equally. Confidence intervals resample whole worlds. Some scenarios contain more forecast cases because they include a source launch; counts and independent-world totals are explicit. This small suite cannot establish a universal winner. Check interval width/coverage and numerical mean stability together.

## Joint portfolio and offered-cost query

`portfolio_summary.csv` and `portfolio_draws.csv` sum the same joint predictive draw across four existing established-source cohorts and two conditional future batches. The common calendar horizon is299,180days after the119 cutoff. The constituent ages differ and are stored explicitly; this does not sum age180 margins reached on different dates. The batch costs are included once. These are total contribution paths, not future incremental cash balances.

`quote_sensitivity.csv` asks about a new cohort from the good/bad cheap source at cutoff119 for planned arrival126. It prices the SAME posterior outcome paths at offered costs5/10/15/20/30cents per lead. Quality inference and outcomes stay fixed; no already purchased cohort is repriced. Every path remains in the first-payback denominator. An absent median payback means fewer than half of draws pay back within the finite policy, not missing data that should be averaged away.

## Computation, corrections and reproduction

Each as-of input is copied from immutable serialized bytes. Every fit starts at the same fixed hyperprior. Final source/output hashes and prediction seals accompany the output. Held-out forecast rows are written before scoring. Demo truth is separately available to plotting/evaluation and is excluded from fitting; the demo does not claim all truth was inaccessible to every function before its global output seal.

Before finalization, review corrected a zero-count log-variance boundary, launch-date calendar uncertainty, a cutoff-dependent prior knot grid, fractional-payout input handling and individual negative-payment rejection. The fixed weekly grid now has invariant historical prior/posterior states when only the cutoff changes with no new data. Prior scale supports and masses were not tuned to observed held-out coverage. Superseded aggregate results are retained as review history and must not be combined with final scores.

Run `make reproduce` from the extension folder after installing the repository's existing requirements. There is no MCMC sampler: Gaussian calculations and discrete hyperprior normalization do not have R-hat or MCMC ESS diagnostics. Component covariance tests, grid weights, predictive Monte Carlo errors and analytical prior moments are the appropriate checks here. `prior_tail_summary.json` includes512/2048/8192-draw checks and smaller/default/larger scale sensitivity. Finite moments do not guarantee stable expected-profit estimates from128 or256 draws.

No raw original row-level export is copied or repackaged. The extension writes only simulation code, aggregated forecasts/evaluation, calendar/measurement summaries, posterior portfolio draws, and figures. All source data is fictional.
