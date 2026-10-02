# Independent review: existing information flow and calendar extension

Review date: 2026-10-02. Scope: read-only inspection of the frozen published
`bayesian-email-cohort-lab`, plus a design/input-contract review of the separate
extension. Boundary tests invoke the unchanged v1 forecaster. No new global or
dynamic model, publication, or changes to the published repository were made.

## Verified existing information flow

**The first cohort has no older cohort history at launch, but its later forecasts
are not target-only.** The published cold-start world has eight weekly cohorts
per source, acquired on days 0, 7, ..., 49. Every source has a w00 cohort at
calendar day 0. `World.as_of` includes all leads acquired on or before the
cutoff, not just the requested forecast target. Observations must also have
occurred and been reported by that cutoff.

| w00 age / calendar cutoff | Visible cohorts per source | Other same-source cohorts | All-source cohort count | Same-source ages at cutoff |
| --- | ---: | --- | ---: | --- |
| 0 (launch) | 1 | none | 3 | 0 |
| 3 | 1 | none | 3 | 3 |
| 7 | 2 | w01 | 6 | 7, 0 |
| 14 | 3 | w01–w02 | 9 | 14, 7, 0 |
| 30 | 5 | w01–w04 | 15 | 30, 23, 16, 9, 2 |
| 60 | 8 | w01–w07 | 24 | 60, 53, 46, 39, 32, 25, 18, 11 |

The count is `min(8, floor(cutoff / 7) + 1)` per source. In particular, day 60
does not include w08: the published world ends acquisition at w07. The inclusive
cutoff means a newly acquired cohort's age-zero data can contribute immediately.
These counts were checked from the existing `leads.csv.gz`, then from snapshots
of all five existing observation tables. All counted cohorts at the published
origins have usable exposure cells; they are not merely metadata placeholders.

`bayes_cohort/model.py` fits each source independently. For that source, every
visible same-source cohort's response likelihood updates one shared posterior
over response mean and age-curve parameters. The target's Gamma response
amplitude then conditions on its own observations and a draw from that shared
source posterior. Exit and acceptance parameters also pool same-source exposure.
Monetary marks pool mature, observed, qualified-human base receipts within the
source. No learned global hyperposterior transfers information across sources;
each source begins with the same fixed declared configuration.

Reconstructed event counts and usable exposure cells exactly matched the frozen
`artifacts/cold_start/diagnostics.json`. At age 7, source-wide/target-only counts
are low 75/58, middle 162/111, and high 226/157. At age 60 they are low 1128/207,
middle 2348/390, and high 4691/671. These are qualified response events, including
repeats, rather than distinct people. Missing exposure cells remain excluded.

Payout learning starts later than response pooling: there are no mature marks at
ages 3, 7, or 14; age 30 has mature marks from w00 only; age 60 has mature marks
from w00–w04. Their later acquisition is valid contemporaneous information,
not future information relative to the calendar cutoff.

**Recommended wording:** “First-cohort launch with no older same-source history;
later as-of forecasts borrow information from subsequently acquired same-source
cohorts.” If “target-only cold start” is intended, it requires a separate explicit
ablation that removes those cohorts' contribution. The existing cold-start
trajectory cannot establish the isolated gain from pooling.

Primary code evidence in the frozen repository:

- `cohort_lab/accounting.py:39`: inclusive, report-time-aware `World.as_of`
- `cohort_lab/generator.py:53`: weekly acquisition schedule
- `run_bayesian_demo.py:96`: eight-cohort demo override; `:111`: full-world cutoff
- `bayes_cohort/model.py:47`: metadata and qualified-event construction;
  `:69`: source loop; `:73`: all same-source cohorts; `:103`: mature mark filtering
- `docs/MODEL_IMPLEMENTATION.md`: declared within-source pooling and event estimand

## Assessment of the supplied DESIGN.md

**The revised design resolves the three original probability-model findings.**
It is suitable for the staged contract/inference plan, subject to the remaining
implementation gates below. It contains no fitted extension or improved result.
Existing information scope, independent cutoff refits, event versus person
estimands, shared portfolio paths, as-of quotes, and unconditional payback
semantics meet the stated design requirements. The original-model ablation
precedes claims about its pooling gain.

Resolved findings:

1. **Payout hierarchy and clocks.** Positive amount now has global, source,
   cohort, and shared calendar effects, with fresh new-source/new-cohort payout
   draws. Initially global paid/reversal/fraction parameters are an explicit
   simplification to ablate. Response intensity uses send date; payout state and
   its applicable contract use occurrence date. Settlement/report dates remain
   accounting/visibility clocks. Exactly known contracted rates are not fitted
   again as unconstrained latent multipliers.
2. **Finite prior-predictive moments.** Fixed-variance Normal locations and
   feature coefficients, bounded source/cohort effect scales and calendar
   innovation variances, bounded features, and Gamma shapes bounded away from
   zero give finite first/second count and payment moments over a finite horizon.
   In particular, `E(theta^2 | m,k) = m^2 (1 + 1/k)` and
   `E(amount^2 | eta,kappa) = exp(2 eta) (1 + 1/kappa)`; the bounded shape factors
   and finite Normal exponential moments survive the declared mixing. The
   argument includes uncertain calendar covariance, rather than merely its
   conditional variance. Numerical prior parameters still must be frozen,
   sensitivity checked, and Monte Carlo tail stability established before
   decision claims. Finite moments do not imply small Monte Carlo error.
3. **Observation/audit likelihood.** Stage 3 explicitly uses known instrument
   inputs; Stage 4 jointly models latent human/bot types, delays, classifier
   categories, known-probability audit selection, and delayed imperfect audit
   labels. Its single joint categorical thinning construction avoids counting
   raw events, production labels, and audited subsamples as independent evidence.
   Independent error calibration/informative priors and identification bounds
   are correctly required. Superseded labels refer to one underlying observation.

The Negative Binomial augmentation itself is coherent: unit-mean Gamma cell
frailty mixed with conditional Poisson counts gives NB2 variance
`lambda + lambda^2 / r`, while a shared Gamma cohort amplitude induces additional
cross-day dependence. Given complete human counts and cell frailties, the
displayed Gamma full conditional for the cohort amplitude is correct. A
collapsed sampler based on visible counts would instead require their report
probabilities in the exposure term. This is counted clustering, rather than a
renaming of uncertainty in a Poisson predictive distribution.

Initial APC identification is defensible under the stated fixed age curve,
reference-date calendar anchor, and exchangeable cohort distribution without a
cohort-date trend. For precision, `theta_c / m_s ~ Gamma(k,k)` is centered at
one on the multiplicative scale; its mean log is `digamma(k) - log(k)`, not
zero. The revised design now declares that centering scale. Future learned age
bases require the promised
rank/null-space check; fixing `f(0)=1` alone removes a level ambiguity, not a
free linear age/calendar/cohort trend. Alternate-anchor sensitivity is properly
required and the driver decomposition remains assumption-dependent.

Remaining implementation gates are actual joint inference; frozen numerical
hyperprior/delay/audit specifications; learned-shape rank/null-space and anchor
checks; tail/sampler diagnostics; and held-out matched ablations/calibration.
Unseen-source transfer must implement the declared full predictive variance.
No static hierarchy, dynamic posterior, pooling benefit, calibration, or policy
improvement is verified by this review.

## Independent prototype checks

Ran `PYTHONDONTWRITEBYTECODE=1 python -m unittest -v` from `prototype/`:
**all 13 final tests passed**, including the added revision-chain regression.
The original 12 tests also passed before that repair. Recomputed the manifest rows in memory from the declared
fixture/cutoffs; they exactly match `replay_input_manifest.json`. These establish
only the tested input/accounting properties; `DynamicHierarchy.predict` remains
unimplemented and the shared paths are illustrative scenarios.

**The material revision-chain issue is resolved.** The initial implementation
returned both x1 and x2 for `x → x1 → x2`, duplicating one measurement. The final
`as_of` resolves visible ancestors to a stable original root, requires matching
kind/source/cohort/effective date and advancing versions/report times, and
rejects missing ancestors/cycles. The new test returns only x2 at the latest
cutoff and only x1 at the earlier cutoff. I inspected the repair and independently
reran the full 13-test suite. No material contract flaw remains identified in
this limited prototype review. I did not edit the prototype implementation.

The acceptance criteria for that assessment are:

1. A learned global → source → cohort hierarchy for human-response rate/shape
   and payout, with assumptions about sharing made explicit. Forecasting a new
   source integrates the global hyperposterior and draws full new-source and
   new-cohort variation. Plug-in global means or existing-cohort centering must
   not suppress predictive variance.
2. Negative-binomial response-event counts and binomial distinct-human-user
   counts have separate estimands and denominators. If both use the same events,
   their likelihood must be coherent or fitted as explicitly separate models;
   multiplying independent likelihoods for overlapping data would double count.
3. One shared calendar response/value state realization per predictive draw
   across the whole portfolio, preserving cross-cohort/source covariance and
   calendar uncertainty. Summing independently simulated cohort shock paths is
   invalid for portfolio risk. The event-to-value calendar reference must be
   declared (send, occurrence, or settlement).
4. Fixed/anchored age curves and centered cohort effects address age-period-
   cohort nonidentifiability. A prior alone does not identify unrestricted age,
   period, and cohort trends. Calendar level, drift, age normalization, and
   cohort-effect constraints must be stated so that rates have one interpretation.
5. Each cutoff refits all immutable as-of data from the same fixed hyperprior.
   A previous posterior is not used as a prior while its observations are reused.
   Warm starts, if allowed, are computational initialization only. Qualification
   revisions and observation/report delays follow the current snapshot.
6. Human qualification/error and delay assumptions are audited, with bot-rate
   and false-positive/false-negative sensitivity. Latest human labels are not
   silently treated as verified ground truth. Missing receipts are not zeros;
   pending receipts and signed revisions are counted exactly once.
7. Historical acquisition costs remain sunk ledger components. Future-source
   quotes apply only to future leads; continuing existing sends uses genuinely
   incremental costs and outcomes. Quote availability must itself be as-of.
8. A target-only versus pooled ablation is required before any claimed pooling
   gain. Match cutoffs, exposure, payout/measurement assumptions, inference
   budget, and target economics. Separate same-source, cross-source, calendar,
   and new-source effects where the claim depends on them. Independent worlds,
   not repeated origins, remain the validation unit.
9. Preserve existing failures and unconditional first-payback semantics: first
   margin crossing at or before age 425, including the never-under-policy mass.
   First crossing differs from positive terminal margin and durable payback.
   CDFs must not condition on eventual success or normalize away failures.

This design review is not empirical calibration or evidence of better forecasts.

## Final bounded v1 calendar replay review

Independently ran the three new `test_v1_replay` boundary tests: all passed.
Together with the 13 previously checked contract tests, the packet now has 16
tests. `select` restricts every one of the five likelihood tables to the target
or its visible same-source cohorts. The immutable serialized input remains
unchanged if a reconstructed DataFrame is mutated. Each forecast reconstructs a
fresh snapshot, clears the v1 fit cache, and receives the identical fixed
original prior/configuration; no previous posterior is reused.

Code inspection confirms prediction CSVs are written and hashed before complete
evaluator paths are accessed. The forecasting boundary receives no World or
GeneratorTruth. Independently verified all 20 prediction/outcome rows, the
prediction seal in every evaluator row, fixed-prior/source-code hashes, matched
scope-pair seeds, 64-draw budgets, same-source cohort censuses (2/3/5/9/13), and
newest-target age 3 at each declared calendar cutoff. I did not perform another
full replay run; the two byte-identical full reruns are reported by the producer.

`REPLAY_CHECKPOINT.md` correctly limits the evidence to an executable
unchanged-v1 information ablation in one development world. Sixty-four draws
cannot establish reliable tail quantiles or stable heavy-tailed means. This
replay demonstrates neither pooling benefit nor held-out calibration, learned
global transfer, calendar-state inference, or improved decisions. No additional
material boundary flaw was identified in this bounded review.
