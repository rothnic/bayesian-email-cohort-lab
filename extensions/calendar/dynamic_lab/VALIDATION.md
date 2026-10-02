# Independent validation of the implemented dynamic approximation

Reviewed 2026-10-02. **The hierarchy, calendar process, and joint predictive
paths are implemented and executable. They are not validated production
probabilities.** This review checks the declared Gaussian approximation,
information boundaries, and accounting; it does not turn the synthetic
comparison into proof of calibration or a universal winner.

Final model code checked:
`ad89ab0f55bb9ef6d76ede414777168f2b6592541e35306a13122c438444cb94`.
The original published v1 model/results were not edited. Review changes are
limited to this report and `test_model_contract.py`.

## Executable checks

Independently ran:

    OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m unittest -v dynamic_lab.test_model dynamic_lab.test_simulation dynamic_lab.test_model_contract

**All 24 tests passed:** 12 model tests, five simulator boundary tests, and seven
independent contract tests. The model suite includes 60 populated fits over six
scenarios, two cutoffs, and five modes. These are implementation checks, not
calibration observations or 60 independent experiments.

The conditional Gaussian calculations use the correct precision, posterior
mean, covariance factor, and integrated marginal likelihood for the stated
diagonal approximate likelihood. The finite grid is the actual proper discrete
hyperprior; its effective number of components is not MCMC ESS. Target-only
and source-pooling blocks use local dispersion posteriors, so unrelated blocks
do not transfer parameter evidence into those comparators.

## Findings resolved during review

- Zero-count transformed response bins retain half-count variance 2 before
  the NB variance term, rather than falsely receiving the .02 precision floor.
  This repairs a numerical/statistical boundary defect; it does not make the
  sparse log-count approximation exact.
- Fractional settlement is now rejected. A visible monetary cell is either
  unavailable or fully settled; an observed fraction greater than zero cannot
  silently fix incomplete revenue/counts as complete.
- Each visible negative daily net-revenue row is rejected before aggregation.
  A negative row can no longer hide inside a positive weekly sum. Signed
  reversal-ledger modeling remains outside this unsigned net-mark model.
- Calendar knots now use one fixed global weekly grid, including a latent
  endpoint beyond a nonweekly cutoff. The prior law for an existing day no
  longer changes merely because a later cutoff is inserted as a new knot.
  Prediction reuses that endpoint, draws further weekly increments, and uses
  the same interpolation law. Independent checks establish prior variance
  invariance at days 9/10 and posterior state mean/covariance invariance when
  the same data are refit at cutoffs 10, 14, and 21. Source/cohort/calendar
  scale supports were not tuned to repair these defects.

## Information, transfer, and economic boundaries

The model imports no simulator or evaluator truth. It retains only declared
observation fields, selects by acquisition/occurrence/report clocks, masks
unavailable audits and monetary fields before validation, and privately
serializes its input. Mutating the caller's DataFrame cannot change the fit.
Duplicate cohort/day versions are rejected; revisions must be resolved upstream.
Each fit starts from the same declared hyperprior and all current as-of data,
without reusing a previous posterior as a prior.

Unseen-source prediction samples the global/feature posterior and scale
hyperposterior, a fresh source effect, and a fresh cohort effect for both
response and payout. Two new cohorts in one new source share the source draw
while retaining separate cohort variability. Independent Monte Carlo covariance
checks confirm this distinction from two different new sources. Equal cheap
quotes/features and identical prelaunch histories cannot disclose whether the
evaluator's unseen source is good or bad.

Each predictive draw has one response calendar path and one value calendar path
shared across targets on absolute calendar dates. The two processes themselves
are independent; their response/value covariance is not learned. The contribution
path includes acquisition at age 0, declared sends through 365, and a finite
425-day policy. Mature past counts/revenue remain fixed; pending imputations are
approximate and do not add a second copy of an observed receipt.

The evaluator reads actual historical purchase costs from its registry. Future
conditional purchases use the quote available at the decision cutoff; their
evaluator path is adjusted to that hypothetical purchase cost only. Changing a
quote does not enter response/value inference. The public model accepts total
target costs supplied by its caller; it does not independently authenticate a
registry, future contract, or quote.

First payback retains all draws, including never-crossing paths at infinity.
An independent test keeps a reversed early crossing, a late crossing, and a
never-crossing draw distinct from positive contribution at age 180. CDFs must
end at one minus the never mass, rather than condition on successful draws.

Held-out forecast rows are written and hashed before their truth scores are
computed. Demo truth is excluded from the model and forecast inputs, but is
accessed to construct separate illustrations before the final demo seal. The
supported claim is sealed prediction-before-scoring plus a truth-free forecasting
boundary; this is not complete process-level pre-seal isolation of demo truth.

## Statistical scope and unresolved uncertainty

The exact Gaussian/grid calculation operates on an **approximate** likelihood.
Audit sensitivity/specificity and response/payment clocks are known controlled
instruments. Human-event estimation smooths and corrects a nested audit fraction,
then applies a clipped log/delta-method transformation. Weekly aggregation,
sparse/zero counts, audit clipping, and correlation with commercial human counts
can bias uncertainty. Future events use NB frailty; that does not make the fitted
response likelihood an exact latent NB/audit model. Distinct human users are
not an additional independent fitted outcome.

Net CPC uses a transformed mature mean and an assumed CV, with Gamma future net
marks. The model does not separately infer unpaid-event probability, reversals,
unknown instrument errors, attrition, delayed occurrence, learned age shapes,
or joint response/value shock covariance. Fixed age shape, calendar-zero anchor,
and exchangeable source/cohort effects identify the decomposition by assumption.

Bounded variance hyperpriors and fixed-variance Normal locations give finite
exponential moments at the finite horizon. They do not guarantee stable means
at 128/256 draws. The separate prior check changes scale supports without using
held-out outcomes; its tail shares and Monte Carlo error must remain visible.
`continue_recent_trend` and `fixed_price` are conditional external scenarios,
not learned knowledge of future deterioration or its duration.

The synthetic evaluation has three independent worlds per scenario and 18 in
the aggregate. Forecasts/cohorts/cutoffs within a world are dependent. World-level
resampling preserves that dependence but three worlds cannot establish robust
scenario calibration. Results must retain interval misses and distinguish
contribution coverage, distributional accuracy, and acquisition decision loss.
Different measures can favor different models; implementing the dynamic model
does not justify selecting one universally superior model.

## Regenerated output checks

Checked the final regenerated run, rather than the earlier pre-correction
scores. All four scientific source hashes and all 14 CSV hashes match
`artifacts/run_manifest.json`. Both forecast seals match their CSV bytes and row
counts. The current run configuration matches the recorded configuration. The
two superseded numerical versions remain separately preserved in
`review_history/`.

The held-out output has **1,530 forecasts, 306 per model, across 18 independent
worlds**. All 210 reported metric summaries recompute from the corresponding
world means, with matching world/forecast denominators. Cheap-source rows are
present at origins 69, 91, and 119. Every one of the 18 illustrated payback CDFs
is monotone, has all ages 0–425, and ends at one minus its never mass.

Independently regenerated all **768 joint portfolio draws** (256 for each of
three future scenarios). They match the archived draws and summary quantiles.
The portfolio is the six declared selected targets, including conditional
future purchases, evaluated at the **same calendar day 299**. Their differing
ages are 299, 187, and 173; this does not sum contribution at unrelated
age-180 calendar dates or resample shocks independently across cohorts. It is
an illustrated selected-target portfolio, not the sum of every acquired cohort.

Independently regenerated all **10 offered-CPL sensitivity rows** for the
future new-source batch born at day 126, priced at origin 119. The five quotes
5/10/15/20/30 cents per lead shift the same posterior outcome paths by their
purchase-cost difference. Monetary means/quantiles and unconditional payback
probabilities match; higher quotes never increase payback probability. This
exercise reprices a prospective purchase only and does not change fitted
quality, old acquisition charges, or source evidence.

**Calibration failure remains material.** Dynamic nominal 95% contribution
coverage is **56.48% under equal weighting of world means**. The raw row fraction
is 46.41%; it weights worlds with more evaluated targets/origins more heavily
and is a different statistic. Static hierarchy has equal-world coverage 47.74%
and target-only 73.25%, also below nominal. The dynamic model's aggregate CRPS
point estimate is lower than the alternatives, but the static hierarchy's
acquisition decision-loss point estimate is slightly lower: $0.049957 versus
$0.050042 per lead. These point estimates and limited-world uncertainty do not
establish a universally superior model or a reliable decision advantage.

The illustrative hold-current portfolio also retains a miss: its 95% interval
is approximately −$44.77 to $116.51, while evaluator contribution is −$70.44.
An implemented shared-state forecast can therefore remain confidently wrong.
The richer synthetic stress evaluation is not simulation-based calibration from
the model's own prior/likelihood. No full calibration claim is supported.

No remaining material implementation defect was identified within the declared
controlled input contract. The approximation and conditional-scenario limits
above remain substantive; passing tests and valid output hashes do not remove
them.

## Article factual review

Checked the 3,150-word article and its `claims.json` against the sealed outputs:
all seven binding-source hashes, 14 selected CSV rows, and two refreshed
prior-tail rows match. The planned-purchase price comparisons, transferred
new-source prior/updates, subsequent interval misses, future-only offered costs,
and common-date portfolio are consistent with the implementation. The old v1
component/stress coverage ranges also match their retained source records.

The article correctly distinguishes equal-world coverage from raw-row coverage,
retains undercoverage and illustrative misses, notes overlapping CRPS uncertainty,
and does not claim a dynamic decision advantage. It states the favorable known
instruments, fixed age/exposure, approximate likelihood, and omitted attrition,
unpaid atom, future signed reversals, and response/value covariance.

Editorial corrections were reported to the parent without editing the article:
remove spaces from the historical commit URL/identifiers; replace the inaccurate
"small" audit sample with the actual roughly 50% random event sampling; describe
hold-current as no deterministic **log-state** drift, rather than constant
arithmetic price levels in expectation; remove missing-selector placeholders;
and state three worlds per scenario versus 18 aggregate worlds clearly. Remote
verification of the historical commit URL was unavailable in this review.
These do not require changes to the frozen numerical results.
