# Implemented bounded dynamic cohort model

This extension actually fits a global → source → cohort hierarchy for human
response and net human CPC, plus shared calendar response/value states. It uses
an explicitly approximate Gaussian measurement likelihood and exact conditional
Gaussian posterior calculations with a finite integrated hyperprior. It is not
a negative-binomial latent-event sampler, an exact audit likelihood, or the full
Bernoulli/Gamma/reversal model proposed in `DESIGN.md`. Original v1 code and its
failed coverage results are unchanged. Comparisons made by the separate evaluator
must establish performance; implementing this model does not establish calibration.

## Public boundary and immutable information

`fit(data, cutoff, mode='dynamic_hierarchy', seed=0, config=None)` accepts an
independent pandas copy, row records, or the public `FrozenSnapshot.frame()`
interface. `predict(targets, draws=256, horizon=425, scenario='hold_current',
seed=None)` returns joint draw paths. Only declared observation columns are kept.
There is no simulator, generator, evaluator, or truth import in `model.py`.

Required daily columns are `cohort_id`, `source_id`, `birth_day`, `calendar_day`,
`age`, `accepted_exposure`, `raw_events`, `audit_n`, `audit_human`,
`net_human_revenue`, `net_human_count`, `response_report_fraction`,
`payout_complete_fraction`, and `observed_at`. Optional provenance clocks
`audit_observed_at` and `payout_observed_at` independently mask late audit/monetary
measurements; late payments also set payout completeness to zero. Payout
completeness must be exactly0 or1: this bounded API does not treat a fractionally
settled receipt as final. Each individual visible negative net-revenue row is
rejected before aggregation, so offsetting positives cannot hide an unsupported
signed reversal. Optional
`source_feature` is one fixed bounded scalar in [−1,1], known before acquisition.
Optional `net_human_revenue_sum_squares` replaces the assumed monetary CV in
the approximate likelihood with an observed second moment. No absent label or
pending receipt is turned into a zero observation. One final revision per
cohort/calendar day is required; revisions must be resolved upstream.

Rows are selected by occurrence, birth and observation clocks at the cutoff.
The fit then serializes its private copy, hashes it, and independently refits
from the same fixed configuration. Later DataFrame mutation cannot change the
fit or its prediction. No previous posterior is used as a repeated-data prior.

A target supplies `cohort_id`, `source_id`, `birth_day`, `leads`,
`acquisition_cost` (the TOTAL purchase cost), `daily_send_cost` (the TOTAL batch
send cost/day), and optional `source_feature`. Default accepted exposure equals
leads through send age365; a scalar or age-specific `accepted_exposure` can be
given. This is a declared operational policy, with no dropout inference.
The fixed public age curve is

    f(a) = .75 exp(−a/3) + .25 exp(−a/90), f(0)=1

Costs are excluded from inference, so changing a new quote cannot alter fitted
quality. The evaluator supplies historical purchase costs from its immutable
registry and only visible quotes for prospective acquisitions. The model does
not itself verify the registry's quote provenance.

## What is learned

For each outcome separately, hierarchical modes fit

    response log-rate[c,t] = μ_R + β_R z_s + u_R,s + e_R,c + g_t
    net human log-CPC[c,t] = μ_V + β_V z_s + u_V,s + e_V,c + v_t

Global locations and feature coefficients are learned jointly with source and
cohort effects. Effects have independent zero-centered Normal population
priors, without a cohort-birth trend or a sum-to-zero projection on unseen
sources. The age curve is fixed. Calendar states are anchored to zero at
calendar0. Inference uses independent weekly random-walk increments of variance
q²×7, with piecewise linear state interpolation on a FIXED global grid at
days0,7,14,... . Fitting retains the endpoint at ceil(cutoff/7)×7 even when that
endpoint is later than the cutoff: it is a latent state, not a future
observation. Prediction reuses that sampled endpoint and draws subsequent
weekly increments on the same grid. There is no inserted cutoff knot and no
fitted free calendar drift. Conditional on q, prior covariance at any historical
day is therefore invariant to which later cutoff is selected. For example,
Var(x_9|q)=q²×(7+4/7) at both cutoffs10 and14; the within-week interpolation has
Var(x_t|q)=q²×[7 floor(t/7)+(t mod7)²/7]. It is a smooth weekly process rather
than a daily Brownian process. These restrictions
resolve the unrestricted age/period/cohort trend degeneracy by assumption;
decomposed drivers remain dependent on the fixed shape and calendar anchor.

The response and value processes are conditionally independent, with separate
hyperposteriors. Their covariance is NOT learned. Each draw has one response
trajectory and one value trajectory shared by all targets on their actual
calendar dates. This retains within-process common shocks and portfolio
dependence. It cannot represent additional correlation between response and
value calendar shocks.

All modes use the same measurement transformations and conditional economics:

| Mode | Information and fitted effects |
|---|---|
| `target_only` | Independent cohort intercept and dispersion posterior; no other cohort affects this target's parameter distribution |
| `source_pooling` | Independent source intercept and dispersion posterior, complete pooling of that source's cohorts |
| `global_pooling` | One global intercept and count-dispersion posterior, complete pooling |
| `static_hierarchy` | Global, feature, source and cohort effects; learned source/cohort scale mixture |
| `dynamic_hierarchy` | Static hierarchy plus shared response/value calendar states and learned calendar scale mixture |

`target_only`/`source_pooling` use separate local dispersion hyperposteriors,
even though all blocks are fitted in one call. Grid-weight diagnostics for these
modes are averages of local posterior probabilities, not a joint hyperposterior.
These modes offer no learned population transfer to a new independent cohort
or source. A globally pooled forecast, by definition, omits source/cohort
predictive heterogeneity rather than secretly adding hierarchy uncertainty.

For a genuinely unseen source, hierarchical prediction draws the global and
feature parameters AND the scale hyperparameters from the posterior, then a
fresh source effect with sampled source SD, then a fresh cohort effect with
sampled cohort SD, separately for response/value. Multiple cohorts in the same
new source share its new source draw but get independent cohort residuals.
An existing source/new cohort uses its fitted source draw and a fresh cohort
residual. This preserves the full source and cohort predictive variation.
Source descriptors and cheap quotes do not reveal evaluator quality: equally
cheap good/bad sources with identical visible histories/features have the same
prelaunch posterior predictive distribution.

## Approximate audit and count measurement

Daily rows are aggregated only within a cohort and calendar-week bin to reduce
the severe sparse-count problems of fitting one daily log rate. Effective
response exposure is Σ accepted_exposure×f(age)×report_fraction. Audited labels
are a nested sample of raw response EVENTS. Repeat events are allowed. Distinct
human responders are neither used nor counted as a second independent outcome.

Within a bin, the observed audit label proportion gets a Beta(1,1) smoothing
prior. The instrument has declared sensitivity .95 and specificity .98; the
estimated human proportion is the corrected label proportion
(p_label−.02)/(.95+.98−1), bounded to [.005,.995] for the log transform. Its
delta-method variance divides the Beta variance by the instrument determinant
squared. These are approximate corrected fractions, not error-free true labels.
Audit arrivals absent at T contribute no labels. This assumes known-probability
representative selection and approximately constant human fraction within a
cohort/week; uncertain instruments, unknown nonrandom sampling and within-bin
drift require sensitivity or a future joint latent audit model.

Estimated human events are raw_events×corrected_human_fraction. The likelihood
uses log((estimated_humans+.5)/effective_exposure). Approximate variance is

    (estimated_humans+.5 + raw_events² Var(human_fraction)) / (estimated_humans+.5)²
      + (Σ daily_effective_exposure² / (Σ daily_effective_exposure)²) / r

The last term approximates independent daily NB2 cell frailty with variance
mean+mean²/r. It is not the exact NB likelihood; uncertain fractions, clipping,
aggregation and the log transform can bias sparse bins. Count and audit
variance are combined into this ONE approximate response measurement. Raw
counts and their audited subsample are not multiplied as independent response
likelihoods. Commercially verified mature human counts are used as monetary
denominators and to condition realized past predictions, not as an additional
response likelihood. Their correlation with audit/count measurements is only
approximately handled by this construction and is an important limitation.

Half-count smoothing enters both the transformed mean and its count variance.
Using Gamma(shape=estimated_humans+.5) moments as a delta-method approximation
gives the count term1/(estimated_humans+.5), hence variance2 at zero events
before NB variance. This is an approximate regularized log-count measurement,
not a second Gamma rate prior in the fitted hierarchy. An initial development
version accidentally used an unsmoothed count numerator: at zero counts it
reduced variance to the .02 floor and made zero bins spuriously precise. That
boundary defect was corrected, regression-tested and all final scoring must
refit with the corrected likelihood. It was not a coverage-tuned prior change.

Known report fractions enter the exposure offset. This does not learn response
or reporting delays. The controlled replay supplies fully reported daily cells,
response visibility at occurrence+2 days, independent audit labels at +7 days
and mature settlement summaries at +14 days. Payment reports are observed
instruments available at their clocks, not evaluator truth inputs.

## Approximate monetary measurement and predictions

Available mature monetary bins measure net human CPC = settled net revenue /
commercially valid human event count, including unpaid/zero net marks. The
Gaussian likelihood is log((net revenue+.005)/human count), with approximate
variance CV²/human count and floor .02. The half-cent pseudocount only keeps
all-zero bins finite; all-zero revenue bins get variance at least4. Default
net-mark CV is1.5, a declared conservative assumption, not generator truth.
An observed sum of squared marks can instead supply the delta-method variance.
Negative net receipts are outside this bounded net-mark model and are rejected.
There are no separately learned payment probability, reversal probability or
positive-amount components. This distinction from `DESIGN.md` is deliberate.

Future counts draw NB cell frailty and Poisson events using the sampled r,
calendar states, cohort amplitudes, accepted exposure and fixed age curve.
Aggregate net marks draw Gamma(shape=human_count/CV²,
scale=mean_net_CPC×CV²). Conditional mean is human_count×mean_net_CPC.
This approximate Gamma mark model has no explicit atom at an unpaid event or
signed reversal ledger. Mature observed past revenue and commercially valid
counts remain fixed. Pending past counts are imputed using the observed raw
events and a corrected Beta audit draw; missing/unreported response cells get
predictive imputation. Pending past marks then use the monetary posterior.
This is explicitly approximate past imputation, not a jointly augmented latent
event posterior. It can over/understate conditional uncertainty.

Economic attribution is to response occurrence/send day in this controlled
no-occurrence-delay experiment. The fourteen-day settlement lag controls
availability, not cash-flow attribution. These paths are contribution economics,
not bank-balance forecasts. Acquired cost is deducted at age0, send costs run
through365, and revenue paths extend to the finite policy horizon425. The model
does not invent post365 sends. A supplied known contracted CPC multiplier is a
policy input applied once to predicted net CPC, with no additional fitted
unconstrained parameter for that multiplier.

## Proper frozen hyperpriors and posterior computation

The model has a finite discrete hyperprior; its probabilities are not fitted
empirical Bayes estimates, and it is not presented as a converged quadrature of
a continuous scaled-Beta prior. Conditional Gaussian posterior calculations
and marginal-likelihood grid weights are exact for the APPROXIMATE Gaussian
likelihood. Defaults are frozen in `ModelConfig`:

| Quantity | Prior |
|---|---|
| Global response log location | Normal(log(.012),1²) |
| Global net-CPC log location | Normal(log(.65),1²) |
| Independent nonhierarchical intercepts | Same location prior, independently |
| Bounded-feature coefficient | Normal(0,.5²) |
| Source SD | {.12,.40,.85}, masses {.25,.50,.25} |
| Cohort SD | {.10,.30,.65}, masses {.25,.50,.25} |
| Calendar innovation SD per √day | {.004,.015,.035}, masses {.35,.45,.20} |
| NB daily event dispersion r | {8,40}, masses {.5,.5} |
| Audit label fraction smoothing | Beta(1,1) |

There are54 response grid components and27 payout components in the full
dynamic model. The static hierarchy has18/9. Normal observation covariance is
diagonal, conditional priors are diagonal, and inference computes the precision
matrix X'WX+D⁻¹ and its Cholesky factor. The integrated marginal likelihood
includes the prior determinant, data determinant and normalizing constants.
Prediction samples a grid component, then the whole joint conditional Gaussian
parameter vector. There is no iterative sampler, no MCMC ESS/R-hat and no
convergence claim. `grid_probability_ess` is simply 1/Σ grid_probability²;
it measures concentration over the finite support, not sampling efficiency.

`ModelConfig` makes prior sensitivity executable. Alternative source/cohort
supports must remain in (0,1.5], calendar supports in (0,.1], and NB dispersion
in [.25,100], with positive normalized probabilities. Smaller/larger scientifically
plausible grids can therefore be refitted without changing data or using
held-out outcomes to tune a fit. The separate evaluator should report sensitivity.

All log locations/effects are Normal with finite fixed or bounded variance.
For every finite calendar date, sums of bounded-variance innovations also have
finite variance. A finite mixture preserves the exponential moments
E(exp(qX))=exp(qμ+q²σ²/2). The conditional trend scenario below is a linear
function of Normal states and likewise has finite variance at every finite
horizon. NB dispersion stays away from zero; Gamma marks have finite moments
given their lognormal mean. Thus first and second monetary moments exist over
the finite policy horizon. Finite moments DO NOT imply stable Monte Carlo means.
Returned diagnostics include the share of total simulated revenue supplied by
the largest1% of draws; evaluation must compare draw budgets and prior grids.

## Conditional future calendar scenarios

Only the historical calendar states and their innovation scale distribution are
learned. Future external conditions are unknown; all future paths are conditional
on one explicitly named policy:

- `hold_current`: continue the sampled latent weekly endpoint spanning the
  cutoff, then draw shared zero-mean weekly innovations with sampled q and
  interpolate the whole path on the same fixed calendar grid
- `continue_recent_trend`: derive EACH draw's last28-calendar-day slope from
  its posterior historical states, then add that slope after the cutoff to the
  shared weekly path. The cutoff state is evaluated at its interpolation basis,
  not taken as the sum of latent increments when the endpoint lies in the
  future. This is a conditional extrapolation rule. It is not a
  learned drift parameter, a prediction of future deterioration duration, or
  knowledge of when the simulated squeeze stabilizes
- `fixed_price`: future payout calendar state is frozen at the learned
  calendar-zero baseline, and future response uses `hold_current`. Actual
  historical monetary observations remain in every scenario. This comparator
  can be optimistic once current conditions have deteriorated

These rules introduce scenario/model uncertainty that a within-scenario interval
cannot resolve. The model never receives evaluator future calendars/contracts.
Higher uncertainty under extrapolation can generate large lognormal tail draws;
the evaluator should disclose this rather than clip or silently normalize them.

## Returned arrays and interpretation

`margin_paths`, `revenue_paths`, `human_count_paths`, and `net_human_cpc` have
shape draws×targets×(horizon+1). `calendar_dates` indexes absolute calendar day.
`calendar_response` and `calendar_cpc` have shape draws×calendar dates and contain
LOG state deviations from calendar0. `*_factor` arrays are their exponentials.
Calendar-state/amplitude intervals are posterior parameter intervals; margins
include future count/mark noise and are posterior PREDICTIVE paths conditional
on the scenario. Source/cohort amplitudes and count-dispersion draws are also
returned for checks. The evaluator must compute an unconditional first-payback
CDF across ALL draws, keeping never-crossing paths at infinity, and distinguish
first payback from terminal positivity and durable payback.

This bounded approximation addresses actual fitted hierarchy and shared paths,
while leaving learned age shapes, unknown calibration instruments, joint
response/value covariance, response-occurrence delay, signed reversals,
operational attrition and a full latent count/audit/payment likelihood for future
work. Claims of improvement require held-out world-level evidence and all
approximation/conditional-policy limitations above must accompany it.

## Reproducible implementation and prior checks

From the extension root, run:

    OPENBLAS_NUM_THREADS=1 python -m unittest -v dynamic_lab.test_model

All twelve tests pass on the development container. They include
sixty finite populated fits (six scenarios×two cutoffs×five modes), unchanged
target/source parameter distributions after excluding unrelated blocks, copied
input mutation isolation, late audit/payment clock masking, identical prelaunch
predictions for equally cheap good/bad sources, shared calendar reconstruction,
all three conditional policies, and common-random accounting monotonicity.
The source-free amplitude check used12,000 draws: response log-amplitude
variance1.4092 versus analytical1.41735, retaining global+source+cohort variation.
The launch regression also checks that cutoff0 has nonzero FUTURE calendar
uncertainty from the declared q hyperprior while calendar0 remains anchored.
Exact covariance checks compare days9/10 at cutoffs10/14/15/21. An additional
same-data regression verifies past-state posterior means, variances and grid
weights are unchanged when a later cutoff adds unobserved calendar knots.
These are software/component checks, not model calibration or held-out scoring.

Reproduce the separate prior-tail and scale-sensitivity evidence with:

    OPENBLAS_NUM_THREADS=1 python -m dynamic_lab.test_model --prior-tail-check dynamic_lab/prior_tail_summary.json

This creates a summary-only JSON with the fixed configuration, seed1777,
no observed rows, cutoff1, one100-exposure new cohort,425-day economic horizon,
sends through365 and the zero-drift `hold_current` scenario. The analytic mean
integrates all discrete source/cohort/calendar scales and Gaussian locations.
No held-out outcomes enter this check or its alternative scale supports.

| Scale-support multiplier | Draws | Mean lifetime revenue | Analytical mean | MC standard error | Revenue share from largest1% of draws |
|---|---:|---:|---:|---:|---:|
| .5 | 8192 | 58.501 | 59.036 | 1.979 | 21.7% |
| 1 | 512 | 68.704 | 84.397 | 8.850 | 23.2% |
| 1 | 2048 | 79.791 | 84.397 | 4.691 | 21.1% |
| 1 | 8192 | 83.436 | 84.397 | 4.908 | 30.6% |
| 1.35 | 8192 | 126.683 | 129.587 | 11.831 | 40.2% |

The small-budget mean is visibly unstable and the alternative priors materially
change prior expected revenue. Default posterior intervals in held-out replay
need their own scoring; these finite-prior computations do not establish
accurate monetary expectations, a pooling benefit, or a decision advantage.
