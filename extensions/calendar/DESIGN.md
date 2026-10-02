# Learning across arriving email cohorts

Status: initial broader design, retained for comparison with the subsequently implemented bounded approximation. The actual fitted extension is specified in [dynamic_lab/MODEL.md](dynamic_lab/MODEL.md); not every latent-count, audit or payment mechanism proposed below was implemented. Its results and limitations are in [dynamic_lab/EXPERIMENT.md](dynamic_lab/EXPERIMENT.md). The original v1 and its failed coverage checks remain frozen.

## What the current experiment actually learns

The first cohort w00 has no older cohort history at launch. Later forecasts also use the observations available from younger cohorts in the same source. At ages 3, 7, 14, 30 and 60 the existing seed-211 run has 1, 2, 3, 5 and 8 visible cohorts per source. Every counted cohort contributes usable exposure. The positive-payment module has no mature marks at 3/7/14, only w00 at 30, and w00–w04 at 60. Different modules therefore learn from different effective histories.

The current response model partially pools cohorts within each source. Source fits start independently from identical fixed priors. There is no learned global distribution across sources, no evolving calendar state, and no forecast for an unobserved source. The original article charts cannot establish the value of pooling: they contain no target-only counterfactual. The separate one-world development replay in this packet executes that comparison but is insufficient to establish a pooling benefit.

## Deliver in bounded stages

1. **Information contract and falsifiable fixtures, this packet.** Create an immutable aggregate observation stream with weekly acquisitions, revisions, reporting delays, audit uncertainty, dated acquisition quotes and a later cheap source. Replay calendar cutoffs into content-hashed snapshots. Verify exact input selection, purchase-cost immutability, shared future paths and accounting monotonicity. These are software/data-contract checks, not inference or calibration evidence.
2. **Existing-model information ablation.** At identical calendar cutoffs and target ages, compare v1 with only target records versus all visible same-source records. Use a fixed original prior in both cases, the same seed schedule and evaluator-only outcomes. Include an input census by likelihood module. Keep truth and later records inaccessible until predictions are sealed. This measures the existing pooling mechanism; it cannot establish the benefit of a new global hierarchy.
3. **Static global→source→cohort model.** Implement actual inference with fixed age shapes first. Transfer a predictive distribution to genuinely unseen sources, with feature covariates specified before held-out evaluation. Compare target-only, globally pooled, source-only and static hierarchical alternatives using the same observations and likelihood. Use a stationary generator aligned with this model before testing misspecification. Do not label a hand-set prior as learned global information.
4. **Shared calendar response and monetary states.** Add a constrained time process and jointly forecast all active and prospective cohorts. Run one declared calendar deterioration scenario and a stationary control. Then introduce good and bad cheap-source arrivals plus bot/measurement stress. Each extra mechanism earns its own ablation; do not fit all additions at once and attribute an improvement to hierarchy.

Stage 1 can proceed without choosing a sampler. Stages 3–4 require a separate inference implementation and reviewed computational diagnostics. This packet contains no substitute fitted values.

## Proposed probability model

Indices: source s, cohort c, acquisition calendar date b_c, calendar date t, cohort age a=t−b_c. E_ct is eligible accepted exposure, observed with its own availability clock. Counts are response **events**, so repeat responses can exceed the number of leads.

For an anchored nonnegative age curve f(a;w_s), set f(0;w_s)=1. Use a fixed, declared library of early-drop and long-tail bases; initially fix their weights. A later model may learn source and cohort simplex weights through a global→source→cohort logistic-normal hierarchy with bounded effect scales. Its cohort coefficients are exchangeable without a birth-date trend, with a fixed reference log-ratio. That is a separately reviewed learned-shape extension, not an initial implementation shortcut. Do not add an unrestricted age drift.

```
log m_s = μ + βᵀ z_s + u_s,       u_s ~ Normal(0, τ_source²)
θ_c | m_s,k ~ Gamma(k, rate k/m_s)
ξ_ct ~ Gamma(r, rate r)
H_ct | θ_c,ξ_ct ~ Poisson(θ_c E_ct f(a;w_s) exp(g_t) ξ_ct)
```

Marginalizing ξ makes event counts negative binomial conditional on θ, with variance mean+mean²/r. The shared cohort amplitude induces dependence across days. This is distinct from the negative-binomial **predictive** count distribution produced by mixing a Poisson model over a Gamma posterior. If overdispersion is omitted in an initial controlled component check, state that explicitly; event-level independence is not an empirical discovery.

With latent complete human counts and ξ fixed, the useful conditional update survives:

```
θ_c | rest ~ Gamma(k + Σ_t H_ct,
                  rate k/m_s + Σ_t E_ct f(a;w_s) exp(g_t) ξ_ct)
```

With partially observed responses, either integrate the report likelihood correctly or sample the latent counts. The global/source/shape/calendar posterior is new inference. Reusing the conditional Gamma formula alone does not fit it.

Global priors are fixed before replay. Use proper priors for μ, β, source variability, cohort dispersion and count dispersion. With only a few sources, τ_source is weakly identified: show prior sensitivity and leave-one-source-out predictions. The Gamma cohort amplitude is centered at m_s on the multiplicative mean scale, not at log m_s on the log scale; its distribution has no free acquisition-date trend. Parameterization constraints must not erase new-source or new-cohort variance.

### Predict a source that has no observed cohort

At a cutoff T, draw hyperparameters from p(μ,β,τ_source,k,w,… | D_≤T), draw a **new** u_new from its population distribution, then draw a new cohort amplitude conditional on that source. Integrate all three uncertainties. Do not copy an old source's posterior, use only hyperposterior means, or impose an existing-sources sum-to-zero projection on the new effect. This applies the usual posterior-predictive integration principle to the proposed hierarchy; see the [Stan posterior predictive sampling guide](https://mc-stan.org/docs/stan-users-guide/posterior-prediction.html).

Covariates z must be information known before the purchase: acquisition channel/format or other prespecified observable descriptors. The price quote enters acquisition economics. Being cheap is not a causal quality feature. A price feature may be considered only as a separately validated observational predictor with support/shift checks. Test equally cheap good and bad new sources so transfer cannot assume one answer.

### Calendar changes and positive revenue

Use a shared state x_t=(g_t,v_t), with g driving human response intensity and v driving net payment opportunity. Anchor x at the declared reference date. Begin with a regularized bivariate random walk or local-level process; innovations have a sampled covariance. This allows response and payout to deteriorate together without asserting that one causes the other.

```
x_t = x_(t−1) + ε_t,     ε_t ~ Normal(0, Σ_calendar)
log positive-payment mean_c,t = α_global + γᵀ z_s + α_source,s + α_cohort,c + v_t
α_source,s ~ Normal(0, τ_pay_source²)
α_cohort,c ~ Normal(0, τ_pay_cohort²)
```

For a new source and cohort, jointly draw payout hyperparameters from their hyperposterior, then a fresh α_source,new, then a fresh α_cohort,new. Retain uncertainty in positive-payment probability and reversals as well. For an existing source/new cohort, retain the source posterior and draw a new cohort residual. Do not reuse the realized residual of an old cohort. Shared future v_t is indexed by the actual future calendar date b_new+a, not by age alone.

Precisely, the response intensity above is indexed by **send date** t and send-age a. A sampled response delay yields occurrence date u=t+d_response. The payout opportunity/state and applicable CPC contract use **occurrence date u**; settlement and later report dates are accounting/visibility clocks, not additional draws of commercial opportunity. Thus η_c,u uses v_u. A different contract basis would be a separately declared policy. The same future x trajectory supplies g_t and v_u across all cohorts and delays.

Positive-payment probability, positive amount and signed reversals need separately defined likelihoods. A first monetary component uses a Bernoulli paid indicator, a Gamma positive amount with shape κ_pay and mean exp(η_c,u), and a Bernoulli reversal indicator with a Beta reversed fraction. Initially the paid probability, reversal probability and fraction parameters are global, while the positive amount has the explicit source/cohort hierarchy above; this sharing choice is disclosed and ablated before expanding them too. The settled net mark is positive_amount×(1−reversal_fraction) after a reversal, with zero for an unpaid valid human response; separate signed entries preserve accounting/report clocks. Other commercial negative fees require their own declared model. These assumptions deliberately keep this first component smaller than all possible contracts.

The following **candidate prior family** has finite moments at the finite policy horizon and must be frozen numerically before inference: global log-rate/log-value locations and feature coefficients are Normal with fixed finite variances; features are bounded observed constants; source and cohort Normal-effect standard deviations have support [0,1.5]; calendar innovation standard deviations have support [0,0.1] per day with correlation restricted to [−0.95,0.95]; Gamma shape/dispersion parameters have compact positive support [0.25,100]. Use proper scaled-Beta densities on these supports, independent fixed-scale Normal location priors, and a proper bounded correlation density. Bernoulli probabilities use proper Beta priors, and reversal fractions stay in [0,1]. The particular bounds are declared modeling choices, not inferred commercial facts or tuned to held-out coverage.

Conditional Normal exponential moments are exp(qμ+q²σ²/2). Bounded variance parameters give a finite upper bound on this expression after mixing at each finite future date; summing finitely many state innovations preserves that property. Gamma moments exist with shapes bounded away from zero and lognormal mean moments finite. Consequently finite-horizon count/payment moments exist under this candidate hierarchy. This does not establish that 500 Monte Carlo draws give stable means. Derive and check the combined prior predictive, simulate tail contributions and compare smaller/larger scientifically plausible bounds before decision claims. Merely calling a prior proper is insufficient. Do not use an unbounded inverse-Gamma variance or Student-t log price without an exponential-moment analysis. Existing unstable prior-mean results remain visible.

Known contracted/current CPC schedules are inputs available at T. A free latent multiplier must not duplicate an exactly known rate. Model uncertain acceptance/value deviations where relevant. For future dates, separate a contracted schedule from a stochastic forecast or explicit scenario. Never pass simulated future contracts as observed facts. Old acquisition costs remain fixed in the signed economic record. A later quote affects only a prospective purchase or a cohort actually acquired at that quote.

For every posterior draw, sample **one** future calendar state trajectory and reuse it across all cohorts. Sum their economic paths within that draw. Independently resampling a calendar shock for each cohort would invent diversification and understate portfolio risk.

## Identification and measurement

Because t=b_c+a, three unrestricted age, calendar and acquisition-date trends are not identifiable. Overlapping cohorts alone does not solve this. The initial design fixes/anchors the age curve, anchors calendar levels, and restricts cohort residuals to exchangeable centered effects without a cohort-date trend. A learned-age extension uses a constrained basis with proper shrinkage and an explicit design-matrix rank/null-space check. Repeat fits under plausible alternate anchors. Decomposed drivers remain assumption-dependent even when outcome prediction works. Random effects also impose identification constraints rather than making the issue vanish; see [Constraints in Random Effects Age-Period-Cohort Models](https://arxiv.org/abs/1904.07672).

Raw activity combines latent human H_ct and bot B_ct events. A concrete known-instrument Stage 3 likelihood draws H from the response model and B from a separate NB nuisance process. Given event type, response and report delay bins follow a declared categorical distribution. Only bins whose report dates are at or before T are visible; the unreported category is integrated or augmented. Conditional on type, the production classifier reports human with probability Se for a human event and 1−Sp for a bot. These probabilities are fixed known instrument inputs in the initial controlled model, rather than learned from simulator truth. This simplification is labeled as such.

In Stage 4, Se/Sp and delay probabilities receive proper Beta/Dirichlet priors and learn from independent calibration records plus known-probability audit selections. Draw selection A~Bernoulli(π_stratum) within logged classifier/time strata. An independent delayed audit label uses its own measured error matrix; audit labels are not automatically truth. Aggregate the joint categories (latent type, report-delay bin, classifier result, selection, audit result/arrival) into a single multinomial/Poisson-thinning likelihood conditional on H/B. Fit their joint likelihood or augment the latent cells; do not treat raw counts, classifier labels and audited subsamples as independent replicated data. Revision records supersede a previously reported label instead of introducing a second event.

Independent error-rate calibration or justified informative priors are required to learn two imperfect classifiers without assuming identifiability. Store π, classifier version and label-arrival times. Nonrandom audits with unknown selection remain a sensitivity case. If audit evidence cannot distinguish declining humans from increasing bots or classifier drift, show sensitivity bounds and call the components unidentified. Unlabeled and unreported events are missing information, not zero humans. Raw bot traffic receives no automatic revenue credit; economics use qualified settled payments and explicit signed corrections.

A chart of distinct human responders has a different numerator and an explicit original-lead or eligible-lead denominator. Give it a Binomial/Beta-Binomial measurement model if it becomes a fitted outcome. Do not multiply a distinct-person likelihood and a repeat-event likelihood as independent evidence when both come from the same events. Either build a coherent joint model or retain the unique-human metric as a separately audited descriptive series.

## Immutable calendar replay

Each input has effective/occurrence time, observed time, immutable record ID and revision information. At T, select only facts that occurred/effected by T and were observed by T. Observation versions replace the same observation; signed monetary corrections add distinct economic entries. Maintain those two semantics separately. Cache predictions by a canonical snapshot hash, target, policy, prior/model configuration hash and draw seed. Returned snapshots expose immutable values, not mutable DataFrames hidden inside a frozen dataclass.

At each origin refit **all** included as-of data from the same fixed hyperprior. Never use yesterday's posterior as today's prior while including yesterday's likelihood again. A future incremental algorithm must process disjoint new information and handle revision retractions explicitly. It is outside this initial scope.

Forecast APIs receive only the snapshot and declared future policy/scenarios. The evaluator receives complete simulator truth only after predictions and input hashes are sealed. Adding a late report may change a later origin; it must not change an earlier origin, cache identity or archived prediction. Acquisition continues weekly while existing cohorts mature. Score a matrix of calendar dates and cohort ages, rather than a single cohort age plot with a new label.

## Evaluation and chart acceptance

Freeze design/tuning seeds separately from held-out worlds before looking at comparative outcomes. Retain old 12-world failures as the original model's results. New experiments get separate versioned outputs. Calibrate a correctly specified stationary scenario before stress scenarios, then test calendar revenue squeeze, acquisition inflation, human decline/bot rise, and new cheap good/bad sources. Use independent worlds as resampling clusters, not correlated cohorts/forecasts.

Compare target-only, global complete pooling, source-only, static partial pooling and dynamic hierarchy with the same information clocks. Report prospective coverage, widths, Brier scores for contribution/payback, CRPS, explicit decision loss and calendar shift detection lag. Distinguish full model performance from a component sampler check. Show Monte Carlo uncertainty, effective sample size/convergence diagnostics where the chosen algorithm needs them, prior predictive moments and heavy-tail stability before expected-value policy claims.

Acquisition loss concerns the next purchasable batch with the quote visible at the decision. Sending loss concerns future avoidable costs/revenue for an already purchased cohort. Preserve unconditional first-payback CDFs with every draw, including paths without a crossing by age425; keep first payback, positive contribution at180, and continued positivity separate. A common-random-path accounting check must show that reducing known acquisition cost or increasing nonnegative human net payout cannot delay first payback, holding every other mechanism fixed. This is an accounting monotonicity claim, not a claim about behavioral or refund changes.

Required visual evidence includes calendar acquisition arrivals, evolving information counts, common-age predictions for new cohorts, new-source prior/likelihood/posterior separation, shared response/value calendar states with identification caveats, and held-out calibration/decision comparison. Every figure must identify calendar date versus cohort age, available data scope, posterior parameter interval versus predictive interval, policy horizon, numerical denominator and immutable output rows. Labels alone cannot substitute for a implemented model or a replay result.

## Implementation status

The information-scope clarification and unchanged-v1 twenty-forecast replay are retained in REPLAY_CHECKPOINT.md. The later user-requested implementation now fits a smaller dynamic hierarchy with Gaussian log-measurement approximations and an integrated discrete hyperprior, rather than the full latent likelihood proposed above. Its six scenarios, actual held-out scores, failed nominal coverage, shared portfolio paths and offered-cost query are separate versioned artifacts. Follow dynamic_lab/MODEL.md for the implemented specification and dynamic_lab/VALIDATION.md for the checked boundaries.
