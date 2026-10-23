# The fitted cohort model

This extension fits a bounded Bayesian approximation to fictional email-cohort data. It separates subscriber contactability, human engagement, repeated event counts and net CPC; hierarchical modes share evidence at population, source and cohort levels. The model has shared calendar response and payout states. It is not an exact subscriber-level posterior: the fitted likelihood uses Gaussian approximations to aggregate measurements, and it does not jointly infer human status, survival, engagement, audits and payments. See `DESIGN.md` for planned extensions that are not implemented.

## Decision quantity and information boundary

`fit(data, cutoff, mode='dynamic_hierarchy', seed=0, config=None)` fits only the declared observation columns available by the cutoff. `predict(targets, draws=256, horizon=425, scenario='hold_current', seed=None, cpc_shock=None)` returns posterior and predictive paths. The function snapshots its input, filters delayed audit and payout records by their own clocks, and never imports simulation truth. A target gives its cohort/source IDs, acquisition day, acquired leads, total acquisition cost, and either a total daily send cost or send cost per contactable address. Existing cohort contactability observations anchor forecasts; a new batch starts with its acquired leads. Acquisition cost is deducted once. Per-address send costs apply only to addresses still contactable and only through send age 365.

The principal profitability quantity is defined at a fixed horizon and against an explicit buffer:

\[
p_{c,T}=P\!\left(M_c(180)/N_c>\$0.02\mid D_{\leq T},\text{future policy}\right).
\]

Here \(M_c(180)\) is cumulative contribution at cohort age 180 and \(N_c\) is its original acquired lead count. For a prospective purchase, the demonstration rule buys at \(p\geq .95\), skips at \(p\leq .05\), and otherwise waits. For an already acquired cohort the same cutoffs describe “likely above target,” “likely below target,” or “inconclusive”; they do not recommend buying it again. A posterior threshold is not proof of empirical reliability. The held-out replay measures false declarations, reversals, undecided share and delay. The code does not model the value of waiting, a lost offer, later bidding or an optimized stopping policy.

First crossing of zero by age 425 is a separate quantity from day-180 profitability. A path can cross zero and later fall below it. A path with no crossing by 425 remains in the denominator; the finite policy says nothing about infinite lifetime.

## Subscriber and cohort data

The simulator carries persistent subscriber identity internally, with a fixed human/bot status, a person-specific engagement frailty, and an absorbing contactability exit. Every acquisition cohort has a different survival half-life. Eligibility is recorded separately from activity: an inactive human may respond later, a bot can remain contactable, and zero clicks do not mean churn. People may produce repeat clicks, so raw event counts are not unique-person counts. The exported observation rows contain cohort-day aggregates: accepted contactable exposure, raw events, daily raw unique responders, separate event and unique-person audits, settled human events and net revenue. Subscriber traits, latent rates, and oracle expected outcomes are held in evaluator-only objects and excluded from fitting.

For measurement, the model builds these distinct streams:

1. **Contactability:** measured exposure losses over observed person-days estimate a piecewise-exponential cohort exit hazard. It is partially pooled across sources and cohorts. The fitted model treats the hazard as constant within a cohort; it does not learn calendar-time deterioration in unsubscribe/bounce risk.
2. **Unique engagement:** audited daily unique responders per contactable exposure estimate the rate that distinct people respond. The model uses this as a separate likelihood stream from repeat event counts. It does not deduplicate identities in its posterior likelihood across days.
3. **Event response:** raw event counts, corrected for a representative sample of audited events, estimate human event rate per contactable exposure. Repeated events are allowed. Audits are nested samples, not independent extra response counts.
4. **Net CPC:** mature settled net revenue divided by commercially verified human events estimates net value per event. Negative daily receipts and partial settlement are outside this unsigned bounded model.

For contactability, the simulator samples an exit age from a geometric lifetime with cohort-specific hazard \(\lambda_c\); expected contactable share is \(S_c(a)=\exp(-\lambda_c a)\). Observed risk sets contribute a log-hazard approximation, using exits plus 0.5 over person-days. This is not an exact discrete survival likelihood and treats observations as final daily exposure snapshots. Pending future exposure is right-censored by the cutoff, not turned into churn.

The engagement generator is heterogeneous: a cohort receives source and cohort rate effects, and each person has persistent Gamma frailty. Human response occurs conditionally on being contactable; repeat-event counts follow an overdispersed process. A separate bot-event process can change over calendar time. Inference sees raw response events and imperfect delayed audits, not human/bot truth. The model’s unique-responder approximation smooths the audited human share and fits a log response rate. It does not jointly model a person’s repeated history, so its uncertainty can be miscalibrated when person-level frailty is strong.

## Hierarchy and pooling

The fitted log-rate structures are:

\[
\begin{aligned}
\log \lambda_c &= \mu_\lambda+\beta_\lambda z_s+u_{\lambda,s}+e_{\lambda,c},\\
\log r^{(event)}_{c,t} &= \mu_R+\beta_R z_s+u_{R,s}+e_{R,c}+g_t,\\
\log r^{(unique)}_{c,t} &= \mu_U+\beta_U z_s+u_{U,s}+e_{U,c}+h_t,\\
\log CPC_{c,t} &= \mu_V+\beta_V z_s+u_{V,s}+e_{V,c}+v_t.
\end{aligned}
\]

Each process has its own global, source and cohort effects. The configured source feature \(z_s\) is a bounded pre-acquisition descriptor; source price is not a quality feature. Global/source/cohort effects are Normal with finite, discrete scale mixtures. The source and cohort populations are exchangeable: there is no fitted acquisition-vintage trend in these effects. Contactability has no calendar state. Event response, unique engagement and payout have separate shared weekly calendar states anchored to day zero. The model does not learn covariance between these processes.

| Mode | Fitted information sharing |
|---|---|
| `target_only` | Separate cohort intercepts; other cohorts do not update this cohort. |
| `source_pooling` | A source intercept pools that source’s cohorts; sources do not share learned population effects. |
| `global_pooling` | One intercept across sources and cohorts; no between-source or cohort prediction spread. |
| `static_hierarchy` | Global, feature, source and cohort effects with partial pooling. |
| `dynamic_hierarchy` | Static hierarchy plus shared event-response, unique-engagement and payout calendar states. |

For a new cohort at a known source, forecasts retain the fitted source distribution and sample a fresh cohort residual. For a new source, they sample global, feature, source-scale and cohort-scale uncertainty, then a fresh source and cohort effect. A cheap quote alone cannot update quality. Separate datasets that share visible prelaunch information receive the same prior predictive distribution.

Every cutoff refits from the same declared hyperprior and currently visible rows. It does not feed yesterday’s posterior back as a prior while counting yesterday’s data again. This gives a coherent sequence of snapshots, but not a sequentially optimized decision policy.

## Observation approximations

The default synthetic audit assumptions are 50% event and unique-responder sampling, sensitivity .95 and specificity .98, with audit reports arriving seven days after activity. Raw activity reports arrive after two days; mature monetary/count summaries after fourteen. For event and unique counts, the likelihood pools evidence within cohort and calendar week, applies a Beta(1,1) smoothing prior to the audited label fraction, corrects for known sensitivity/specificity, and uses a half-count log-rate approximation. The audit is assumed representative and its accuracy known. Nonrandom review selection, changing instrumentation and within-week quality shifts are not learned.

The event-count measurement is approximately

\[
y=\log\frac{\widehat H+0.5}{\sum_d E_d},\quad
Var(y)\approx\frac{\widehat H+0.5+R^2 Var(\hat p_H)}{(\widehat H+0.5)^2}
+\frac{\sum_d E_d^2}{(\sum_d E_d)^2 r},
\]

where \(R\) is raw events, \(E_d\) is contactable reported exposure and \(r\in\{8,40\}\) is an overdispersion support. The same delta-method structure is used for unique responders, without treating the daily identity sets as independent people. This is one approximate response likelihood: raw counts and their audit subsample are not multiplied as independent evidence. Half-count smoothing gives a zero-event bin variance of 2 before overdispersion, instead of a falsely precise numerical floor.

The contactability likelihood groups exposure snapshots into approximately weekly risk intervals. It uses the decline from starting to ending active addresses and observed person-days as an offset. The Gaussian log-hazard measurement has variance approximately \(1/(deaths+0.5)\), floored at the configured measurement variance. This approximation discards some daily survival detail and does not model audit error in eligibility counts.

Mature CPC bins use \(\log((revenue+0.005)/human\ events)\), approximate variance \(CV^2/n\), configured \(CV=1.5\), and a conservative floor; zero-revenue settled bins get broad variance. This is not an explicit unpaid-payment probability, reversal ledger, or Gamma mark likelihood in inference. Negative receipts are rejected.

Conditional Gaussian posterior calculations are exact for these approximate Gaussian measurements. Hyperparameters are integrated over finite proper priors, not tuned to held-out outcomes. Key default supports include source SD \(.12,.40,.85\), cohort SD \(.10,.30,.65\), calendar innovation SD \(.004,.015,.035\), and event dispersion \(8,40\), with configured probabilities in `ModelConfig`. There is no iterative MCMC sampler, so R-hat/ESS are inapplicable. Grid probability ESS measures finite-grid concentration only.

## Calendar and price scenarios

The shared log-state process has weekly Normal increments on a fixed global knot grid, variance \(q^2\times7\). It is not a fitted free drift. A partially observed endpoint can lie after the cutoff; that latent endpoint is preserved and later innovations use the same grid. The day-zero state is anchored. Response, unique engagement and payout calendars are modeled independently, so correlated market shocks remain unrepresented.

`hold_current` means no deterministic future drift plus shared future weekly innovations. `continue_recent_trend` adds each posterior draw’s recent 28-day slope as a conditional extrapolation. `fixed_price` freezes future payout to the learned day-zero reference while response uses hold-current. None of these is knowledge of future truth.

The optional `cpc_shock={effective_day,drop_fraction,ramp_days}` is a controlled what-if. It must begin after the information cutoff, multiplies only future CPC after the stated effective day/ramp, and does not alter people, retention, event counts, purchase price or sending cost. Use the same fitted model and random seed for baseline and shock to create paired paths. A CPL what-if changes only the prospective cohort’s total acquisition cost. Existing cohorts’ costs remain immutable.

## Prediction arrays, truth and limitations

The returned draw arrays separate realized predictive quantities from expected rates: `contactable_count_paths` contains future survival draws, `expected_contactable_count_paths` its fitted conditional mean, `unique_human_count_paths` future realized distinct responders, and `expected_unique_response_paths` rate expectations using expected contactability. `human_count_paths` are event counts, not people. Revenue and cumulative margin include count/mark noise and booked past amounts. Calendar/amplitude bands describe parameter states; contribution intervals are posterior predictive intervals.

The evaluator separately exports oracle expected contactability/engagement, realized unique responders, realized economic paths and as-of posterior forecasts. Oracle expected curves must be compared with matching expected-rate bands; realized events with predictive distributions. Hidden truth must never be supplied to `fit`. Synthetic human/bot labels are known to the evaluator only.

Remaining limitations include approximate aggregate likelihoods, no joint posterior over subscriber traits, no long-memory subscriber response model, known audit quality, an unsigned payment model, no learned response/CPC dependence, no calendar-varying retention hazard, and conditional rather than optimized future purchase policy. A smaller held-out demonstration can reveal these failures; it cannot establish production calibration or business results.

## Reproduction

From `extensions/calendar`, install the declared Python requirements and run:

```sh
OPENBLAS_NUM_THREADS=1 python -m unittest -v dynamic_lab.test_simulation dynamic_lab.test_model_contract dynamic_lab.test_model
python -m dynamic_lab.evaluate --phase all
python -m dynamic_lab.figures
```

The figures are rendered from sealed synthetic outputs and include plotted numeric tables and source hashes. The article remains a draft; it is not deployed to nickroth.com.
