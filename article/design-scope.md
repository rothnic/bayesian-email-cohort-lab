# Design scope: Bayesian cohort decisions and visual explanations

Status: proposed design for the next implementation. No new fitted results, chart renders or validated decision thresholds are claimed here.

This scope builds on the [source review](review-2026-10-20.md). The objective is to show how evidence changes a business decision: acquire another batch, negotiate its price, collect more evidence, or continue/stop sending to an already acquired cohort. Existing accounting experiments and failed calibration results remain available separately.

## 1. Keep business drivers separate

CPC here means net earnings per qualified human click, not acquisition spend per click. Acquisition uses cost per lead (CPL). These units must appear explicitly on charts.

The business flow is:

~~~mermaid
flowchart LR
  A[Acquired leads] --> B[Contactable subscribers]
  B --> C[Qualified human engagement]
  C --> D[Valid human clicks]
  D --> E[Earned revenue]
  V[Net earnings per valid click] --> E
  E --> P[Contribution]
  A --> AC[Acquisition cost]
  B --> SC[Attempted-send cost]
  AC --> P
  SC --> P
~~~

Contactability, probability of any response, repeated-click frequency and earnings per click are different quantities. In particular, a CPC shock must not silently change the other three.

For one cohort's cumulative contribution through the chosen horizon, a useful accounting identity is:

    Revenue = valid human clicks × net earnings per valid human click
    Contribution = revenue − acquisition cost − attempted-send cost − other included costs.

Net earnings per click is click-weighted. It is not the unweighted mean of subscriber or cohort CPCs. Across heterogeneous people, compute expected revenue by summing person-level expectations or using correctly conditional/weighted rates. Do not multiply unrelated posterior means: calculate each outcome from a matching joint posterior draw, then summarize outcomes.

Deduct acquisition cost once. Attempted-send costs include unsuccessful attempts; the sending policy must use eligibility information available at that time, not hidden future contactability.

Bots may remain contactable, receive mail, incur costs and produce raw telemetry. They do not automatically disappear from sending costs because they are excluded from qualified human revenue.

## 2. Subscriber, cohort and source definitions

- **Person:** stable synthetic ID, acquisition episode, source, acquisition time, contactability history and activity/payment/reporting records.
- **Cohort:** source × acquisition week. A person has one acquisition episode in the basic experiment; resubscription/reacquisition requires an explicit attribution rule.
- **Source:** a population of cohorts, with source-level behavior and value effects learned from observations.
- **Acquisition-month group:** a reporting rollup of weekly cohorts. Changing the acquisition grouping does not automatically change the activity window.
- **Calendar environment:** shared market and performance conditions experienced by multiple cohorts at once.

Use unequal cohort sizes and a mix of weak, strong and borderline performance. Keep persistent differences in subscribers and cohort decay behavior; redrawing independent noise around one known curve is insufficient.

### Proposed statistical components

| Component | Minimal generative model | Meaning and what is learned |
|---|---|---|
| Acquisitions and CPL | Prespecified acquisition schedule with unequal counts; observed quotes and fixed purchase charges | Price and volume are business inputs. Cheapness is not a quality observation. |
| Human/bot status | Bernoulli status with source/cohort variation | A persistent trait, not independently redrawn for each event. Begin with known qualification in a controlled case; later use imperfect audits. |
| Contactability | Survival time with a cohort-specific timescale; Weibull is one initial family | Unsubscribe/bounce is permanent exit in the basic model. Learn global/source/cohort parameters from eligibility histories and right-censored follow-up. |
| Human engagement | Bernoulli weekly response conditional on human status and contactability; hierarchical logit with persistent person frailty | Different people have different propensities. A zero week does not imply churn; a person can respond again later. |
| Repeat clicks | Zero when not engaged; otherwise 1 + negative-binomial repeat count | Separates unique responders from repeated events and permits bursty counts. Start with source/cohort intensity; add a second persistent person click-intensity trait if model checks require it. |
| Value per valid click | A separate positive monetary mark model, such as Gamma, centered on calendar/source/cohort CPC | Price variation is independent of activity in the controlled CPC-only experiment. Future market innovations may be added without fixing future prices in advance. |
| Costs and reporting | Actual attempted-send costs and acquisition charges, with occurrence/report/settlement clocks | Costs, revenue recognition and cash arrival are not interchangeable. Only reports available at the cutoff enter inference. |

A hierarchical lognormal distribution can give cohort retention timescales different positive values. Source/cohort engagement deviations and person frailty can use proper Normal distributions on logit/log scales. Use one varying decay parameter initially rather than many unidentifiable shapes. Estimate its population distribution instead of supplying the generator's true age curve.

The contactability and interest components require different evidence. Without eligibility records, the model generally cannot distinguish people who cannot receive mail from people who can receive it but do not respond. Event-only audits also cannot identify the human fraction among all purchased leads without additional assumptions or representative lead-level evidence.

An explicit active/dormant hidden Markov model is optional, not required for the first version: Bernoulli engagement with persistent person heterogeneity already permits inactivity followed by return.

The survival and event-count foundations are discussed in the primary [Stan survival guide](https://mc-stan.org/docs/stan-users-guide/survival.html) and [hurdle-model guide](https://mc-stan.org/docs/stan-users-guide/finite-mixtures.html#hurdle-models). The proposed 1 + negative-binomial count is specifically a shifted positive-count model, not a claim that it equals a zero-truncated negative binomial.

Introduce unpaid outcomes and signed reversals as a later accounting extension. Either model net marks directly, or separately model payment acceptance, positive amounts and reversals, deriving net CPC from them. Do not multiply by an extra payment probability when the net-CPC mean already incorporates unpaid clicks.

## 3. Bayesian learning that supports the decision

Use global distributions for retention, engagement, click intensity and value; source deviations; cohort deviations; and persistent person heterogeneity. New cohorts borrow information without losing their own predictive variation. A new source needs fresh source and cohort effects, not a copied successful source posterior.

The first controlled model uses known qualification and recorded eligibility. Later experiments add incomplete/noisy audits, delayed reporting and harder calendar shifts. Use likelihoods suitable for sparse and zero observations; do not rely on a convenient log-rate transformation being automatically calibrated. Choose scale priors that retain finite required monetary moments after integrating all hierarchical uncertainty. Inspect prior-predictive tails and Monte Carlo stability, not merely numerical finiteness.

Fit all as-of evidence jointly from the fixed hyperprior. An educational global/source-prior comparison can hold the target cohort's observations out, then show their update. Do not count those observations twice. Keep the age/calendar/acquisition-vintage identification restrictions explicit.

Keep outcome definitions fixed as evidence changes:

    p(c,T) = P[contribution at age H / original acquisitions > buffer
               | reports available at T, sending policy, future-price scenario].

Use H = 180 for the first comparison with the existing experiment. Terminal age-425 profitability is a separate export. First payback, terminal positivity, positive expected margin and durable positivity are distinct.

A 95%/5% declaration rule is a candidate policy to evaluate, not an already validated guarantee. Pair it with expected downside and the business costs of wrong decisions. Some cohorts should remain undecided. Reviews before H are forecasts; later reviews with incomplete reports are nowcasts.

A fresh purchase includes CPL. Continuing sends to an existing cohort compares incremental future receipts with avoidable costs because acquisition is sunk.

## 4. Five main decision views

### A. Which cohorts are weakening, and what does pooling hide?

Use two aligned age-based panels: contactable share and unique-human engagement per original acquired lead. Thin gray lines represent observed cohorts; the selected cohort is a thick charcoal line. Facet by source instead of assigning a rainbow color to every cohort. End each line at its available information boundary.

Pair this with weekly-acquisition, monthly-acquisition and all-history rollups, using small multiples with the same scales. Pool distinct-person numerators and original-acquisition denominators at the same age window. Show contributing cohorts and original-lead counts below the plot. Retain a recent-vintage comparison so a stable historical average cannot hide deterioration.

Keep a seven-day activity window while switching acquisition grouping. A true monthly-activity view requires deduplication over the entire month; it cannot sum weekly unique responders. Immature/unreported windows are missing, never zero. Comparing curve shapes may require a fixed mature cohort set.

### B. What might this particular cohort earn?

Use one selected cohort throughout. Stack a behavior-learning panel above a cumulative-contribution panel. X is cohort age; contribution Y is dollars per original acquired lead, with a clear zero line.

Before the information cutoff, show reported observations as charcoal points/solid segments and mark pending reports. After it, show the predictive median and two labeled central bands, initially 50% and 95%. These are pointwise quantiles, not a promise that 95% of entire paths remain inside the band at every age. Sampled paths must be temporally coherent joint draws. A small number of faint sampled future paths can be an optional overlay; never combine every path and every band by default.

In evaluation mode, reveal the latent expected engagement curve in the upper panel and the realized future contribution path below. They answer different questions and use different uncertainty comparisons.

### C. Has enough evidence arrived to act?

Plot the same cohort's probability of meeting the fixed target against successive information cutoffs, also labeled with cohort age. Use a 0–100% axis, candidate threshold lines, and a strip reading candidate profitable / insufficient evidence / candidate below target. Curves may rise, fall or widen in uncertainty as conditions change.

A single focal curve is the default. Optional global-only/source-sharing/cohort-informed comparisons use the same target, cost basis, cutoff and future assumptions. Display the first qualifying prespecified review and any later reversal.

Keep a short reliability warning beside the threshold until sequential evaluation supports it. Do not portray 256/256 sampled successes as certainty or a noisy Monte Carlo threshold crossing as a business discovery.

### D. What changes if CPC falls independently?

Use aligned baseline-versus-shock panels for future earnings per valid click and contribution. Mark the shock date and information cutoff. State explicitly that subscribers, contactability, engagement and valid clicks are unchanged in this isolated experiment.

Show a paired distribution of change in contribution at H, then a sensitivity curve: X is CPC reduction, Y is probability of meeting the profitability target. An expanded companion heatmap can put offered CPL on X, CPC reduction on Y, and profitability probability in each cell. A labeled candidate risk contour supports quote negotiation.

Each comparison names the affected future period, amount, step/ramp policy and all unchanged mechanisms. Scenario lines are conditional alternatives, not confidence-interval bounds.

### E. Which offer deserves attention now?

Use a compact source/new-batch board with current quote, proposed volume, available evidence, expected contribution per lead, predictive interval, probability above the buffer, expected downside and status. All rows share an information cutoff, purchase policy and outcome horizon.

Show insufficient evidence prominently. Allow inspection of the cohort forecast behind each row. Candidate buy/decline statuses require a validated rule; a suggestion to collect more data is not an optimal experimentation policy unless information value and pilot costs are modeled.

## 5. Visual grammar and layout

| Object | Appearance | Meaning |
|---|---|---|
| Reported observations | Charcoal solid marks | Information available to the analyst |
| Descriptive rollup | Thick charcoal line, directly labeled | A pooled statistic, not a Bayesian forecast |
| Learned latent performance | Blue estimate and credible band | Uncertainty about the underlying rate or parameter |
| Future outcome forecast | Blue predictive fan after the cutoff | Includes future event/payment noise under the stated scenario |
| Conditional what-if | Orange dashed line, directly labeled | An imposed alternative, not interval bounds |
| Latent simulator truth | Gold dotted line in evaluation mode | True underlying expected behavior for the defined population |
| Realized future truth | Dark dashed path or endpoint cross | One realized simulated outcome, revealed for evaluation |

Use only the relevant truth overlay in each panel. Prefer direct labels and aligned small multiples to long legends or dual axes. Keep observed cohort spaghetti and one-cohort predictive spaghetti in separate panels.

Default money units are contribution per original acquired lead; offer totals are separately labeled. Use plain percentages for engagement/probability and dollars or cents per valid click for CPC. Show denominator, source/cohort, information cutoff, horizon and future assumptions near the chart.

Use a light background, quiet grid lines, consistent typography and full-width article evidence plots. On narrow screens, stack panels and retain readable labels rather than shrinking type. Show only a few comparable series at once.

A companion can expose source/cohort selection, acquisition grouping, information-cutoff replay, target horizon, future-price assumptions and an evaluator-truth reveal. Cutoff replay must load actual as-of fits, not relabel a final full-data posterior as an earlier forecast.

## 6. What-if experiments and their controls

| Experiment | Change | Keep fixed |
|---|---|---|
| CPC-only squeeze | Net future value per valid click falls x% after a date, abruptly or gradually | Acquired people, contactability, response/click paths and historical earnings/costs |
| CPL-only increase | Quote for a prospective purchase | Same planned batch size, quality inference and all existing paid acquisition charges |
| Lower engagement | Probability of a human responding | CPC and contactability unless another change is declared |
| Higher churn | Contactability hazard/timescale | Conditional response propensity and CPC |
| Cheap new source | Its prospective quote; its quality remains unknown to inference | No hidden good/bad label supplied to the model |
| Combined stress | An explicit combination of the above | Any remaining mechanism, with joint dependence declared |

Use stored paths or independent/keyed random streams for people, contactability, activity and monetization. Merely reusing one overall seed is insufficient when changed branches consume random draws differently. For CPC-only shocks, reuse exactly the same posterior/person/click paths and standardized monetary noise, changing only the declared monetary mechanism. A separate fixed-budget scenario can let a CPL change alter purchase volume; that is not the pure CPL comparison.

A price change applies to clicks occurring after its effective date. A later settlement date does not reprice earlier clicks. If unpaid outcomes, fees or refunds are modeled, declare which amounts scale with price. Do not silently treat a shock to positive gross price as the same shock to expected net CPC.

Illustrative arithmetic only, not a lab result: with 1,000 valid clicks, net CPC of $0.20 and fixed included costs of $150, contribution is $50. A 20% CPC reduction gives $160 revenue and $10 contribution while the people and clicks remain identical. Profit falls 80%, showing why the decision needs economics rather than an activity chart alone.

This isolated intervention assumes price does not cause changes in content, sending or engagement. It is not an identified production causal effect. Correlated market/quality shocks can be a separate explicitly modeled stress.

## 7. Three truths and an inference boundary

Retain three distinct objects:

1. **Latent truth:** the generator's known parameters, person traits and performance paths. An oracle expected engagement curve integrates future contactability and response randomness for a declared population, conditional on its fixed traits and stated environment, such as the actual acquired people. Conditioning on their realized future eligibility path would define a different target.
2. **Realized truth:** the unique people, events, revenue and contribution produced in one simulated world, including its future.
3. **Posterior forecasts:** estimates and future distributions computed only from the reported prefix available to the analyst.

Underlying expected-rate truth belongs against latent-rate credible bands. A realized future outcome belongs against predictive bands. A single world's binary profitable/unprofitable result is not its true profitability probability. Do not draw it as an oracle probability curve; such a curve would need a separately defined conditional probability calculation.

~~~mermaid
flowchart LR
  G[Generator and hidden truth] -->|reported prefix only| O[As-of observation snapshot]
  O --> B[Bayesian inference]
  B --> F[Sealed forecasts and candidate decisions]
  G --> E[Evaluator]
  F --> E
  E --> V[Truth overlays and reliability checks]
~~~

Keep the evaluator store/API separate from fit inputs. Hidden source quality labels, person frailties, unreported classifications and future market paths must not enter inference. The what-if assumption supplied by the analyst is distinct from the hidden future realized in the generator.

Seal forecasts and configuration/source hashes before joining future outcomes. A leakage check changes future truth while preserving the same reported prefix and supplied future assumption; earlier forecasts must be unchanged for the same inference call/seed. Freeze the relevant information set when showing overlays.

Operator-style views default to truth hidden. Educational figures can reveal it with prominent labeling. Real business data would not offer this oracle.

## 8. Validation and delivery scope

Keep probability reliability and first-declaration diagnostics easy to find, not buried behind technical fit metrics. Show held-out predicted-probability bins against realized profitable fractions and the ideal diagonal, with counts and independent-world uncertainty.

Evaluate one first declaration per cohort/policy, reversals, false profitable declarations, false negative declarations, decision age, undecided share and economic loss. Preserve shared calendar shocks in joint portfolio draws and resample independent worlds for uncertainty.

The existing nominal 95% interval coverage of 56.5% and the source review's high-probability forecast failures remain evidence about the old model. New diagrams do not repair those probabilities. Threshold validation and untouched held-out worlds are release requirements.

The first implementation should establish a controlled known-qualification case with eligibility records, persistent person variation, varying cohort decay and independent CPC/CPL changes. Once its behavior and decisions are checked, layer imperfect audits, delayed settlement, signed reversals and correlated shocks onto the same visual story. Keep favorable assumptions visible at every stage.

Deliver a reproducible generator, as-of projection, fitted hierarchy, truth store, scenario configuration, paired outputs, chart-ready aggregates and synchronized article/MDX/HTML/assets. No rendering or simulation work is implied by publication of this design scope.
