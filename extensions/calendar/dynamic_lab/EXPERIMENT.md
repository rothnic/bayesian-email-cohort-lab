# Fictional worlds and decision replay

This experiment generates synthetic subscriber-level truth, exports delayed aggregate records, fits the model specified in `MODEL.md`, seals predictions, and only then scores them against hidden truth. Every price, subscriber and outcome is fictional. No original email or company records are used.

## Data-generating process

Two established sources acquire equal-sized cohorts weekly from calendar day0 through126. Cohort size defaults to100 leads. Source and cohort effects vary; each cohort receives its own contactability half-life, person engagement frailty and persistent human/bot assignments. Human and bot events use separate streams. Human responders may create repeat clicks, while unique people and event totals are recorded separately. Contactability is absorbing after exit. Sending cost is incurred per contactable address, and acquisition cost is paid once at cohort birth.

Net CPC has a shared calendar component and source/cohort marks. Each scenario is an independent, named mechanism; the seed-keyed components make suitable comparisons paired. In particular, CPC stress does not quietly add CPL or bot changes.

| Scenario | Mechanism changed |
|---|---|
| `fixed_price` | Stable baseline CPC, CPL, human engagement and bot activity apart from shared modest process noise. |
| `cpc_squeeze` | Net human CPC falls log-linearly from calendar day42 to60% of its day42 level at day126, then holds there. Subscribers, click paths and acquisition costs match the fixed-price world under the same seed. |
| `rising_cpl` | New purchase quotes rise by0.8% per day after day42; CPC and quality mechanisms remain at baseline. Old acquisition costs stay fixed. |
| `human_decline_bot_rise` | Calendar human engagement falls to72% of its day42 level by day126; bot event rate rises from0.4% to6%; later-vintage cohorts have declining human share. This does not also squeeze CPC or increase CPL. Contactability exit hazard stays cohort-specific but does not worsen over calendar time. |
| `cheap_good` | A new source appears at day70 at a low quote, with higher hidden response quality. |
| `cheap_bad` | Same visible source descriptors and quote as `cheap_good`, but lower hidden response quality. |

Source/cohort traits and response processes contain random variability. The truth objects preserve the latent expected rate, realized person-level activity, contactability and economic path separately. None is included in the model’s as-of snapshot. Audit/event observations arrive after2 days, event and unique-person audit labels after7, and mature net-revenue/human-count summaries after14. The audit labels are sampled with assumed sensitivity95% and specificity98%; these are known fixture assumptions, not learned production characteristics.

## Aggregation without erasing cohorts

`retention_truth.csv` contains per-cohort age-aligned latent and realized contactability/response curves. `cohort_retention_rollup_detail.csv` contains complete, nonoverlapping 7- and30-day response windows deduplicated within each person across that full window. The response denominator is the number contactable at window start. `cohort_retention_rollups.csv` pools counts across acquisition cohorts before dividing: summed distinct responders / summed eligible people, and summed contactable at window end / summed original acquired leads. It records the number of contributing cohorts and windows. Daily unique counts are never summed and mislabeled as weekly unique people.

These views reorganize the same people; daily, weekly, monthly and all-history summaries are not independent observations. Averaging within an acquisition month can reduce visible noise, but also hide a low-performing source or a recent shift. The per-cohort spaghetti view preserves that dispersion beside denominator-weighted pooled lines. Only complete windows are used, and each plotted source table retains the cohort counts and denominators.

## Price counterfactuals

The model accepts an explicit post-cutoff CPC shock in `predict`. The demonstration applies a0%,15%,30% or45% reduction beginning at calendar day140, with a28-day ramp, to a prospective established-source cohort. Each what-if reuses the same posterior parameters and random marks. It changes only the value per post-shock human event; human activity, exits, bot traffic, acquisition quote and send costs are held constant. It reports probability of clearing the day-180 two-cent-per-lead buffer, contribution quantiles, expected shortfall below the buffer, and the paired margin difference from no shock.

The separate offer-price table varies only a new source’s prospective CPL while holding its posterior outcome paths fixed. It reports fixed-horizon profitability probability and expected downside as well as first-payback probabilities. No already acquired cohort is repriced. These controlled counterfactuals quantify consequences under stated assumptions; they do not predict which future market change will occur or establish causal effects of an acquisition channel.

## Fixed-cohort confidence and held-out evaluation

The demonstration follows the same established cohort acquired on day70 at information cutoffs77,84,91,98,105,112 and119. It holds the target age at180, future policy at `hold_current`, and the profitability buffer at$0.02 per acquired lead. At each cutoff the model refits the newly visible data from the same declared hyperprior. Its probability is classified as above target at95% or more, below target at5% or less, and inconclusive otherwise. These are posterior classification thresholds, not a claim of95% empirical reliability. The later outcome is hidden from the fit but used by the evaluator.

Held-out worlds use three base seeds per six mechanisms and all five model scopes for common cutoffs. The sequential replay separately tracks one fixed established cohort at prespecified weekly cutoffs in the four mechanism worlds that affect the established source. It records first declaration, declaration age, reversals, false profitable/unprofitable classifications and the share left undecided. Only whole worlds are independent evaluation units; repeated cutoffs and cohorts are correlated.

The probability-reliability table bins day-180 buffer probabilities and compares their mean with the realized profitable share. Both predictive and observed intervals resample independent worlds, not rows. Bins can contain few worlds, so the result is a diagnostic, not proof of calibration. Coverage, CRPS and decision loss also summarize the distribution and the declared immediate buy/skip rule. In this evaluator, “wait” is treated like no immediate purchase; delayed information, missed offers and an optimal value-of-information policy are not modeled.

The configuration is frozen in `run_config.json`. Outputs include source hashes and a forecast seal before scoring. Demo truth figures are labelled evaluator-only. Changing configuration or source requires regenerating artifacts, figures, article numeric claims and manifests together. Never cite old CSVs as results of the updated person-level simulator.
