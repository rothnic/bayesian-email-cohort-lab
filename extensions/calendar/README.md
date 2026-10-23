# Changing-price Bayesian cohort experiment

This extension models synthetic paid-email cohorts as people with persistent human or bot status, person-level engagement variation, cohort-specific contactability lifetimes, distinct responders and repeat clicks. A partially pooled global/source/cohort model learns alongside shared calendar response and CPC states. The Gaussian aggregate likelihood is an approximation, not a full subscriber-level posterior.

Start with [the experiment](dynamic_lab/EXPERIMENT.md), [implemented math](dynamic_lab/MODEL.md), and [current validation](dynamic_lab/VALIDATION.md). Eleven source-bound figures and their mobile PNG/SVG versions are in [artifacts/figures](dynamic_lab/artifacts/figures), with captions, alt text, numerical tables, mobile-size checks and chart provenance.

The six named synthetic worlds isolate fixed economics, a future CPC squeeze, later CPL inflation, human/bot deterioration, and two equally priced new sources with different hidden performance. The prospective CPC what-if reuses the same posterior draws and subscriber activity while changing only future CPC. A separate quote sensitivity changes only the purchase cost of a future batch. Fixed-cohort charts show both the evolving probability of clearing a day-180 target and the evolving days-from-cutoff-to-break-even distribution (negative means the estimated crossing has passed), with no-break-even paths reported separately; the held-out reliability chart checks probabilities and sequential thresholds against separate synthetic worlds.

To reproduce from the repository root, install the existing `requirements.txt`, then:

```sh
cd extensions/calendar
make reproduce
```

The dynamic model tests, simulator boundary checks, independent input-contract checks and preserved prototype replay tests are included. `dynamic_lab/artifacts/run_manifest.json` binds the scientific source and every output CSV hash. The extension never uses private or real email data and requires no model API, hosted service, paid computation or GitHub Actions.

The original v1 accounting experiments and their failure evidence remain preserved separately. `DESIGN.md`, `prototype/`, `REPLAY_CHECKPOINT.md` and `INDEPENDENT_REVIEW.md` describe earlier stages; `dynamic_lab/review_history/` retains superseded numerical runs. Do not confuse those artifacts with the current subscriber-level approximation.

All observations are synthetic. The original six excluded row-level exports remain excluded; this extension publishes aggregate person/cohort summaries, forecasts, evaluation scores, shared portfolio draws and figures, not person/event ledgers. The visual article is a draft in `article/`; it has not been deployed to nickroth.com.
