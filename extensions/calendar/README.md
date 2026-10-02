# Changing-price Bayesian cohort experiment

This extension actually fits global → source → cohort effects and shared calendar states, then predicts joint contribution and unconditional first payback. It adds six nested fictional scenarios: fixed economics, CPC squeeze, higher new-purchase prices, human decline/bot activity, and equally cheap good/bad sources arriving later. The likelihood is an explicitly disclosed Gaussian approximation with an integrated finite hyperprior.

Start with [the experiment](dynamic_lab/EXPERIMENT.md), [implemented math](dynamic_lab/MODEL.md), and [independent validation](dynamic_lab/VALIDATION.md). The six figure families and their mobile PNG/SVG variants are in [artifacts/figures](dynamic_lab/artifacts/figures). Each includes full numeric tables, captions, alt text and source hashes. The [offered-cost table](dynamic_lab/artifacts/quote_sensitivity.csv) prices the same future-cohort outcome paths at several offers; it never reprices an existing cohort.

The final comparison has 18 independent worlds and 1,530 forecasts across five information-sharing models. The dynamic model's nominal 95% intervals cover only 56.5% by equal-world averaging (46.4% if every forecast row is equally weighted). Its CRPS is lower than the static hierarchy's in this suite, but the static hierarchy has slightly lower acquisition-loss point estimate. Neither yields validated uncertainty or a demonstrated production policy. These failures remain alongside the learned price and source differences.

From the repository root, install the existing `requirements.txt`, then:

```
cd extensions/calendar
make reproduce
```

Forty extension tests cover the new dynamic model and the earlier replay contract. The original v1's 60 tests and published results remain separate. No model API, hosted service, new account or paid computation is needed. `dynamic_lab/artifacts/run_manifest.json` binds final scientific source and CSV output hashes. `make reproduce` runs tests, evaluation and figure generation; `article/build_article.py` rebuilds the article from these outputs.

The original development stages remain inspectable: `DESIGN.md` is the broader proposed design, `prototype/calendar_replay.py` is an unfitted input-contract fixture, and `prototype/replay_v1_ablation.py` replays unchanged v1 on one development world. `REPLAY_CHECKPOINT.md` and `INDEPENDENT_REVIEW.md` describe those earlier stages. Do not confuse them with the implemented dynamic approximation. `dynamic_lab/review_history` retains superseded scores and mathematical boundary corrections; it is not combined with final results.

All observations are synthetic. The original six excluded row-level exports stay excluded. This extension publishes aggregate forecasts/evaluation, calendar/measurement summaries, joint portfolio draws and figures; it does not copy the original individual/event ledgers. No live website deployment or Actions configuration is included.
