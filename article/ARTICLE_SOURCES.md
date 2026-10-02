# Evidence map for the article

Scientific source snapshot: [research commit 8e3443fe](https://github.com/rothnic/bayesian-email-cohort-lab/tree/8e3443fe287d8e3ee29156d628df931b35fad9d5). No model was refitted for this article. Chart generation uses existing outputs only.

| Claim or figure | Saved input | Exact selection or calculation |
|---|---|---|
| Original engagement denominator | `artifacts/demo/synthetic_observations/leads.csv.gz` | All eight 180-lead cohorts per source: 1,440 originally acquired leads/source |
| Reported daily human-qualified clickers | `artifacts/demo/synthetic_observations/engagement_events.csv.gz` | Exclude missing report timestamps; latest reported revision/event; qualification human; distinct lead/source/occurrence age; divide by original source denominator; trailing seven available days |
| Middle-source learning probabilities | `artifacts/cold_start/summary.csv` | source middle, horizon180, origins3/14/30: .364/.176/.994 |
| Middle-source intervals and medians | `artifacts/cold_start/summary.csv` | source middle, horizon180, all origins; dollars divided by n_leads180 |
| Economic path at age60 | `artifacts/cold_start/trajectories.csv.gz` | source middle, origin60, all ages0–425; dollar quantiles and truth divided by180 |
| Unconditional first-payback CDF | `artifacts/cold_start/payback_cdf.csv.gz` | source middle, origins3/14/60; day425 age3 CDF .406, un-crossed mass .594; all500draws remain in denominator |
| Like-for-like source comparison | `artifacts/cold_start/summary.csv` | all three first cohorts w00, origin60, horizon180; values/180 |
| Narrower mature-history misses | `artifacts/demo/summary.csv` | all three later cohorts w07, origin60, horizon180; complete truth lies outside each95% interval; values/180 |
| Controlled95% coverage | `artifacts/controlled_scenario/metrics_by_origin.csv` | coverage95 over2000 worlds at origins3/7/14/30/60: .953/.942/.9475/.955/.953 |
| Controlled width and Brier improvement | Same controlled metrics | age3→60: mean_width95 12.2568125→3.83475; Brier approximately .106016→.031422; known exposure100/day, fixed value and costs |
| Richer-model coverage | `artifacts/calibration/scores.csv` and `metrics_by_origin.csv` | horizon180: covered95 counts23/36,21/36,19/36,22/36,22/36;12 independent worlds, source forecasts correlated within world; whole-world bootstrap |
| Reactivation misses at age60 | `artifacts/calibration/scores.csv` | scenario late_reactivation, origin60,horizon180:1 of9 covered;3 independent worlds |
| Baseline comparison | `artifacts/calibration/baseline_scores.csv` and `scores.csv` | horizon180; mean absolute error per lead, equal weighting complete worlds; source-aware baseline lower at3/7/14/60, Bayesian mean lower at30 |
| Cold-start high-source extreme mean | `artifacts/cold_start/summary.csv` | source high, origin3,horizon180; mean9.8405/lead, median.0756,95%[-.1665,4.5004];500 predictive draws |
| Monetary-prior moments | `artifacts/sensitivity/positive_mark_prior_moments.csv` | Caps1/4: positive-mark mean .19138/.31454; SD .48885/283.8017; variance-bound model described in MODEL_IMPLEMENTATION.md |
| Finite policy and clock assumptions | `docs/MODEL_IMPLEMENTATION.md`, `cohort_lab/accounting.py`, `bayes_cohort/config.py` | Daily sends through365; response/accounting closure425; known synthetic clocks; modular finite-grid/conjugate inference, no MCMC |
| Sixty tests | Research `tests/` and `docs/INDEPENDENT_REVIEW.md` | Independently rerun before article delivery; passing implementation checks do not imply full-pipeline calibration |

`chart_provenance.json` includes SHA-256 hashes for each chart's input files. `essential_values.csv` contains selected normalized values used by the HTML's readable tables. The five main figures and their mobile variants show the same underlying data; the optional appendix diagnostic retains all nine late-reactivation cases at one fixed origin.

## Information available to the first cohort

The first cohort w00 has no older cohort history at launch. Later forecasts also use the observations available from younger cohorts in the same source. `World.as_of(cutoff)` includes every acquired cohort by the cutoff; `_fit` visits every visible cohort within each source. Source fits start from independent copies of the same fixed priors. There is within-source partial pooling and no learned global hyperprior across sources. These figures therefore do not isolate target-only learning or demonstrate a pooling benefit; that requires an ablation. This clarification changes no simulation, forecast or plotted value.
