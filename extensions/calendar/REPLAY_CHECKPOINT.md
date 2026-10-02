# A real calendar replay of the unchanged model

The separate aggregate fixture tests the planned input contract. This checkpoint additionally executes the existing published v1 Bayesian model, without changing its likelihood or priors.

It uses one development world, seed884901, the existing collapse_tail simulator, 13 weekly cohorts per source and 50 leads per cohort. Acquisition dates run from calendar day0 through84. At calendar cutoffs10,17,31,59 and87, it forecasts both the first middle-source cohort and the newest cohort at age3. Each target is evaluated with target-only observations and with all visible same-source observations. This produces 20 forecasts, each with64 posterior predictive draws, predicting contribution at age180 and unconditional first payback through425.

The 64-draw budget makes this an execution example. It is insufficient for accurate tail quantiles or stable heavy-tailed expected values. The CSV retains mean Monte Carlo standard error. No forecast was selected by looking at its outcome, but one development world is still not a held-out multi-world comparison or a calibration study. Do not report a pooling benefit from this packet.

`prototype/v1_replay/predictions.csv` records the calendar cutoff, target age, information scope, input hash, fixed-prior hash, draw seed, cohort census and forecast summary. The function receiving each immutable serialized snapshot has no World or GeneratorTruth argument. It reconstructs a fresh working view, starts from the same frozen original prior at every cutoff and checks that inference did not mutate the input. Monetary and exposure rows stay in memory; no original raw exports are written.

Predictions are written and hashed before the complete evaluator outcomes are read. `evaluator_comparison.csv` stores that prediction-file hash with each outcome, interval inclusion and absolute mean error. `manifest.json` identifies the source/configuration hashes and declares that no new hierarchy was fitted. Two complete reruns produced byte-identical prediction and evaluator CSVs.

To reproduce alongside the research repository:

```
cd prototype
python -m unittest -v
python replay_v1_ablation.py --lab /path/to/bayesian-email-cohort-lab
```

The additional v1 boundary tests verify that mutating a reconstructed DataFrame cannot change archived snapshot bytes, that a future-reported ledger entry cannot change an earlier prediction, and that target-only filtering removes other cohorts from every likelihood table. The separate input-contract suite verifies revisions, arrivals, quotes, audit availability, shared scenario paths and first-payback accounting.

This replay answers whether the intended comparison is executable with the existing model and realistic information cutoffs. It does not fit a learned global prior, learn calendar states, predict an unseen source or demonstrate superior decisions. Those remain the bounded next stages in DESIGN.md. The original article's undercoverage and prior-instability findings remain unchanged.
