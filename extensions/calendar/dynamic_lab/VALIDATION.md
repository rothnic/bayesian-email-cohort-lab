# Refreshed dynamic model validation

Run date: 2026-10-23. This records a reproducibility and implementation check of the synthetic extension; it is not an independent commercial calibration study. The Gaussian measurement model is approximate. The full evaluator used 18 held-out worlds across six mechanisms, 306 fixed-target forecasts per model (1,530 rows across five model variants), with truth kept outside each as-of fit. The sealed output hashes and methods are in `artifacts/run_manifest.json`; detailed world-level scores are in `artifacts/heldout_scores.csv` and `artifacts/metrics.csv`.

## Checks completed

- 29 dynamic model and simulation unit tests passed. The separate prototype replay suite passed 13 tests; its legacy-v1 test module is skipped because the original `cohort_lab` package is not in this public snapshot.
- Ten figures were regenerated from sealed outputs. All 40 desktop/mobile PNG/SVG variants exist, all ten mobile SVGs pass the 390px text-size check (minimum effective size: 12.58 CSS px), and chart provenance verifies source data hashes.
- The article builder generated the 3,263-word draft, MDX, static HTML and standalone preview. The browser layout has not been checked in a full browser engine.

## Held-out results and interpretation

The dynamic model's equal-world mean 95% interval coverage was 92.0% (world-bootstrap 95% interval 87.0%–96.2%; 18 worlds). Row-weighted coverage was 93.5% across 306 dependent forecasts. The interval is somewhat under nominal, but the small world-level interval includes 95%. Against the static hierarchy, dynamic CRPS was $0.040 per lead versus $0.041 (overlapping world-bootstrap intervals). Dynamic decision loss was $0.031 per lead versus $0.028 for static; the static point estimate is lower. These results show no general decision advantage and do not validate a 95% decision threshold.

In the highest posterior-probability bin, predicted profitability averaged 96.4% while the realized share was 69.2%; its world-bootstrap interval was 45.0%–91.2% across 13 worlds and 132 nested forecasts. This is a visible calibration warning. In a separate weekly fixed-cohort replay, 10 of 12 worlds declared below-target, two remained unresolved, and none declared above-target. Zero false declarations and zero reversals in such a small sample are not evidence of a safe business policy.

The model now fits changing calendar conditions, subscriber/cohort heterogeneity, source and cohort partial pooling, distinct people-level unique response and repeat-event rates, net CPC, and acquisition cost separately. Held-out scenarios include CPC squeeze, rising CPL, human decline with increasing bot share, and a cheaper good or bad source. The illustrative CPC what-if applies after its information cutoff without changing subscriber behavior or repricing earlier clicks. The controlled comparison is conditional on the shock; it does not assign a probability to the shock happening.

## Limits that remain

Contactability uses a constant within-cohort exit hazard. The model's unique-response, repeat-event and monetary processes do not form a fully joint person-level posterior; dependence across these processes is incomplete. Audit selection and audit accuracy are favorable declared assumptions. Unknown audit errors, signed monetary reversals, unrestricted aging/calendar/birth-cohort trends, and more flexible dropout remain unmodeled. There are only three held-out worlds per mechanism and 12 worlds in the sequential replay. The prior has heavy tails and finite simulated moments are less stable than its bounded parameterization may suggest.

The historical v1 stress failure (56.5% nominal 95% coverage in its frozen 12-world suite) is retained as a distinct earlier experiment, not pooled with this refreshed model. All data and outcomes in this extension are fictional.
