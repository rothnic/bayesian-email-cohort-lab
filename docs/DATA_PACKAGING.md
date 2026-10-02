# Reproducible data packaging

The release bundle includes fictional lead, engagement and operational-exit records, posterior draws, trajectory quantiles, summaries, calibration scores and charts. All observations can be regenerated from the declared seeds and fictional configuration with no external data source.

The following row-level exports are **generated locally and excluded from the release bundle**:

- `artifacts/demo/synthetic_observations/ledger.csv.gz`
- `artifacts/demo/evaluator_only_truth/complete_ledger.csv.gz`
- `artifacts/demo/synthetic_observations/exposure_daily.csv.gz`
- `artifacts/cold_start/synthetic_observations/ledger.csv.gz`
- `artifacts/cold_start/evaluator_only_truth/complete_ledger.csv.gz`
- `artifacts/cold_start/synthetic_observations/exposure_daily.csv.gz`

Recreate them locally with:

```sh
python run_bayesian_demo.py demo
python run_bayesian_demo.py cold-start
```

The commands generate a complete fictional world afresh before taking any as-of snapshot. Neither command downloads nor needs an existing row-level ledger or exposure file. The evaluator-only ledger remains separate from model inputs. `.gitignore` keeps these generated exports out of an ordinary subsequent source commit.

`PUBLIC_MANIFEST.json` records the files and SHA-256 hashes in the release bundle, excluding its own hash, together with these omissions. The checked-in summaries, draw files and figures are inspectable without the omitted exports. The standard test suite and public-artifact validator do not require those exports to be present; full reproduction regenerates them.
