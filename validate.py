#!/usr/bin/env python3
"""Run local tests and validate the emitted versioned point-forecast contract."""
import hashlib
import json
import unittest
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent
suite=unittest.defaultTestLoader.discover(str(ROOT/"tests"))
result=unittest.TextTestRunner(verbosity=2).run(suite)
validation={"unit_tests_run":result.testsRun,"unit_tests_successful":result.wasSuccessful(),"artifact_checks":[]}
for mode in ("smoke","heldout"):
    base=ROOT/"artifacts"/mode
    if not (base/"forecasts.csv").exists():continue
    forecasts=pd.read_csv(base/"forecasts.csv")
    contract=json.loads((ROOT/"contract.json").read_text())
    assert set(contract["tables"]["forecasts"]) <= set(forecasts.columns)
    assert forecasts.probability_profitable.isna().all()
    assert forecasts.forecast_type.eq("point").all() and forecasts.draw_id.eq(0).all()
    assert forecasts.payback_state.isin(["by_180","later_than_180","never_under_policy"]).all()
    assert forecasts.first_payback_age.isna().equals(forecasts.payback_state.eq("never_under_policy"))
    assert not forecasts.duplicated(["model","scenario","seed","cutoff","cohort_id","horizon","draw_id"]).any()
    scores=pd.read_csv(base/"margin_scores.csv")
    assert (scores.horizon > scores.origin_age).all()
    decisions=pd.read_csv(base/"decisions.csv")
    assert ((decisions.losses_avoided_vs_always_continue-decisions.forgone_profitable_margin)-(decisions.chosen_realized_margin-decisions.always_continue_realized_margin)).abs().max() < 1e-8
    meta=json.loads((base/"run_metadata.json").read_text())
    current_hash=hashlib.sha256((ROOT/"parameters.json").read_bytes()).hexdigest()
    assert meta["parameter_sha256"]==current_hash
    for rel,sha in meta["source_sha256"].items():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==sha, f"Source changed after {mode} run: {rel}"
    validation["artifact_checks"].append({"mode":mode,"forecast_rows":len(forecasts),"prospective_scores":len(scores),"status":"passed"})
(ROOT/"artifacts"/"validation.json").write_text(json.dumps(validation,indent=2))
if not result.wasSuccessful():raise SystemExit(1)
print(json.dumps(validation,indent=2))
