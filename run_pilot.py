#!/usr/bin/env python3
"""Run bounded smoke or held-out fictional worlds; no inference API calls."""
import argparse
import hashlib
import importlib
import json
import platform
import os
import resource
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from cohort_lab.accounting import load_parameters, summarize_path
from cohort_lab.generator import generate_world

ROOT=Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"artifacts"/"mpl-cache"))

def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [clean(v) for v in value]
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value

def versions():
    out={"python":platform.python_version(),"platform":platform.platform()}
    for name in ("numpy","pandas","scipy","matplotlib","pyarrow","pytest"):
        try:out[name]=importlib.import_module(name).__version__
        except ImportError:out[name]="not installed"
    return out

def world_run(args):
    from cohort_lab.evaluation import evaluate_world
    params=load_parameters()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    world,truth,policy=generate_world(args.seed,args.scenario,params,small=args.small)
    generated=time.perf_counter()
    result=evaluate_world(world,truth,policy,params)
    evaluated=time.perf_counter()
    for key in ("forecasts","margin_scores","decisions","allocations"):
        frame=result.get(key)
        if isinstance(frame,pd.DataFrame):frame.to_csv(output/f"{key}.csv",index=False)
    truth_rows=[]
    for cohort in world.leads.cohort_id.unique():
        n=int(world.leads.cohort_id.eq(cohort).sum())
        for row in summarize_path(truth.cohort_path(world.leads,cohort,policy)):
            truth_rows.append(dict(scenario=args.scenario,seed=args.seed,cohort_id=cohort,source_id=cohort.split('-')[0],n_leads=n,**row))
    pd.DataFrame(truth_rows).to_csv(output/"truth_cohort_summary.csv",index=False)
    # Bounded artifacts: raw observations only for one smoke world, never every full world.
    if args.export_small and args.small:
        observations=output/"synthetic_observations";observations.mkdir(exist_ok=True)
        for key in ("leads","exposure_daily","engagement_events","ledger","operational_exits"):
            getattr(world,key).to_csv(observations/f"{key}.csv.gz",index=False,compression="gzip")
        evaluator_only=output/"evaluator_only_truth";evaluator_only.mkdir(exist_ok=True)
        truth.complete_ledger.to_csv(evaluator_only/"complete_ledger.csv.gz",index=False,compression="gzip")
        truth.calendar_shocks.to_csv(evaluator_only/"calendar_shocks.csv",index=False)
    measurement={"scenario":args.scenario,"seed":args.seed,"small":args.small,
                 "generator_seconds":generated-start,"evaluation_seconds":evaluated-generated,
                 "total_seconds":time.perf_counter()-start,"peak_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                 "rows":{key:len(getattr(world,key)) for key in ("leads","exposure_daily","engagement_events","ledger","operational_exits")},
                 "parameter_sha256":hashlib.sha256((ROOT/"parameters.json").read_bytes()).hexdigest(),
                 "forecast_type":"point; probability and interval calibration not implemented",
                 "parquet":"pending optional pyarrow dependency; schema-identical CSV currently emitted",
                 "summary":result.get("summary",{})}
    (output/"summary.json").write_text(json.dumps(clean(measurement),indent=2,allow_nan=False))
    print(json.dumps(clean({k:v for k,v in measurement.items() if k != "summary"}),allow_nan=False),flush=True)
    return measurement

def suite_run(args):
    import subprocess
    params=load_parameters()
    output=Path(args.output or ROOT/"artifacts"/args.mode);output.mkdir(parents=True,exist_ok=True)
    seeds=params["smoke_seeds"] if args.mode == "smoke" else params["heldout_seeds"]
    metadata={"synthetic_only":True,"mode":args.mode,"seeds":seeds,"scenarios":params["scenarios"],"versions":versions(),
              "parameter_sha256":hashlib.sha256((ROOT/"parameters.json").read_bytes()).hexdigest(),
              "source_sha256":{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted((ROOT/"cohort_lab").glob("*.py"))}}
    (output/"run_metadata.json").write_text(json.dumps(metadata,indent=2))
    begin=time.perf_counter()
    for scenario in params["scenarios"]:
        for seed in seeds:
            target=output/f"{scenario}-seed{seed}"
            cmd=[sys.executable,str(ROOT/"run_pilot.py"),"world","--scenario",scenario,"--seed",str(seed),"--output",str(target)]
            if args.mode == "smoke":cmd += ["--small"]
            if args.mode == "smoke" and scenario == "collapse_tail":cmd += ["--export-small"]
            subprocess.run(cmd,check=True,cwd=ROOT)
    for key in ("forecasts","margin_scores","decisions","allocations","truth_cohort_summary"):
        frames=[pd.read_csv(path) for path in sorted(output.glob(f"*/{key}.csv")) if path.stat().st_size>1]
        if frames:pd.concat(frames,ignore_index=True).to_csv(output/f"{key}.csv",index=False)
    measurements=[json.loads(path.read_text()) for path in sorted(output.glob("*/summary.json"))]
    pd.DataFrame([{k:v for k,v in m.items() if k not in ("summary","rows")} for m in measurements]).to_csv(output/"resource_measurements.csv",index=False)
    metadata["suite_wall_seconds"]=time.perf_counter()-begin
    metadata["world_count"]=len(measurements)
    (output/"run_metadata.json").write_text(json.dumps(metadata,indent=2))
    print(f"Completed {len(measurements)} {args.mode} worlds in {metadata['suite_wall_seconds']:.1f}s; outputs {output}")

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["smoke","heldout","world"])
    parser.add_argument("--scenario",default="collapse_tail")
    parser.add_argument("--seed",type=int,default=11)
    parser.add_argument("--small",action="store_true")
    parser.add_argument("--export-small",action="store_true")
    parser.add_argument("--output")
    args=parser.parse_args()
    if args.mode=="world":
        if not args.output:args.output=str(ROOT/"artifacts"/"one_world")
        world_run(args)
    else:suite_run(args)
