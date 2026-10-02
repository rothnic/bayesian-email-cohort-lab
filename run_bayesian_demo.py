#!/usr/bin/env python3
"""Reproduce fictional Bayesian forecasts and honest out-of-model stress checks.

No network, credentials, inference service or paid infrastructure is used.
Forecasting only receives an as-of Snapshot; complete truth is evaluator-only.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import os
import platform
import resource
import time
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/cohort-lab-matplotlib')
import numpy as np
import pandas as pd
from cohort_lab.accounting import load_parameters, ledger_path
from cohort_lab.generator import generate_world
from cohort_lab.baselines import forecast_cohort as baseline_forecast, clear_fit_cache as clear_baselines
from bayes_cohort import forecast_cohort, default_config_dict

ROOT = Path(__file__).resolve().parent


def json_safe(value):
    if isinstance(value, dict): return {str(k): json_safe(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray): return json_safe(value.tolist())
    if isinstance(value, np.generic): return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + '\n')


def crps(draws, truth):
    """Empirical CRPS in outcome units; O(B log B), including finite B."""
    x = np.sort(np.asarray(draws, dtype=float))
    b = len(x)
    return float(np.mean(np.abs(x - truth)) - np.sum((2*np.arange(1,b+1)-b-1)*x)/(b*b))


def first_payback(paths):
    hit = np.asarray(paths) >= 0
    found = hit.any(axis=1)
    return np.where(found, hit.argmax(axis=1), np.inf)


def summarize_draws(paths, truth_path, origins, n_leads):
    rows = []
    tau = first_payback(paths)
    truth_tau = first_payback(truth_path[None,:])[0]
    for h in origins:
        x = paths[:,h]
        qs = np.quantile(x, [.025,.1,.25,.5,.75,.9,.975])
        positive = x > 0
        p = float(positive.mean())
        remains = np.array([np.isfinite(t) and t <= h and np.all(path[int(t):h+1]>=0)
                            for t,path in zip(tau,paths)])
        rows.append(dict(horizon=h, mean=float(x.mean()), mean_mcse=float(x.std(ddof=1)/np.sqrt(len(x))) if len(x)>1 else None, q025=qs[0], q10=qs[1], q25=qs[2],
                         median=qs[3], q75=qs[4], q90=qs[5], q975=qs[6],
                         probability_profitable=p, probability_payback_by_horizon=float(np.mean(tau<=h)),
                         probability_remains_positive=float(remains.mean()),
                         probability_never=float(np.mean(~np.isfinite(tau))),
                         probability_later_than_180=float(np.mean((tau>180)&np.isfinite(tau))),
                         true_margin=float(truth_path[h]), true_payback_age=truth_tau if np.isfinite(truth_tau) else None,
                         true_never=not np.isfinite(truth_tau),
                         absolute_error=abs(float(x.mean())-truth_path[h]),
                         absolute_error_per_lead=abs(float(x.mean())-truth_path[h])/n_leads,
                         crps=crps(x,truth_path[h]), crps_per_lead=crps(x,truth_path[h])/n_leads,
                         brier=(p-float(truth_path[h]>0))**2,
                         payback_brier=(float(np.mean(tau<=h))-float(truth_tau<=h))**2,
                         coverage80=float(qs[1]<=truth_path[h]<=qs[5]),
                         coverage95=float(qs[0]<=truth_path[h]<=qs[6]),
                         width80=float(qs[5]-qs[1]), width95=float(qs[6]-qs[0])))
    return rows


def source_hashes():
    files = [ROOT/'run_bayesian_demo.py',ROOT/'run_sensitivity.py',ROOT/'demo_config.json',ROOT/'parameters.json']
    files += list((ROOT/'bayes_cohort').glob('*.py')) + list((ROOT/'cohort_lab').glob('*.py'))
    files += list((ROOT/'scripts').glob('*.py'))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def run_world(scenario, seed, config, output, calibration=False, quick=False):
    begin = time.perf_counter()
    output.mkdir(parents=True,exist_ok=True)
    prefix = 'calibration' if calibration else 'demo'
    params = load_parameters()
    params['leads_per_cohort'] = config[f'{prefix}_leads_per_cohort']
    params['cohorts_per_source'] = config[f'{prefix}_cohorts_per_source']
    if quick:
        params['leads_per_cohort'] = 30
        params['cohorts_per_source'] = 4
    n = params['leads_per_cohort']
    ndraw = 32 if quick else config['calibration_draws' if calibration else 'posterior_draws']
    world, truth, policy = generate_world(seed,scenario,params)
    generated = time.perf_counter()
    summaries, scores, trajectory, cdf, records, baseline_rows, diagnostics = [], [], [], [], [], [], []
    sources = [s['source_id'] for s in params['sources']]
    origin_ages = [3,60] if quick else config['origin_ages']
    week = config.get(f'{prefix}_target_cohort_index',params['cohorts_per_source']-1)
    for age in origin_ages:
        cutoff = week*params['cohort_spacing_days'] + age
        snapshot = world.as_of(cutoff)
        for source_i, source in enumerate(sources):
            cohort = f'{source}-w{week:02d}'
            base = dict(scenario=scenario,seed=seed,world_id=f'{scenario}-{seed}',source_id=source,
                        cohort_id=cohort,origin_age=age,cutoff=cutoff,n_leads=n,
                        policy_id=policy.policy_id,draw_count=ndraw)
            result = forecast_cohort(snapshot,policy,cohort,draws=ndraw,
                                     seed=seed*100000+source_i*10000+age*10)
            paths = np.asarray(result['margin_paths'])
            if paths.shape != (ndraw,policy.terminal_age+1) or not np.isfinite(paths).all():
                raise ValueError(f'Invalid forecast shape/values: {paths.shape}')
            # Complete economics enter only AFTER forecasting, for scoring.
            truth_path = truth.cohort_path(world.leads,cohort,policy)
            visible = ledger_path(snapshot.ledger.loc[snapshot.ledger.cohort_id.eq(cohort)],
                                  week*params['cohort_spacing_days'],policy.terminal_age)
            tau = first_payback(paths)
            diagnostics.append(dict(**base,diagnostics=result['diagnostics']))
            rows = summarize_draws(paths,truth_path,config['horizons'],n)
            for row in rows:
                item = dict(**base,model=result['model'],model_version=result['model_version'],forecast_type='posterior_predictive',**row)
                item['evaluation_role'] = 'prospective' if row['horizon']>age else 'nowcast_or_retrospective'
                summaries.append(item)
                if row['horizon']>age: scores.append(item)
            for model in ('fixed_aggregate','observable_survival'):
                point = baseline_forecast(snapshot,policy,cohort,model,params)['margin_path']
                for h in config['horizons']:
                    if h<=age: continue
                    baseline_rows.append(dict(**base,model=model,horizon=h,prediction=point[h],
                        true_margin=truth_path[h],absolute_error=abs(point[h]-truth_path[h]),
                        absolute_error_per_lead=abs(point[h]-truth_path[h])/n,
                        forecast_type='point',probability_profitable=None))
            if not calibration:
                quantiles = np.quantile(paths,[.025,.1,.25,.5,.75,.9,.975],axis=0)
                for h in range(policy.terminal_age+1):
                    trajectory.append(dict(**base,age=h,truth=truth_path[h],observed=visible[h],
                        **dict(zip(['q025','q10','q25','median','q75','q90','q975'],quantiles[:,h]))))
                    cdf.append(dict(**base,age=h,cdf=float(np.mean(tau<=h)),
                                    probability_never=float(np.mean(~np.isfinite(tau)))))
                for draw in range(ndraw):
                    first = tau[draw]
                    state = 'never_under_policy' if not np.isfinite(first) else ('by_180' if first<=180 else 'later_than_180')
                    for h in config['horizons']:
                        records.append(dict(**base,model=result['model'],model_version=result['model_version'],draw_id=draw,horizon=h,
                            margin=paths[draw,h],first_payback_age=int(first) if np.isfinite(first) else None,
                            payback_state=state,positive_at_horizon=bool(paths[draw,h]>0)))
            print(f'{scenario} seed={seed} {cohort} origin={age} draws={ndraw}',flush=True)
        clear_baselines()
    pd.DataFrame(summaries).to_csv(output/'summary.csv',index=False)
    pd.DataFrame(scores).to_csv(output/'scores.csv',index=False)
    pd.DataFrame(baseline_rows).to_csv(output/'baseline_scores.csv',index=False)
    write_json(output/'diagnostics.json',diagnostics)
    if not calibration:
        pd.DataFrame(trajectory).to_csv(output/'trajectories.csv.gz',index=False,compression={'method':'gzip','mtime':0})
        pd.DataFrame(cdf).to_csv(output/'payback_cdf.csv.gz',index=False,compression={'method':'gzip','mtime':0})
        pd.DataFrame(records).to_csv(output/'posterior_predictive_draws.csv.gz',index=False,compression={'method':'gzip','mtime':0})
        obs = output/'synthetic_observations'; obs.mkdir(exist_ok=True)
        for name in ('leads','exposure_daily','engagement_events','ledger','operational_exits'):
            getattr(world,name).to_csv(obs/f'{name}.csv.gz',index=False,compression={'method':'gzip','mtime':0})
        evaluator=output/'evaluator_only_truth'; evaluator.mkdir(exist_ok=True)
        truth.complete_ledger.to_csv(evaluator/'complete_ledger.csv.gz',index=False,compression={'method':'gzip','mtime':0})
        truth.calendar_shocks.to_csv(evaluator/'calendar_shocks.csv',index=False)
    result = dict(scenario=scenario,seed=seed,draws=ndraw,cohorts_per_source=params['cohorts_per_source'],
                  leads_per_cohort=n,forecasts=len(sources)*len(origin_ages),
                  generator_seconds=generated-begin,total_seconds=time.perf_counter()-begin,
                  peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    write_json(output/'run.json',result)
    return result


def calibration_report(output):
    frames = [pd.read_csv(p) for p in sorted(output.glob('*/scores.csv'))]
    scores = pd.concat(frames,ignore_index=True)
    scores.to_csv(output/'scores.csv',index=False)
    base = pd.concat([pd.read_csv(p) for p in sorted(output.glob('*/baseline_scores.csv'))],ignore_index=True)
    base.to_csv(output/'baseline_scores.csv',index=False)
    metrics = ['absolute_error_per_lead','crps_per_lead','brier','payback_brier','coverage80','coverage95','width80','width95']
    # Bootstrap whole worlds, not correlated horizons/sources/origins as iid rows.
    rng = np.random.default_rng(777)
    rows=[]
    for age,g in scores.loc[scores.horizon.eq(180)].groupby('origin_age'):
        world_mean = g.groupby('world_id')[metrics].mean()
        boot_idx = rng.integers(0,len(world_mean),size=(2000,len(world_mean)))
        values=world_mean.to_numpy()
        boot=values[boot_idx].mean(axis=1)
        for j,m in enumerate(metrics):
            lo,hi=np.quantile(boot[:,j],[.025,.975])
            rows.append(dict(origin_age=age,metric=m,mean=values[:,j].mean(),lower95=lo,upper95=hi,
                             worlds=len(world_mean),forecasts=len(g)))
    pd.DataFrame(rows).to_csv(output/'metrics_by_origin.csv',index=False)
    scores.groupby(['scenario','origin_age','horizon'])[metrics].mean().reset_index().to_csv(output/'metrics_by_scenario.csv',index=False)
    # Reliability bins are descriptive, with counts. No iid binomial confidence claim.
    rel=scores.loc[scores.horizon.eq(180)].copy()
    rel['probability_bin']=pd.cut(rel.probability_profitable,[-.001,.2,.4,.6,.8,1.001],labels=['0–.2','.2–.4','.4–.6','.6–.8','.8–1'])
    rel['realized_profitable']=rel.true_margin>0
    rel.groupby('probability_bin',observed=False).agg(mean_prediction=('probability_profitable','mean'),
        realized_fraction=('realized_profitable','mean'),forecasts=('seed','size')).reset_index().to_csv(output/'reliability.csv',index=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['demo','cold-start','calibration'])
    parser.add_argument('--quick',action='store_true',help='Small wiring test, not calibration evidence')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    config=json.loads((ROOT/'demo_config.json').read_text())
    output=args.output or ROOT/'artifacts'/('quick' if args.quick else args.mode.replace('-','_'))
    if args.mode=='cold-start': config['demo_target_cohort_index']=0
    output.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    if args.mode in ('demo','cold-start'):
        runs=[run_world(config['demo_scenario'],config['demo_seed'],config,output,quick=args.quick)]
    else:
        scenarios=config['calibration_scenarios'][:1] if args.quick else config['calibration_scenarios']
        seeds=config['calibration_seeds'][:1] if args.quick else config['calibration_seeds']
        worlds=[(sc,seed+i*config['calibration_seed_stride']) for i,sc in enumerate(scenarios) for seed in seeds]
        runs=[run_world(sc,seed,config,output/f'{sc}-{seed}',calibration=True,quick=args.quick) for sc,seed in worlds]
        calibration_report(output)
    meta=dict(synthetic_only=True,mode=args.mode,quick=args.quick,runs=runs,
              versions={m:importlib.import_module(m).__version__ for m in ('numpy','pandas','scipy','matplotlib')},
              python=platform.python_version(),source_sha256=source_hashes(),wall_seconds=time.perf_counter()-start,
              calibration_note='Stress validation under four deliberately misspecified generators, not simulation-based calibration from the Bayesian prior. Worlds are the bootstrap unit; only three seeds per scenario.',
              config=config,model_config=default_config_dict())
    write_json(output/'run_metadata.json',meta)
    from scripts.plot_results import render
    render(output,'calibration' if args.mode=='calibration' else 'demo')
    print(f'Wrote {output}; elapsed {meta["wall_seconds"]:.1f}s',flush=True)

if __name__=='__main__': main()
