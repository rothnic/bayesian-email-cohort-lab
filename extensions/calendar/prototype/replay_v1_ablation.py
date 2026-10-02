"""Execute unchanged-v1 calendar replay; one development world, no gain claim."""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
_CANDIDATES=(*HERE.parents,HERE.parents[1]/'bayesian-email-cohort-lab')
DEFAULT_LAB=next((p for p in _CANDIDATES if (p/'bayes_cohort/model.py').exists()),
                 HERE.parents[1]/'bayesian-email-cohort-lab')
TABLES=('leads','exposure_daily','engagement_events','ledger','operational_exits')


@dataclass(frozen=True)
class FrozenInput:
    payload: bytes

    @property
    def digest(self):return sha256(self.payload).hexdigest()

    def thaw(self, snapshot_type):
        obj=json.loads(self.payload)
        frames={name:pd.DataFrame(obj[name]['rows'],columns=obj[name]['columns']) for name in TABLES}
        return snapshot_type(cutoff=obj['cutoff'],**frames)


def freeze(snapshot):
    obj={'cutoff':int(snapshot.cutoff)}
    for name in TABLES:
        frame=getattr(snapshot,name)
        values=frame.astype(object).where(pd.notna(frame),None)
        obj[name]={'columns':list(frame.columns),'rows':values.values.tolist()}
    return FrozenInput(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode())


def select(snapshot, target, scope, snapshot_type):
    meta=snapshot.leads.loc[snapshot.leads.cohort_id.eq(target)]
    if meta.empty:raise ValueError('Target does not yet exist')
    source=meta.source_id.iloc[0]
    if scope=='target_only':chosen={target}
    elif scope=='same_source':chosen=set(snapshot.leads.loc[snapshot.leads.source_id.eq(source),'cohort_id'])
    else:raise ValueError('v1 can compare only target and same-source information')
    return snapshot_type(snapshot.cutoff,**{name:getattr(snapshot,name).loc[getattr(snapshot,name).cohort_id.isin(chosen)].copy() for name in TABLES})


def forecast_frozen(frozen, policy, target, fixed_config, draws, seed, snapshot_type, forecaster, clear):
    # Function accepts no World or GeneratorTruth; each fit starts at the fixed prior.
    snapshot=frozen.thaw(snapshot_type)
    clear()
    result=forecaster(snapshot,policy,target,config=json.loads(fixed_config),draws=draws,seed=seed)
    assert freeze(snapshot).digest==frozen.digest, 'The model mutated its as-of input'
    paths=np.asarray(result['margin_paths'])
    x=paths[:,180]
    crossed=paths>=0
    tau=np.where(crossed.any(axis=1),crossed.argmax(axis=1),np.inf)
    return dict(mean180=float(x.mean()),mean_mcse180=float(x.std(ddof=1)/np.sqrt(len(x))),
                median180=float(np.median(x)),q025_180=float(np.quantile(x,.025)),q975_180=float(np.quantile(x,.975)),
                p_positive180=float(np.mean(x>0)),p_first_payback425=float(np.mean(tau<=425)),
                p_no_first_payback425=float(np.mean(~np.isfinite(tau))))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--lab',type=Path,default=DEFAULT_LAB)
    parser.add_argument('--output',type=Path,default=HERE/'v1_replay');parser.add_argument('--draws',type=int,default=64)
    args=parser.parse_args();sys.path.insert(0,str(args.lab.resolve()))
    from cohort_lab.accounting import Snapshot,load_parameters
    from cohort_lab.generator import generate_world
    from bayes_cohort import forecast_cohort,default_config_dict
    from bayes_cohort.model import clear_fit_cache
    started=time.perf_counter();args.output.mkdir(parents=True,exist_ok=True)
    params=load_parameters(args.lab/'parameters.json');params.update(leads_per_cohort=50,cohorts_per_source=13)
    world,truth,policy=generate_world(seed=884901,scenario='collapse_tail',parameters=params)
    fixed=json.dumps(default_config_dict(),sort_keys=True,separators=(',',':')).encode()
    prior_hash=sha256(fixed).hexdigest();rows=[]
    source_files=['bayes_cohort/model.py','bayes_cohort/config.py','cohort_lab/generator.py','cohort_lab/accounting.py','parameters.json']
    source_hashes={name:sha256((args.lab/name).read_bytes()).hexdigest() for name in source_files}
    for cutoff in [10,17,31,59,87]:
        snap=world.as_of(cutoff)
        for target in ['middle-w00',f'middle-w{(cutoff-3)//7:02d}']:
            birth=int(snap.leads.loc[snap.leads.cohort_id.eq(target),'acquired_at'].iloc[0])
            forecast_seed=884901+cutoff*100+birth
            for scope in ['target_only','same_source']:
                frozen=freeze(select(snap,target,scope,Snapshot))
                copied=frozen.thaw(Snapshot)
                pred=forecast_frozen(frozen,policy,target,fixed,args.draws,forecast_seed,Snapshot,forecast_cohort,clear_fit_cache)
                row=dict(world_seed=884901,scenario='collapse_tail',calendar_cutoff=cutoff,target=target,
                         origin_age=cutoff-birth,scope=scope,cohorts=int(copied.leads.cohort_id.nunique()),
                         exposure_rows=len(copied.exposure_daily),observed_event_rows=len(copied.engagement_events),
                         input_sha256=frozen.digest,prior_sha256=prior_hash,draws=args.draws,draw_seed=forecast_seed,
                         forecast_type='unchanged_v1_posterior_predictive',evaluation_role='development_illustration',**pred)
                rows.append(row)
                print(cutoff,target,scope,'cohorts',row['cohorts'],flush=True)
    # Seal forecast output before touching complete evaluator outcomes.
    predictions=pd.DataFrame(rows);pred_path=args.output/'predictions.csv';predictions.to_csv(pred_path,index=False)
    seal=sha256(pred_path.read_bytes()).hexdigest()
    evaluation=[]
    for row in rows:
        true_path=truth.cohort_path(world.leads,row['target'],policy)
        actual=float(true_path[180]);hit=np.flatnonzero(true_path>=0)
        evaluation.append(dict(calendar_cutoff=row['calendar_cutoff'],target=row['target'],scope=row['scope'],
                               origin_age=row['origin_age'],true_margin180=actual,
                               absolute_mean_error180=abs(row['mean180']-actual),
                               covered95_180=bool(row['q025_180']<=actual<=row['q975_180']),
                               truth_first_payback425=int(hit[0]) if len(hit) else None,
                               prediction_file_sha256=seal))
    pd.DataFrame(evaluation).to_csv(args.output/'evaluator_comparison.csv',index=False)
    manifest=dict(stage='unchanged_v1_information_ablation',development_only=True,world_seed=884901,
                  scenario='collapse_tail',cohorts_per_source=13,leads_per_cohort=50,
                  calendar_cutoffs=[10,17,31,59,87],source='middle',draws=args.draws,forecasts=len(rows),
                  original_source_sha256=source_hashes,prior_sha256=prior_hash,prediction_sha256=seal,
                  terminal_age=425,raw_row_exports_written=False,new_hierarchical_model_fitted=False,
                  claimed_pooling_benefit=False,total_seconds=time.perf_counter()-started)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Sealed',len(rows),'forecasts. One development world is not calibration or a pooling-value estimate.',flush=True)


if __name__=='__main__':main()
