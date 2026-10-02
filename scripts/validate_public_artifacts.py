#!/usr/bin/env python3
"""Check generated evidence, provenance and accidental private-path leakage."""
import json
import hashlib
import re
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from run_bayesian_demo import source_hashes


def main():
    for mode in ['demo','cold_start','calibration']:
        out=ROOT/'artifacts'/mode
        meta=json.loads((out/'run_metadata.json').read_text())
        assert meta['synthetic_only'] is True and meta['quick'] is False
        assert meta['source_sha256']==source_hashes(), f'Source drift since {mode} run'
        summary=pd.read_csv(out/'summary.csv') if mode!='calibration' else pd.read_csv(out/'scores.csv')
        numeric=summary[['mean','median','q025','q10','q25','q75','q90','q975','true_margin','crps','brier']]
        assert np.isfinite(numeric.to_numpy()).all()
        qs=summary[['q025','q10','q25','median','q75','q90','q975']].to_numpy()
        assert (np.diff(qs,axis=1)>=0).all()
        for name in ['probability_profitable','probability_never','probability_payback_by_horizon','probability_later_than_180','probability_remains_positive']:
            assert summary[name].between(0,1).all(),name
        assert summary.crps.min()>=-1e-10
        if mode=='calibration':
            assert len(meta['runs'])==12
            assert len({r['seed'] for r in meta['runs']})==12,'World random streams must be distinct'
            assert (summary.horizon>summary.origin_age).all()
            assert summary.world_id.nunique()==12
        else:
            cdf=pd.read_csv(out/'payback_cdf.csv.gz')
            for _,g in cdf.groupby(['cohort_id','origin_age']):
                g=g.sort_values('age')
                assert (np.diff(g.cdf)>=-1e-12).all()
                assert abs(g.cdf.iloc[-1]+g.probability_never.iloc[-1]-1)<1e-10
            draws=pd.read_csv(out/'posterior_predictive_draws.csv.gz')
            assert (draws.loc[draws.payback_state.eq('never_under_policy'),'first_payback_age'].isna()).all()
            assert draws.loc[draws.payback_state.eq('later_than_180'),'first_payback_age'].between(181,425).all()
            assert set(draws.payback_state)<=set(['by_180','later_than_180','never_under_policy'])
        assert list((out/'figures').glob('*.png')),f'No figures for {mode}'
        assert (out/'RESULTS.md').exists()
    controlled=ROOT/'artifacts'/'controlled_scenario'
    cm=json.loads((controlled/'run_metadata.json').read_text())
    assert cm['independent_worlds']==2000 and cm['all_worlds_retained'] is True
    assert cm['module_crosscheck']['passed'] is True
    for name,digest in cm['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,f'Controlled source drift: {name}'
    assert hashlib.sha256((ROOT/cm['paired_stress_scores']).read_bytes()).hexdigest()==cm['paired_stress_scores_sha256']
    assert len(pd.read_csv(controlled/'exact_forecasts.csv.gz'))==10000
    assert len(list(controlled.glob('*.png')))==2
    forbidden=re.compile(r'(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}|/home/[A-Za-z0-9_-]+/|/Users/[A-Za-z0-9_-]+/')
    for path in ROOT.rglob('*'):
        if not path.is_file() or any(x in path.parts for x in ['__pycache__','.git','.venv']) or 'tmp-' in str(path):continue
        if path==Path(__file__).resolve():continue
        if path.suffix in ['.py','.md','.json','.toml','.txt','.csv']:
            assert not forbidden.search(path.read_text()),f'Public leakage scan: {path.relative_to(ROOT)}'
    assert not (ROOT/'.github'/'workflows').exists(),'No Actions workflow is authorized for this demo'
    print('PASS: provenance, finite outputs, probability/payback semantics, independent seeds, figures, public content scan')

if __name__=='__main__':main()
