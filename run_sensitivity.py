#!/usr/bin/env python3
"""Prespecified age-tail support sensitivity on the exact same fictional data."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from cohort_lab.accounting import load_parameters
from cohort_lab.generator import generate_world
from bayes_cohort import forecast_cohort, default_config_dict
from scipy.integrate import quad
from scipy.stats import invgamma
from run_bayesian_demo import first_payback,write_json,source_hashes

ROOT=Path(__file__).resolve().parent

def main():
    setup=json.loads((ROOT/'demo_config.json').read_text())
    params=load_parameters()
    params['leads_per_cohort']=setup['demo_leads_per_cohort']
    params['cohorts_per_source']=setup['demo_cohorts_per_source']
    world,truth,policy=generate_world(setup['demo_seed'],setup['demo_scenario'],params)
    week=params['cohorts_per_source']-1
    supports={'default':{}, 'short_tail':{'tail_timescales':[20.,40.,80.]},
              'long_tail':{'tail_timescales':[80.,180.,400.]},
              'denser_grid':{'tail_timescales':[20.,40.,60.,90.,120.,180.,270.,400.]},
              'variance_cap_1':{'max_log_mark_variance':1.},
              'variance_cap_9':{'max_log_mark_variance':9.}}
    rows=[]
    for age in [3,60]:
        snap=world.as_of(week*params['cohort_spacing_days']+age)
        for source in ['low','middle','high']:
            cohort=f'{source}-w{week:02d}'
            for name,model_config in supports.items():
                result=forecast_cohort(snap,policy,cohort,config=model_config,draws=1000,seed=54321+age)
                paths=result['margin_paths'];tau=first_payback(paths)
                for h in [180,425]:
                    x=paths[:,h]/params['leads_per_cohort']
                    rows.append(dict(prior=name,source_id=source,origin_age=age,horizon=h,
                        mean=x.mean(),median=np.median(x),q025=np.quantile(x,.025),q975=np.quantile(x,.975),
                        probability_profitable=np.mean(x>0),probability_never=np.mean(~np.isfinite(tau)),
                        true_margin_per_lead=truth.cohort_path(world.leads,cohort,policy)[h]/params['leads_per_cohort']))
    output=ROOT/'artifacts'/'sensitivity';output.mkdir(exist_ok=True,parents=True)
    df=pd.DataFrame(rows);df.to_csv(output/'tail_sensitivity.csv',index=False)
    write_json(output/'assumptions.json',dict(seed=setup['demo_seed'],draws=1000,prior_settings=supports,source_sha256=source_hashes(),
        note='Same observed data; each setting changes only the declared tail support or variance cap. This is sensitivity analysis, not optimized model selection. All six alternatives are retained.'))
    lines=['# Tail and revenue-variance prior sensitivity','',
           'All inputs are fictional. The same data are fitted with four prespecified discrete tail supports and log-variance caps of 1, 4 (default) and 9; each setting changes only the named assumption. These are different models, not a search for the most favorable result. Finite Monte Carlo errors remain.','',
           'Posterior predictive day-180 contribution per acquired lead (fictional USD):','',
           '| Source | Origin | Prior setting | Mean | 95% predictive interval | P(profit) |','|---|---:|---|---:|---|---:|']
    for r in df.loc[df.horizon.eq(180)].itertuples():
        lines.append(f'| {r.source_id} | {r.origin_age} | {r.prior} | {r.mean:.3f} | [{r.q025:.3f}, {r.q975:.3f}] | {r.probability_profitable:.1%} |')
    lines += ['', 'A large sensitivity to prior support means that tail assumptions remain decision-relevant. A small difference here does not validate omitted reactivation, calendar shocks or informative missingness. See `tail_sensitivity.csv` for the terminal-horizon results and evaluator-only truth.','', 'Reproduce: `python run_sensitivity.py`','']
    cold_rows=[]
    snap=world.as_of(3)
    for source in ['low','middle','high']:
        for cap in [1.,4.,9.]:
            result=forecast_cohort(snap,policy,f'{source}-w00',config={'max_log_mark_variance':cap},draws=2000,seed=34567)
            x=result['margin_paths'][:,180]/params['leads_per_cohort']
            cold_rows.append(dict(source_id=source,origin_age=3,variance_cap=cap,draws=2000,mean=x.mean(),
                mean_mcse=x.std(ddof=1)/np.sqrt(len(x)),median=np.median(x),q025=np.quantile(x,.025),
                q975=np.quantile(x,.975),probability_profitable=np.mean(x>0)))
    pd.DataFrame(cold_rows).to_csv(output/'cold_start_variance_sensitivity.csv',index=False)
    prior=default_config_dict()
    moments=[]
    for cap in [1.,4.,9.]:
        a,b=prior['log_mark_prior_shape'],prior['log_mark_prior_scale']
        mu,k=prior['log_mark_prior_mean'],prior['log_mark_prior_precision']
        z=invgamma.cdf(cap,a,scale=b)
        m=quad(lambda v: np.exp(mu+.5*v*(1+1/k))*invgamma.pdf(v,a,scale=b)/z,0,cap)[0]
        second=quad(lambda v: np.exp(2*mu+2*v*(1+1/k))*invgamma.pdf(v,a,scale=b)/z,0,cap)[0]
        moments.append(dict(variance_cap=cap,positive_mark_prior_mean=m,positive_mark_prior_sd=np.sqrt(second-m*m)))
    pd.DataFrame(moments).to_csv(output/'positive_mark_prior_moments.csv',index=False)
    lines += ['## Cold-start monetary-prior sensitivity','',
        'The mature-history comparisons above barely respond to variance caps because observed mature marks dominate that part of the prior. The first cohort at age 3 has no mature marks. Here are all three prespecified caps on that same cold-start snapshot, with 2,000 draws per setting. These heavy-tailed sample means and their empirical MCSE can still be unstable; neither is a reliable bound on unseen extremes.','',
        '| Source | Variance cap | Mean / lead | Median / lead | 95% interval | P(profit) |',
        '|---|---:|---:|---:|---|---:|']
    for r in cold_rows:
        lines.append(f"| {r['source_id']} | {r['variance_cap']:g} | {r['mean']:.3f} | {r['median']:.3f} | [{r['q025']:.3f}, {r['q975']:.3f}] | {r['probability_profitable']:.1%} |")
    lines += ['','Analytic positive-mark prior moments below integrate the bounded variance mixture, including uncertainty in the log mean. They describe a single **positive** click payment before mature data, not total cohort profit. These quadrature results make the prior tail risk visible without relying on a few Monte Carlo draws.','',
        '| Variance cap | Prior mean positive payment | Prior standard deviation |','|---|---:|---:|']
    for r in moments:
        lines.append(f"| {r['variance_cap']:g} | {r['positive_mark_prior_mean']:.4f} | {r['positive_mark_prior_sd']:.3g} |")
    lines += ['','All moments are finite, yet a large standard deviation can make finite-draw economic means unreliable. This is why a finite mathematical expectation alone is not an adequate prior-predictive check. The default prior is retained to make its cold-start fragility inspectable; it is not recommended as a production monetary prior.','']
    (output/'RESULTS.md').write_text('\n'.join(lines))
    print(f'Wrote {output}')

if __name__=='__main__': main()
