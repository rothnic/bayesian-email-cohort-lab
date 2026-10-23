"""Run declared fictional experiments; seal predictions before truth scoring."""
from pathlib import Path
from hashlib import sha256
import argparse
import json
import time
import numpy as np
import pandas as pd
from .simulation import generate,conditions,freeze

ROOT=Path(__file__).resolve().parent
PROFIT_BUFFER_PER_LEAD=.02
BUY_THRESHOLD=.95
NO_BUY_THRESHOLD=.05


def crps(x,y):
    x=np.sort(np.asarray(x));n=len(x)
    return float(np.mean(abs(x-y))-np.sum((2*np.arange(1,n+1)-n-1)*x)/(n*n))


def json_write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def targets_for(world,cutoff,include_prospective=True):
    rows=[]
    sources=sorted(world.quotes.loc[world.quotes.observed_at<=cutoff,'source_id'].unique())
    for source in sources:
        first_birth=int(world.cohort_registry.loc[world.cohort_registry.source_id==source,'birth_day'].min())
        existing=[(first_birth,'first'),(cutoff-7,'newest')]
        for birth,role in existing:
            if birth>cutoff:continue
            match=world.cohort_registry.loc[(world.cohort_registry.source_id==source)&(world.cohort_registry.birth_day==birth)]
            if len(match):
                t=match.iloc[0].to_dict();t['target_role']=role;rows.append(t)
        if include_prospective:
            # A conditional purchase decision: commit to next batch at today's observed quote.
            birth=cutoff+7
            match=world.cohort_registry.loc[(world.cohort_registry.source_id==source)&(world.cohort_registry.birth_day==birth)]
            if len(match):
                t=match.iloc[0].to_dict();t['target_role']='next_purchase'
                t['acquisition_cost']=world.quote_at(source,cutoff)*t['leads']
                rows.append(t)
            # The quote preceding launch is known, but no new-source outcomes are.
            if first_birth>cutoff and first_birth<=cutoff+7:
                match=world.cohort_registry.loc[(world.cohort_registry.source_id==source)&(world.cohort_registry.birth_day==first_birth)]
                t=match.iloc[0].to_dict();t['target_role']='next_purchase'
                t['acquisition_cost']=world.quote_at(source,cutoff)*t['leads']
                rows.append(t)
    return rows


def summarize(paths,target,base):
    x=paths[:,180];tau=np.where((paths>=0).any(axis=1),(paths>=0).argmax(axis=1),np.inf)
    q=np.quantile(x,[.025,.1,.5,.9,.975])
    return {**base,'cohort_id':target['cohort_id'],'source_id':target['source_id'],
            'birth_day':target['birth_day'],'origin_age':base['origin_day']-target['birth_day'],
            'target_role':target.get('target_role','existing'),'target_age':180,'n_leads':target['leads'],
            'acquisition_cost':target['acquisition_cost'],'mean':float(x.mean()),
            'mean_mcse':float(x.std(ddof=1)/np.sqrt(len(x))),'q025':float(q[0]),'q10':float(q[1]),
            'median':float(q[2]),'q90':float(q[3]),'q975':float(q[4]),
            'profit_buffer_per_lead':PROFIT_BUFFER_PER_LEAD,
            'p_positive180':float(np.mean(x>0)),
            'p_margin_above_buffer180':float(np.mean(x>PROFIT_BUFFER_PER_LEAD*target['leads'])),
            'decision_at_95_5':('buy' if np.mean(x>PROFIT_BUFFER_PER_LEAD*target['leads'])>=BUY_THRESHOLD else
                                'skip' if np.mean(x>PROFIT_BUFFER_PER_LEAD*target['leads'])<=NO_BUY_THRESHOLD else 'wait'),
            'p_payback425':float(np.mean(tau<=425)),
            'p_no_payback425':float(np.mean(~np.isfinite(tau))),'draws':len(x)}


def evaluate_rows(pending):
    scored=[]
    for row,paths,evaluator_input in pending:
        truth_path=actual_path(*evaluator_input)
        actual=float(truth_path[180]);x=paths[:,180];n=row['n_leads']
        buy=bool(row['decision_at_95_5']=='buy')
        regret=(max(actual,0)-float(buy)*actual)/n if row['target_role']=='next_purchase' else np.nan
        scored.append({**row,'true_margin':actual,'coverage95':float(row['q025']<=actual<=row['q975']),
                       'coverage80':float(row['q10']<=actual<=row['q90']),
                       'crps_per_lead':crps(x,actual)/n,'mae_per_lead':abs(row['mean']-actual)/n,
                       'brier':(row['p_positive180']-float(actual>0))**2,
                       'decision_loss_per_lead':regret})
    return pd.DataFrame(scored)


def actual_path(world,truth,target):
    path=truth.cohort_paths[target['cohort_id']].copy()
    if target.get('target_role')=='next_purchase':
        # Only future hypothetical purchases use the as-of quote. Old paths stay fixed.
        original=float(world.cohort_registry.loc[world.cohort_registry.cohort_id==target['cohort_id'],'acquisition_cost'].iloc[0])
        path+=original-target['acquisition_cost']
    return path


def fit_predict(frame,cutoff,mode,targets,draws,seed,future):
    from .model import fit
    before=freeze(frame,cutoff).digest
    fitted=fit(frame,cutoff,mode=mode,seed=seed)
    result=fitted.predict(targets,draws=draws,horizon=425,scenario=future,seed=seed+1)
    if freeze(frame,cutoff).digest!=before:raise AssertionError('Inference modified its supplied input')
    paths=np.asarray(result['margin_paths'])
    if paths.shape!=(draws,len(targets),426) or not np.isfinite(paths).all():
        raise AssertionError(f'Bad predictive paths: {paths.shape}')
    return fitted,result


def demo(config,out):
    forecasts=[];pending=[];trajectories=[];cdfs=[];calendar=[];measurement=[];states=[];transfer=[];diagnostics=[];portfolio=[];portfolio_draws=[];quote_sensitivity=[];retention_truth=[];retention_forecast=[];retention_rollups=[];retention_rollup_detail=[];demo_confidence=[];cpc_shocks=[]
    for scenario in config['scenarios']:
        world,truth=generate(config['demo_seed'],scenario,config['leads_per_cohort'],config['last_birth'])
        wid=f'demo-{scenario}-{config["demo_seed"]}'
        econ=truth.calendar.loc[truth.calendar.calendar_day<=126].copy()
        byday=world.cohort_registry.loc[world.cohort_registry.source_id=='established_a'].groupby('birth_day').agg(new_purchases=('leads','sum'),acquisition_total=('acquisition_cost','sum'))
        for r in econ.itertuples():
            daily=byday.loc[r.calendar_day] if r.calendar_day in byday.index else None
            calendar.append(dict(world=wid,scenario=scenario,source_id='established_a',calendar_day=r.calendar_day,price_per_send=.00025,
                                 human_cpc=r.human_cpc,cpl_quote=world.quote_at('established_a',r.calendar_day),
                                 new_purchases=0 if daily is None else int(daily.new_purchases),
                                 acquisition_total=0. if daily is None else float(daily.acquisition_total),
                                 value_role='evaluator_only_calendar_and_observed_quotes'))
        q=truth.quality.loc[truth.quality.calendar_day<=126].groupby('calendar_day')[['true_human_events','raw_events','audit_n','audit_human']].sum()
        for day,r in q.iterrows():
            n=int(r.audit_n);labels=int(r.audit_human)
            # Display only audits available by this plotted calendar day, using their report clock.
            mature=world.as_of(int(day)).frame();m=mature.loc[mature.audit_n.notna()]
            recent=m.loc[m.calendar_day>=day-14]
            an=float(recent.audit_n.sum());ah=float(recent.audit_human.sum())
            labelp=(ah+1)/(an+2);humanp=np.clip((labelp-.02)/.93,0,1)
            reported=world.observations.loc[world.observations.observed_at==day,'raw_events'].sum()
            measurement.append(dict(world=wid,scenario=scenario,calendar_day=day,
                                    true_human_events=int(r.true_human_events),reported_events=int(reported),
                                    audited_bot_fraction=float(1-humanp),audit_n=int(an),
                                    measurement_note='Reported activity uses report date (2daylag); truth uses occurrence date. Audits use available trailing14day cells; knownSe95Sp98.'))
        for cutoff in config['demo_cutoffs']:
            frozen=world.as_of(cutoff);targets=targets_for(world,cutoff)
            for mode in ['static_hierarchy','dynamic_hierarchy']:
                seed=config['demo_seed']+cutoff*100+(0 if mode=='static_hierarchy' else 1)
                fitted,pred=fit_predict(frozen.frame(),cutoff,mode,targets,config['demo_draws'],seed,'hold_current')
                variants={'hold_current':pred}
                if mode=='dynamic_hierarchy' and cutoff==119:
                    for future in ['fixed_price','continue_recent_trend']:
                        variants[future]=fitted.predict(targets,draws=config['demo_draws'],horizon=425,scenario=future,seed=seed+1)
                diagnostics.append(dict(world=wid,scenario=scenario,origin_day=cutoff,model=mode,diagnostics=pred.get('diagnostics',{})))
                for future,result in variants.items():
                    if scenario=='human_decline_bot_rise' and mode=='dynamic_hierarchy' and cutoff==119:
                        common_day=cutoff+180
                        target_ages=np.array([common_day-t['birth_day'] for t in targets],dtype=int)
                        assert np.all((target_ages>=0)&(target_ages<=425))
                        joint=np.asarray(result['margin_paths'])[:,np.arange(len(targets)),target_ages].sum(axis=1)
                        true_joint=sum(actual_path(world,truth,t)[a] for t,a in zip(targets,target_ages))
                        qj=np.quantile(joint,[.025,.5,.975])
                        portfolio.append(dict(world=wid,scenario=scenario,model=mode,origin_day=cutoff,
                            calendar_horizon=common_day,future_scenario=future,target_count=len(targets),
                            total_leads=sum(t['leads'] for t in targets),mean=float(joint.mean()),
                            mean_mcse=float(joint.std(ddof=1)/np.sqrt(len(joint))),q025=qj[0],median=qj[1],q975=qj[2],
                            p_positive=float(np.mean(joint>0)),true_margin=float(true_joint),draws=len(joint),
                            target_ids=';'.join(t['cohort_id'] for t in targets),
                            target_ages=';'.join(str(a) for a in target_ages),
                            note='Common calendar horizon; shared states per draw; includes conditional future purchases'))
                        for draw,value in enumerate(joint):
                            portfolio_draws.append(dict(world=wid,origin_day=cutoff,calendar_horizon=common_day,
                                                        future_scenario=future,draw_id=draw,margin=float(value)))
                    for j,t in enumerate(targets):
                        paths=np.asarray(result['margin_paths'])[:,j,:]
                        base=dict(world=wid,scenario=scenario,model=mode,origin_day=cutoff,future_scenario=future,input_sha256=frozen.digest)
                        row=summarize(paths,t,base);forecasts.append(row)
                        pending.append((row,paths,(world,truth,t)))
                        if mode=='dynamic_hierarchy' and cutoff==119 and t['source_id']=='established_a' and t['target_role']=='newest':
                            qq=np.quantile(paths,[.025,.1,.5,.9,.975],axis=0)
                            tau=np.where((paths>=0).any(axis=1),(paths>=0).argmax(axis=1),np.inf)
                            tp=actual_path(world,truth,t)
                            for age in range(426):
                                shared={**base,'cohort_id':t['cohort_id'],'source_id':t['source_id'],'birth_day':t['birth_day'],'n_leads':t['leads'],'age':age}
                                trajectories.append({**shared,'q025':qq[0,age],'q10':qq[1,age],'median':qq[2,age],'q90':qq[3,age],'q975':qq[4,age],'true_margin':tp[age]})
                                cdfs.append({**shared,'cdf':float(np.mean(tau<=age)),'p_no_payback425':float(np.mean(~np.isfinite(tau)))})
                if mode=='dynamic_hierarchy':
                    dates=np.asarray(pred.get('calendar_dates',[]),dtype=int)
                    for key,label,truth_col in [('calendar_response','response','human_response_multiplier'),('calendar_cpc','value','human_cpc')]:
                        arr=np.asarray(pred.get(key,[]))
                        if len(dates) and arr.shape==(config['demo_draws'],len(dates)):
                            qq=np.quantile(arr,[.025,.5,.975],axis=0)
                            for i,day in enumerate(dates):
                                tv=truth.calendar.loc[truth.calendar.calendar_day==day,truth_col]
                                reference=float(truth.calendar.loc[truth.calendar.calendar_day==0,truth_col].iloc[0])
                                states.append(dict(world=wid,scenario=scenario,origin_day=cutoff,calendar_day=day,state=label,
                                                   true_value=float(np.log(tv.iloc[0]/reference)) if len(tv) else np.nan,
                                                   median=qq[1,i],q025=qq[0,i],q975=qq[2,i],interval_kind='parameter_state',
                                                   state_scale='log_ratio_reference_day0'))
        if scenario in ['cheap_good','cheap_bad']:
            for cutoff,stage in [(69,'prior'),(91,'posterior'),(119,'posterior')]:
                frozen=world.as_of(cutoff)
                target=world.cohort_registry.loc[world.cohort_registry.cohort_id=='new_cheap-b070'].iloc[0].to_dict()
                target['target_role']='new_source'
                if cutoff==69:target['acquisition_cost']=world.quote_at('new_cheap',69)*target['leads']
                learned,result=fit_predict(frozen.frame(),cutoff,'dynamic_hierarchy',[target],config['demo_draws'],config['demo_seed']+cutoff,'hold_current')
                paths=result['margin_paths'][:,0,:]
                row=summarize(paths,target,dict(world=wid,scenario=scenario,model='dynamic_hierarchy',origin_day=cutoff,future_scenario='hold_current',input_sha256=frozen.digest))
                transfer.append({**row,'stage':stage,'new_source_kind':scenario.removeprefix('cheap_'),
                                 'observed_events':int(frozen.frame().loc[lambda x:x.source_id=='new_cheap','raw_events'].sum()),
                                 'true_margin':float(truth.cohort_paths[target['cohort_id']][180])})
                if cutoff==119:
                    next_target=world.cohort_registry.loc[world.cohort_registry.cohort_id=='new_cheap-b126'].iloc[0].to_dict()
                    next_target['acquisition_cost']=world.quote_at('new_cheap',cutoff)*next_target['leads']
                    next_target['target_role']='next_purchase'
                    next_draws=learned.predict([next_target],draws=config['demo_draws'],horizon=425,scenario='hold_current',seed=config['demo_seed']+1119)
                    base_paths=np.asarray(next_draws['margin_paths'])[:,0,:]
                    previous=None
                    for offered_quote in [.05,.10,.15,.20,.30]:
                        changed=base_paths+next_target['acquisition_cost']-offered_quote*next_target['leads']
                        tau=np.where((changed>=0).any(axis=1),(changed>=0).argmax(axis=1),np.inf)
                        probability=float(np.mean(tau<=425))
                        p_buffer=float(np.mean(changed[:,180]>PROFIT_BUFFER_PER_LEAD*next_target['leads']))
                        if previous is not None:assert probability<=previous
                        previous=probability
                        qu=np.quantile(changed[:,180],[.025,.5,.975]);med=float(np.median(tau))
                        quote_sensitivity.append(dict(world=wid,scenario=scenario,new_source_kind=scenario.removeprefix('cheap_'),
                            origin_day=cutoff,birth_day=126,cohort_id=next_target['cohort_id'],n_leads=next_target['leads'],
                            offered_cpl=offered_quote,total_purchase_cost=offered_quote*next_target['leads'],
                            mean_margin180=float(changed[:,180].mean()),median_margin180=qu[1],q025_margin180=qu[0],q975_margin180=qu[2],
                            p_margin_above_buffer180=p_buffer,
                            expected_shortfall_below_buffer_per_lead=float(np.maximum(
                                PROFIT_BUFFER_PER_LEAD*next_target['leads']-changed[:,180],0).mean()/next_target['leads']),
                            p_payback180=float(np.mean(tau<=180)),p_payback425=probability,p_no_payback425=1-probability,
                            median_first_payback=None if not np.isfinite(med) else med,draws=len(tau),
                            future_scenario='hold_current',input_sha256=frozen.digest,
                            quality_inference='identical posterior and common outcome draws across quotes',
                            cost_scope='prospective new cohort only; no existing cost altered'))
            target=world.cohort_registry.loc[world.cohort_registry.cohort_id=='new_cheap-b070'].iloc[0].to_dict()
            target['target_role']='same_fixed_cohort';target['acquisition_cost']=world.quote_at('new_cheap',69)*target['leads']
            actual=float(actual_path(world,truth,target)[180])
            for cutoff in [69,77,84,91,98,105,112,119]:
                frozen=world.as_of(cutoff)
                _,pred=fit_predict(frozen.frame(),cutoff,'dynamic_hierarchy',[target],config['demo_draws'],
                    config['demo_seed']+cutoff+1701,'hold_current')
                x=np.asarray(pred['margin_paths'])[:,0,180]
                probability=float(np.mean(x>PROFIT_BUFFER_PER_LEAD*target['leads']))
                demo_confidence.append(dict(world=wid,scenario=scenario,model='dynamic_hierarchy',
                    cohort_id=target['cohort_id'],source_id=target['source_id'],birth_day=target['birth_day'],
                    origin_day=cutoff,origin_age=cutoff-target['birth_day'],target_age=180,
                    profit_buffer_per_lead=PROFIT_BUFFER_PER_LEAD,posterior_probability=probability,
                    state='above_target' if probability>=BUY_THRESHOLD else
                    'below_target' if probability<=NO_BUY_THRESHOLD else 'undecided',true_margin180=actual,
                    true_above_buffer=actual>PROFIT_BUFFER_PER_LEAD*target['leads'],draws=len(x)))
        if scenario=='cpc_squeeze':
            retention_truth.extend(truth.cohort_truth.loc[truth.cohort_truth.age<=180].to_dict('records'))
            raw_rollups=truth.cohort_rollups
            for r in raw_rollups.itertuples(index=False):
                retention_rollup_detail.append(dict(world=wid,scenario=scenario,**r._asdict(),
                    contactable_share_of_acquired=r.contactable_at_end/r.acquired_leads,
                    unique_response_rate_of_eligible=(r.distinct_human_responders/r.eligible_at_start
                        if r.eligible_at_start else np.nan)))
            for (source,window,start),g in raw_rollups.groupby(['source_id','window_days','window_start_age']):
                acquired=float(g.acquired_leads.sum());eligible=float(g.eligible_at_start.sum())
                retention_rollups.append(dict(world=wid,scenario=scenario,source_id=source,
                    window_days=int(window),window_start_age=int(start),window_end_age=int(g.window_end_age.max()),
                    cohort_count=int(g.cohort_id.nunique()),acquired_leads=int(acquired),
                    eligible_people_at_start=int(eligible),contactable_people_at_end=int(g.contactable_at_end.sum()),
                    distinct_human_responders=int(g.distinct_human_responders.sum()),
                    contactable_share_of_acquired=float(g.contactable_at_end.sum()/acquired),
                    unique_response_rate_of_eligible=float(g.distinct_human_responders.sum()/eligible) if eligible else np.nan,
                    window_completeness='complete_windows_only',truth_role='evaluator_only; weighted pooled counts'))
            cohort='established_a-b070'
            target=world.cohort_registry.loc[world.cohort_registry.cohort_id==cohort].iloc[0].to_dict()
            target['target_role']='sequential_candidate'
            actual=float(truth.cohort_paths[cohort][180])
            for cutoff in [77,84,91,98,105,112,119]:
                frozen=world.as_of(cutoff)
                learned,pred=fit_predict(frozen.frame(),cutoff,'dynamic_hierarchy',[target],
                    config['demo_draws'],config['demo_seed']+cutoff+909,'hold_current')
                paths=np.asarray(pred['margin_paths'])[:,0,:];x=paths[:,180]
                probability=float(np.mean(x>PROFIT_BUFFER_PER_LEAD*target['leads']))
                demo_confidence.append(dict(world=wid,scenario=scenario,model='dynamic_hierarchy',cohort_id=cohort,
                    source_id=target['source_id'],birth_day=target['birth_day'],origin_day=cutoff,
                    origin_age=cutoff-target['birth_day'],target_age=180,profit_buffer_per_lead=PROFIT_BUFFER_PER_LEAD,
                    posterior_probability=probability,state='above_target' if probability>=BUY_THRESHOLD else
                    'below_target' if probability<=NO_BUY_THRESHOLD else 'undecided',true_margin180=actual,
                    true_above_buffer=actual>PROFIT_BUFFER_PER_LEAD*target['leads'],draws=len(x)))
                if cutoff==119:
                    pred_ret=pred['expected_contactable_count_paths'][:,0,:]/target['leads']
                    pred_ret_predictive=pred['contactable_count_paths'][:,0,:]/target['leads']
                    pred_uni=pred['expected_unique_response_paths'][:,0,:]/target['leads']
                    truth_curve=truth.cohort_truth.loc[truth.cohort_truth.cohort_id==cohort].sort_values('age')
                    for age in range(181):
                        retention_forecast.append(dict(cohort_id=cohort,source_id=target['source_id'],
                            birth_day=target['birth_day'],age=age,origin_day=cutoff,
                            contactable_q05=np.quantile(pred_ret[:,age],.05),contactable_median=np.median(pred_ret[:,age]),
                            contactable_q95=np.quantile(pred_ret[:,age],.95),
                            contactable_predictive_q05=np.quantile(pred_ret_predictive[:,age],.05),
                            contactable_predictive_median=np.median(pred_ret_predictive[:,age]),
                            contactable_predictive_q95=np.quantile(pred_ret_predictive[:,age],.95),
                            expected_unique_response_q05=np.quantile(pred_uni[:,age],.05),
                            expected_unique_response_median=np.median(pred_uni[:,age]),
                            expected_unique_response_q95=np.quantile(pred_uni[:,age],.95),
                            true_contactable_fraction=float(truth_curve.loc[truth_curve.age==age,'true_contactable_fraction'].iloc[0]),
                            true_expected_contactable_fraction=float(truth_curve.loc[
                                truth_curve.age==age,'true_expected_contactable_fraction'].iloc[0]),
                            true_expected_unique_response_per_lead=float(truth_curve.loc[
                                truth_curve.age==age,'true_expected_unique_response_per_lead'].iloc[0]),
                            true_realized_unique_response_per_lead=float(truth_curve.loc[
                                truth_curve.age==age,'true_unique_response_per_lead'].iloc[0])))
                    # Use the higher-cost established source near the decision
                    # boundary so a CPC stress test can visibly change profit
                    # probability while holding its people and clicks in common.
                    next_target=world.cohort_registry.loc[world.cohort_registry.cohort_id=='established_b-b126'].iloc[0].to_dict()
                    next_target['acquisition_cost']=world.quote_at('established_b',cutoff)*next_target['leads']
                    for drop in [0.,.15,.30,.45]:
                        shock=None if drop==0 else {'effective_day':140,'drop_fraction':drop,'ramp_days':28}
                        baseline_whatif=learned.predict([next_target],draws=config['demo_draws'],horizon=425,
                            scenario='hold_current',seed=config['demo_seed']+1919)
                        whatif=learned.predict([next_target],draws=config['demo_draws'],horizon=425,
                            scenario='hold_current',seed=config['demo_seed']+1919,cpc_shock=shock)
                        values=whatif['margin_paths'][:,0,180];q=np.quantile(values,[.05,.5,.95])
                        delta=values-baseline_whatif['margin_paths'][:,0,180]
                        dq=np.quantile(delta,[.05,.5,.95])
                        cpc_shocks.append(dict(cohort_id=next_target['cohort_id'],origin_day=cutoff,
                            source_id=next_target['source_id'],birth_day=next_target['birth_day'],
                            cpc_drop_fraction=drop,effective_day=140,ramp_days=28,
                            p_margin_above_buffer180=float(np.mean(values>PROFIT_BUFFER_PER_LEAD*next_target['leads'])),
                            expected_shortfall_below_buffer_per_lead=float(np.maximum(
                                PROFIT_BUFFER_PER_LEAD*next_target['leads']-values,0).mean()/next_target['leads']),
                            paired_margin_delta_q05=dq[0],paired_margin_delta_median=dq[1],paired_margin_delta_q95=dq[2],
                            margin_q05=q[0],margin_median=q[1],margin_q95=q[2],draws=len(values),
                            common_draw_seed=config['demo_seed']+1919,
                            note='Paired posterior draws; only future CPC changes; human activity, retention, CPL and costs are held fixed'))
    pd.DataFrame(forecasts).to_csv(out/'forecasts_unscored.csv',index=False)
    json_write(out/'forecast_seal.json',{'sha256':sha256((out/'forecasts_unscored.csv').read_bytes()).hexdigest(),'rows':len(forecasts)})
    evaluate_rows(pending).to_csv(out/'cohort_predictions.csv',index=False)
    for name,rows in [('calendar_economics',calendar),('measurement',measurement),('calendar_states',states),
                      ('trajectories',trajectories),('payback_cdf',cdfs),('transfer',transfer),
                      ('portfolio_summary',portfolio),('portfolio_draws',portfolio_draws),('quote_sensitivity',quote_sensitivity),
                      ('retention_truth',retention_truth),('retention_forecast',retention_forecast),
                      ('cohort_retention_rollups',retention_rollups),
                      ('cohort_retention_rollup_detail',retention_rollup_detail),
                      ('demo_confidence',demo_confidence),('cpc_shock_sensitivity',cpc_shocks)]:
        pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False)
    json_write(out/'diagnostics.json',diagnostics)


def heldout(config,out):
    pending=[];forecasts=[];timings=[];sequential=[]
    for si,scenario in enumerate(config['scenarios']):
        for base_seed in config['heldout_base_seeds']:
            seed=base_seed+si*config['scenario_seed_stride'];started=time.perf_counter()
            world,truth=generate(seed,scenario,config['leads_per_cohort'],config['last_birth'])
            wid=f'heldout-{scenario}-{seed}'
            cutoffs=config['evaluation_cutoffs']
            if scenario in ('cheap_good','cheap_bad'):
                cutoffs=sorted(set(cutoffs+config['cheap_source_evaluation_cutoffs']))
            for cutoff in cutoffs:
                frozen=world.as_of(cutoff);targets=targets_for(world,cutoff)
                for mi,mode in enumerate(config['models']):
                    _,pred=fit_predict(frozen.frame(),cutoff,mode,targets,config['evaluation_draws'],seed*100+cutoff+mi,'hold_current')
                    for j,t in enumerate(targets):
                        paths=np.asarray(pred['margin_paths'])[:,j,:]
                        row=summarize(paths,t,dict(world=wid,scenario=scenario,model=mode,origin_day=cutoff,future_scenario='hold_current',input_sha256=frozen.digest))
                        forecasts.append(row);pending.append((row,paths,(world,truth,t)))
            if scenario in ('fixed_price','cpc_squeeze','rising_cpl','human_decline_bot_rise'):
                match=world.cohort_registry.loc[world.cohort_registry.cohort_id=='established_a-b070']
                if len(match):
                    target=match.iloc[0].to_dict();target['target_role']='sequential_candidate'
                    actual=float(actual_path(world,truth,target)[180])
                    for cutoff in config['sequential_cutoffs']:
                        frozen=world.as_of(cutoff)
                        _,pred=fit_predict(frozen.frame(),cutoff,'dynamic_hierarchy',[target],
                            config['sequential_draws'],seed*100+cutoff+701,'hold_current')
                        paths=np.asarray(pred['margin_paths'])[:,0,:]
                        x=paths[:,180];prob=float(np.mean(x>PROFIT_BUFFER_PER_LEAD*target['leads']))
                        state='above_target' if prob>=BUY_THRESHOLD else 'below_target' if prob<=NO_BUY_THRESHOLD else 'undecided'
                        sequential.append(dict(world=wid,scenario=scenario,model='dynamic_hierarchy',
                            cohort_id=target['cohort_id'],source_id=target['source_id'],birth_day=target['birth_day'],
                            origin_day=cutoff,origin_age=cutoff-target['birth_day'],target_age=180,
                            profit_buffer_per_lead=PROFIT_BUFFER_PER_LEAD,
                            p_margin_above_buffer180=prob,state=state,posterior_draws=len(x),
                            true_margin180=actual,true_above_buffer=actual>PROFIT_BUFFER_PER_LEAD*target['leads'],
                            input_sha256=frozen.digest,interval_mcse=float(np.sqrt(prob*(1-prob)/len(x)))))
            timings.append(dict(world=wid,seconds=time.perf_counter()-started))
            print(wid,'finished',timings[-1]['seconds'],flush=True)
    pd.DataFrame(forecasts).to_csv(out/'heldout_unscored.csv',index=False)
    json_write(out/'heldout_seal.json',{'sha256':sha256((out/'heldout_unscored.csv').read_bytes()).hexdigest(),'rows':len(forecasts)})
    scored=evaluate_rows(pending);scored.to_csv(out/'heldout_scores.csv',index=False)
    reliability=[]
    edges=np.linspace(0,1,11);scored['probability_bin']=pd.cut(scored.p_margin_above_buffer180,
        bins=edges,include_lowest=True,labels=False)
    scored['observed_above_buffer']=(scored.true_margin>PROFIT_BUFFER_PER_LEAD*scored.n_leads).astype(float)
    for key,g in scored.dropna(subset=['probability_bin']).groupby('probability_bin'):
        per_world=g.groupby('world').agg(predicted=('p_margin_above_buffer180','mean'),
            observed=('observed_above_buffer','mean'),forecast_rows=('world','size')).reset_index()
        rng_bin=np.random.default_rng(18000+int(key));n=len(per_world)
        boot_pred=rng_bin.choice(per_world.predicted.to_numpy(),size=(2000,n),replace=True).mean(axis=1)
        boot_obs=rng_bin.choice(per_world.observed.to_numpy(),size=(2000,n),replace=True).mean(axis=1)
        reliability.append(dict(probability_bin=int(key),probability_lower=edges[int(key)],
            probability_upper=edges[int(key)+1],mean_predicted_probability=float(per_world.predicted.mean()),
            predicted_world_bootstrap_lower95=float(np.quantile(boot_pred,.025)),
            predicted_world_bootstrap_upper95=float(np.quantile(boot_pred,.975)),
            realized_profitable_share=float(per_world.observed.mean()),
            realized_world_bootstrap_lower95=float(np.quantile(boot_obs,.025)),
            realized_world_bootstrap_upper95=float(np.quantile(boot_obs,.975)),
            independent_worlds=n,forecast_rows=int(per_world.forecast_rows.sum()),
            rule='M180 / acquired leads > $0.02; intervals bootstrap independent worlds'))
    pd.DataFrame(reliability).to_csv(out/'probability_reliability.csv',index=False)
    seq=pd.DataFrame(sequential)
    if not seq.empty:
        seq=seq.sort_values(['world','origin_day'])
        decision_rows=[]
        for wid,g in seq.groupby('world',sort=False):
            decisive=g.loc[g.state!='undecided']
            first='undecided' if decisive.empty else decisive.state.iloc[0]
            first_day=np.nan if decisive.empty else float(decisive.origin_day.iloc[0])
            reversals=bool((decisive.state!=first).any()) if first!='undecided' else False
            g=g.copy();g['first_decision']=first;g['first_decisive_day']=first_day
            g['reversal_after_first']=((g.state!='undecided')&(first!='undecided')&(g.state!=first))
            seq.loc[g.index,['first_decision','first_decisive_day','reversal_after_first']]=g[
                ['first_decision','first_decisive_day','reversal_after_first']]
            decision_rows.append(dict(world=wid,scenario=g.scenario.iloc[0],cohort_id=g.cohort_id.iloc[0],
                first_decision=first,first_decisive_day=first_day,final_state=g.state.iloc[-1],
                any_reversal=reversals,true_above_buffer=bool(g.true_above_buffer.iloc[0]),
                true_margin180=float(g.true_margin180.iloc[0]),birth_day=int(g.birth_day.iloc[0]),
                decision_age=np.nan if np.isnan(first_day) else first_day-int(g.birth_day.iloc[0])))
        seq.to_csv(out/'sequential_confidence.csv',index=False)
        decisions=pd.DataFrame(decision_rows)
        decisions['false_positive']=(decisions.first_decision=='above_target')&(~decisions.true_above_buffer)
        decisions['false_negative']=(decisions.first_decision=='below_target')&(decisions.true_above_buffer)
        decisions['undecided']=decisions.first_decision=='undecided'
        decisions.to_csv(out/'sequential_decisions.csv',index=False)
        summary=[];srng=np.random.default_rng(2927)
        for scenario_name,g in pd.concat([decisions,decisions.assign(scenario='all_scenarios')]).groupby('scenario'):
            for metric in ['false_positive','false_negative','undecided','any_reversal','decision_age']:
                vals=g[metric].astype(float).to_numpy();valid=vals[np.isfinite(vals)]
                if len(valid):
                    boot=srng.choice(valid,size=(2000,len(valid)),replace=True).mean(axis=1)
                    summary.append(dict(scenario=scenario_name,metric=metric,mean=float(valid.mean()),
                        lower95=float(np.quantile(boot,.025)),upper95=float(np.quantile(boot,.975)),
                        n_worlds=len(valid),rule='first weekly 95%/5% declaration; fixed day-180 margin vs $0.02/lead'))
        pd.DataFrame(summary).to_csv(out/'sequential_metrics.csv',index=False)
    json_write(out/'heldout_runtime.json',timings)
    rows=[];rng=np.random.default_rng(92271)
    allscored=pd.concat([scored,scored.assign(scenario='all_scenarios')],ignore_index=True)
    for (scenario,model),group in allscored.groupby(['scenario','model']):
        for metric in ['coverage95','coverage80','crps_per_lead','mae_per_lead','brier','decision_loss_per_lead']:
            valid=group.loc[group[metric].notna()];v=valid.groupby('world')[metric].mean().to_numpy()
            boot=rng.choice(v,size=(2000,len(v)),replace=True).mean(axis=1)
            rows.append(dict(scenario=scenario,model=model,metric=metric,mean=float(v.mean()),
                             lower95=float(np.quantile(boot,.025)),upper95=float(np.quantile(boot,.975)),
                             n_worlds=len(v),n_forecasts=len(valid),n_leads=config['leads_per_cohort'],
                             units='USD per lead' if metric.endswith('_per_lead') else 'fraction'))
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['demo','heldout','all'],default='all');p.add_argument('--output',type=Path,default=ROOT/'artifacts');p.add_argument('--quick',action='store_true')
    args=p.parse_args();config=json.loads((ROOT/'run_config.json').read_text());args.output.mkdir(parents=True,exist_ok=True)
    source_names=['model.py','simulation.py','evaluate.py','run_config.json']
    source_hashes={n:sha256((ROOT/n).read_bytes()).hexdigest() for n in source_names}
    if args.quick:
        config.update(demo_cutoffs=[63],demo_draws=16,evaluation_draws=16,heldout_base_seeds=[4101],
            evaluation_cutoffs=[63],sequential_cutoffs=[77,91],sequential_draws=32,scenarios=['fixed_price'])
    json_write(args.output/'run_config.json',config)
    if args.phase in ['demo','all']:demo(config,args.output)
    if args.phase in ['heldout','all']:heldout(config,args.output)
    assert source_hashes=={n:sha256((ROOT/n).read_bytes()).hexdigest() for n in source_names},'Scientific source changed during run'
    outputs={p.name:sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.glob('*.csv'))}
    json_write(args.output/'run_manifest.json',{'scientific_source_sha256':source_hashes,
        'output_csv_sha256':outputs,'phase':args.phase,'quick':args.quick,
        'heldout_worlds':len(config['heldout_base_seeds'])*len(config['scenarios']),
        'method':'Approximate Gaussian measurement likelihood with integrated finite hyperprior',
        'truth_boundary':'Model receives as-of observations only; forecasts sealed before scoring; demo evaluator truth separately plotted'})


if __name__=='__main__':main()
