"""Fictional subscriber-level worlds with aggregate as-of observations."""
from dataclasses import dataclass
from hashlib import sha256
import json
import numpy as np
import pandas as pd

SCENARIOS=('fixed_price','cpc_squeeze','rising_cpl','human_decline_bot_rise','cheap_good','cheap_bad')
OBS_COLUMNS=['cohort_id','source_id','source_feature','birth_day','calendar_day','age',
             'acquired_leads','accepted_exposure','raw_events','raw_unique_responders',
             'audit_n','audit_human','unique_audit_n','unique_audit_human','net_human_revenue',
             'net_human_count','response_report_fraction','payout_complete_fraction',
             'observed_at','audit_observed_at','unique_audit_observed_at','payout_observed_at']

def conditions(day,scenario):
    day=np.asarray(day,dtype=float)
    elapsed=np.clip(day-42,0,84)
    # Each benchmark changes one declared business mechanism. In particular,
    # CPC-only does not perturb subscribers, engagement, or click paths.
    cpc=np.exp(np.log(.60)*elapsed/84) if scenario=='cpc_squeeze' else np.ones_like(day)
    quality_clock=np.clip((day-42)/84,0,1)
    human=np.exp(np.log(.72)*quality_clock) if scenario=='human_decline_bot_rise' else np.ones_like(day)
    bot=.004+.056*np.clip(day/126,0,1) if scenario=='human_decline_bot_rise' else np.full_like(day,.004)
    return human,cpc,bot


def _rng(seed,*parts):
    """Stable independent random streams make paired price changes auditable."""
    key='|'.join([str(int(seed)),*(str(x) for x in parts)]).encode()
    stream=int.from_bytes(sha256(key).digest()[:8],'little')
    return np.random.default_rng(stream)


@dataclass(frozen=True)
class FrozenSnapshot:
    payload: bytes

    @property
    def digest(self):return sha256(self.payload).hexdigest()

    def frame(self):
        p=json.loads(self.payload)
        return pd.DataFrame(p['rows'],columns=p['columns'])

    @property
    def cutoff(self):return json.loads(self.payload)['cutoff']


def freeze(frame,cutoff):
    obj={'cutoff':int(cutoff),'columns':list(frame.columns),
         'rows':frame.astype(object).where(pd.notna(frame),None).values.tolist()}
    return FrozenSnapshot(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode())


@dataclass(frozen=True)
class World:
    observations: pd.DataFrame
    cohort_registry: pd.DataFrame
    quotes: pd.DataFrame
    seed: int
    scenario: str

    def as_of(self,cutoff):
        f=self.observations.loc[(self.observations.calendar_day<=cutoff)&(self.observations.observed_at<=cutoff)].copy()
        f.loc[f.audit_observed_at>cutoff,['audit_n','audit_human']]=np.nan
        f.loc[f.unique_audit_observed_at>cutoff,['unique_audit_n','unique_audit_human']]=np.nan
        pending=f.payout_observed_at>cutoff
        f.loc[pending,['net_human_revenue','net_human_count']]=np.nan
        f.loc[pending,'payout_complete_fraction']=0.
        return freeze(f[OBS_COLUMNS],cutoff)

    def targets(self,cutoff,cohort_ids=None):
        f=self.cohort_registry.loc[self.cohort_registry.birth_day<=cutoff].copy()
        if cohort_ids is not None:f=f.loc[f.cohort_id.isin(cohort_ids)]
        return f.to_dict('records')

    def quote_at(self,source,cutoff):
        f=self.quotes.loc[(self.quotes.source_id==source)&(self.quotes.observed_at<=cutoff)]
        if f.empty:raise ValueError('No quote known at cutoff')
        return float(f.sort_values('calendar_day').iloc[-1].cpl_quote)


@dataclass(frozen=True)
class Truth:
    cohort_paths: dict
    calendar: pd.DataFrame
    quality: pd.DataFrame
    cohort_truth: pd.DataFrame
    parameters: pd.DataFrame
    cohort_rollups: pd.DataFrame


def generate(seed,scenario,leads=100,last_birth=126):
    if scenario not in SCENARIOS:raise ValueError(scenario)
    root=_rng(seed,'calendar')
    sources=[('established_a',-1.,0,.004*np.exp(-.35),.18,205.,.92),
             ('established_b',1.,0,.004*np.exp(.35),.24,150.,.86)]
    if scenario in ('cheap_good','cheap_bad'):
        sources.append(('new_cheap',0.,70,.010 if scenario=='cheap_good' else .0015,.08,175.,.88))
    rows=[];registry=[];quotes=[];paths={};quality=[];cohort_truth=[];parameters=[];cohort_rollups=[]
    days=np.arange(last_birth+426)
    human,cpc,bot=conditions(days,scenario)
    # Shared response and price shocks are evaluator-only; keyed streams make
    # the fixed/CPC worlds identical in every nonprice mechanism.
    noise=root.normal(0,.025,size=(2,(len(days)+6)//7))
    response_calendar=human*np.exp(np.repeat(noise[0],7)[:len(days)])
    cpc_calendar=.20*cpc*np.exp(np.repeat(noise[1],7)[:len(days)])
    calendar=pd.DataFrame({'calendar_day':days,'human_response_multiplier':response_calendar,
                           'human_cpc':cpc_calendar,'bot_rate':bot})
    for source,z,start,base_rate,base_cpl,base_half_life,base_human_share in sources:
        source_rng=_rng(seed,'source',source)
        source_gain=source_rng.lognormal(-.5*.12**2,.12)
        source_value=source_rng.lognormal(.08*z-.5*.08**2,.08)
        # A prospective quote for a new source is available one day before launch.
        for day in range(max(0,start-1),last_birth+1):
            inflation=1+.008*max(day-42,0) if scenario=='rising_cpl' else 1.
            quotes.append(dict(source_id=source,calendar_day=day,observed_at=day,cpl_quote=base_cpl*inflation))
        for birth in range(start,last_birth+1,7):
            cohort=f'{source}-b{birth:03d}'
            crng=_rng(seed,'cohort',cohort)
            cohort_gain=crng.lognormal(-.5*.28**2,.28)
            cohort_value=crng.lognormal(-.5*.16**2,.16)
            half_life=float(base_half_life*crng.lognormal(0,.32))
            survival_hazard=float(np.log(2)/half_life)
            human_share=float(np.clip(base_human_share*np.exp(-.003*max(birth-42,0))
                if scenario=='human_decline_bot_rise' else base_human_share, .45,.99))
            status_rng=_rng(seed,'status',cohort)
            is_human=status_rng.random(leads)<human_share
            activity_rng=_rng(seed,'activity',cohort)
            frailty=activity_rng.gamma(shape=4.,scale=.25,size=leads)
            retention_rng=_rng(seed,'retention',cohort)
            exit_uniform=retention_rng.random(leads)
            # Geometric lifetime: an address is active through its sampled
            # exit age, then stays inactive. All subscriber traits are fixed.
            daily_hazard=1-np.exp(-survival_hazard)
            exit_age=np.floor(np.log(np.maximum(exit_uniform,1e-15))/np.log1p(-daily_hazard)).astype(int)
            # The new source's first batch honors its day-before-launch firm quote.
            pricing_day=birth-1 if source=='new_cheap' and birth==start else birth
            inflation=1+.008*max(pricing_day-42,0) if scenario=='rising_cpl' else 1.
            actual_cost=leads*base_cpl*inflation
            registry.append(dict(cohort_id=cohort,source_id=source,source_feature=z,birth_day=birth,
                                 leads=leads,acquisition_cost=actual_cost,daily_send_cost=leads*.00025,
                                 send_cost_per_contactable=.00025))
            a=np.arange(366);t=birth+a
            human_rng=_rng(seed,'human-engagement',cohort)
            bot_rng=_rng(seed,'bot-events',cohort)
            click_rng=_rng(seed,'repeat-clicks',cohort)
            audit_rng=_rng(seed,'audits',cohort)
            unique_audit_rng=_rng(seed,'unique-audits',cohort)
            money_rng=_rng(seed,'monetization',cohort)
            attempts=exit_age[:,None]>=a[None,:]
            active_human=attempts & is_human[:,None]
            active_bot=attempts & (~is_human)[:,None]
            contactable=attempts.sum(axis=0).astype(int)
            active_human_count=active_human.sum(axis=0).astype(int)
            p_engage=np.clip(base_rate*source_gain*cohort_gain*frailty[:,None]*
                             response_calendar[t][None,:],0,.8)
            human_responders=active_human & (human_rng.random((leads,len(a)))<p_engage)
            # A responding person can produce several clicks; repeat clicks
            # are a separate count process from unique weekly response.
            repeated=click_rng.negative_binomial(2.,.65,size=(leads,len(a)))
            h=(human_responders*(1+repeated)).sum(axis=0).astype(int)
            bot_responders=active_bot & (bot_rng.random((leads,len(a)))<bot[t][None,:])
            b=bot_responders.sum(axis=0).astype(int)
            raw=h+b
            raw_unique=human_responders.sum(axis=0).astype(int)+bot_responders.sum(axis=0).astype(int)
            audit_n=audit_rng.binomial(raw,.5)
            audit_h=audit_rng.hypergeometric(h,b,audit_n)
            labels=audit_rng.binomial(audit_h,.95)+audit_rng.binomial(audit_n-audit_h,.02)
            unique_audit_n=unique_audit_rng.binomial(raw_unique,.5)
            true_human_unique=human_responders.sum(axis=0).astype(int)
            true_bot_unique=bot_responders.sum(axis=0).astype(int)
            unique_audit_h=unique_audit_rng.hypergeometric(true_human_unique,true_bot_unique,unique_audit_n)
            unique_labels=unique_audit_rng.binomial(unique_audit_h,.95)+unique_audit_rng.binomial(
                unique_audit_n-unique_audit_h,.02)
            mean_mark=cpc_calendar[t]*source_value*cohort_value
            revenue=np.where(h>0,money_rng.gamma(np.maximum(h*4.,.001),mean_mark/4.),0.)
            increments=np.zeros(426);increments[0]-=actual_cost
            increments[:366]+=revenue-contactable*.00025
            paths[cohort]=increments.cumsum()
            parameters.append(dict(cohort_id=cohort,source_id=source,birth_day=birth,
                true_contactable_half_life=half_life,true_human_share=human_share,
                true_initial_engagement_rate=base_rate*source_gain*cohort_gain,
                true_net_cpc_reference=.20*source_value*cohort_value,
                truth_role='evaluator_only'))
            for i in range(366):
                rows.append(dict(cohort_id=cohort,source_id=source,source_feature=z,birth_day=birth,
                    calendar_day=int(t[i]),age=int(a[i]),acquired_leads=leads,
                    accepted_exposure=int(contactable[i]),raw_events=int(raw[i]),
                    raw_unique_responders=int(raw_unique[i]),
                    audit_n=int(audit_n[i]),audit_human=int(labels[i]),
                    unique_audit_n=int(unique_audit_n[i]),unique_audit_human=int(unique_labels[i]),
                    net_human_revenue=float(revenue[i]),net_human_count=int(h[i]),
                    response_report_fraction=1.,payout_complete_fraction=1.,observed_at=int(t[i]+2),
                    audit_observed_at=int(t[i]+7),unique_audit_observed_at=int(t[i]+7),
                    payout_observed_at=int(t[i]+14)))
                quality.append(dict(cohort_id=cohort,source_id=source,calendar_day=int(t[i]),age=int(a[i]),
                                    true_human_events=int(h[i]),true_bot_events=int(b[i]),raw_events=int(raw[i]),
                                    raw_unique_responders=int(raw_unique[i]),
                                    true_unique_human_responders=int(true_human_unique[i]),
                                    true_contactable=int(contactable[i]),true_active_humans=int(active_human_count[i]),
                                    unique_human_responders=int(human_responders[:,i].sum()),
                                    true_expected_unique_response=float(np.sum(p_engage[:,i]*active_human[:,i])),
                                    true_expected_human_events=float(np.sum(p_engage[:,i]*active_human[:,i])*2.077),
                                    audit_n=int(audit_n[i]),audit_human=int(labels[i])))
                cohort_truth.append(dict(cohort_id=cohort,source_id=source,calendar_day=int(t[i]),age=int(a[i]),
                    acquired_leads=leads,contactable=int(contactable[i]),active_humans=int(active_human_count[i]),
                    unique_human_responders=int(human_responders[:,i].sum()),
                    true_contactable_fraction=float(contactable[i]/leads),
                    true_expected_contactable_fraction=float(np.exp(-survival_hazard*a[i])),
                    true_unique_response_per_lead=float(human_responders[:,i].sum()/leads),
                    true_expected_unique_response_per_lead=float(np.sum(
                        p_engage[:,i]*is_human)*np.exp(-survival_hazard*a[i])/leads),
                    true_expected_human_events_per_lead=float(np.sum(
                        p_engage[:,i]*is_human)*np.exp(-survival_hazard*a[i])*2.077/leads),
                    true_net_cpc=float(mean_mark[i]),truth_role='evaluator_only'))
            # Unique responders are deduplicated within each complete age
            # window, so aggregation never mistakes respondent-days for
            # distinct people. Denominators are eligible people at window start.
            for window in (7,30):
                for window_start in range(0,181,window):
                    window_end=window_start+window
                    if window_end>181:continue
                    unique_people=(human_responders[:,window_start:window_end].any(axis=1)&
                                   attempts[:,window_start])
                    cohort_rollups.append(dict(cohort_id=cohort,source_id=source,birth_day=birth,
                        window_days=window,window_start_age=window_start,window_end_age=window_end-1,
                        acquired_leads=leads,eligible_at_start=int(attempts[:,window_start].sum()),
                        contactable_at_end=int(attempts[:,window_end-1].sum()),
                        distinct_human_responders=int(unique_people.sum()),
                        truth_role='evaluator_only'))
    return World(pd.DataFrame(rows)[OBS_COLUMNS],pd.DataFrame(registry),pd.DataFrame(quotes),seed,scenario), Truth(
        paths,calendar,pd.DataFrame(quality),pd.DataFrame(cohort_truth),pd.DataFrame(parameters),
        pd.DataFrame(cohort_rollups))
