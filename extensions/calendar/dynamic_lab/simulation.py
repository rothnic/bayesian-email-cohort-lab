"""New fictional aggregate observation streams; evaluator truth stays separate."""
from dataclasses import dataclass
from hashlib import sha256
import json
import numpy as np
import pandas as pd

SCENARIOS=('fixed_price','cpc_squeeze','rising_cpl','human_decline_bot_rise','cheap_good','cheap_bad')
OBS_COLUMNS=['cohort_id','source_id','source_feature','birth_day','calendar_day','age',
             'accepted_exposure','raw_events','audit_n','audit_human','net_human_revenue',
             'net_human_count','response_report_fraction','payout_complete_fraction',
             'observed_at','audit_observed_at','payout_observed_at']


def age_curve(age):
    age=np.asarray(age)
    return .75*np.exp(-age/3)+.25*np.exp(-age/90)


def conditions(day,scenario):
    day=np.asarray(day,dtype=float)
    elapsed=np.clip(day-42,0,84)
    cpc=np.exp(-.008*elapsed) if scenario!='fixed_price' else np.ones_like(day)
    human=np.exp(-.006*elapsed) if scenario in SCENARIOS[3:] else np.ones_like(day)
    bot=.004+.056*np.clip(day/126,0,1) if scenario in SCENARIOS[3:] else np.full_like(day,.004)
    return human,cpc,bot


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


def generate(seed,scenario,leads=100,last_birth=126):
    if scenario not in SCENARIOS:raise ValueError(scenario)
    rng=np.random.default_rng(seed)
    sources=[('established_a',-1.,0,.095*np.exp(-.35),.18),
             ('established_b',1.,0,.095*np.exp(.35),.24)]
    if scenario in ('cheap_good','cheap_bad'):
        sources.append(('new_cheap',0.,70,.15 if scenario=='cheap_good' else .024,.08))
    rows=[];registry=[];quotes=[];paths={};quality=[]
    days=np.arange(last_birth+426)
    human,cpc,bot=conditions(days,scenario)
    # A modest shared latent weekly calendar disturbance is evaluator-only.
    noise=rng.normal(0,.025,size=(2,(len(days)+6)//7))
    response_calendar=human*np.exp(np.repeat(noise[0],7)[:len(days)])
    cpc_calendar=.20*cpc*np.exp(np.repeat(noise[1],7)[:len(days)])
    calendar=pd.DataFrame({'calendar_day':days,'human_response_multiplier':response_calendar,
                           'human_cpc':cpc_calendar,'bot_rate':bot})
    for source,z,start,base_rate,base_cpl in sources:
        source_gain=rng.lognormal(-.5*.12**2,.12)
        source_value=rng.lognormal(.08*z-.5*.08**2,.08)
        # A prospective quote for a new source is available one day before launch.
        for day in range(max(0,start-1),last_birth+1):
            inflation=1+.008*max(day-42,0) if scenario in SCENARIOS[2:] else 1.
            quotes.append(dict(source_id=source,calendar_day=day,observed_at=day,cpl_quote=base_cpl*inflation))
        for birth in range(start,last_birth+1,7):
            cohort=f'{source}-b{birth:03d}'
            cohort_gain=rng.lognormal(-.5*.20**2,.20)
            cohort_value=rng.lognormal(-.5*.10**2,.10)
            # The new source's first batch honors its day-before-launch firm quote.
            pricing_day=birth-1 if source=='new_cheap' and birth==start else birth
            inflation=1+.008*max(pricing_day-42,0) if scenario in SCENARIOS[2:] else 1.
            actual_cost=leads*base_cpl*inflation
            registry.append(dict(cohort_id=cohort,source_id=source,source_feature=z,birth_day=birth,
                                 leads=leads,acquisition_cost=actual_cost,daily_send_cost=leads*.00025))
            a=np.arange(366);t=birth+a
            lam=leads*base_rate*source_gain*cohort_gain*age_curve(a)*response_calendar[t]
            # NB2 count clustering; audits sample events, not distinct people.
            h=rng.negative_binomial(20.,20./(20.+lam))
            b=rng.negative_binomial(15.,15./(15.+leads*bot[t]))
            raw=h+b
            audit_n=rng.binomial(raw,.5)
            audit_h=rng.hypergeometric(h,b,audit_n)
            labels=rng.binomial(audit_h,.95)+rng.binomial(audit_n-audit_h,.02)
            mean_mark=cpc_calendar[t]*source_value*cohort_value
            revenue=np.where(h>0,rng.gamma(np.maximum(h*4.,.001),mean_mark/4.),0.)
            increments=np.zeros(426);increments[0]-=actual_cost
            increments[:366]+=revenue-leads*.00025
            paths[cohort]=increments.cumsum()
            for i in range(366):
                rows.append(dict(cohort_id=cohort,source_id=source,source_feature=z,birth_day=birth,
                    calendar_day=int(t[i]),age=int(a[i]),accepted_exposure=leads,
                    raw_events=int(raw[i]),audit_n=int(audit_n[i]),audit_human=int(labels[i]),
                    net_human_revenue=float(revenue[i]),net_human_count=int(h[i]),
                    response_report_fraction=1.,payout_complete_fraction=1.,observed_at=int(t[i]+2),
                    audit_observed_at=int(t[i]+7),payout_observed_at=int(t[i]+14)))
                quality.append(dict(cohort_id=cohort,source_id=source,calendar_day=int(t[i]),age=int(a[i]),
                                    true_human_events=int(h[i]),true_bot_events=int(b[i]),raw_events=int(raw[i]),
                                    audit_n=int(audit_n[i]),audit_human=int(labels[i])))
    return World(pd.DataFrame(rows)[OBS_COLUMNS],pd.DataFrame(registry),pd.DataFrame(quotes),seed,scenario), Truth(paths,calendar,pd.DataFrame(quality))
