"""Render article-only views of frozen simulation outputs; never refit a model."""
from pathlib import Path
import os,json,hashlib
os.environ.setdefault('MPLCONFIGDIR','/tmp/article-matplotlib')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter,FuncFormatter
ROOT=Path(__file__).resolve().parent
LAB=ROOT.parent if (ROOT.parent/'parameters.json').exists() else ROOT.parent/'bayesian-email-cohort-lab'
OUT=ROOT/'assets';OUT.mkdir(exist_ok=True)
INK='#283641';GREEN='#2d679d';ORANGE='#b17721';MUTED='#687580';PALE='#dce7f0';BLUE='#6085aa'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.labelsize':13,'axes.titlesize':15,
    'axes.edgecolor':'#b8c2ba','axes.labelcolor':INK,'xtick.color':MUTED,'ytick.color':MUTED,
    'text.color':INK,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white',
    'axes.facecolor':'white','savefig.facecolor':'white','svg.fonttype':'none'})
records=[]
for prior in OUT.glob('*'):
 if prior.suffix in ('.png','.svg'): prior.unlink()
def source(rel):
 p=LAB/rel;return pd.read_csv(p),{'path':rel,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def save(fig,name,inputs,filter_description,alt):
 for ext in ['png','svg']:
  fig.savefig(OUT/f'{name}.{ext}',dpi=180,bbox_inches='tight',pad_inches=.22)
 width,height=fig.get_size_inches()
 fig.set_size_inches(4.6,max(height,4.4))
 for axis in fig.axes: axis.title.set_wrap(True)
 fig.tight_layout(rect=(0,0,1,.96),pad=1.1)
 for ext in ['png','svg']:
  fig.savefig(OUT/f'{name}-mobile.{ext}',dpi=180,bbox_inches='tight',pad_inches=.14)
 plt.close(fig)
 for svg_path in [OUT/f'{name}.svg',OUT/f'{name}-mobile.svg']:
  import html
  txt=svg_path.read_text();key=svg_path.stem;at=txt.index('<svg ');txt=txt[:at]+txt[at:].replace('<svg ',f'<svg role="img" aria-labelledby="{key}-title {key}-desc" ',1);end=txt.index('>',at)+1
  txt=txt[:end]+f'<title id="{key}-title">{html.escape(alt)}</title><desc id="{key}-desc">Synthetic demonstration. {html.escape(filter_description)}</desc>'+txt[end:];svg_path.write_text(txt)
 records.append({'id':name,'files':[f'assets/{name}.png',f'assets/{name}.svg',f'assets/{name}-mobile.png',f'assets/{name}-mobile.svg'],'source_files':inputs,'selection':filter_description,'alt':alt})
def setup(ax):
 ax.set_axisbelow(True);ax.grid(axis='y',color='#e4e9e4',linewidth=.8);ax.tick_params(length=0,pad=8)
def label(fig,text):fig.text(.02,1.0,'SYNTHETIC DATA  ·  '+text,fontsize=10.5,color=MUTED,weight='medium')
# Snapshot intervals: each x position is an explicitly named forecast, not elapsed time.
d,s=source('artifacts/cold_start/summary.csv');d=d[(d.source_id=='middle')&(d.horizon==180)].sort_values('origin_age')
fig,ax=plt.subplots(figsize=(6.4,4.6));setup(ax);x=np.arange(len(d));n=d.n_leads.to_numpy()
ax.vlines(x,d.q025/n,d.q975/n,color='#adc8b9',lw=5,zorder=2)
ax.vlines(x,d.q10/n,d.q90/n,color=GREEN,lw=13,zorder=3)
ax.plot(x,d['median']/n,color=INK,marker='o',lw=1.5,ms=7,zorder=5)
ax.axhline(d.true_margin.iloc[0]/n[0],color=ORANGE,lw=1.8,ls=(0,(4,3)))
ax.axhline(0,color='#b8c2ba',lw=.8);ax.set_xticks(x,[f'Day {a}' for a in d.origin_age]);ax.set_ylabel('Day-180 contribution / acquired lead')
ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'${v:,.2f}'));ax.set_xlim(-.45,4.45);ax.set_xlabel('Forecast snapshot')
ax.legend([Line2D([],[],color=INK,marker='o'),Line2D([],[],color=GREEN,lw=7),Line2D([],[],color='#adc8b9',lw=4),Line2D([],[],color=ORANGE,ls='--')],['Median','80% interval','95% interval','Outcome'],ncol=2,frameon=False,fontsize=10.8,loc='lower left',bbox_to_anchor=(0,1.02))
label(fig,'MIDDLE SOURCE · FIRST COHORT');fig.tight_layout()
save(fig,'02-learning',[s],'source_id=middle; horizon=180; all five origin ages; normalized by n_leads','Median and 80% and 95% predictive intervals for day-180 contribution per acquired lead across five forecast snapshots.')
# Properly separated controlled component and richer-model stress coverage.
a,sa=source('artifacts/controlled_scenario/metrics_by_origin.csv');b,sb=source('artifacts/calibration/metrics_by_origin.csv');b=b[b.metric=='coverage95'].sort_values('origin_age');a=a.sort_values('origin_age')
fig,ax=plt.subplots(figsize=(6.4,4.6));setup(ax);x=np.arange(len(a))
ax.errorbar(x-.07,a.coverage95,yerr=np.vstack([a.coverage95-a.coverage95_wilson_lower,a.coverage95_wilson_upper-a.coverage95]),fmt='o-',color=GREEN,lw=2,ms=6,capsize=4,label='Controlled component')
ax.errorbar(x+.07,b['mean'],yerr=np.vstack([b['mean']-b.lower95,b.upper95-b['mean']]),fmt='s-',color=ORANGE,lw=2,ms=6,capsize=4,label='Richer model')
ax.axhline(.95,color=INK,ls=(0,(4,3)),lw=1.3,label='Nominal 95%')
ax.set_xticks(x,[f'Day {v}' for v in a.origin_age]);ax.set_ylim(0,1.05);ax.set_xlabel('Forecast snapshot');ax.set_ylabel('Outcomes inside the predictive interval')
ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(frameon=False,loc='lower left',fontsize=11)
label(fig,'95% INTERVAL COVERAGE · DAY 180');fig.tight_layout()
save(fig,'05-coverage',[sa,sb],'controlled: coverage95 with Wilson bounds; stress: metric=coverage95, whole-world bootstrap bounds','Controlled 95% intervals cover about 94–96% of outcomes while the richer model covers about 53–64%.')
# All nine late-reactivation forecasts at one fixed origin.
d,s=source('artifacts/calibration/scores.csv');d=d[(d.scenario=='late_reactivation')&(d.origin_age==60)&(d.horizon==180)].copy()
d['source_order']=d.source_id.map({'low':0,'middle':1,'high':2});d=d.sort_values(['source_order','seed'])
fig,ax=plt.subplots(figsize=(6.5,5.3));ax.set_axisbelow(True);ax.grid(axis='x',color='#e4e9e4');y=np.arange(len(d));n=d.n_leads.to_numpy()
ax.hlines(y,d.q025/n,d.q975/n,color=GREEN,lw=5,alpha=.7);ax.plot(d['mean']/n,y,'o',color=GREEN,ms=7,label='Forecast mean + 95% interval')
ax.plot(d.true_margin/n,y,marker='x',ls='',color=ORANGE,mew=2.5,ms=9,label='Realized outcome')
ax.set_yticks(y,[f'{r.source_id.title()} · {r.seed}' for r in d.itertuples()]);ax.invert_yaxis();ax.tick_params(length=0,pad=8)
ax.set_xlabel('Day-180 contribution / acquired lead');ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'${v:.2f}'))
ax.legend(frameon=False,fontsize=10.4,loc='lower left',bbox_to_anchor=(0,1.02));ax.axvline(0,color='#aab9ae',lw=.8)
label(fig,'LATE REACTIVATION · FORECAST AT DAY 60');fig.tight_layout()
save(fig,'appendix-reactivation',[s],'scenario=late_reactivation; origin_age=60; horizon=180; all 9 source/seed rows; normalized by n_leads','Eight of nine outcomes lie outside nominal 95% intervals in three late-reactivation worlds.')
# Cumulative economic path and unconditional payback, with the full denominator.
d,s=source('artifacts/cold_start/payback_cdf.csv.gz');d=d[d.source_id=='middle']
t,st=source('artifacts/cold_start/trajectories.csv.gz');t=t[(t.source_id=='middle')&(t.origin_age==60)].sort_values('age');n=t.n_leads.iloc[0]
fig,(top,ax)=plt.subplots(2,1,figsize=(6.6,8.3),gridspec_kw={'hspace':.38})
setup(top);setup(ax)
top.fill_between(t.age,t.q025/n,t.q975/n,color='#dce7f0',label='95% interval')
top.fill_between(t.age,t.q10/n,t.q90/n,color='#aac5dc',label='80% interval')
top.plot(t.age,t['median']/n,color=GREEN,lw=2,label='Predictive median')
top.plot(t.age,t.truth/n,color=ORANGE,ls='--',lw=1.8,label='Outcome')
top.axhline(0,color=INK,lw=.8);top.axvline(60,color='#7c8892',lw=1,ls=':')
top.set_xlim(0,425);top.set_xticks([0,60,180,300,425]);top.set_ylabel('Contribution / acquired lead')
top.set_title('Cumulative contribution forecast at day 60',loc='left',fontsize=13,pad=12)
top.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'${v:.2f}'))
top.legend(frameon=False,fontsize=9.5,ncol=2,loc='lower right')
for origin,color,ls in [(3,GREEN,'-'),(14,BLUE,'--'),(60,ORANGE,'-.')]:
 a=d[d.origin_age==origin].sort_values('age');ax.plot(a.age,a.cdf,color=color,ls=ls,lw=2.4,label=f'Forecast at day {origin}')
ax.set_xlim(0,425);ax.set_ylim(0,1.04);ax.set_xticks([0,90,180,300,425]);ax.set_xlabel('Cohort age');ax.set_ylabel('Share of all predictive paths')
ax.set_title('Probability of first payback by each age',loc='left',fontsize=13,pad=12)
ax.yaxis.set_major_formatter(PercentFormatter(1))
for y,text,color in [(.455,'Day 3 · 40.6%',GREEN),(.24,'Day 14 · 19.0%',BLUE),(.945,'Day 60 · 100%',ORANGE)]:
 ax.text(410,y,text,ha='right',fontsize=10.7,color=color)
label(fig,'MIDDLE SOURCE · FINITE POLICY TO DAY 425')
fig.subplots_adjust(left=.17,right=.98,top=.94,bottom=.08)
save(fig,'03-margin-payback',[s,st],'middle source; margin forecast at origin60; payback origins3/14/60; full ages0–425; every predictive path retained','Two panels show the day-sixty cumulative contribution forecast and unconditional payback curves; the day-three curve ends at40.6%.')
# Like-for-like source comparison at a common origin and horizon.
d,s=source('artifacts/cold_start/summary.csv');d=d[(d.origin_age==60)&(d.horizon==180)].copy();d['order']=d.source_id.map({'low':0,'middle':1,'high':2});d=d.sort_values('order');n=d.n_leads.to_numpy();x=np.arange(3)
fig,ax=plt.subplots(figsize=(6.4,4.3));setup(ax)
ax.vlines(x,d.q025/n,d.q975/n,color=GREEN,lw=7,alpha=.6);ax.plot(x,d['mean']/n,'o',color=GREEN,ms=8,label='Predictive mean')
ax.plot(x,d.true_margin/n,'x',color=ORANGE,ms=10,mew=2.4,label='Outcome');ax.axhline(0,color=INK,lw=.9)
ax.set_xticks(x,[v.title() for v in d.source_id]);ax.set_xlim(-.55,2.55);ax.set_xlabel('Fictional acquisition source');ax.set_ylabel('Day-180 contribution / acquired lead')
ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'${v:.2f}'));ax.legend(frameon=False,fontsize=10,loc='lower left',bbox_to_anchor=(0,1.02),ncol=2)
label(fig,'FIRST COHORTS · COMMON FORECAST AT DAY 60');fig.tight_layout()
save(fig,'04-sources',[s],'origin_age=60; horizon=180; all three first source cohorts, same seed211; normalized by n_leads','A common-age comparison shows different day-180 modeled contribution for low, middle and high sources.')
# Observable engagement uses original acquired leads, not clicks or current eligibility.
l,sl=source('artifacts/demo/synthetic_observations/leads.csv.gz');e,se=source('artifacts/demo/synthetic_observations/engagement_events.csv.gz')
e=e[e.observed_at.notna()].sort_values(['event_id','revision_sequence','observed_at']).drop_duplicates('event_id',keep='last');e=e[e.qualification=='human']
e=e.merge(l[['lead_id','acquired_at']],on='lead_id',how='left',validate='many_to_one');e['age']=e.occurred_at-e.acquired_at
counts=e.groupby(['source_id','age']).lead_id.nunique();denom=l.groupby('source_id').size()
fig,ax=plt.subplots(figsize=(6.4,4.4));setup(ax)
rows=[]
for src,color,ls in [('low','#88a7c1',':'),('middle',GREEN,'-'),('high',ORANGE,'--')]:
 daily=counts.loc[src].reindex(np.arange(181),fill_value=0)/denom[src]
 smooth=daily.rolling(7,min_periods=1).mean();ax.plot(smooth.index,smooth.values,color=color,ls=ls,lw=2.3,label=f'{src.title()} source')
 for age in daily.index:rows.append({'source_id':src,'cohort_age':int(age),'daily_reported_unique_clicker_fraction':float(daily[age]),'trailing_7day_average':float(smooth[age]),'original_acquired_leads':int(denom[src])})
ax.set_xlim(0,180);ax.set_xlabel('Cohort age (days)');ax.set_ylabel('Daily clickers / original acquired leads');ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0));ax.legend(frameon=False,fontsize=11)
label(fig,'REPORTED HUMAN CLICKERS · 7-DAY AVERAGE');fig.tight_layout()
pd.DataFrame(rows).to_csv(ROOT/'engagement_values.csv',index=False)
save(fig,'01-engagement',[sl,se],'latest reported human qualification; unique lead per occurrence-age day; all8cohorts/source; denominator1440original leads/source; trailing7day average','Daily reported human clickers divided by original acquired leads show early decline and a long response tail.')
(ROOT/'chart_provenance.json').write_text(json.dumps({'synthetic_only':True,'simulation_refit':False,'charts':records},indent=2)+'\n')
print('Rendered',len(records),'source-derived charts')
