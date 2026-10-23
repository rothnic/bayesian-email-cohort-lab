"""Bind the reviewed draft to sealed scientific outputs and build offline HTML."""
from pathlib import Path
from html.parser import HTMLParser
import argparse,base64,hashlib,html,json,re,shutil
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent


def inline(s):
    s=html.escape(s,quote=False)
    s=re.sub(r'`([^`]+)`',r'<code>\1</code>',s)
    s=re.sub(r'\[([^]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',s)
    s=re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',s)
    return s


def table(frame):
    def cell(v):
        if pd.isna(v):return '—'
        if isinstance(v,(int,np.integer)):return str(v)
        if isinstance(v,(float,np.floating)):return f'{v:.4g}'
        return str(v)
    return '<div class="table-wrap"><table><thead><tr>'+''.join('<th scope="col">'+html.escape(str(c))+'</th>' for c in frame.columns)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(cell(v))+'</td>' for v in row)+'</tr>' for row in frame.itertuples(index=False,name=None))+'</tbody></table></div>'


def main():
    p=argparse.ArgumentParser();p.add_argument('--extension',type=Path,default=ROOT.parent/'cohort-calendar-extension');a=p.parse_args()
    ext=a.extension.resolve();art=ext/'dynamic_lab/artifacts';fig=art/'figures'
    metadata=json.loads((fig/'chart_provenance.json').read_text());assets=ROOT/'assets';assets.mkdir(exist_ok=True);tables=ROOT/'tables';tables.mkdir(exist_ok=True)
    for c in metadata['charts']:
        for rel in c['files']:shutil.copy2(ext/rel,assets/Path(rel).name)
        shutil.copy2(ext/c['table'],tables/Path(c['table']).name)
    predictions=pd.read_csv(art/'cohort_predictions.csv');metrics=pd.read_csv(art/'metrics.csv');trans=pd.read_csv(art/'transfer.csv')
    claims={};source_paths=[p.name for p in art.glob('*.csv')]
    confidence=pd.read_csv(art/'demo_confidence.csv')
    detail=pd.read_csv(art/'cohort_retention_rollup_detail.csv')
    rolls=pd.read_csv(art/'cohort_retention_rollups.csv')
    seq=pd.read_csv(art/'sequential_confidence.csv')
    seqm=pd.read_csv(art/'sequential_metrics.csv')
    metrics=pd.read_csv(art/'metrics.csv')
    quotes=pd.read_csv(art/'quote_sensitivity.csv')
    shutil.copy2(art/'quote_sensitivity.csv',tables/'offered-cost.csv')
    # Bind narrative claims directly to the regenerated synthetic outputs.
    target=confidence[(confidence.source_id=='new_cheap')&(confidence.birth_day==70)]
    good=target[target.scenario=='cheap_good'].sort_values('origin_day')
    bad=target[target.scenario=='cheap_bad'].sort_values('origin_day')
    assert not good.empty and not bad.empty
    firstg=good.iloc[0]; lastg=good.iloc[-1]; lastb=bad.iloc[-1]
    crossings=good[good.posterior_probability>=.95]
    first_cross=int(crossings.origin_day.min()) if len(crossings) else None
    crossing_note=(f"The good-source line first crosses the 95% decision threshold at day {first_cross}. " if first_cross is not None else "The good-source line does not cross the 95% threshold in the plotted window. ")
    confidence_text=(f"The chart follows the same day-70 cohort in two equally priced fictional source worlds. Before launch their forecasts match. {crossing_note}By cutoff day {int(lastg.origin_day)}, the good-source estimate has {lastg.posterior_probability:.0%} probability of clearing the $0.02-per-lead buffer, compared with {lastb.posterior_probability:.0%} for the bad source, which remains unresolved above the 5% below-target cutoff. These posterior probabilities are not demonstrated calibration; the held-out check below assesses that separately.")
    claims['confidence_learning']={'good':good.to_dict('records'),'bad':bad.to_dict('records')}
    conf_rule="At each information cutoff I classify a cohort as above target only when its posterior probability is at least 95%, and below target only when it is at most 5%. The target is day-180 contribution above $0.02 per originally acquired lead. Values between the thresholds remain unresolved. Those probability cutoffs are a decision convention, not a claim of 95% empirical accuracy."
    shocks=pd.read_csv(art/'cpc_shock_sensitivity.csv')
    s0=shocks[shocks.cpc_drop_fraction==0].iloc[0]; s30=shocks[shocks.cpc_drop_fraction==.30].iloc[0]
    cpc_text=(f"For a planned 100-lead {s30.source_id} cohort, a 30% future CPC reduction changes median day-180 contribution by ${s30.paired_margin_delta_median:.2f} for the whole cohort relative to no extra squeeze (paired 90% range ${s30.paired_margin_delta_q05:.2f} to ${s30.paired_margin_delta_q95:.2f}). The chance of clearing the two-cent-per-lead buffer moves from {s0.p_margin_above_buffer180:.0%} to {s30.p_margin_above_buffer180:.0%}, with fitted draws and subscriber activity held in common. This scenario starts after the cutoff and leaves past clicks, subscriber activity and acquisition costs fixed.")
    claims['cpc_what_if']={'rows':shocks.to_dict('records')}
    q10=quotes[np.isclose(quotes.offered_cpl,.1)]
    qg=q10[q10.new_source_kind=='good'].iloc[0]; qb=q10[q10.new_source_kind=='bad'].iloc[0]
    transfer=(f"For a prospective 100-lead batch offered at $0.10 per lead, estimated probability of clearing the fixed day-180 target is {qg.p_margin_above_buffer180:.0%} for the good source and {qb.p_margin_above_buffer180:.0%} for the bad source. The offer-price table changes only purchase cost; inferred quality and outcome draws stay fixed.")
    claims['offered_cost']={'rows':q10.to_dict('records')}
    qt=quotes[['new_source_kind','offered_cpl','p_margin_above_buffer180','expected_shortfall_below_buffer_per_lead','median_margin180']].copy()
    qt['median_margin180']/=quotes['n_leads'].to_numpy()
    qt.columns=['New source','Offered $ / lead','P(day-180 margin > $0.02)','Expected shortfall / lead','Median day-180 margin / lead']
    prior=json.loads((ext/'dynamic_lab/prior_tail_summary.json').read_text())
    large=next(r for r in prior['rows'] if r['scale_sensitivity']=='default' and r['draws']==8192)
    small=next(r for r in prior['rows'] if r['scale_sensitivity']=='default' and r['draws']==512)
    prior_text=(f"Under the bounded scale priors, the largest 1% of 8,192 draws contribute about {100*large['largest_one_percent_draw_revenue_share']:.0f}% of sampled prior revenue. A separate 512-draw mean is ${small['mean_lifetime_revenue']:.2f}, versus an independent reference simulation mean of ${small['reference_mc_mean_lifetime_revenue']:.2f}; its estimated Monte Carlo standard error is ${small['monte_carlo_standard_error']:.2f}. These fictional prior diagnostics are not commercial estimates; heavy tails make numerical error visible.")
    claims['prior_tail']={'source':'extensions/calendar/dynamic_lab/prior_tail_summary.json','rows':[large,small]}
    def metric(model,key):return metrics[(metrics.scenario=='all_scenarios')&(metrics.model==model)&(metrics.metric==key)].iloc[0]
    dc=metric('dynamic_hierarchy','coverage95'); sc=metric('static_hierarchy','coverage95')
    de=metric('dynamic_hierarchy','crps_per_lead'); se=metric('static_hierarchy','crps_per_lead')
    dl=metric('dynamic_hierarchy','decision_loss_per_lead'); sl=metric('static_hierarchy','decision_loss_per_lead')
    heldout=pd.read_csv(art/'heldout_scores.csv')
    raw=float(heldout[heldout.model=='dynamic_hierarchy'].coverage95.mean())
    reliability=pd.read_csv(art/'probability_reliability.csv')
    topbin=reliability.sort_values('probability_bin').iloc[-1]
    evaluation=(f"Across {int(dc.n_worlds)} independent held-out worlds, nominal 95% contribution intervals covered {100*dc['mean']:.1f}% of outcomes for the dynamic model under equal-world averaging (world-bootstrap 95% interval {100*dc.lower95:.1f}%–{100*dc.upper95:.1f}%), versus {100*sc['mean']:.1f}% for the static hierarchy. Weighting forecast rows gives {100*raw:.1f}% dynamic coverage. The dynamic point estimate is below nominal, though its broad world-bootstrap interval includes 95%; this small experiment cannot establish precise calibration. Dynamic CRPS was ${de['mean']:.3f} per lead versus ${se['mean']:.3f} for the static hierarchy, with overlapping uncertainty intervals. Decision loss was ${dl['mean']:.3f} versus ${sl['mean']:.3f} per lead, with the static point estimate lower. Neither average score validates a decision rule.")
    evaluation+=(f" In the highest predicted-probability bin, the mean forecast was {100*topbin.mean_predicted_probability:.1f}% while {100*topbin.realized_profitable_share:.1f}% of outcomes cleared the buffer; the observed-share world-bootstrap interval was {100*topbin.realized_world_bootstrap_lower95:.1f}%–{100*topbin.realized_world_bootstrap_upper95:.1f}% ({int(topbin.independent_worlds)} independent worlds, {int(topbin.forecast_rows)} nested forecasts). A high posterior cutoff is therefore an illustrative policy convention here, not a validated 95%-reliable business decision.")
    claims['evaluation']={'rows':[dc.to_dict(),sc.to_dict(),de.to_dict(),se.to_dict(),dl.to_dict(),sl.to_dict()],'dynamic_row_coverage':raw,'top_probability_bin':topbin.to_dict(),'all_reliability_bins':reliability.to_dict('records')}
    if len(seqm):
        sm=seqm[(seqm.scenario=='all_scenarios')].set_index('metric')
        decisions=pd.read_csv(art/'sequential_decisions.csv')
        below=int((decisions.first_decision=='below_target').sum())
        above=int((decisions.first_decision=='above_target').sum())
        unresolved=int((decisions.first_decision=='undecided').sum())
        false_below=int(decisions.false_negative.sum())
        reversals=int(decisions.any_reversal.sum())
        evaluation+=(f" In the weekly replay of one already-acquired established cohort, the first declaration was below-target in {below} of {len(decisions)} independent worlds; {above} were above-target and {unresolved} stayed unresolved. There were {false_below} false below-target declarations and {reversals} later reversals in this small replay. It does not test a positive-purchase decision and is not policy validation.")
    else:
        decisions=pd.DataFrame()
    claims['sequential_reliability']={'metrics':seqm.to_dict('records'),'confidence_rows':seq.to_dict('records'),'decisions':decisions.to_dict('records')}
    complete=detail[(detail.window_days==7)&(detail.truth_role=='evaluator_only')]
    cohort_variability=(f"The exported detail includes {int(complete.cohort_id.nunique())} cohorts and {len(complete)} complete seven-day cohort windows. Each window counts a person once and divides by the contactable population at the window start. The pooled curve adds numerators and denominators before dividing; thin paths preserve cohort variation rather than hiding it in an average.")
    claims['cohort_rollup']={'cohort_count':int(complete.cohort_id.nunique()),'seven_day_windows':len(complete),'rollups':rolls.to_dict('records')}
    policy="The synthetic sender mails each contactable address once per day. Humans can create repeat clicks and revenue; bots are counted separately and earn no human CPC. Acquisition cost is booked once, while send costs follow contactable addresses. This separates survival, unique people, repeat activity and economics under explicit assumptions."
    clocks="Event time and report time differ: reported events arrive after two days, audit labels after seven, and mature revenue and human-count summaries after fourteen. A forecast at a cutoff uses only information visible by then. Evaluator truth is kept separately to compare predictions with eventual simulated outcomes."
    comparison="The same day-70 cohort's probability evolves as evidence accumulates, while acquisition age and the day-180 target stay fixed. New-source quality remains unknown before launch; after observations arrive, the hierarchical model updates its population, source and cohort components with partial pooling."
    legacy="The earlier v1 stress test remains a separate historical result: nominal 95% intervals covered 56.5% in its frozen 12-world stress suite. That is a failure of the earlier model and should not be confused with the refreshed subscriber model's reliability results."
    portfolio=pd.read_csv(art/'portfolio_summary.csv'); po=portfolio[portfolio.future_scenario=='hold_current'].iloc[0]
    portfolio_text=(f"For the selected illustrative portfolio, cohort contributions are summed on common calendar day {int(po.calendar_horizon)} using shared future calendar paths. Median total contribution is ${po['median']:.2f}, with a 95% interval from ${po.q025:.2f} to ${po.q975:.2f}; simulated truth is ${po.true_margin:.2f}. Joint uncertainty and aggregation do not guarantee calibrated forecasts.")
    claims['portfolio']={'row':po.to_dict()}
    replacements={'COHORT_VARIABILITY':cohort_variability,'POLICY_DESCRIPTION':policy,'OBSERVATION_CLOCKS':clocks,'CONFIDENCE_RULE':conf_rule,'CONFIDENCE_COMPARISON':comparison,'PRICE_COMPARISON':'Figure 1 isolates CPC decline, later CPL inflation, and human/bot deterioration in separate paired synthetic worlds. A later quote does not reprice existing cohorts. Figure 6 tests the resulting predictions against held-out worlds.','CPC_WHAT_IF':cpc_text,'TRANSFER_COMPARISON':transfer,'EVALUATION_COMPARISON':evaluation,'SEQUENTIAL_RELIABILITY':'The fixed-cohort replay separately reports false above-target calls, false below-target calls, no-decision cases, and reversals after an earlier decisive call. These answer both whether the threshold was reached and whether that early classification agreed with eventual hidden outcomes.','LEGACY_VALIDATION':legacy,'PORTFOLIO_COMPARISON':portfolio_text,'PRIOR_CHECK':prior_text}
    for c in metadata['charts']:
        replacements[f"CAPTION_{c['id'][:2]}"]='*Figure '+c['id'][:2]+'. '+c['caption']+'*'
    text=(ROOT/'article.template.md').read_text()
    for k,v in replacements.items():text=text.replace('{{'+k+'}}',v)
    assert '{{' not in text
    # Normalize prose spacing without corrupting commit SHAs, URLs or cohort IDs.
    prose_words='day|days|age|calendar|through|by|cutoff|variance|floor|nominal|across|of|before|after|with|about|roughly|planned|both|at|on|have|is|versus|gives|including|cohort|draws|from|for|two|largest|contains|supply|uses|use|and|are|arrival|cover|first|has|new|only|or|reported|respective|state|the|trailing'
    text=re.sub(r'\b('+prose_words+r')(?=[0-9])',r'\1 ',text)
    text=re.sub(r'(?<=[0-9])(worlds|days|leads|cohorts|draws|tests|cents)\b',r' \1',text)
    text=re.sub(r'(?<=[A-Za-z])(?=\$)',' ',text)
    text=text.replace('A512','A 512').replace('a95%','a 95%')
    text=text.replace('is$','is $').replace('of$','of $').replace('with$','with $').replace('and$','and $')
    text=re.sub(r'\$-([0-9]+(?:\.[0-9]+)?)',r'−$\1',text)
    text=text.replace('../extensions/calendar/dynamic_lab/MODEL.md','https://github.com/rothnic/bayesian-email-cohort-lab/blob/main/extensions/calendar/dynamic_lab/MODEL.md')
    (ROOT/'article.md').write_text(text);(ROOT/'article.mdx').write_text(text.replace('```math','```text'))
    body=text.split('---',2)[2].strip();lines=body.splitlines();out=[];i=0;byid={c['id']:c for c in metadata['charts']}
    while i<len(lines):
        l=lines[i]
        if not l.strip() or l.startswith('# '):i+=1;continue
        if l.startswith('## '):out.append('<h2>'+inline(l[3:])+'</h2>');i+=1;continue
        if l.startswith('```'):
            parts=[];i+=1
            while i<len(lines) and not lines[i].startswith('```'):parts.append(lines[i]);i+=1
            out.append('<pre class="equation"><code>'+html.escape('\n'.join(parts))+'</code></pre>');i+=1;continue
        img=re.match(r'!\[([^]]+)\]\(([^)]+)\)',l)
        if img:
            alt,path=img.groups();key=Path(path).stem;i+=1
            while i<len(lines) and not lines[i].strip():i+=1
            cap=lines[i].strip('*');i+=1
            data=pd.read_csv(tables/(key+'.csv'))
            column_sets={
                '01-calendar-economics':['panel','calendar_day','human_cpc','cpl_quote','paid_cpl','new_purchases','acquisition_total'],
                '02-human-bot-evidence':['calendar_day','reported_events','true_human_events','audited_bot_fraction','audit_n'],
                '03-shared-calendar-states':['origin_day','calendar_day','state','median','q025','q975','true_value','state_scale'],
                '04-cohort-contribution-payback':['panel','origin_day','model','future_scenario','age','median_per_lead','q025_per_lead','q975_per_lead','cdf','crossing_paths','draws'],
                '05-new-source-transfer':['panel','new_source_kind','stage','origin_day','median_per_lead','q025_per_lead','q975_per_lead','true_margin_per_lead'],
                '06-heldout-comparison':['panel','scenario','model','metric','mean','lower95','upper95','units','n_worlds','n_forecasts'],
                '07-cohort-variability':['panel','source_id','cohort_id','age','contactable_share_of_acquired','unique_response_rate_of_eligible','cohort_count','window_days','window_start_age','window_end_age','acquired_leads','eligible_people_at_start','distinct_human_responders'],
                '08-confidence-over-time':['scenario','cohort_id','origin_day','origin_age','posterior_probability','state','true_above_buffer','draws'],
                '11-break-even-learning':['scenario','cohort_id','origin_day','origin_age','conditional_median_days_from_cutoff','conditional_q25_days_from_cutoff','conditional_q75_days_from_cutoff','conditional_q10_days_from_cutoff','conditional_q90_days_from_cutoff','p_no_break_even_by_horizon','true_first_break_even_days_from_cutoff','crossing_draws','draws'],
                '09-cpc-what-if':['cpc_drop_fraction','effective_day','ramp_days','p_margin_above_buffer180','expected_shortfall_below_buffer_per_lead','paired_margin_delta_median','paired_margin_delta_q05','paired_margin_delta_q95','margin_median','draws'],
                '10-decision-reliability':['panel','model','metric','mean','lower95','upper95','n_worlds','n_forecasts','probability_bin','predicted_probability','observed_rate','state']}
            cols=[c for c in column_sets.get(key,list(data.columns)) if c in data]
            shown=data[cols].dropna(how='all').iloc[np.unique(np.linspace(0,len(data)-1,min(10,len(data)),dtype=int))]
            values=table(shown)
            if key=='05-new-source-transfer':values='<h3>Offered cost for a future 100-lead batch</h3><p>Calendar-day evidence and prospective cohort economics; historical acquisition costs stay fixed.</p>'+table(qt)+values
            out.append(f'<figure id="{key}"><div class="chart"><picture><source media="(max-width:600px)" srcset="assets/{key}-mobile.svg"><source srcset="assets/{key}.svg"><img src="{path}" alt="{html.escape(alt,quote=True)}"></picture></div><figcaption>{inline(cap)}</figcaption><div class="figure-tools"><a href="assets/{key}.svg">Full-size SVG</a><a href="assets/{key}.png" download>PNG</a><a href="tables/{key}.csv" download>All plotted values</a></div><details class="values"><summary>Read numerical values</summary>{values}</details></figure>')
            continue
        if l.startswith('- '):
            items=[]
            while i<len(lines) and lines[i].startswith('- '):items.append('<li>'+inline(lines[i][2:])+'</li>');i+=1
            out.append('<ul>'+''.join(items)+'</ul>');continue
        parts=[l];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','!','```','- ')):parts.append(lines[i]);i+=1
        out.append('<p>'+inline(' '.join(parts))+'</p>')
    css=(ROOT/'style.css').read_text()
    htmltext='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="robots" content="noindex,nofollow"><title>When does an email subscriber pay for itself? · Draft</title><style>'+css+'</style></head><body><header><a class="name" href="#top">NICK ROTH</a><span class="status">Article draft</span></header><div class="hero" id="top"><p class="eyebrow">Bayesian cohorts · Changing economics</p><h1>When does an email subscriber pay for itself?</h1><p class="dek">Prices move, cohorts mature, and new sources arrive. What can the evidence tell us before the next purchase?</p><div class="byline"><strong>Nick Roth</strong><span>October 2026 · Draft</span><span>Entirely synthetic analysis</span></div></div><main>'+''.join(out)+'<div class="preview-note">Local review copy. <a href="article.md" download>Markdown</a> · <a href="article.mdx" download>MDX</a> · <a href="tables/offered-cost.csv" download>Offered-cost table</a>. Browser rendering has not been verified in this environment.</div></main><footer>Draft for review · All analytical results are fictional simulation outputs</footer></body></html>'
    (ROOT/'index.html').write_text(htmltext)
    (ROOT/'chart_provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    sources={n:sha256((art/n).read_bytes()) for n in source_paths}
    sources['extensions/calendar/dynamic_lab/prior_tail_summary.json']=sha256((ext/'dynamic_lab/prior_tail_summary.json').read_bytes())
    (ROOT/'claims.json').write_text(json.dumps(json_clean({'source_sha256':sources,'claims':claims}),indent=2,allow_nan=False)+'\n')
    # One-file preview includes assets AND linked source/table downloads.
    embedded={}
    def embed(m):
        attr,path=m.groups()
        if path.startswith(('#','https:','http:','data:')):return m.group(0)
        local=ROOT/path
        if not local.is_file():raise ValueError('Unresolved standalone asset '+path)
        b=local.read_bytes();mime={'.svg':'image/svg+xml','.png':'image/png','.md':'text/markdown','.mdx':'text/plain','.csv':'text/csv'}[local.suffix]
        embedded[path]=sha256(b)
        return attr+'="data:'+mime+';base64,'+base64.b64encode(b).decode()+'"'
    standalone=re.sub(r'(src|srcset|href)="([^"]+)"',embed,htmltext)
    (ROOT/'standalone.html').write_text(standalone)
    (ROOT/'STANDALONE_VALIDATION.json').write_text(json.dumps({'embedded_files':embedded,'unresolved_local_assets':0,'remote_runtime_assets':0,'browser_render_verified':False},indent=2)+'\n')
    print('Built article, MDX, HTML and standalone preview:',len(text.split()),'words,',len(metadata['charts']),'figures')


def sha256(data):return hashlib.sha256(data).hexdigest()


def json_clean(value):
    if isinstance(value,dict):return {k:json_clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_clean(v) for v in value]
    if value is pd.NA:return None
    if isinstance(value,(np.integer,)):return int(value)
    if isinstance(value,(np.floating,float)):
        return None if not np.isfinite(value) else float(value)
    return value


if __name__=='__main__':main()
