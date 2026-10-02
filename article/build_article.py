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
    claims={};source_paths=['cohort_predictions.csv','metrics.csv','heldout_scores.csv','transfer.csv','quote_sensitivity.csv','portfolio_summary.csv']
    select=dict(scenario='human_decline_bot_rise',origin_day=119,source_id='established_a',target_role='next_purchase',model='dynamic_hierarchy')
    d=predictions
    for k,v in select.items():d=d[d[k]==v]
    h=d[d.future_scenario=='hold_current'].iloc[0];f=d[d.future_scenario=='fixed_price'].iloc[0];t=d[d.future_scenario=='continue_recent_trend'].iloc[0]
    price=(f"For one planned100-lead batch, forecast on calendar day119 for acquisition on126, the observed quote is${h.acquisition_cost:.3f} in total. Holding future CPC at the learned early reference gives mean day180 contribution of${f['mean']:.2f} and a {100*f.p_payback425:.1f}% first-payback probability by425. The current-state future gives${h['mean']:.2f} and{100*h.p_payback425:.1f}%. These are total cohort dollars. The fictional realized day180 contribution is${h.true_margin:.2f}, outside the current-state95% interval [${h.q025:.2f},${h.q975:.2f}].")
    price+=f" The recent-state extrapolation has mean${t['mean']:.2f}, median${t['median']:.2f} and mean Monte Carlo standard error${t.mean_mcse:.2f}. It is not uniformly more pessimistic: uncertain future slopes create an asymmetric upper tail."
    claims['price_comparison']={'selector':select,'rows':d.to_dict('records')}
    prior=trans[trans.stage=='prior'];assert np.allclose(prior['mean'],prior['mean'].iloc[0]);after=trans[trans.origin_day==119]
    good=after[after.new_source_kind=='good'].iloc[0];bad=after[after.new_source_kind=='bad'].iloc[0];pr=prior.iloc[0]
    transfer=(f"Before launch, both sources have the same forecast, including{100*pr.p_payback425:.1f}% first-payback probability by425. After their own observations arrive, their forecasts separate. At calendar day119, the good source's first cohort has mean day180 contribution of${good['mean']:.2f}; the bad source's is${bad['mean']:.2f}. Both are100-lead cohorts. Both realized day180 outcomes still fall below their respective95% predictive intervals: ${good.true_margin:.2f} versus [${good.q025:.2f},${good.q975:.2f}], and${bad.true_margin:.2f} versus [${bad.q025:.2f},${bad.q975:.2f}]. Learning a difference is not the same as quantifying its uncertainty correctly.")
    claims['transfer']={'selector':{'origin_day':119},'rows':after.to_dict('records'),'prior_rows':prior.to_dict('records')}
    def metric(model,key):return metrics[(metrics.scenario=='all_scenarios')&(metrics.model==model)&(metrics.metric==key)].iloc[0]
    dc=metric('dynamic_hierarchy','coverage95');sc=metric('static_hierarchy','coverage95');de=metric('dynamic_hierarchy','crps_per_lead');se=metric('static_hierarchy','crps_per_lead');dl=metric('dynamic_hierarchy','decision_loss_per_lead');sl=metric('static_hierarchy','decision_loss_per_lead')
    heldout=pd.read_csv(art/'heldout_scores.csv');raw_coverage=float(heldout.loc[heldout.model=='dynamic_hierarchy','coverage95'].mean())
    evaluation=(f"The dynamic model's nominal95% intervals cover an equal-world average of{100*dc['mean']:.1f}% of outcomes, compared with{100*sc['mean']:.1f}% for the static hierarchy. Each model has{int(dc.n_forecasts)} forecast cases from{int(dc.n_worlds)} independent worlds. Pooling all forecast rows instead gives{100*raw_coverage:.1f}% dynamic coverage; that weights the source-launch worlds more heavily because they contain extra forecasts. Both summaries show severe undercoverage. Dynamic CRPS, a score for the whole predictive distribution where lower is better, is${de['mean']:.3f} per lead versus${se['mean']:.3f}. Their world-bootstrap intervals overlap. Conditional acquisition loss is${dl['mean']:.3f} per lead versus${sl['mean']:.3f}; the static point estimate is slightly lower before rounding, so this does not establish a dynamic decision advantage. These are failures of nominal coverage, alongside a limited improvement in average predictive error.")
    claims['evaluation']={'selector':{'scenario':'all_scenarios'},'rows':[r.to_dict() for r in [dc,sc,de,se,dl,sl]],'dynamic_raw_row_coverage':raw_coverage}
    quotes=pd.read_csv(art/'quote_sensitivity.csv');shutil.copy2(art/'quote_sensitivity.csv',tables/'offered-cost.csv')
    qt=quotes[['new_source_kind','offered_cpl','p_payback425','p_no_payback425','median_margin180']].copy()
    qt['median_margin180']/=100
    qt.columns=['New source','Offered $ / lead','P(payback by 425)','P(no payback by 425)','Median day 180 $ / lead']
    q10=quotes[quotes.offered_cpl==.1]
    qg=q10[q10.new_source_kind=='good'].iloc[0];qb=q10[q10.new_source_kind=='bad'].iloc[0]
    transfer+=f" For the next new-source batch, offered at$0.10 per lead on calendar119 for arrival126, estimated first-payback probability is{100*qg.p_payback425:.1f}% for the good source and{100*qb.p_payback425:.1f}% for the bad source. The table under this figure varies the offer while keeping quality inference and outcome draws fixed."
    claims['offered_cost']={'selector':{'origin_day':119,'offered_cpl':.1},'rows':q10.to_dict('records')}
    prior=json.loads((ext/'dynamic_lab/prior_tail_summary.json').read_text())
    large=next(r for r in prior['rows'] if r['scale_sensitivity']=='default' and r['draws']==8192)
    small=next(r for r in prior['rows'] if r['scale_sensitivity']=='default' and r['draws']==512)
    prior_text=(f"The implementation has finite monetary moments under its bounded scale priors. That is still a low bar. In a prior check, the largest1% of8,192 sampled draws supply roughly{100*large['largest_one_percent_draw_revenue_share']:.0f}% of revenue under the default prior. A512-draw mean is${small['mean_lifetime_revenue']:.2f} against an analytical expectation of${small['analytic_mean_lifetime_revenue']:.2f}, with an estimated Monte Carlo standard error of${small['monte_carlo_standard_error']:.2f}. Those are fictional prior properties, not plausible commercial payment estimates. Numerical uncertainty belongs beside expected-profit claims.")
    claims['prior_tail']={'source':'extensions/calendar/dynamic_lab/prior_tail_summary.json','rows':[large,small]}
    portfolio=pd.read_csv(art/'portfolio_summary.csv');po=portfolio[portfolio.future_scenario=='hold_current'].iloc[0]
    portfolio_text=(f"For an illustrative portfolio of four existing cohorts and two planned batches, the common endpoint is calendar day{int(po.calendar_horizon)}. Cohort ages differ at that date. The current-state scenario gives median total contribution of${po['median']:.2f}, with a95% interval [${po.q025:.2f},${po.q975:.2f}]. The simulated outcome is${po.true_margin:.2f}, below that interval. Correctly sharing calendar paths does not rescue a misspecified forecast.")
    claims['portfolio']={'selector':{'future_scenario':'hold_current','calendar_horizon':299},'row':po.to_dict()}
    replacements={'PRICE_COMPARISON':price,'TRANSFER_COMPARISON':transfer,'EVALUATION_COMPARISON':evaluation,'PRIOR_CHECK':prior_text,'PORTFOLIO_COMPARISON':portfolio_text}
    for i,c in enumerate(metadata['charts'],1):replacements[f'CAPTION_{i:02}']='*Figure '+str(i)+'. '+c['caption']+'*'
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
                '06-heldout-comparison':['panel','scenario','model','metric','mean','lower95','upper95','units','n_worlds','n_forecasts']}
            cols=[c for c in column_sets[key] if c in data]
            shown=data[cols].dropna(how='all').iloc[np.unique(np.linspace(0,len(data)-1,min(10,len(data)),dtype=int))]
            values=table(shown)
            if key=='05-new-source-transfer':values='<h3>Offered cost for a future100-lead batch</h3><p>Calendar119 evidence; planned arrival126; all paths remain in the denominator. Historical acquisition costs stay fixed.</p>'+table(qt)+values
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
    htmltext='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="robots" content="noindex,nofollow"><title>When does an email subscriber pay for itself? · Draft</title><style>'+css+'</style></head><body><header><a class="name" href="#top">NICK ROTH</a><span class="status">Article draft</span></header><div class="hero" id="top"><p class="eyebrow">Bayesian cohorts · Changing economics</p><h1>When does an email subscriber pay for itself?</h1><p class="dek">Prices move, cohorts mature, and new sources arrive. What can the evidence tell us before the next purchase?</p><div class="byline"><strong>Nick Roth</strong><span>October2026 · Draft</span><span>Entirely synthetic analysis</span></div></div><main>'+''.join(out)+'<div class="preview-note">Local review copy. <a href="article.md" download>Markdown</a> · <a href="article.mdx" download>MDX</a> · <a href="tables/offered-cost.csv" download>Offered-cost table</a>. Browser rendering has not been verified in this environment.</div></main><footer>Draft for review · All analytical results are fictional simulation outputs</footer></body></html>'
    (ROOT/'index.html').write_text(htmltext)
    (ROOT/'chart_provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    sources={n:sha256((art/n).read_bytes()) for n in source_paths}
    sources['extensions/calendar/dynamic_lab/prior_tail_summary.json']=sha256((ext/'dynamic_lab/prior_tail_summary.json').read_bytes())
    (ROOT/'claims.json').write_text(json.dumps({'source_sha256':sources,'claims':claims},indent=2,allow_nan=False)+'\n')
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


if __name__=='__main__':main()
