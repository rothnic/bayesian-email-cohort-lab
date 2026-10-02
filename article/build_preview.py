"""Portable static preview for this draft; no deployment and no external assets."""
from pathlib import Path
import re,html,json,hashlib,shutil
import pandas as pd
ROOT=Path(__file__).resolve().parent;LAB=ROOT.parent if (ROOT.parent/'parameters.json').exists() else ROOT.parent/'bayesian-email-cohort-lab'
raw=(ROOT/'article.md').read_text();body=raw.split('---',2)[2].strip()
def money(v):return ('−' if v<0 else '')+f'${abs(v):.3f}'
def pct(v):return f'{100*v:.1f}%'
def table(headers,rows):
 return '<div class="table-wrap"><table><thead><tr>'+''.join('<th scope="col">'+html.escape(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
essential={};long=[]
e=pd.read_csv(ROOT/'engagement_values.csv');rows=[]
for src in ['low','middle','high']:
 d=e[e.source_id==src].set_index('cohort_age');rows.append([src.title(),pct(d.loc[3,'trailing_7day_average']),pct(d.loc[30,'trailing_7day_average']),pct(d.loc[180,'trailing_7day_average'])])
 for age in [3,30,180]:long.append({'figure':'01-engagement','source':src,'age':age,'metric':'trailing_daily_clicker_fraction','value':d.loc[age,'trailing_7day_average']})
essential['01-engagement']=table(['Source','Age 3*','Age 30','Age 180'],rows)+'<p class="table-note">Reported daily clickers / original acquired leads. *Age 3 averages the four available days; later rows are seven-day trailing averages.</p>'
s=pd.read_csv(LAB/'artifacts/cold_start/summary.csv');m=s[(s.source_id=='middle')&(s.horizon==180)].sort_values('origin_age');rows=[]
for r in m.itertuples():
 rows.append([r.origin_age,money(r.median/r.n_leads),money(r.q025/r.n_leads)+' to '+money(r.q975/r.n_leads),pct(r.probability_profitable)])
 for k in ['median','q025','q975','probability_profitable']:long.append({'figure':'02-learning','source':'middle','age':r.origin_age,'metric':k+('_per_lead' if k!='probability_profitable' else ''),'value':getattr(r,k)/(r.n_leads if k!='probability_profitable' else 1)})
essential['02-learning']=table(['Forecast age','Median / lead','95% predictive interval','P(positive at 180)'],rows)
c=pd.read_csv(LAB/'artifacts/cold_start/payback_cdf.csv.gz');rows=[]
for age in [3,14,60]:
 d=c[(c.source_id=='middle')&(c.origin_age==age)].set_index('age');rows.append([age,pct(d.loc[180,'cdf']),pct(d.loc[425,'cdf']),pct(d.loc[425,'probability_never'])])
 for h in [180,425]:long.append({'figure':'03-margin-payback','source':'middle','age':age,'metric':f'first_payback_by_{h}','value':d.loc[h,'cdf']})
essential['03-margin-payback']=table(['Forecast age','First payback by 180','First payback by 425','No first payback by 425'],rows)
rows=[]
for src in ['low','middle','high']:
 r=s[(s.source_id==src)&(s.origin_age==60)&(s.horizon==180)].iloc[0]
 rows.append([src.title(),money(r['mean']/r.n_leads),money(r.q025/r.n_leads)+' to '+money(r.q975/r.n_leads),money(r.true_margin/r.n_leads)])
 for k in ['mean','q025','q975','true_margin']:long.append({'figure':'04-sources','source':src,'age':60,'metric':k+'_per_lead','value':r[k]/r.n_leads})
essential['04-sources']=table(['Source','Mean / lead','95% predictive interval','Outcome / lead'],rows)
older=pd.read_csv(LAB/'artifacts/demo/summary.csv');older_rows=[]
for src in ['low','middle','high']:
 r=older[(older.source_id==src)&(older.origin_age==60)&(older.horizon==180)].iloc[0]
 older_rows.append([src.title(),money(r.q025/r.n_leads)+' to '+money(r.q975/r.n_leads),money(r.true_margin/r.n_leads),'Outside'])
 for k in ['q025','q975','true_margin']:long.append({'figure':'04-sources-mature-counterexample','source':src,'age':60,'metric':k+'_per_lead','value':r[k]/r.n_leads})
essential['04-sources']+='<h4>Mature-history counterexample</h4><p class="table-note">Later cohort w07, with older cohorts available; same forecast age60 and horizon180. All three outcomes miss the interval.</p>'+table(['Source','95% predictive interval','Outcome / lead','Covered?'],older_rows)
a=pd.read_csv(LAB/'artifacts/controlled_scenario/metrics_by_origin.csv');b=pd.read_csv(LAB/'artifacts/calibration/metrics_by_origin.csv');b=b[b.metric=='coverage95'].set_index('origin_age');rows=[]
for r in a.itertuples():
 v=b.loc[r.origin_age];rows.append([r.origin_age,pct(r.coverage95),pct(v['mean']),pct(v.lower95)+' to '+pct(v.upper95)])
 for kind,value in [('controlled_95_coverage',r.coverage95),('stress_95_coverage',v['mean'])]:long.append({'figure':'05-coverage','source':kind,'age':r.origin_age,'metric':'coverage95','value':value})
essential['05-coverage']=table(['Forecast age','Controlled coverage','Stress coverage','Stress bootstrap interval'],rows)
pd.DataFrame(long).to_csv(ROOT/'essential_values.csv',index=False)
(ROOT/'essential_values.json').write_text(json.dumps(long,indent=2)+'\n')
def inline(s):
 s=html.escape(s,quote=False)
 s=re.sub(r'`([^`]+)`',r'<code>\1</code>',s)
 s=re.sub(r'\[([^]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',s)
 s=re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',s)
 s=re.sub(r'\*([^*]+)\*',r'<em>\1</em>',s)
 return s
lines=body.splitlines();output=[];i=0;toc=[]
while i<len(lines):
 l=lines[i]
 if not l.strip():i+=1;continue
 if l.startswith('# '): i+=1;continue
 if l.startswith('## '):
  title=l[3:];slug=re.sub(r'[^a-z0-9]+','-',title.lower()).strip('-');toc.append((slug,title));output.append(f'<h2 id="{slug}">{inline(title)}</h2>');i+=1;continue
 if l.startswith('### '):output.append('<h3>'+inline(l[4:])+'</h3>');i+=1;continue
 if l.startswith('```'):
  code=[];i+=1
  while i<len(lines) and not lines[i].startswith('```'):code.append(lines[i]);i+=1
  output.append('<pre class="equation"><code>'+html.escape('\n'.join(code))+'</code></pre>');i+=1;continue
 img=re.match(r'!\[([^]]+)\]\(([^)]+)\)',l)
 if img:
  alt,path=img.groups();key=Path(path).stem;caption='';i+=1
  while i<len(lines) and not lines[i].strip():i+=1
  if i<len(lines) and lines[i].startswith('*Figure '):caption=lines[i].strip('*');i+=1
  output.append(f'<figure id="{key}"><div class="chart"><a href="assets/{key}.svg" aria-label="Open full-size chart: {html.escape(alt,quote=True)}"><picture><source media="(max-width: 600px)" srcset="assets/{key}-mobile.svg" type="image/svg+xml"><source srcset="assets/{key}.svg" type="image/svg+xml"><img src="{path}" alt="{html.escape(alt,quote=True)}" loading="eager"></picture></a></div><figcaption>{inline(caption)}</figcaption><div class="figure-tools"><a href="assets/{key}.svg">Full-size SVG</a><a href="assets/{key}.png" download>PNG</a></div><details class="values"><summary>Read the chart values</summary>{essential.get(key,"")}</details></figure>')
  continue
 if l.startswith('- '):
  items=[]
  while i<len(lines) and lines[i].startswith('- '):items.append('<li>'+inline(lines[i][2:])+'</li>');i+=1
  output.append('<ul>'+''.join(items)+'</ul>');continue
 paragraph=[l];i+=1
 while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','!','```','- ')):
  paragraph.append(lines[i]);i+=1
 output.append('<p>'+inline(' '.join(paragraph))+'</p>')
css='''
:root{--paper:#fbfaf6;--ink:#263540;--muted:#62717b;--blue:#2d679d;--amber:#b17721;--line:#d8dedf}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font-family:Georgia,"Times New Roman",serif;font-size:20px;line-height:1.7}a{color:var(--blue);text-decoration-thickness:1px;text-underline-offset:3px}a:hover{color:#19486e}a:focus-visible,summary:focus-visible{outline:3px solid #b17721;outline-offset:5px}header{max-width:1160px;margin:0 auto;padding:28px 38px;display:flex;justify-content:space-between;align-items:center;font:13px/1.3 system-ui,sans-serif;letter-spacing:.04em}header .name{font-weight:750;letter-spacing:.16em;font-size:13px;color:var(--ink);text-decoration:none}.status{font-size:11px;text-transform:uppercase;border:1px solid var(--line);padding:7px 11px;border-radius:100px;color:var(--muted)}.hero{max-width:900px;margin:45px auto 55px;padding:0 32px;text-align:left}.eyebrow{font:700 12px/1.4 system-ui,sans-serif;letter-spacing:.14em;text-transform:uppercase;color:var(--amber);margin:0 0 20px}h1{font-weight:500;font-size:clamp(42px,5.4vw,66px);line-height:1.08;letter-spacing:-.035em;margin:0 0 28px;max-width:850px}.dek{font-size:23px;line-height:1.5;color:#596a74;max-width:740px;margin:0 0 26px}.byline{font:13px/1.5 system-ui,sans-serif;color:var(--muted);display:flex;gap:15px;flex-wrap:wrap}.byline strong{color:var(--ink);font-weight:650}main{max-width:760px;margin:0 auto;padding:0 28px 80px}p{margin:0 0 1.35em}h2{font-weight:500;letter-spacing:-.015em;font-size:32px;line-height:1.2;margin:2.3em 0 .7em;scroll-margin-top:35px}h3{font:650 17px/1.4 system-ui,sans-serif;margin:2em 0 .7em}strong{font-weight:700}ul{padding-left:1.3em;margin:1em 0 1.5em}li{padding:0 0 .55em .2em}code{font:85%/1.5 ui-monospace,monospace}.equation{padding:22px 26px;background:#eff3f5;border-radius:4px;white-space:pre-wrap;overflow-wrap:anywhere;font-size:18px;line-height:1.6;margin:1.7em 0;border-left:3px solid #abc6dc}.equation code{font-size:17px}figure{margin:44px -110px 56px;padding:0;background:white;border:1px solid #e3e6e3;border-radius:4px;overflow:hidden}.chart{padding:16px 18px 0}.chart a{display:block}.chart img{display:block;width:100%;height:auto}.chart picture{display:block}figcaption{font:14px/1.6 system-ui,sans-serif;color:#52616b;padding:10px 28px 13px}.figure-tools{font:12px/1.4 system-ui,sans-serif;display:flex;gap:20px;padding:0 28px 17px}.values{border-top:1px solid #e6ebec;padding:0 28px 13px;font:14px/1.5 system-ui,sans-serif}.values summary{padding:15px 0 0;cursor:pointer;font-weight:650;color:var(--blue)}.values[open] summary{padding-bottom:15px}.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}th{font-weight:650;text-align:left;background:#f3f6f7;color:#415565}td,th{padding:10px 9px;border-bottom:1px solid #e1e7e9;vertical-align:top}td{font-feature-settings:"tnum"}.table-note{font:12px/1.5 system-ui,sans-serif;margin:12px 0 0}.preview-note{font:13px/1.5 system-ui,sans-serif;color:var(--muted);border-top:1px solid var(--line);padding-top:24px;margin-top:45px}.jump{font:13px/1.4 system-ui,sans-serif;display:inline-block;margin-top:10px}footer{border-top:1px solid var(--line);padding:25px;text-align:center;font:12px/1.5 system-ui,sans-serif;color:var(--muted)}@media(max-width:1000px){figure{margin-left:-25px;margin-right:-25px}}@media(max-width:600px){body{font-size:18px;line-height:1.65}header{padding:20px 22px}.status{font-size:9px;padding:6px 8px}.hero{margin:32px auto 35px;padding:0 23px}h1{font-size:43px;line-height:1.1;letter-spacing:-.04em}.dek{font-size:20px}.eyebrow{font-size:10px;margin-bottom:16px}.byline{font-size:12px}main{padding:0 23px 50px}h2{font-size:29px;margin-top:2em}figure{margin:32px -13px 42px}.chart{padding:12px 2px 0}figcaption{padding:8px 14px 11px;font-size:13px}.figure-tools{padding:0 14px 15px}.values{padding:0 12px 13px}.values summary{font-size:14px}td,th{padding:8px 6px;font-size:13px}.equation{padding:18px 15px}.equation code{font-size:15px}}@media print{header,.status,.figure-tools,.values,.preview-note,footer{display:none}body{background:white;font-size:12pt}main{max-width:none;padding:0}.hero{margin:0 0 30px}h1{font-size:35pt}figure{margin:25px 0;break-inside:avoid}h2{break-after:avoid}a{color:inherit}}
'''
page='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="robots" content="noindex,nofollow"><title>When does an email subscriber pay for itself? · Draft</title><meta name="description" content="A synthetic Bayesian experiment in cohort payback and uncertainty."><style>'+css+'</style></head><body><header><a class="name" href="#top">NICK ROTH</a><span class="status">Local draft preview</span></header><div class="hero" id="top"><p class="eyebrow">Research notes · Bayesian decision making</p><h1>When does an email subscriber pay for itself?</h1><p class="dek">What early evidence can tell us about a cohort, and why a narrower forecast can still be wrong.</p><div class="byline"><strong>Nick Roth</strong><span>Draft · October 2026</span><span>Synthetic demonstration</span></div></div><main>'+''.join(output)+'<div class="preview-note">Prepared as an article draft. This local preview has no analytics, remote fonts, or live-site dependencies. <a href="article.md">Markdown source</a> · <a href="article.mdx">MDX source</a> · <a href="essential_values.csv">Chart values</a></div></main><footer>Draft for review · All analytical results are fictional simulation outputs</footer></body></html>'
(ROOT/'index.html').write_text(page)
(ROOT/'article.mdx').write_text(raw.replace('```math','```text'))
print('Built static HTML and MDX; essential values',len(long))
