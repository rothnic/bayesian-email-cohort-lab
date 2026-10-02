#!/usr/bin/env python3
"""Build a compact source-backed report from held-out artifacts, without inference."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent

def main():
    base=ROOT/"artifacts"/"heldout"
    forecasts=pd.read_csv(base/"forecasts.csv")
    scores=pd.read_csv(base/"margin_scores.csv")
    decisions=pd.read_csv(base/"decisions.csv")
    allocations=pd.read_csv(base/"allocations.csv")
    truth=pd.read_csv(base/"truth_cohort_summary.csv")
    resources=pd.read_csv(base/"resource_measurements.csv")
    metadata=json.loads((base/"run_metadata.json").read_text())
    lines=["# Fictional cohort profitability: first proof", "", "## Bottom line", "", "The generator, as-of signed accounting, two comparable point baselines and rolling-origin replay execute locally. This is a wiring/assumption proof, not evidence of Bayesian superiority or calibrated uncertainty. All data and outcomes below are fictional.", "", "## Reproduce", "", "```sh", "python run_pilot.py smoke", "python run_pilot.py heldout", "python build_report.py", "python diagnose_engagement.py", "python validate.py", "```", "", "The held-out suite uses 3 fixed seeds × 4 preregistered scenarios, with 18,000 leads/world. Models replay ages 3, 7, 14, 30 and 60 for later cohorts. Only timestamps and revisions visible at each actual calendar cutoff enter model fits. Earlier horizons are omitted from prospective scores. Truth remains evaluator-only.", "", "## Version and resources", "", f"Parameter SHA-256: `{metadata['parameter_sha256']}`", "", f"Versions: {', '.join(f'{k}={v}' for k,v in metadata['versions'].items() if k!='platform')}", "", f"Measured held-out suite wall time: {metadata['suite_wall_seconds']:.1f} seconds. Per-world runtime range {resources.total_seconds.min():.1f}–{resources.total_seconds.max():.1f} seconds; median {resources.total_seconds.median():.1f}. Peak per-world RSS range {resources.peak_rss_mib.min():.1f}–{resources.peak_rss_mib.max():.1f} MiB. Fresh subprocesses isolate RSS measurements.", "", "The combined 21-test suite covers no-future perturbation invariance, ledger reconciliation, missing telemetry, pending settlement after final send, operational contactability, null probability fields, source allocation constraints and decision arithmetic. `validate.py` also verifies emitted schema fields and unchanged parameter/source hashes.", "", "CSV/JSON output preserves contract fields. Optional Parquet serialization is pending pyarrow; no packages were installed. Tests use unittest, not pytest.", "", "## Accounting and observed states", "", "Immutable acquisition/send costs and revenue/reversal/invoice entries reconcile to complete-policy cohort margins. Human→bot reclassifications append later-observed versions. Event occurrence, economic settlement and reporting clocks remain distinct. Missing exposure counts are explicit unavailable values, never fabricated zeros. Inactivity/dormancy does not remove contactability; operational unsubscribe/permanent suppression does. Delayed responses to prior messages continue after sends stop.", "", "Policy sends at ages 0–365 inclusive, allows response lags up to 30 days and accounting/reporting up to another 30 days, and terminates at age 425. First payback can be reversed by later costs/revisions; horizon positivity and remains-positive flags are separate. Never-under-policy makes no claim about unrestricted lifetime."]
    terminal=truth[truth.horizon.eq(425)].copy()
    terminal["unit_margin"]=terminal.margin/terminal.n_leads
    category=np.where(terminal.unit_margin < -.03,"losing",np.where(terminal.unit_margin > .03,"profitable","near-break-even (±$0.03/lead)"))
    lines += ["", "Fictional terminal cohort categories: "+", ".join(f"{k}: {v}" for k,v in pd.Series(category).value_counts().items()), "", "Truth payback states: "+", ".join(f"{k}: {v}" for k,v in terminal.payback_state.value_counts().items())]
    for category_name, mask in (("Losing",terminal.unit_margin.lt(-.03)),("Near break-even",terminal.unit_margin.abs().le(.03)),("Profitable",terminal.unit_margin.gt(.03))):
        candidates=terminal.loc[mask].sort_values("unit_margin",key=(lambda s:s.abs()) if category_name=="Near break-even" else None)
        if len(candidates):
            r=candidates.iloc[0]
            lines += [f"- {category_name}: {r.scenario}, seed {int(r.seed)}, {r.cohort_id}; terminal margin ${r.margin:.2f} (${r.unit_margin:.4f}/lead); {r.payback_state}"]
    crossed_then_lost=terminal.loc[terminal.first_payback_age.notna() & ~terminal.positive_at_horizon]
    lines += [f"- {len(crossed_then_lost)} cohorts crossed payback then ended with nonpositive contribution; first crossing alone is insufficient"]
    lines += ["", "The fictional reference age-three per-send propensity is 4.118%; occurrence-day unique engagement differs after response lags, source/lead variation and repeat responses. The small smoke diagnostic shows roughly 75–77% contactability at day 180 even when day-180 human engagement is 0–0.625%. Those are different observables, not a survival estimate."]
    # Keep scored-column names inspectable, and summarize world means, not correlated rows as independent simulations.
    lines += ["", "## Forecast accuracy and failures", "", "All forecasts are deterministic points: draw_id=0, probability_profitable is unavailable. Brier/log scores, CRPS, predictive coverage/width and payback-CDF calibration are intentionally unavailable. Three held-out seeds per scenario are insufficient for a strong statistical ranking.", "", "Detailed source × origin × horizon errors are in the CSV artifacts and per-world JSON summaries. All worlds remain in the comparison, including adverse outcomes.", ""]
    error_column=next((c for c in ("absolute_error","abs_error","margin_absolute_error") if c in scores),None)
    if error_column:
        grouped=scores.groupby(["scenario","model","origin_age"])[error_column].mean().reset_index()
        grouped.to_csv(base/"mae_by_scenario_model_age.csv",index=False)
        lines += ["Mean absolute cohort margin error in dollars, averaged over prospective horizons:", "", grouped.to_markdown(index=False) if False else "```text", grouped.pivot(index=["scenario","model"],columns="origin_age",values=error_column).round(2).to_string(), "```"]
        # Biggest misses must remain visible even if they make a baseline look bad.
        bad=scores.sort_values(error_column,ascending=False).head(6)
        lines += ["", "Largest individual misses (no seeds discarded):", "", "```text",bad[[c for c in ("scenario","seed","model","cohort_id","origin_age","horizon","predicted_margin","true_margin",error_column) if c in bad]].round(2).to_string(index=False),"```"]
        worldmeans=scores.groupby(["scenario","seed","model"])[error_column].mean().reset_index()
        worldmeans.to_csv(base/"world_mae.csv",index=False)
        intervals=[]
        for (scenario,model),g in worldmeans.groupby(["scenario","model"]):
            mean=g[error_column].mean();se=g[error_column].std(ddof=1)/np.sqrt(len(g))
            intervals.append({"scenario":scenario,"model":model,"independent_worlds":len(g),"mean_mae":mean,"world_standard_error":se})
        uncertainty=pd.DataFrame(intervals);uncertainty.to_csv(base/"world_mae_uncertainty.csv",index=False)
        lines += ["", "Simulation variability: standard errors below use independent worlds, not thousands of correlated forecast rows; n=3 per scenario/model is descriptive, not validation.", "", "```text",uncertainty.round(2).to_string(index=False),"```"]
    failure=decisions.loc[(decisions.scenario=="late_reactivation") & (decisions.origin_age==3) & (decisions.model=="observable_survival")]
    lines += ["", f"Failure case: at age three the observable baseline falsely stops {int(failure.profitable_opportunity_falsely_stopped.sum())} profitable late-reactivating opportunities. Including partial curtailment, forgone profitable margin totals ${failure.forgone_profitable_margin.sum():.2f} in this alternative acquisition replay. Its better aggregate point error does not remove tail/decision risk."]
    lines += ["", "## Decisions and allocation", "", "Stopping acquisition is replayed separately from stopping sends. Unit-margin acquisition replay evaluates stop/small/normal batch choices and charges forgone profitable margin. Existing cohorts continue sending. Allocation has a fixed acquisition budget and source capacity; oracle outcomes come only from simulation truth. These are simplified counterfactual bookkeeping demonstrations with constant per-lead marginal value, not validated causal next-batch or saturation models.", "", "Replay rule: stop when predicted full-policy unit margin ≤0; buy 25% of a 500-lead normal batch for positive margins up to $0.01/lead; otherwise buy the normal batch. Allocation budget is $100 with capacity 500 leads/source. Fractional allocation units are permitted for this diagnostic. Every origin is an alternative replay of the same cohorts; do not add origins as cumulative real spending.", ""]
    numeric=[c for c in decisions.select_dtypes(include=np.number).columns if any(x in c for x in ("regret","avoided","forgone","savings","false_stop"))]
    if numeric:
        groupkeys=[c for c in ("scenario","model","origin_age") if c in decisions]
        ds=decisions.groupby(groupkeys)[numeric].sum().reset_index()
        ds.to_csv(base/"decision_totals.csv",index=False)
        lines += ["", "Age-three totals across the three worlds per scenario/model (alternative replay, not historical spend):", "", "```text",ds.loc[ds.origin_age.eq(3)].round(2).to_string(index=False),"```"]
    av=allocations.loc[allocations.origin_age.eq(3)].groupby(["scenario","model"])[["chosen_realized_margin","always_continue_realized_margin","oracle_realized_margin","allocation_regret","source_pairwise_ranking_accuracy"]].mean().reset_index()
    av.to_csv(base/"allocation_mean_age3.csv",index=False)
    lines += ["", "Mean age-three allocation outcomes in dollars (fixed $100 acquisition budget):", "", "```text",av.round(2).to_string(index=False),"```"]
    lines += ["", "## Limitations and next stage", "", "- This baseline-only report compares two deterministic models. It does not evaluate the separate Bayesian implementation", "- Extrapolation beyond observed support is assumption-driven. Late reactivation and future shared price/attribution shocks intentionally challenge it", "- Mature-history delay correction and pending-receivable treatment are simplified; informative/value-dependent missingness requires a stronger observation model", "- Forecast accuracy is assessed against complete finite-policy economic truth, while models see delayed as-of records; a real pilot needs mature/censoring-aware outcome adjudication", "- Point acquisition decisions and stop-sending diagnostics are not value-of-information optimization; source capacities/budgets are illustrative", "- See the separate Bayesian demo for posterior predictive intervals and modest stress-validation results; full negative-binomial regression, learned calendar effects and rigorous experimental stopping remain extensions", "", "## Traceability", "", "This report covers the deterministic baseline-only suite. The Bayesian implementation and its independently reproduced results are documented separately in README.md and artifacts/demo/RESULTS.md. All simulations are fictional."]
    (ROOT/"REPORT.md").write_text("\n".join(lines)+"\n")
    print(ROOT/"REPORT.md")
if __name__=="__main__":main()
