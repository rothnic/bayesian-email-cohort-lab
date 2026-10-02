#!/usr/bin/env python3
"""Complete synthetic outcomes diagnostic; never a forecast-model input."""
from pathlib import Path
import numpy as np
import pandas as pd
from cohort_lab.generator import generate_world
from cohort_lab.accounting import load_parameters

root=Path(__file__).resolve().parent
params=load_parameters()
w,t,p=generate_world(11,"collapse_tail",params,small=True)
meta=w.leads.groupby("cohort_id").agg(acquired_at=("acquired_at","first"),source_id=("source_id","first"),n_leads=("lead_id","size"))
e=t.final_human_events.copy()
e["age"]=e.occurred_at-e.cohort_id.map(meta.acquired_at)
clicks=e.groupby(["cohort_id","age"]).lead_id.nunique().rename("unique_human_clickers")
ages=[0,1,3,7,30,90,180,365]
index=pd.MultiIndex.from_product([meta.index,ages],names=["cohort_id","age"])
daily=clicks.reindex(index,fill_value=0).reset_index().merge(meta,left_on="cohort_id",right_index=True)
daily["daily_engagement_per_original_lead"]=daily.unique_human_clickers/daily.n_leads
eligibility=w.exposure_daily.groupby(["cohort_id","age"]).eligible.sum().rename("eligible")
daily=daily.merge(eligibility,left_on=["cohort_id","age"],right_index=True)
daily["contactability_per_original_lead"]=daily.eligible/daily.n_leads
summary=daily.groupby(["source_id","age"])[["daily_engagement_per_original_lead","contactability_per_original_lead"]].mean().reset_index()
output=root/"artifacts"/"smoke";output.mkdir(parents=True,exist_ok=True)
summary.to_csv(output/"complete_truth_engagement_diagnostic.csv",index=False)
curve=params["reference_response_curve"]
reference=curve["fast_amplitude"]*np.exp(-3/curve["fast_timescale"])+curve["tail_amplitude"]*np.exp(-3/curve["tail_timescale"])
print(f"Reference per-send propensity at age 3: {reference:.4%}; actual occurrence-day engagement differs after response lags and heterogeneity")
print(summary[summary.age.isin([3,180])].to_string(index=False))
