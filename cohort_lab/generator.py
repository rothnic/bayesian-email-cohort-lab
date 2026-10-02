"""Sparse fictional generator. Truth is returned separately and never used by models."""
from dataclasses import replace
from math import exp
import numpy as np
import pandas as pd
from .accounting import World, GeneratorTruth, Policy, load_parameters

LEAD_COLUMNS = ["lead_id","source_id","cohort_id","acquired_at","acquisition_cost","currency","arm","assignment_probability","eligibility_rule_version"]
EXPOSURE_COLUMNS = ["cohort_id","source_id","date","age","arm","attempted","accepted","eligible","exits","observed_at","telemetry_status","content_id"]
EVENT_COLUMNS = ["record_id","event_id","lead_id","cohort_id","source_id","message_id","arm","send_at","occurred_at","observed_at","qualification","revision_sequence","filter_version"]
LEDGER_COLUMNS = ["entry_id","event_id","cohort_id","source_id","arm","amount","economic_at","observed_at","kind","revision_id","revises_entry_id","currency"]
EXIT_COLUMNS = ["lead_id","cohort_id","source_id","occurred_at","observed_at","reason"]

def _distribution(mapping):
    return np.array(list(map(int, mapping))), np.array(list(mapping.values()), dtype=float)

def generate_world(seed=11, scenario="collapse_tail", parameters=None, small=False):
    params = dict(parameters or load_parameters())
    if scenario not in params["scenarios"]:
        raise ValueError(f"Unknown preregistered scenario: {scenario}")
    n = 80 if small else int(params["leads_per_cohort"])
    ncohorts = 6 if small else int(params["cohorts_per_source"])
    params["leads_per_cohort"] = n
    params["cohorts_per_source"] = ncohorts
    rng = np.random.default_rng(seed)
    policy = Policy(last_send_age=params["last_send_age"], max_response_delay=params["max_response_delay"],
                    max_accounting_delay=params["max_accounting_delay"], terminal_age=params["terminal_age"], send_cost=params["send_cost"])
    calendar_length = (ncohorts - 1) * params["cohort_spacing_days"] + policy.terminal_age + 1
    # One world-wide realization per date, shared across every source/cohort/arm.
    z = rng.normal(size=(2, calendar_length))
    shock = np.zeros_like(z)
    for t in range(calendar_length):
        shock[:,t] = .55 * shock[:,t-1] + .12 * z[:,t] if t else .12*z[:,t]
    click_shock, value_shock = np.exp(shock[0]), np.exp(shock[1])
    if scenario == "calendar_attribution_shock":
        click_shock[90:130] *= .85
        value_shock[90:130] *= .25
        value_shock[160:220] *= 1.3
    response_days, response_probs = _distribution(params["response_delay_probabilities"])
    posting_days, posting_probs = _distribution(params["posting_delay_probabilities"])
    lead_rows, exposure_rows, event_rows, ledger_rows, exit_rows = [], [], [], [], []
    latent_rows, final_events = [], []
    next_lead = next_event = next_entry = next_record = 0
    content_response = np.array([1., .9, 1.1, .95])
    content_value = np.array([1., 1.15, .85, 1.05])
    def entry(event_id, cohort, source, arm, amount, economic, observed, kind, revision_id="", revises=""):
        nonlocal next_entry
        identifier = f"l{next_entry}"; next_entry += 1
        ledger_rows.append((identifier,event_id,cohort,source,arm,float(amount),int(economic),observed,kind,revision_id,revises,"USD"))
        return identifier
    for s in params["sources"]:
        source = s["source_id"]
        for week in range(ncohorts):
            cohort = f"{source}-w{week:02d}"
            acquired = week * params["cohort_spacing_days"]
            ids = np.arange(next_lead, next_lead+n); next_lead += n
            arms = rng.integers(0, 2, n)
            quality = rng.lognormal(-.5*.6**2, .6, n)
            cohort_quality = rng.lognormal(-.5*.30**2, .30)
            dormant = rng.random(n) < .35
            eligible = np.ones(n, dtype=bool)
            per_lead_cost = s["acquisition_cost"] * rng.uniform(.95,1.05)
            for j in range(n):
                lead_id = f"u{ids[j]}"
                lead_rows.append((lead_id,source,cohort,acquired,per_lead_cost,"USD",int(arms[j]),.5,"eligible/1"))
                latent_rows.append((lead_id,cohort,float(quality[j]),bool(dormant[j]),float(cohort_quality)))
            for arm in (0,1):
                count = int((arms == arm).sum())
                base = entry("",cohort,source,arm,-count*per_lead_cost,acquired,acquired,"acquisition")
                # Final invoice is an appended cost adjustment, invisible to early snapshots.
                adjustment = -count*per_lead_cost*rng.normal(.012,.02)
                entry("",cohort,source,arm,adjustment,acquired+21,acquired+24,"acquisition_revision",f"invoice-{cohort}-{arm}",base)
            for age in range(policy.last_send_age+1):
                date = acquired + age
                exits = eligible & (rng.random(n) < params["operational_exit_hazard"])
                eligible[exits] = False
                for j in np.flatnonzero(exits):
                    reason = "unsubscribe" if rng.random() < .7 else "permanent_suppression"
                    exit_rows.append((f"u{ids[j]}",cohort,source,date,date+int(rng.choice([0,1])),reason))
                accepted = eligible & (rng.random(n) < params["acceptance_probability"])
                content = date % 4
                telemetry_missing = rng.random() < .003
                for arm in (0,1):
                    arm_mask = arms == arm
                    attempted_count = int((eligible & arm_mask).sum())
                    exposure_rows.append((cohort,source,date,age,arm,attempted_count,int((accepted & arm_mask).sum()),
                                          attempted_count,int((exits & arm_mask).sum()),date,
                                          "missing" if telemetry_missing else "observed",f"content-{content}"))
                    entry("",cohort,source,arm,-attempted_count*policy.send_cost,date,date,"sending")
                if scenario == "stationary":
                    reference = .010 * exp(-age/250.)
                else:
                    curve = params["reference_response_curve"]
                    reference = curve["fast_amplitude"]*exp(-age/curve["fast_timescale"]) + curve["tail_amplitude"]*exp(-age/curve["tail_timescale"])
                p = reference * s["response_multiplier"] * cohort_quality * quality * click_shock[date] * content_response[content]
                if scenario == "late_reactivation":
                    if 7 <= age < 80:
                        p[dormant] *= .03
                    elif 80 <= age <= 150:
                        p[dormant] += .045 * s["response_multiplier"] * quality[dormant]
                responders = accepted & (rng.random(n) < np.minimum(p,.90))
                counts = np.zeros(n,dtype=int)
                responders_i = np.flatnonzero(responders)
                counts[responders_i] = 1 + rng.poisson(params["repeat_click_mean"],len(responders_i))
                # Imperfect bot classification is observable and may change later.
                bots = accepted & (rng.random(n) < .0006)
                for j in np.flatnonzero((counts>0) | bots):
                    for click_number in range(int(counts[j]) + int(bots[j])):
                        human = click_number < counts[j]
                        event_id = f"e{next_event}"; next_event += 1
                        delay = int(rng.choice(response_days,p=response_probs))
                        occurred = date + delay
                        event_observed = occurred + int(rng.choice([0,1,2],p=[.85,.10,.05]))
                        message_id = f"m:{ids[j]}:{date}"
                        initial_qualification = "human" if human or rng.random() < .55 else "bot"
                        event_rows.append((f"r{next_record}",event_id,f"u{ids[j]}",cohort,source,message_id,int(arms[j]),date,occurred,event_observed,initial_qualification,0,"filter/1")); next_record += 1
                        if not human and initial_qualification == "human":
                            event_rows.append((f"r{next_record}",event_id,f"u{ids[j]}",cohort,source,message_id,int(arms[j]),date,occurred,event_observed+7,"bot",1,"filter/2")); next_record += 1
                        if not human:
                            continue
                        final_events.append((event_id,f"u{ids[j]}",cohort,source,int(arms[j]),date,occurred))
                        amount = 0.
                        if rng.random() < params["positive_value_probability"]:
                            median = s["positive_value_median"] * content_value[content] * value_shock[date]
                            amount = float(rng.lognormal(np.log(median),params["log_value_sigma"]))
                        # Economic settlement and report arrival are separate clocks.
                        settlement = int(rng.choice(posting_days,p=posting_probs))
                        economic = occurred + settlement
                        observed = economic + int(rng.choice([0,1,3],p=[.8,.15,.05]))
                        base = entry(event_id,cohort,source,int(arms[j]),amount,economic,observed,"revenue")
                        if rng.random() < .06 and amount:
                            revision_age = max(settlement + 1, int(rng.integers(8,26)))
                            fraction = rng.uniform(.25,1.)
                            entry(event_id,cohort,source,int(arms[j]),-amount*fraction,occurred+revision_age,occurred+revision_age+3,
                                  "revenue_revision",f"revision-{event_id}",base)
                        if scenario == "calendar_attribution_shock" and 100 <= date < 125 and amount:
                            entry(event_id,cohort,source,int(arms[j]),-.45*amount,occurred+25,occurred+28,
                                  "revenue_revision",f"shock-{event_id}",base)
    leads = pd.DataFrame(lead_rows,columns=LEAD_COLUMNS)
    exposure = pd.DataFrame(exposure_rows,columns=EXPOSURE_COLUMNS)
    events = pd.DataFrame(event_rows,columns=EVENT_COLUMNS)
    ledger = pd.DataFrame(ledger_rows,columns=LEDGER_COLUMNS)
    exits = pd.DataFrame(exit_rows,columns=EXIT_COLUMNS)
    truth = GeneratorTruth(scenario,seed,ledger.copy(deep=True),
                           pd.DataFrame(final_events,columns=["event_id","lead_id","cohort_id","source_id","arm","send_at","occurred_at"]),
                           pd.DataFrame(latent_rows,columns=["lead_id","cohort_id","persistent_quality","dormant_returner","cohort_quality"]),
                           pd.DataFrame({"date":np.arange(calendar_length),"click_multiplier":click_shock,"value_multiplier":value_shock}),dict(params))
    # Missing telemetry retains an explicit flag; no truth-only fields enter World.
    # A few missing event reports demonstrate unavailable != zero, without changing economics.
    if len(events):
        hidden_ids = rng.choice(events.event_id.unique(),size=max(1,int(.002*events.event_id.nunique())),replace=False)
        events.loc[events.event_id.isin(hidden_ids),"observed_at"] = np.nan
    world = World(scenario,seed,leads,exposure,events,ledger,exits)
    return world, truth, policy
