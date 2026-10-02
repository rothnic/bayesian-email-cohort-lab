"""Two deterministic as-of baselines; no GeneratorTruth is accepted here.

A forecasts pooled signed revenue per original lead and send age. B forecasts
source-pooled operational eligibility, clicks per accepted exposure and signed
revenue per human click. Both keep immutable observed accounting, predict
outstanding old-message outcomes, and use exactly the same future send policy.
These point paths are neither draws nor calibrated profitability probabilities.
"""
from collections import OrderedDict
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .accounting import SCHEMA_VERSION, Snapshot, Policy, cohort_metadata, ledger_path, summarize_path

MODELS = ("fixed_aggregate", "observable_survival")
MODEL_VERSION = "fictional-point-baselines/1.0.0"
# Small cache retains the snapshot itself, avoiding id reuse and unlimited worlds.
_FIT_CACHE = OrderedDict()
_AGE_BINS = ((0, 0), (1, 1), (2, 3), (4, 7), (8, 14), (15, 30),
             (31, 60), (61, 90), (91, 180), (181, 365))

@dataclass
class _Fit:
    snapshot: Snapshot
    metadata: pd.DataFrame
    exposure: pd.DataFrame
    human_events: pd.DataFrame
    operational: dict
    response_kernel: list
    marks: dict
    aggregate_curve: np.ndarray
    click_curves: dict
    assumptions: list
    pooled_mark: list


def clear_fit_cache():
    """Release transient fit/snapshot references, e.g. between independent worlds."""
    _FIT_CACHE.clear()


def _human_events(snapshot):
    frame = snapshot.engagement_events
    if frame.empty:
        return frame.copy()
    qualification = frame.qualification.astype(str).str.lower()
    human = qualification.isin(("human", "qualified_human", "true", "1"))
    return frame.loc[human].copy()


def _probability_kernel(frame, columns):
    if frame.empty:
        return [(0, 0, 1.0)]
    frequencies = frame.groupby(list(columns)).size()
    total = float(frequencies.sum())
    return [(int(a), int(b), float(count / total)) for (a, b), count in frequencies.items()]


def _mark_kernel(events, revenue, cutoff, maturity, source=None):
    """Expected signed ledger components/click, including zero-value clicks.

    Mature visible clicks are the denominator; revisions remain signed. Missing
    event reports cannot be inferred from the observed tables and are disclosed.
    """
    selected = events.loc[events.occurred_at.le(cutoff - maturity)]
    if source is not None:
        selected = selected.loc[selected.source_id.eq(source)]
    if selected.empty:
        return []
    entries = revenue.merge(selected[["event_id", "occurred_at"]], on="event_id", how="inner")
    if entries.empty:
        return [(0, 0, 0.0)]
    entries["economic_delay"] = (entries.economic_at - entries.occurred_at).clip(lower=0).astype(int)
    entries["report_delay"] = (entries.observed_at - entries.economic_at).clip(lower=0).astype(int)
    grouped = entries.groupby(["economic_delay", "report_delay"]).amount.sum() / len(selected)
    return [(int(e), int(r), float(v)) for (e, r), v in grouped.items()]


def _tail_curve(values, support, terminal, timescale):
    curve = np.zeros(terminal + 1, dtype=float)
    if support < 0:
        return curve
    curve[:support + 1] = values[:support + 1]
    # A preregistered seven-day anchor reduces sensitivity to a single zero day.
    start = max(0, support - 6)
    ages = np.arange(start, support + 1)
    anchor = float(np.mean(curve[ages] * np.exp((ages - support) / timescale)))
    if support < terminal:
        future = np.arange(support + 1, terminal + 1)
        curve[future] = anchor * np.exp(-(future - support) / timescale)
    return curve


def _fit(snapshot, policy, parameters):
    if not isinstance(snapshot, Snapshot):
        raise TypeError("Model inputs must be an as-of Snapshot, not a World or GeneratorTruth")
    fixed_tail = float(parameters.get("fixed_baseline_tail_timescale", 75.0))
    observable_tail = float(parameters.get("observable_baseline_tail_timescale", 75.0))
    if fixed_tail <= 0 or observable_tail <= 0:
        raise ValueError("Frozen tail timescales must be positive")
    key = (id(snapshot), policy, fixed_tail, observable_tail)
    if key in _FIT_CACHE:
        _FIT_CACHE.move_to_end(key)
        return _FIT_CACHE[key]
    meta = cohort_metadata(snapshot)
    meta_index = meta.set_index("cohort_id")
    exposure = snapshot.exposure_daily.copy()
    if len(exposure):
        exposure = exposure.groupby(["cohort_id", "source_id", "date", "age"], as_index=False)[
            ["attempted", "accepted", "eligible", "exits"]].sum(min_count=1)
    events = _human_events(snapshot)
    revenue = snapshot.ledger.loc[snapshot.ledger.kind.isin(("revenue", "revenue_revision"))].copy()
    assumptions = [
        "Fictional deterministic point extrapolation; bootstrap uncertainty, probability calibration and interval scores are deferred",
        "Only the supplied as-of Snapshot is used; future calendar shocks, final revisions and latent lead states are unavailable",
        "Inactivity never defines operational exit; missing exposure telemetry is excluded from rate denominators, not converted to observed zero",
        "Both models send daily through the same policy end using the same source-pooled operational exit hazard; acquisition cost is sunk for stop-sending diagnostics",
        "Visible signed ledger entries are retained; unavailable future acquisition-invoice revisions are not forecast",
        "Empirical delay kernels assume transportability and independent delays; permanently unavailable events and value-dependent reporting are not identifiable",
        "Rates pool visible historical and target-cohort information; no fitted cohort effect is transported to a new cohort",
    ]
    # Fit event response/reporting delays using sends with the full response window.
    mature_response = events.loc[events.send_at.le(snapshot.cutoff - policy.max_response_delay)].copy()
    if mature_response.empty:
        mature_response = events.copy()
        assumptions.append("No fully response-mature clicks: the observed, right-truncated delay histogram is used as a disclosed fallback")
    if len(mature_response):
        mature_response["response_delay"] = (mature_response.occurred_at - mature_response.send_at).clip(lower=0, upper=policy.max_response_delay).astype(int)
        mature_response["event_report_delay"] = (mature_response.observed_at - mature_response.occurred_at).clip(lower=0).astype(int)
    response = _probability_kernel(mature_response, ("response_delay", "event_report_delay"))
    pooled_mark = _mark_kernel(events, revenue, snapshot.cutoff, policy.max_accounting_delay)
    if not pooled_mark:
        # No generator value prior is substituted when available history is thin.
        pooled_mark = _mark_kernel(events, revenue, snapshot.cutoff, 0) or [(0, 0, 0.0)]
        assumptions.append("No accounting-mature click history: signed revenue/click uses incomplete visible history, or zero if no history exists")
    marks = {}
    operational = {}
    click_curves = {}
    sources = sorted(meta.source_id.unique())
    observed_cells = exposure.dropna(subset=["accepted", "attempted", "eligible", "exits"]) if len(exposure) else exposure
    global_hazard = (float(observed_cells.exits.sum()) / float((observed_cells.eligible + observed_cells.exits).sum())) if len(observed_cells) and (observed_cells.eligible + observed_cells.exits).sum() > 0 else 0.0
    global_accept = float(observed_cells.accepted.sum() / observed_cells.attempted.sum()) if len(observed_cells) and observed_cells.attempted.sum() > 0 else 0.0
    event_counts = events.groupby(["cohort_id", "send_at"]).size().rename("clicks") if len(events) else pd.Series(dtype=float, name="clicks")
    cells = exposure.copy()
    if len(cells):
        if len(event_counts):
            cells = cells.merge(event_counts, left_on=["cohort_id", "date"], right_index=True, how="left")
            cells["clicks"] = cells.clicks.fillna(0.0)
        else:
            cells["clicks"] = 0.0
        elapsed = snapshot.cutoff - cells.date.to_numpy(dtype=int)
        fraction = np.zeros(len(cells))
        for delay, report, weight in response:
            fraction += weight * (elapsed >= delay + report)
        cells["estimated_eventual_clicks"] = cells.clicks / np.maximum(fraction, 0.05)
    for source in sources:
        cell = observed_cells.loc[observed_cells.source_id.eq(source)] if len(observed_cells) else observed_cells
        denominator = float((cell.eligible + cell.exits).sum()) if len(cell) else 0.0
        hazard = float(cell.exits.sum() / denominator) if denominator else global_hazard
        acceptance = float(cell.accepted.sum() / cell.attempted.sum()) if len(cell) and cell.attempted.sum() > 0 else global_accept
        operational[source] = (float(np.clip(hazard, 0, 1)), float(np.clip(acceptance, 0, 1)))
        marks[source] = _mark_kernel(events, revenue, snapshot.cutoff, policy.max_accounting_delay, source) or pooled_mark
        rows = cells.loc[cells.source_id.eq(source) & cells.accepted.notna() & cells.accepted.gt(0)] if len(cells) else cells
        support = int(rows.age.max()) if len(rows) else -1
        rates = np.zeros(policy.last_send_age + 1)
        last_rate = 0.0
        for lo, hi in _AGE_BINS:
            if lo > policy.last_send_age:
                break
            selected = rows.loc[rows.age.between(lo, min(hi, policy.last_send_age))] if len(rows) else rows
            denom = float(selected.accepted.sum()) if len(selected) else 0.0
            if denom:
                last_rate = max(0.0, float(selected.estimated_eventual_clicks.sum() / denom))
            rates[lo:min(hi, policy.last_send_age) + 1] = last_rate
        click_curves[source] = _tail_curve(rates, min(support, policy.last_send_age), policy.last_send_age, observable_tail)
    # Aggregate A: age-specific signed revenue/original lead, attributed to send.
    # All visible classifications can provide send linkage; qualification itself
    # is not allowed to delete a signed ledger entry from aggregate accounting.
    all_events = snapshot.engagement_events[["event_id", "send_at"]]
    attributed = revenue.merge(all_events, on="event_id", how="inner")
    unavailable_links = len(revenue) - len(attributed)
    if unavailable_links:
        assumptions.append(f"{unavailable_links} visible revenue entries lack available send linkage and are excluded from rate fitting; target accounting still retains them")
    total_mark = sum(value for _, _, value in pooled_mark)
    mass = sum(abs(value) for _, _, value in pooled_mark)
    raw_curve = np.zeros(policy.last_send_age + 1)
    support = min(policy.last_send_age, int((snapshot.cutoff - meta.acquired_at).max())) if len(meta) else -1
    if len(attributed):
        attributed["acquired_at"] = attributed.cohort_id.map(meta_index.acquired_at)
        attributed["send_age"] = (attributed.send_at - attributed.acquired_at).astype(int)
        # Completion proxy uses absolute signed-revision mass, so late negative
        # revisions cannot produce an invalid observation probability.
        elapsed = snapshot.cutoff - attributed.send_at.to_numpy(dtype=int)
        completeness = np.zeros(len(attributed))
        if mass > 0:
            for delay, _, probability in response:
                for economic, report, value in pooled_mark:
                    completeness += probability * abs(value) / mass * (elapsed >= delay + economic + report)
        else:
            completeness[:] = 1.0
        attributed["corrected_amount"] = attributed.amount / np.maximum(completeness, 0.05)
        sums = attributed.groupby("send_age").corrected_amount.sum()
    else:
        sums = pd.Series(dtype=float)
    for age in range(support + 1):
        denominator = float(meta.loc[meta.acquired_at.le(snapshot.cutoff - age), "n_leads"].sum())
        raw_curve[age] = float(sums.get(age, 0.0)) / denominator if denominator else 0.0
    aggregate_curve = _tail_curve(raw_curve, support, policy.last_send_age, fixed_tail)
    assumptions.extend([
        f"A freezes one pooled daily signed revenue/original-lead curve; beyond age {support}, a seven-day anchor uses fixed exponential timescale {fixed_tail:g} days",
        f"B uses source-pooled piecewise age click/exposure rates and fixed exponential timescale {observable_tail:g} days beyond available age support",
        "A's available-history revenue completion adjustment uses absolute revision mass; B's click completion adjustment uses the empirical response/reporting CDF, floored at 0.05",
        "Revenue marks preserve zeros and signed revisions; source-specific mature marks fall back to pooled mature history rather than generator truth",
    ])
    fit = _Fit(snapshot, meta, exposure, events, operational, response, marks, aggregate_curve, click_curves, assumptions, pooled_mark)
    _FIT_CACHE[key] = fit
    while len(_FIT_CACHE) > 2:
        _FIT_CACHE.popitem(last=False)
    return fit


def _future_exposure(fit, policy, cohort_id, acquired, n, source):
    hazard, acceptance = fit.operational[source]
    origin = fit.snapshot.cutoff - acquired
    target = fit.exposure.loc[fit.exposure.cohort_id.eq(cohort_id) & fit.exposure.eligible.notna()] if len(fit.exposure) else fit.exposure
    exits = fit.snapshot.operational_exits
    exited = exits.loc[exits.cohort_id.eq(cohort_id), "lead_id"].nunique() if len(exits) else 0
    eligible = float(max(0, n - exited))
    if len(target):
        last = target.sort_values("date").iloc[-1]
        eligible = float(last.eligible) * (1.0 - hazard) ** max(0, fit.snapshot.cutoff - int(last.date))
    attempted = np.zeros(policy.last_send_age + 1)
    accepted = np.zeros_like(attempted)
    # Past measured exposure is immutable; missing past cells receive the same
    # observable hazard estimate, disclosed as imputation rather than a zero.
    for age in range(min(origin, policy.last_send_age) + 1):
        past = target.loc[target.age.eq(age)] if len(target) else target
        if len(past):
            attempted[age], accepted[age] = float(past.attempted.iloc[0]), float(past.accepted.iloc[0])
        else:
            attempted[age] = n * (1.0 - hazard) ** (age + 1)
            accepted[age] = attempted[age] * acceptance
    for age in range(max(0, origin + 1), policy.last_send_age + 1):
        eligible *= 1.0 - hazard
        attempted[age] = eligible
        accepted[age] = eligible * acceptance
    return attempted, accepted


def forecast_cohort(snapshot, policy, cohort_id, model, parameters):
    """Return a full point path and a sunk-cost-free stop-sending diagnostic.

    Callable inputs are Snapshot, Policy, cohort ID, one of MODELS, and a frozen
    configuration dictionary. No truth/world or latent parameters are accepted.
    Configuration is used only for the declared tail timescales.
    """
    aliases = {"A": "fixed_aggregate", "B": "observable_survival"}
    model = aliases.get(model, model)
    if model not in MODELS:
        raise ValueError(f"Unknown baseline {model!r}; expected {MODELS}")
    fit = _fit(snapshot, policy, parameters)
    target = fit.metadata.loc[fit.metadata.cohort_id.eq(cohort_id)]
    if len(target) != 1:
        raise ValueError(f"Expected exactly one acquired cohort {cohort_id!r}")
    meta = target.iloc[0]
    acquired, n, source = int(meta.acquired_at), int(meta.n_leads), str(meta.source_id)
    origin = snapshot.cutoff - acquired
    if origin < 0 or origin > policy.terminal_age:
        raise ValueError("Origin must lie within the declared finite policy")
    target_ledger = snapshot.ledger.loc[snapshot.ledger.cohort_id.eq(cohort_id)]
    observed_path = ledger_path(target_ledger, acquired, policy.terminal_age)
    increments = np.diff(np.r_[0.0, observed_path])
    attempted, accepted = _future_exposure(fit, policy, cohort_id, acquired, n, source)
    future_cost = float(attempted[origin + 1:].sum() * policy.send_cost)
    if origin < policy.last_send_age:
        increments[origin + 1:policy.last_send_age + 1] -= attempted[origin + 1:] * policy.send_cost
    marks = fit.pooled_mark if model == "fixed_aggregate" else fit.marks[source]
    total_mark = sum(value for _, _, value in marks)
    if model == "fixed_aggregate":
        # Signed pooled revenue rates are converted into a click-equivalent
        # intensity only to transport the common response/accounting delay kernel.
        intensity = n * fit.aggregate_curve / total_mark if abs(total_mark) > 1e-12 else np.zeros(policy.last_send_age + 1)
    else:
        intensity = accepted * fit.click_curves[source]
    future_send_revenue = 0.0
    pending_prior_send_revenue = 0.0
    # Each historical send contributes only outcomes whose event telemetry AND
    # ledger contribution would not yet be visible. Already observed entries are
    # retained exactly above. A/B keep old-message responses after sending stops.
    for response, event_report, probability in fit.response_kernel:
        for economic, report, mark in marks:
            ages = np.arange(policy.last_send_age + 1)
            occurrence = ages + response
            economics = occurrence + economic
            unseen_event = occurrence + event_report > origin
            unseen_ledger = economics + report > origin
            valid = unseen_event & unseen_ledger & (economics <= policy.terminal_age)
            contribution = intensity * probability * mark
            np.add.at(increments, economics[valid], contribution[valid])
            future = ages > origin
            # Full economic contributions from genuinely future sends have no
            # sunk-cost/pending components in the incremental sending decision.
            future_send_revenue += float(contribution[future & (economics <= policy.terminal_age)].sum())
            pending_prior_send_revenue += float(contribution[valid & ~future].sum())
    # Observed human clicks condition the outstanding revenue components rather
    # than erasing them because no more messages are sent. Signed revisions may
    # remain outstanding even when an original positive payment is visible.
    known = fit.human_events.loc[fit.human_events.cohort_id.eq(cohort_id)]
    if len(known):
        occurrence = known.occurred_at.to_numpy(dtype=int) - acquired
        for economic, report, mark in marks:
            economics = occurrence + economic
            valid = (economics + report > origin) & (economics <= policy.terminal_age)
            np.add.at(increments, economics[valid], mark)
            pending_prior_send_revenue += float(valid.sum() * mark)
    margin_path = increments.cumsum()
    incremental = future_send_revenue - future_cost
    assumptions = list(fit.assumptions)
    if fit.exposure.loc[fit.exposure.cohort_id.eq(cohort_id), "accepted"].isna().any() if len(fit.exposure) else False:
        assumptions.append("Missing target past exposure is imputed from observable operational hazards and acceptance; original missing flags remain in the Snapshot")
    return {
        "schema_version": SCHEMA_VERSION, "model": model, "model_version": MODEL_VERSION,
        "cutoff": int(snapshot.cutoff), "cohort_id": str(cohort_id), "policy_id": policy.policy_id,
        "source_id": source, "origin_age": int(origin), "n_leads": n,
        "known_acquisition_cost_per_lead": float(meta.acquisition_cost / n),
        "margin_path": margin_path, "incremental_continuation_value": float(incremental),
        "stop_sending_value": float(-incremental), "future_send_cost": future_cost,
        "future_send_revenue": float(future_send_revenue),
        "pending_prior_send_revenue": float(pending_prior_send_revenue),
        "assumptions": assumptions,
    }


def forecast_records(forecast, scenario, seed, horizons=(30, 90, 180, 425), include_retrospective=False):
    """Serialize contract point fields; omit horizons before origin by default.

    Past/current horizons are distinguishable via evaluation_role when explicitly
    included. evaluate_world scores only strictly prospective horizons.
    """
    path = np.asarray(forecast["margin_path"], dtype=float)
    horizons = [int(h) for h in horizons if 0 <= int(h) < len(path)
                and (include_retrospective or int(h) >= forecast["origin_age"])]
    rows = []
    for summary in summarize_path(path, horizons):
        horizon = summary["horizon"]
        role = "prospective" if horizon > forecast["origin_age"] else ("at_origin_nowcast" if horizon == forecast["origin_age"] else "retrospective")
        rows.append({
            "schema_version": forecast["schema_version"], "model": forecast["model"],
            "model_version": forecast["model_version"], "scenario": str(scenario), "seed": int(seed),
            "cutoff": forecast["cutoff"], "cohort_id": forecast["cohort_id"],
            "policy_id": forecast["policy_id"], "horizon": horizon,
            "draw_id": 0, "forecast_type": "point", **summary,
            "probability_profitable": None, "evaluation_role": role,
        })
    return rows
