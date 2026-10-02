"""Rolling-origin synthetic outcome replay, with truth restricted to evaluation.

Allocation and acquisition decisions are deliberately simplified: observed
cohort unit margins are reused as hypothetical batch outcomes under unchanged
policy and source exchangeability. They are not causal next-batch validation.
"""
from collections import defaultdict
import math
import numpy as np
import pandas as pd
from .accounting import summarize_path
from .baselines import MODELS, forecast_cohort, forecast_records, clear_fit_cache


def allocate_budget(unit_margins, costs, capacities, budget):
    """Exact continuous-capacity, linear fixed-budget allocation diagnostic.

    Fractional lead units are permitted in this synthetic diagnostic; no actual
    purchase or integer-knapsack optimization is claimed. Capacity and budget
    are identical for predicted, always-continue, and clairvoyant allocations.
    """
    if budget < 0:
        raise ValueError("Budget must be nonnegative")
    sources = sorted(unit_margins)
    quantities = {source: 0.0 for source in sources}
    remaining = float(budget)
    ranking = sorted(sources, key=lambda s: (-unit_margins[s] / costs[s] if costs[s] > 0 else -math.inf, s))
    for source in ranking:
        if costs[source] <= 0 or capacities[source] < 0:
            raise ValueError("Costs must be positive and capacities nonnegative")
        if unit_margins[source] <= 0:
            continue
        quantity = min(float(capacities[source]), remaining / costs[source])
        quantities[source] = max(0.0, quantity)
        remaining -= quantity * costs[source]
    return quantities


def _rank_accuracy(predicted, actual):
    a = np.asarray(predicted, dtype=float)
    b = np.asarray(actual, dtype=float)
    pairs = [(i, j) for i in range(len(a)) for j in range(i + 1, len(a)) if b[i] != b[j]]
    if not pairs:
        return None
    return float(np.mean([1.0 if np.sign(a[i] - a[j]) == np.sign(b[i] - b[j]) else 0.5 if a[i] == a[j] else 0.0 for i, j in pairs]))


def _records(frame, keys, values):
    if frame.empty:
        return []
    grouped = frame.groupby(keys, dropna=False).agg(**values).reset_index()
    # JSON null rather than nonstandard NaN values.
    return grouped.astype(object).where(pd.notnull(grouped), None).to_dict("records")


def evaluate_world(world, truth, policy, parameters, models=MODELS, origin_ages=None, horizons=None):
    """Evaluate later-half weekly cohorts at actual calendar rolling cutoffs.

    Return a dict with forecasts, margin_scores, decisions and allocations
    DataFrames, plus a JSON-ready summary. All truth access stays in this module;
    each baseline receives only world.as_of(cutoff), policy, ID and frozen config.
    Strictly prospective horizons are scored; at-origin nowcasts are emitted and
    labelled separately, while horizons preceding origin are omitted.
    """
    origin_ages = list(parameters.get("rolling_origin_ages", [3, 7, 14, 30, 60]) if origin_ages is None else origin_ages)
    horizons = list(parameters.get("horizons", [30, 90, 180, policy.terminal_age]) if horizons is None else horizons)
    if any(age < 0 or age > policy.terminal_age for age in origin_ages):
        raise ValueError("Rolling origins must lie within the declared policy")
    metadata = world.leads.groupby(["cohort_id", "source_id", "acquired_at"], as_index=False).agg(
        n_leads=("lead_id", "size"), acquisition_cost=("acquisition_cost", "sum"))
    targets = []
    for _, rows in metadata.groupby("source_id"):
        rows = rows.sort_values(["acquired_at", "cohort_id"])
        targets.extend(rows.iloc[len(rows) // 2:].to_dict("records"))
    by_cutoff = defaultdict(list)
    for target in targets:
        for origin_age in origin_ages:
            cutoff = int(target["acquired_at"] + origin_age)
            by_cutoff[cutoff].append((target, int(origin_age)))
    # Small terminal arrays are the only evaluator truth objects retained here.
    truth_paths = {target["cohort_id"]: truth.cohort_path(world.leads, target["cohort_id"], policy) for target in targets}
    truth_summaries = {cohort: {row["horizon"]: row for row in summarize_path(path, horizons)} for cohort, path in truth_paths.items()}
    normal_batch = float(parameters.get("decision_normal_batch", parameters.get("leads_per_cohort", 500)))
    small_fraction = float(parameters.get("decision_small_batch_fraction", 0.25))
    small_threshold = float(parameters.get("decision_small_margin_threshold", 0.01))
    if normal_batch <= 0 or not 0 < small_fraction < 1 or small_threshold < 0:
        raise ValueError("Invalid declared stop/small/normal-batch replay settings")
    budget = float(parameters.get("allocation_budget", 100.0))
    default_capacity = float(parameters.get("allocation_source_capacity", normal_batch))
    capacities_config = parameters.get("allocation_source_capacities", {})
    forecast_rows, score_rows, decisions, allocation_inputs = [], [], [], []
    assumptions = {}
    clear_fit_cache()
    try:
        for cutoff, origins in sorted(by_cutoff.items()):
            snapshot = world.as_of(cutoff)
            for target, age in origins:
                cohort_id = target["cohort_id"]
                true_path = truth_paths[cohort_id]
                n = target["n_leads"]
                true_unit = float(true_path[-1] / n)
                for model in models:
                    forecast = forecast_cohort(snapshot, policy, cohort_id, model, parameters)
                    model_name = forecast["model"]
                    assumptions.setdefault(model_name, forecast["assumptions"])
                    emitted = forecast_records(forecast, world.scenario, world.seed, horizons)
                    forecast_rows.extend(emitted)
                    for row in emitted:
                        if row["evaluation_role"] != "prospective":
                            continue
                        actual = truth_summaries[cohort_id][row["horizon"]]
                        score_rows.append({
                            "model": model_name, "scenario": world.scenario, "seed": world.seed,
                            "cohort_id": cohort_id, "source_id": target["source_id"], "cutoff": cutoff,
                            "origin_age": age, "horizon": row["horizon"],
                            "predicted_margin": row["margin"], "true_margin": actual["margin"],
                            "absolute_error": abs(row["margin"] - actual["margin"]),
                            "absolute_error_per_original_lead": abs(row["margin"] - actual["margin"]) / n,
                            "predicted_payback_state": row["payback_state"], "true_payback_state": actual["payback_state"],
                            "payback_state_correct": row["payback_state"] == actual["payback_state"],
                            "predicted_first_payback_age": row["first_payback_age"], "true_first_payback_age": actual["first_payback_age"],
                            "positive_at_horizon_correct": row["positive_at_horizon"] == actual["positive_at_horizon"],
                            "predicted_positive_at_horizon": row["positive_at_horizon"], "true_positive_at_horizon": actual["positive_at_horizon"],
                            "predicted_remains_positive_after_payback": row["remains_positive_after_payback"],
                            "true_remains_positive_after_payback": actual["remains_positive_after_payback"],
                        })
                    predicted_unit = float(forecast["margin_path"][-1] / n)
                    action = "stop" if predicted_unit <= 0 else ("small_batch" if predicted_unit <= small_threshold else "normal_batch")
                    quantity = 0.0 if action == "stop" else normal_batch * small_fraction if action == "small_batch" else normal_batch
                    unit_cost = float(forecast["known_acquisition_cost_per_lead"])
                    chosen_value = quantity * true_unit
                    oracle_value = normal_batch * max(true_unit, 0.0)
                    avoided_quantity = normal_batch - quantity
                    decisions.append({
                        "model": model_name, "scenario": world.scenario, "seed": world.seed,
                        "cohort_id": cohort_id, "source_id": target["source_id"], "cutoff": cutoff,
                        "origin_age": age, "action": action, "normal_batch": normal_batch,
                        "quantity": quantity, "predicted_full_policy_margin_per_lead": predicted_unit,
                        "heldout_realized_full_policy_margin_per_lead": true_unit,
                        "chosen_realized_margin": chosen_value, "always_continue_realized_margin": normal_batch * true_unit,
                        "oracle_realized_margin": oracle_value, "decision_regret": max(0.0, oracle_value - chosen_value),
                        "gross_acquisition_spend_avoided": avoided_quantity * unit_cost,
                        "losses_avoided_vs_always_continue": avoided_quantity * max(-true_unit, 0.0),
                        "forgone_profitable_margin": avoided_quantity * max(true_unit, 0.0),
                        "profitable_opportunity_falsely_stopped": bool(action == "stop" and true_unit > 0),
                        "profitable_opportunity_curtailed": bool(quantity < normal_batch and true_unit > 0),
                        "incremental_continuation_value_existing_cohort": forecast["incremental_continuation_value"],
                        "stop_sending_value_existing_cohort": forecast["stop_sending_value"],
                        "pending_prior_send_revenue_preserved": forecast["pending_prior_send_revenue"],
                        "replay_type": "simplified_observed_cohort_unit_outcome_no_causal_claim",
                    })
                    allocation_inputs.append({
                        "model": model_name, "acquired_at": int(target["acquired_at"]), "origin_age": age,
                        "source_id": target["source_id"], "predicted_unit": predicted_unit,
                        "true_unit": true_unit, "cost": unit_cost, "cohort_id": cohort_id,
                    })
            # Release a full snapshot before constructing the next cutoff.
            clear_fit_cache()
            del snapshot
    finally:
        clear_fit_cache()
    allocations = []
    frame = pd.DataFrame(allocation_inputs)
    if len(frame):
        for (model, acquired, age), rows in frame.groupby(["model", "acquired_at", "origin_age"]):
            predicted = dict(zip(rows.source_id, rows.predicted_unit))
            actual = dict(zip(rows.source_id, rows.true_unit))
            costs = dict(zip(rows.source_id, rows.cost))
            capacities = {s: float(capacities_config.get(s, default_capacity)) for s in predicted}
            chosen = allocate_budget(predicted, costs, capacities, budget)
            oracle = allocate_budget(actual, costs, capacities, budget)
            # Always-continue comparison spends evenly among sources until a
            # capacity binds; any remainder is filled by cheapest source.
            equal = {s: min(capacities[s], budget / len(costs) / costs[s]) for s in costs}
            remainder = max(0.0, budget - sum(equal[s] * costs[s] for s in costs))
            for s in sorted(costs, key=lambda s: (costs[s], s)):
                extra = min(capacities[s] - equal[s], remainder / costs[s])
                equal[s] += extra
                remainder -= extra * costs[s]
            chosen_value = sum(chosen[s] * actual[s] for s in actual)
            oracle_value = sum(oracle[s] * actual[s] for s in actual)
            allocations.append({
                "model": model, "scenario": world.scenario, "seed": world.seed,
                "acquired_at": int(acquired), "origin_age": int(age), "cutoff": int(acquired + age),
                "budget": budget, "capacity_per_source": capacities,
                "chosen_quantities": chosen, "oracle_quantities": oracle, "always_continue_quantities": equal,
                "chosen_realized_margin": chosen_value, "oracle_realized_margin": oracle_value,
                "always_continue_realized_margin": sum(equal[s] * actual[s] for s in actual),
                "allocation_regret": max(0.0, oracle_value - chosen_value),
                "acquisition_spend": sum(chosen[s] * costs[s] for s in actual),
                "source_pairwise_ranking_accuracy": _rank_accuracy(list(predicted.values()), [actual[s] for s in predicted]),
                "replay_type": "continuous_capacity_linear_budget_observed_cohort_unit_outcome_no_causal_claim",
            })
    forecasts = pd.DataFrame(forecast_rows)
    scores = pd.DataFrame(score_rows)
    decision_frame = pd.DataFrame(decisions)
    allocation_frame = pd.DataFrame(allocations)
    summary = {
        "synthetic_only": True, "scenario": world.scenario, "seed": int(world.seed),
        "target_selection": "Later half of chronological weekly cohorts within each source",
        "uncertainty_status": "Point forecasts only: bootstrap distributions, CRPS, interval coverage, Brier scores, calibration and payback-CDF calibration unavailable",
        "prospective_scoring": "Only horizons strictly greater than origin age; earlier horizons omitted, equal horizons emitted as at_origin_nowcast and excluded from scores",
        "replay_warning": "Simplified decision replay reuses each held-out observed cohort unit outcome for hypothetical next batches under fixed policy and source exchangeability; no causal or independent prospective next-batch validation",
        "allocation_warning": "Continuous (fractional-lead) linear allocation with fixed acquisition budget and source capacities; future send policy identical; no saturation, interference or sending-capacity optimizer",
        "stop_sending_warning": "Diagnostic compares revenue generated by future sends against avoidable future send costs. Acquisition cost is sunk; pending old-message responses, settlements and revisions are preserved",
        "decision_settings": {"normal_batch": normal_batch, "small_batch_fraction": small_fraction, "small_positive_margin_threshold": small_threshold},
        "allocation_settings": {"budget": budget, "default_source_capacity": default_capacity, "source_capacity_overrides": capacities_config},
        "repeat_warning": "Each origin age is an alternative replay of the same cohorts. Totals across ages must not be interpreted as cumulative real spending or independent trials",
        "models": list(models), "n_target_cohorts": len(targets), "n_origins": len(targets) * len(origin_ages),
        "n_forecasts": len(forecasts), "n_prospective_scores": len(scores),
        "margin_mae_by_age_horizon_source": _records(scores, ["model", "origin_age", "horizon", "source_id"], {
            "n": ("absolute_error", "size"), "margin_mae": ("absolute_error", "mean"),
            "margin_mae_per_original_lead": ("absolute_error_per_original_lead", "mean")}),
        "margin_mae_by_age_horizon": _records(scores, ["model", "origin_age", "horizon"], {
            "n": ("absolute_error", "size"), "margin_mae": ("absolute_error", "mean")}),
        "payback_classification": _records(scores.loc[scores.horizon.eq(policy.terminal_age)] if len(scores) else scores,
            ["model", "origin_age", "true_payback_state", "predicted_payback_state"], {"n": ("cohort_id", "size")}),
        "decision_value_by_age_source": _records(decision_frame, ["model", "origin_age", "source_id"], {
            "n": ("cohort_id", "size"), "mean_regret": ("decision_regret", "mean"),
            "mean_chosen_margin": ("chosen_realized_margin", "mean"),
            "mean_always_continue_margin": ("always_continue_realized_margin", "mean"),
            "mean_oracle_margin": ("oracle_realized_margin", "mean"),
            "gross_spend_avoided": ("gross_acquisition_spend_avoided", "sum"),
            "forgone_profitable_margin": ("forgone_profitable_margin", "sum"),
            "losses_avoided": ("losses_avoided_vs_always_continue", "sum"),
            "false_stops": ("profitable_opportunity_falsely_stopped", "sum")}),
        "allocation_value_by_age": _records(allocation_frame, ["model", "origin_age"], {
            "n": ("acquired_at", "size"), "mean_chosen_margin": ("chosen_realized_margin", "mean"),
            "mean_always_continue_margin": ("always_continue_realized_margin", "mean"),
            "mean_oracle_margin": ("oracle_realized_margin", "mean"), "mean_regret": ("allocation_regret", "mean"),
            "mean_pairwise_ranking_accuracy": ("source_pairwise_ranking_accuracy", "mean")}),
        "assumptions": assumptions,
    }
    return {"forecasts": forecasts, "margin_scores": scores, "decisions": decision_frame, "allocations": allocation_frame, "summary": summary}
