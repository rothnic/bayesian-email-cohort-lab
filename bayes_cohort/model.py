"""Snapshot-only finite-grid hierarchical posterior and predictive accounting.

The finite grid is the model, not an MCMC approximation to an unreported
continuous posterior. Conditional cohort amplitudes and nuisance parameters
have conjugate posteriors. Future counts, costs, marks and revisions are sampled.
"""
from collections import OrderedDict
from itertools import product
import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp
from scipy.stats import invgamma
from cohort_lab.accounting import Snapshot, Policy, SCHEMA_VERSION, cohort_metadata, ledger_path
from .config import resolve_config

MODEL_VERSION = "finite-grid-gamma-poisson/1.0.0"
MODEL_NAME = "bayesian_hierarchical"
_FIT_CACHE = OrderedDict()


def clear_fit_cache():
    _FIT_CACHE.clear()


def _pairs(first, second):
    values = np.array([(a, b) for a, _ in first for b, _ in second], dtype=int)
    probs = np.array([pa * pb for _, pa in first for _, pb in second], dtype=float)
    return values, probs


def _human_events(snapshot):
    frame = snapshot.engagement_events
    return frame.loc[frame.qualification.astype(str).str.lower().isin(("human", "qualified_human", "true", "1"))].copy()


def _fit(snapshot, policy, config):
    if not isinstance(snapshot, Snapshot):
        raise TypeError("Model inputs must be an as-of Snapshot, never a World or GeneratorTruth")
    if not isinstance(policy, Policy):
        raise TypeError("policy must be the public Policy object")
    if max(d for d, _ in config.response_delay) > policy.max_response_delay:
        raise ValueError("Declared response kernel exceeds policy.max_response_delay; supply a compatible frozen configuration")
    key = (id(snapshot), policy, config)
    if key in _FIT_CACHE:
        _FIT_CACHE.move_to_end(key)
        return _FIT_CACHE[key]
    meta = cohort_metadata(snapshot)
    human = _human_events(snapshot)
    exposure = snapshot.exposure_daily.copy()
    if len(exposure):
        exposure = exposure.groupby(["cohort_id", "source_id", "date", "age"], as_index=False)[["attempted", "accepted", "eligible", "exits"]].agg(lambda column: column.sum(min_count=len(column)))
    usable = exposure.dropna(subset=["attempted", "accepted", "eligible", "exits"])
    if len(usable):
        usable = usable.loc[usable.age.between(0, policy.last_send_age)].copy()
        counts = human.groupby(["cohort_id", "send_at"]).size().rename("count")
        usable = usable.merge(counts, left_on=["cohort_id", "date"], right_index=True, how="left")
        usable["count"] = usable["count"].fillna(0).astype(int)
    else:
        usable = usable.assign(count=pd.Series(dtype=int))
    grid = np.array(list(product(config.source_means, config.fast_weights, config.fast_timescales, config.tail_timescales)), dtype=float)
    ages = np.arange(policy.last_send_age + 1)
    curves = grid[:, 1, None] * np.exp(-ages / grid[:, 2, None]) + (1 - grid[:, 1, None]) * np.exp(-ages / grid[:, 3, None])
    # A proper discrete source-mean prior and equal mass over the shape grid.
    log_prior = -.5 * ((np.log(grid[:, 0]) - config.log_source_mean_prior_location) / config.log_source_mean_prior_scale) ** 2
    log_prior -= logsumexp(log_prior)
    response_pairs, response_probs = _pairs(config.response_delay, config.event_report_delay)
    ledger_pairs, ledger_probs = _pairs(config.settlement_delay, config.ledger_report_delay)
    source_fits = {}
    for source in sorted(meta.source_id.unique()):
        rows = usable.loc[usable.source_id.eq(source)]
        log_weight = log_prior.copy()
        cohort_fits = {}
        for cohort in meta.loc[meta.source_id.eq(source), "cohort_id"]:
            cells = rows.loc[rows.cohort_id.eq(cohort)]
            observed_count = int(cells["count"].sum())
            if len(cells):
                age = cells.age.to_numpy(dtype=int)
                elapsed = snapshot.cutoff - cells.date.to_numpy(dtype=int)
                q = (elapsed[:, None] >= response_pairs.sum(axis=1)[None, :]) @ response_probs
                impossible = (cells["count"].to_numpy() > 0) & ((cells.accepted.to_numpy() <= 0) | (q <= 0))
                if impossible.any():
                    raise ValueError("Positive response counts require positive accepted exposure and observation probability")
                information = curves[:, age] @ (cells.accepted.to_numpy(dtype=float) * q)
                shape_log_likelihood = np.log(curves[:, age]) @ cells["count"].to_numpy(dtype=float)
            else:
                information = np.zeros(len(grid))
                shape_log_likelihood = np.zeros(len(grid))
            rate = config.cohort_shape / grid[:, 0]
            log_weight += (gammaln(config.cohort_shape + observed_count) - gammaln(config.cohort_shape)
                           + config.cohort_shape * np.log(rate)
                           - (config.cohort_shape + observed_count) * np.log(rate + information)
                           + shape_log_likelihood)
            cohort_fits[str(cohort)] = (observed_count, information)
        weights = np.exp(log_weight - logsumexp(log_weight))
        hazard_success = float(rows.exits.sum())
        hazard_trials = float((rows.eligible + rows.exits).sum())
        acceptance_success = float(rows.accepted.sum())
        acceptance_trials = float(rows.attempted.sum())
        if hazard_success < 0 or hazard_success > hazard_trials or acceptance_success < 0 or acceptance_success > acceptance_trials:
            raise ValueError("Operational exposure counts violate binomial denominators")
        # Source-specific marks: complete-window observed human events with an
        # actual base ledger row. Missing mature receipts are unavailable, not 0.
        mature = human.loc[human.source_id.eq(source) & human.occurred_at.le(snapshot.cutoff - config.mark_maturity_days)]
        bases = snapshot.ledger.loc[snapshot.ledger.kind.eq("revenue")]
        mature_bases = mature[["event_id"]].merge(bases[["event_id", "amount"]], on="event_id", how="inner")
        if mature_bases.event_id.duplicated().any():
            raise ValueError("Expected one base revenue entry per event")
        if (mature_bases.amount < 0).any():
            raise ValueError("Base revenue marks must be nonnegative; corrections belong in signed revision entries")
        values = mature_bases.amount.to_numpy(dtype=float)
        positive = values[values > 0]
        logs = np.log(positive)
        count = len(logs)
        kappa = config.log_mark_prior_precision + count
        sample_mean = float(logs.mean()) if count else config.log_mark_prior_mean
        mu = (config.log_mark_prior_precision * config.log_mark_prior_mean + count * sample_mean) / kappa
        alpha = config.log_mark_prior_shape + .5 * count
        beta = (config.log_mark_prior_scale + .5 * float(((logs - sample_mean) ** 2).sum())
                + .5 * config.log_mark_prior_precision * count / kappa * (sample_mean - config.log_mark_prior_mean) ** 2)
        revision_ids = set(snapshot.ledger.loc[snapshot.ledger.kind.eq("revenue_revision"), "event_id"])
        positive_ids = mature_bases.loc[mature_bases.amount.gt(0), "event_id"]
        revised = sum(event in revision_ids for event in positive_ids)
        source_fits[str(source)] = {
            "weights": weights, "cohorts": cohort_fits,
            "hazard": (config.hazard_alpha + hazard_success, config.hazard_beta + hazard_trials - hazard_success),
            "acceptance": (config.acceptance_alpha + acceptance_success, config.acceptance_beta + acceptance_trials - acceptance_success),
            "positive": (config.positive_alpha + len(positive), config.positive_beta + len(values) - len(positive)),
            "log_mark": (mu, kappa, alpha, beta),
            "revision": (config.revision_alpha + revised, config.revision_beta + len(positive) - revised),
            "mature_marks": len(values), "mature_positive_marks": len(positive),
            "missing_mature_base_rows": int(len(mature) - len(values)),
            "observed_click_count": int(rows["count"].sum()), "observed_exposure_cells": int(len(rows)),
        }
    fit = {"snapshot": snapshot, "meta": meta, "exposure": exposure, "human": human,
           "grid": grid, "curves": curves, "sources": source_fits,
           "response_pairs": response_pairs, "response_probs": response_probs,
           "ledger_pairs": ledger_pairs, "ledger_probs": ledger_probs}
    _FIT_CACHE[key] = fit
    while len(_FIT_CACHE) > 4:
        _FIT_CACHE.popitem(last=False)
    return fit


def _marks(rng, draw_ids, parameters):
    positive = rng.random(len(draw_ids)) < parameters["positive_probability"][draw_ids]
    value = np.zeros(len(draw_ids))
    ids = draw_ids[positive]
    value[positive] = rng.lognormal(parameters["log_mark_mean"][ids], parameters["log_mark_sigma"][ids])
    return value


def _add(increments, draws, economics, amounts, valid):
    valid = valid & (economics >= 0) & (economics < increments.shape[1])
    np.add.at(increments, (draws[valid], economics[valid]), amounts[valid])


def _fresh_revisions(rng, draw_ids, occurrence, settlement, value, parameters, config):
    revised = (value > 0) & (rng.random(len(value)) < parameters["revision_probability"][draw_ids])
    delay = np.maximum(settlement + 1, rng.integers(config.revision_delay_min, config.revision_delay_max + 1, len(value)))
    economic = occurrence + delay
    amount = -value * rng.uniform(config.revision_fraction_min, config.revision_fraction_max, len(value)) * revised
    return economic, amount


def payback_summary(margin_paths):
    """Unconditional finite-policy states, retaining the non-payback atom."""
    paths = np.asarray(margin_paths, dtype=float)
    if paths.ndim != 2 or not len(paths) or not np.isfinite(paths).all():
        raise ValueError("margin_paths must be a finite, nonempty draws-by-age array")
    crossed = paths >= 0
    has_crossed = crossed.any(axis=1)
    first = np.where(has_crossed, crossed.argmax(axis=1), -1)
    return {"first_payback_ages": first, "probabilities": {
        "by_180": float(np.mean((first >= 0) & (first <= 180))),
        "later_than_180": float(np.mean(first > 180)),
        "never_under_policy": float(np.mean(first < 0)),
    }}


def forecast_cohort(snapshot, policy, cohort_id, config=None, draws=300, seed=0):
    """Draw posterior predictive complete economic margin paths, as of cutoff.

    Only Snapshot and public Policy are accepted. Paths include immutable visible
    ledger components plus latent unreported economics and future outcomes.
    They therefore can differ from the visible ledger balance before the cutoff.
    """
    config = resolve_config(config)
    if not isinstance(draws, (int, np.integer)) or draws < 1:
        raise ValueError("draws must be a positive integer")
    fit = _fit(snapshot, policy, config)
    target = fit["meta"].loc[fit["meta"].cohort_id.eq(cohort_id)]
    if len(target) != 1:
        raise ValueError(f"Expected exactly one acquired cohort {cohort_id!r}")
    meta = target.iloc[0]
    acquired, n, source = int(meta.acquired_at), int(meta.n_leads), str(meta.source_id)
    origin = int(snapshot.cutoff - acquired)
    if origin < 0 or origin > policy.terminal_age:
        raise ValueError("Origin must lie within the declared finite policy")
    rng = np.random.default_rng(seed)
    source_fit = fit["sources"][source]
    grid_ids = rng.choice(len(fit["grid"]), size=draws, p=source_fit["weights"])
    grid = fit["grid"][grid_ids]
    count, information = source_fit["cohorts"][str(cohort_id)]
    amplitude = rng.gamma(config.cohort_shape + count, 1 / (config.cohort_shape / grid[:, 0] + information[grid_ids]))
    mark_mu, kappa, alpha, beta = source_fit["log_mark"]
    # An unbounded NIG log-value mixture has infinite exponential moments.
    # The frozen variance bound is part of the prior, so the conjugate posterior
    # is exactly the updated inverse-Gamma restricted to the same support.
    variance_bound_cdf = float(invgamma.cdf(config.max_log_mark_variance, a=alpha, scale=beta))
    if not variance_bound_cdf > 0:
        raise ValueError("Truncated log-mark posterior normalization underflowed; inspect the mark observations and frozen variance bound")
    uniforms = np.maximum(rng.random(draws), np.nextafter(0., 1.)) * variance_bound_cdf
    variance = invgamma.ppf(uniforms, a=alpha, scale=beta)
    if not np.isfinite(variance).all() or (variance > config.max_log_mark_variance).any():
        raise FloatingPointError("Invalid truncated inverse-Gamma variance draws")
    parameters = {
        "source_mean": grid[:, 0], "fast_weight": grid[:, 1], "fast_timescale": grid[:, 2], "tail_timescale": grid[:, 3],
        "cohort_amplitude": amplitude, "exit_hazard": rng.beta(*source_fit["hazard"], size=draws),
        "acceptance_probability": rng.beta(*source_fit["acceptance"], size=draws),
        "positive_probability": rng.beta(*source_fit["positive"], size=draws),
        "log_mark_mean": rng.normal(mark_mu, np.sqrt(variance / kappa)), "log_mark_sigma": np.sqrt(variance),
        "revision_probability": rng.beta(*source_fit["revision"], size=draws),
    }
    target_ledger = snapshot.ledger.loc[snapshot.ledger.cohort_id.eq(cohort_id)]
    observed_path = ledger_path(target_ledger, acquired, policy.terminal_age)
    increments = np.tile(np.diff(np.r_[0., observed_path]), (draws, 1))
    send_ages = np.arange(policy.last_send_age + 1)
    attempted = np.zeros((draws, len(send_ages)), dtype=int)
    accepted = np.zeros_like(attempted)
    target_exposure = fit["exposure"].loc[fit["exposure"].cohort_id.eq(cohort_id)]
    measured = target_exposure.dropna(subset=["attempted", "accepted", "eligible", "exits"])
    measured_index = measured.set_index("age")
    missing_past = 0
    eligible = np.full(draws, n, dtype=int)
    for age in range(min(origin, policy.last_send_age) + 1):
        if age in measured_index.index:
            cell = measured_index.loc[age]
            attempted[:, age] = int(cell.attempted)
            accepted[:, age] = int(cell.accepted)
            eligible[:] = int(cell.eligible)
        else:
            missing_past += 1
            eligible -= rng.binomial(eligible, parameters["exit_hazard"])
            attempted[:, age] = eligible
            accepted[:, age] = rng.binomial(eligible, parameters["acceptance_probability"])
    for age in range(max(0, origin + 1), policy.last_send_age + 1):
        eligible -= rng.binomial(eligible, parameters["exit_hazard"])
        attempted[:, age] = eligible
        accepted[:, age] = rng.binomial(eligible, parameters["acceptance_probability"])
        increments[:, age] -= attempted[:, age] * policy.send_cost
    future_mask = send_ages > origin
    future_send_cost = attempted[:, future_mask].sum(axis=1) * policy.send_cost
    future_send_revenue = np.zeros(draws)
    pending_revenue = np.zeros(draws)
    elapsed = origin - send_ages
    response_pairs, response_probs = fit["response_pairs"], fit["response_probs"]
    q = (elapsed[:, None] >= response_pairs.sum(axis=1)[None, :]) @ response_probs
    missing_fraction = np.where(future_mask, 1., np.maximum(0., 1 - q))
    intensity = accepted * amplitude[:, None] * fit["curves"][grid_ids] * missing_fraction
    predictive_counts = rng.poisson(intensity)
    flat = np.repeat(np.arange(predictive_counts.size), predictive_counts.ravel())
    event_draws = flat // len(send_ages)
    event_send = flat % len(send_ages)
    if len(flat):
        response_ids = rng.choice(len(response_pairs), len(flat), p=response_probs)
        for age in np.unique(event_send[event_send <= origin]):
            mask = event_send == age
            probs = response_probs * (response_pairs.sum(axis=1) > origin - age)
            probs /= probs.sum()
            response_ids[mask] = rng.choice(len(probs), mask.sum(), p=probs)
        response = response_pairs[response_ids, 0]
        ledger_ids = rng.choice(len(fit["ledger_pairs"]), len(flat), p=fit["ledger_probs"])
        settlement, report = fit["ledger_pairs"][ledger_ids].T
        occurrence = event_send + response
        economic = occurrence + settlement
        value = _marks(rng, event_draws, parameters)
        unseen_base = economic + report > origin
        _add(increments, event_draws, economic, value, unseen_base)
        rev_economic, rev_amount = _fresh_revisions(rng, event_draws, occurrence, settlement, value, parameters, config)
        unseen_revision = unseen_base & (rev_economic + config.revision_report_delay > origin)
        _add(increments, event_draws, rev_economic, rev_amount, unseen_revision)
        valid_base = unseen_base & (economic <= policy.terminal_age)
        valid_rev = unseen_revision & (rev_economic <= policy.terminal_age)
        for mask, result in ((event_send > origin, future_send_revenue), (event_send <= origin, pending_revenue)):
            np.add.at(result, event_draws[mask & valid_base], value[mask & valid_base])
            np.add.at(result, event_draws[mask & valid_rev], rev_amount[mask & valid_rev])
    # Known clicks with a genuinely outstanding base receipt, conditioned on
    # missing base reporting. This is not a right-truncated empirical delay fit.
    known_events = fit["human"].loc[fit["human"].cohort_id.eq(cohort_id)]
    bases = target_ledger.loc[target_ledger.kind.eq("revenue")]
    base_ids = set(bases.event_id)
    unmatched_complete_clicks = 0
    pending_known_clicks = 0
    for event in known_events.loc[~known_events.event_id.isin(base_ids)].itertuples():
        occurrence = int(event.occurred_at) - acquired
        pairs, probs = fit["ledger_pairs"], fit["ledger_probs"]
        probs = probs * (pairs.sum(axis=1) > origin - occurrence)
        if probs.sum() <= 0:
            unmatched_complete_clicks += 1
            continue
        pending_known_clicks += 1
        ids = rng.choice(len(pairs), draws, p=probs / probs.sum())
        settlement, report = pairs[ids].T
        draw_ids = np.arange(draws)
        value = _marks(rng, draw_ids, parameters)
        economic = occurrence + settlement
        _add(increments, draw_ids, economic, value, np.ones(draws, dtype=bool))
        rev_economic, rev_amount = _fresh_revisions(rng, draw_ids, np.full(draws, occurrence), settlement, value, parameters, config)
        valid_rev = rev_economic + config.revision_report_delay > origin
        _add(increments, draw_ids, rev_economic, rev_amount, valid_rev)
        pending_revenue += value * (economic <= policy.terminal_age) + rev_amount * valid_rev * (rev_economic <= policy.terminal_age)
    # Every observed base is retained exactly. Future signed revisions are
    # conditional on no revision report yet, including ledger-only events.
    occurrence_map = known_events.set_index("event_id").occurred_at.to_dict()
    revised_ids = set(target_ledger.loc[target_ledger.kind.eq("revenue_revision"), "event_id"])
    ledger_only_bases = 0
    revision_delays = np.arange(config.revision_delay_min, config.revision_delay_max + 1)
    for base in bases.loc[bases.amount.gt(0) & ~bases.event_id.isin(revised_ids)].itertuples():
        economic_base = int(base.economic_at) - acquired
        if base.event_id in occurrence_map:
            occurrence = np.full(draws, int(occurrence_map[base.event_id]) - acquired, dtype=int)
        else:
            ledger_only_bases += 1
            # Missing click telemetry cannot identify occurrence; the declared
            # settlement clock supplies an explicitly approximate back-imputation.
            possible = [(d, p) for d, p in config.settlement_delay if d <= economic_base]
            days, probs = map(np.array, zip(*possible))
            occurrence = economic_base - rng.choice(days.astype(int), draws, p=probs / probs.sum())
        settlement = economic_base - occurrence
        delays = np.maximum(settlement[:, None] + 1, revision_delays[None, :])
        future_revision = occurrence[:, None] + delays + config.revision_report_delay > origin
        remaining = future_revision.mean(axis=1)
        p = parameters["revision_probability"]
        conditional_p = p * remaining / (1 - p * (1 - remaining))
        revise = rng.random(draws) < conditional_p
        active = np.flatnonzero(revise)
        if len(active):
            # Uniform choices over the still-unreported revision-clock states.
            choices = (rng.random(len(active)) * future_revision[active].sum(axis=1)).astype(int)
            ids = (future_revision[active].cumsum(axis=1) > choices[:, None]).argmax(axis=1)
            rev_economic = occurrence[active] + delays[active, ids]
            amount = -float(base.amount) * rng.uniform(config.revision_fraction_min, config.revision_fraction_max, len(active))
            _add(increments, active, rev_economic, amount, np.ones(len(active), dtype=bool))
            np.add.at(pending_revenue, active[rev_economic <= policy.terminal_age], amount[rev_economic <= policy.terminal_age])
    # The initial visible acquisition charge is immutable. The final invoice is
    # a separate uncertain appended adjustment, rather than assumed settled.
    missing_invoices = 0
    if config.forecast_invoice_revisions:
        acquisition = target_ledger.loc[target_ledger.kind.eq("acquisition")]
        invoice_arms = set(target_ledger.loc[target_ledger.kind.eq("acquisition_revision"), "arm"])
        for arm, amount in acquisition.groupby("arm").amount.sum().items():
            if arm in invoice_arms:
                continue
            missing_invoices += 1
            if origin < config.invoice_report_age and config.invoice_economic_age <= policy.terminal_age:
                increments[:, config.invoice_economic_age] += float(amount) * rng.normal(config.invoice_fraction_mean, config.invoice_fraction_sd, draws)
    margin_paths = increments.cumsum(axis=1)
    payback = payback_summary(margin_paths)
    assumptions = [
        "Synthetic demonstration, not validated production decision probabilities; posterior predictive intervals require held-out coverage checks",
        "Only the supplied as-of Snapshot and frozen model configuration are used; no generator truth or generator parameter file is read",
        "Exact posterior over a declared finite grid of source means and early-plus-tail curves, with conditional Gamma cohort amplitudes and genuine within-source partial pooling",
        "Aggregate qualified-response counts per accepted exposure are Poisson; they are not unique-person retention or latent survival",
        "Response and reporting/settlement clocks are declared known synthetic instrument assumptions, not estimated from right-truncated visible delays",
        "Operational exits and acceptance use source-pooled Beta-binomial posterior uncertainty and future binomial outcomes; inactivity never defines an exit",
        "Zero-plus-lognormal monetary marks and revision frequency use source-specific mature-only conjugate posteriors; the inverse-Gamma log-variance prior is bounded above to ensure finite expected positive value",
        "Thin mature mark histories use declared priors rather than incomplete current receipts; the frozen log-variance bound must be tested in prior sensitivity analysis",
        "Observed signed ledger entries are immutable components; latent unreported receipts, revisions and invoices can nowcast economic dates before the cutoff",
        "One ordinary negative revision per positive event is modeled, with declared synthetic delay and fraction distribution; systematic calendar attribution revisions are not modeled",
        "Future calendar/content shocks, persistent lead heterogeneity, correlated repeat clicks, permanently unavailable click telemetry, and temporary human/bot misclassification are omitted model risks",
        "Unreported events with already-visible base receipts are not added twice; their outstanding revisions are predicted from the observed base instead",
        "Missing exposure cells are excluded from likelihood denominators and imputed stochastically for prediction; missing mature revenue receipts are unavailable, not observed zeros",
        "Unmatched ledger-only event occurrences use a disclosed approximate settlement-clock back-imputation for revision prediction",
        "An absent invoice after its declared reporting support is flagged as incomplete data; it is not evidence of economic settlement",
    ]
    horizons = sorted(set(h for h in (30, 90, 180, policy.terminal_age) if h <= policy.terminal_age))
    diagnostics = {
        "inference": "exact finite-grid posterior, conditional conjugate draws, posterior predictive outcome simulation",
        "grid_points": len(fit["grid"]), "source_posterior_effective_grid_points": float(1 / np.sum(source_fit["weights"] ** 2)),
        "source_observed_click_count": source_fit["observed_click_count"], "target_observed_click_count": count,
        "source_observed_exposure_cells": source_fit["observed_exposure_cells"],
        "mature_mark_count": source_fit["mature_marks"], "mature_positive_mark_count": source_fit["mature_positive_marks"],
        "missing_mature_base_rows": source_fit["missing_mature_base_rows"], "mark_prior_only": source_fit["mature_marks"] == 0,
        "missing_target_past_exposure_cells": missing_past, "unmatched_complete_window_clicks": unmatched_complete_clicks,
        "pending_observed_clicks_with_unreported_base": pending_known_clicks,
        "ledger_only_positive_bases_for_revision_imputation": ledger_only_bases, "unobserved_acquisition_invoice_arms": missing_invoices,
        "invoice_reporting_support_passed_without_record": bool(missing_invoices and origin >= config.invoice_report_age),
        "payback_state_probabilities": payback["probabilities"],
        "cohort_amplitude_quantiles": [float(x) for x in np.quantile(amplitude, [.05, .5, .95])],
        "parameter_quantiles": {name: [float(x) for x in np.quantile(value, [.05, .5, .95])] for name, value in parameters.items()},
        "max_log_mark_variance": float(config.max_log_mark_variance),
        "untruncated_log_variance_posterior_mass_below_bound": variance_bound_cdf,
        "grid_boundary_mass": {
            name: {"lower": float(source_fit["weights"][fit["grid"][:, column] == min(values)].sum()),
                   "upper": float(source_fit["weights"][fit["grid"][:, column] == max(values)].sum())}
            for name, column, values in (("source_mean", 0, config.source_means), ("fast_weight", 1, config.fast_weights),
                                        ("fast_timescale", 2, config.fast_timescales), ("tail_timescale", 3, config.tail_timescales))
        },
        "tail_timescale_probabilities": {str(float(t)): float(source_fit["weights"][fit["grid"][:, 3] == t].sum()) for t in config.tail_timescales},
        "horizons": {str(h): {"mean": float(margin_paths[:, h].mean()), "q05": float(np.quantile(margin_paths[:, h], .05)),
                                "q50": float(np.quantile(margin_paths[:, h], .5)), "q95": float(np.quantile(margin_paths[:, h], .95)),
                                "probability_profitable": float(np.mean(margin_paths[:, h] > 0))} for h in horizons},
        "assumptions": assumptions,
    }
    return {
        "schema_version": SCHEMA_VERSION, "model": MODEL_NAME, "model_version": MODEL_VERSION,
        "cohort_id": str(cohort_id), "source_id": source, "cutoff": int(snapshot.cutoff), "origin_age": origin,
        "policy_id": policy.policy_id, "n_leads": n, "draws": int(draws), "seed": int(seed),
        "margin_paths": margin_paths, "margin_path": margin_paths.mean(axis=0), "observed_margin_path": observed_path,
        "first_payback_ages": payback["first_payback_ages"], "payback_state_probabilities": payback["probabilities"],
        "parameter_draws": parameters, "future_send_cost": future_send_cost, "future_send_revenue": future_send_revenue,
        "pending_prior_send_revenue": pending_revenue, "incremental_continuation_value": future_send_revenue - future_send_cost,
        "stop_sending_value": future_send_cost - future_send_revenue,
        "known_acquisition_cost_per_lead": float(meta.acquisition_cost / n), "diagnostics": diagnostics, "assumptions": assumptions,
    }


def forecast_records(forecast, scenario, seed, horizons=(30, 90, 180, 425), include_retrospective=False):
    """Draw-level records with unconditional probabilities and evaluation roles."""
    paths = forecast["margin_paths"]
    first = forecast["first_payback_ages"]
    rows = []
    for horizon in horizons:
        horizon = int(horizon)
        if not 0 <= horizon < paths.shape[1] or (not include_retrospective and horizon < forecast["origin_age"]):
            continue
        role = "prospective" if horizon > forecast["origin_age"] else ("at_origin_nowcast" if horizon == forecast["origin_age"] else "retrospective")
        probability = float(np.mean(paths[:, horizon] > 0))
        for draw_id in range(len(paths)):
            age = int(first[draw_id])
            state = "never_under_policy" if age < 0 else ("by_180" if age <= 180 else "later_than_180")
            rows.append({"schema_version": forecast["schema_version"], "model": forecast["model"], "model_version": forecast["model_version"],
                         "scenario": str(scenario), "seed": int(seed), "cutoff": forecast["cutoff"], "cohort_id": forecast["cohort_id"],
                         "source_id": forecast["source_id"], "policy_id": forecast["policy_id"], "origin_age": forecast["origin_age"],
                         "horizon": horizon, "draw_id": draw_id, "forecast_type": "posterior_predictive", "margin": float(paths[draw_id, horizon]),
                         "first_payback_age": None if age < 0 else age, "payback_state": state,
                         "positive_at_horizon": bool(paths[draw_id, horizon] > 0),
                         "remains_positive_after_payback": bool(age >= 0 and age <= horizon and (paths[draw_id, age:horizon + 1] >= 0).all()),
                         "probability_profitable": probability, "evaluation_role": role})
    return rows
