"""Bounded Bayesian Gaussian approximation for audited email cohort economics.

No simulator/generator imports or truth columns are used.  See MODEL.md for the
measurement approximation, discrete proper hyperpriors and scenario semantics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import product
import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.special import logsumexp


MODES = ("target_only", "source_pooling", "global_pooling",
         "static_hierarchy", "dynamic_hierarchy")
SCENARIOS = ("hold_current", "continue_recent_trend", "fixed_price")
REQUIRED = ("cohort_id", "source_id", "birth_day", "calendar_day", "age",
            "acquired_leads", "accepted_exposure", "raw_events", "audit_n", "audit_human",
            "net_human_revenue", "net_human_count", "response_report_fraction",
            "payout_complete_fraction", "observed_at")


@dataclass(frozen=True)
class ModelConfig:
    # All variance mixtures have finite bounded support, with positive masses.
    source_sds: tuple = (.12, .40, .85)
    cohort_sds: tuple = (.10, .30, .65)
    effect_weights: tuple = (.25, .50, .25)
    calendar_sds: tuple = (.004, .015, .035)  # per sqrt(calendar day)
    calendar_weights: tuple = (.35, .45, .20)
    count_dispersions: tuple = (8., 40.)
    dispersion_weights: tuple = (.5, .5)
    response_log_location: float = float(np.log(.004))
    payout_log_location: float = float(np.log(.20))
    retention_log_hazard: float = float(np.log(.004))
    unique_response_log_location: float = float(np.log(.004))
    location_sd: float = 1.0
    feature_sd: float = .5
    net_mark_cv: float = 1.5
    audit_sensitivity: float = .95
    audit_specificity: float = .98
    calendar_bin_days: int = 7
    audit_beta_alpha: float = 1.
    audit_beta_beta: float = 1.
    send_last_age: int = 365
    trend_window_days: int = 28
    measurement_variance_floor: float = .02

    def validate(self):
        for xs, ws in ((self.source_sds, self.effect_weights),
                       (self.cohort_sds, self.effect_weights),
                       (self.calendar_sds, self.calendar_weights),
                       (self.count_dispersions, self.dispersion_weights)):
            if len(xs) != len(ws) or not len(xs):
                raise ValueError("Hyperprior supports and probabilities must align")
            if not np.all(np.isfinite(xs)) or min(xs) <= 0:
                raise ValueError("Hyperprior supports must be positive and finite")
            if not np.all(np.isfinite(ws)) or min(ws) <= 0 or not np.isclose(sum(ws), 1):
                raise ValueError("Hyperprior probabilities must be positive and sum to one")
        if max(self.source_sds + self.cohort_sds) > 1.5 or max(self.calendar_sds) > .1:
            raise ValueError("Declared finite-moment scale bounds exceeded")
        if min(self.count_dispersions) < .25 or max(self.count_dispersions) > 100:
            raise ValueError("Count dispersion support must lie within [.25,100]")
        if self.audit_sensitivity + self.audit_specificity <= 1:
            raise ValueError("The known audit instrument must distinguish event types")
        if not (0 < self.audit_sensitivity <= 1 and 0 < self.audit_specificity <= 1):
            raise ValueError("Invalid audit instrument")
        if self.location_sd <= 0 or self.feature_sd <= 0 or self.net_mark_cv <= 0:
            raise ValueError("Location/feature scales and mark CV must be positive")
        if self.calendar_bin_days < 1 or self.trend_window_days < 1:
            raise ValueError("Calendar periods must be positive")
        if min(self.audit_beta_alpha, self.audit_beta_beta) <= 0:
            raise ValueError("Audit Beta prior parameters must be positive")


def _frame(data) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy(deep=True)
    if hasattr(data, "frame") and callable(data.frame):
        return data.frame().copy(deep=True)
    if isinstance(data, Mapping):
        data = data.get("rows", data.get("data", data))
    if hasattr(data, "rows"):
        data = data.rows
    return pd.DataFrame(data).copy(deep=True)


def _snapshot(data, cutoff: int) -> pd.DataFrame:
    d = _frame(data)
    missing = set(REQUIRED) - set(d.columns)
    if missing:
        optional_acquired = missing == {"acquired_leads"}
        if not optional_acquired:
            raise ValueError(f"Missing observation fields: {sorted(missing)}")
        d["acquired_leads"] = d.groupby("cohort_id").accepted_exposure.transform("max")
    # Only the declared observation contract is retained, never arbitrary columns.
    keep = list(REQUIRED) + [c for c in ("source_feature", "audit_observed_at",
        "unique_audit_observed_at", "unique_audit_n", "unique_audit_human",
        "raw_unique_responders", "payout_observed_at", "net_human_revenue_sum_squares") if c in d]
    d = d[keep].copy(deep=True)
    d = d[(d.calendar_day <= cutoff) & (d.observed_at <= cutoff) &
          (d.birth_day <= cutoff)].copy()
    for clock, fields in (("audit_observed_at", ("audit_n", "audit_human")),
                          ("unique_audit_observed_at", ("unique_audit_n", "unique_audit_human")),
                          ("payout_observed_at", ("net_human_revenue", "net_human_count"))):
        if clock in d:
            pending = d[clock].isna() | (d[clock] > cutoff)
            d.loc[pending, list(fields)] = np.nan
            if clock == "payout_observed_at":
                d.loc[pending, "payout_complete_fraction"] = 0.
    if "source_feature" not in d:
        d["source_feature"] = 0.
    if d.duplicated(["cohort_id", "calendar_day"]).any():
        raise ValueError("Snapshot must have one final revision per cohort/calendar day")
    if not d.empty:
        if not np.allclose(d.age, d.calendar_day - d.birth_day):
            raise ValueError("age must equal calendar_day minus birth_day")
        if (d[["accepted_exposure", "raw_events", "response_report_fraction"]] < 0).any().any():
            raise ValueError("Counts, exposure and report fractions cannot be negative")
        if (d.accepted_exposure > d.acquired_leads).any():
            raise ValueError("Contactable exposure cannot exceed original acquired leads")
        if (d.response_report_fraction > 1).any() or (d.payout_complete_fraction > 1).any():
            raise ValueError("Completeness fractions cannot exceed one")
        if not d.payout_complete_fraction.isin((0., 1.)).all():
            raise ValueError("Payout completeness must be binary: unavailable or fully settled")
        if (d.net_human_revenue.dropna() < 0).any():
            raise ValueError("Individual negative net-revenue rows are outside this unsigned net-mark model")
        a = d.dropna(subset=["audit_n", "audit_human"])
        if ((a.audit_n < a.audit_human) | (a.audit_human < 0) |
            (a.audit_n > a.raw_events)).any():
            raise ValueError("Audit subsamples must be nested within raw events")
        if (d.source_feature.abs() > 1).any():
            raise ValueError("source_feature must be prespecified and bounded in [-1,1]")
        if d.groupby("cohort_id").source_id.nunique().max() > 1:
            raise ValueError("A cohort cannot change source")
        if d.groupby("source_id").source_feature.nunique().max() > 1:
            raise ValueError("The prespecified feature must be constant within source")
        if d.groupby("cohort_id").birth_day.nunique().max() > 1:
            raise ValueError("A cohort cannot change acquisition date")
    return d.sort_values(["source_id", "cohort_id", "calendar_day"]).reset_index(drop=True)


def _audit_fraction(n, h, config):
    """Single nested audit measurement; raw and audit are not replicate counts."""
    aa = config.audit_beta_alpha + h
    bb = config.audit_beta_beta + n - h
    pa = aa / (aa + bb)
    va = aa * bb / ((aa + bb) ** 2 * (aa + bb + 1))
    det = config.audit_sensitivity + config.audit_specificity - 1
    ph = np.clip((pa - (1 - config.audit_specificity)) / det, .005, .995)
    return ph, va / det ** 2


def _measurements(d, config, outcome):
    records = []
    groups = (d.groupby("cohort_id", sort=True) if outcome == "retention" else
              d.assign(_bin=d.calendar_day // config.calendar_bin_days).groupby(
                  ["cohort_id", "_bin"], sort=True))
    for key, g in groups:
        cohort = key if isinstance(key, str) else key[0]
        if outcome == "retention":
            # Group consecutive exposure snapshots into approximately weekly
            # risk intervals. Exposure is the operationally observed active
            # mailing list; generator truth is never read by this likelihood.
            g = g.sort_values("age")
            active = g.accepted_exposure.to_numpy(float)
            ages = g.age.to_numpy(int)
            days = g.calendar_day.to_numpy(float)
            leads = float(g.acquired_leads.iloc[0])
            for end in range(1, len(g), config.calendar_bin_days):
                start = end - 1
                stop = min(end + config.calendar_bin_days - 1, len(g) - 1)
                elapsed = int(ages[stop] - ages[start])
                if elapsed <= 0:
                    continue
                exits = max(active[start] - active[stop], 0.)
                person_days = float(active[start:stop].sum())
                if person_days <= 0:
                    continue
                # Piecewise-exponential survival likelihood: deaths are
                # Poisson with the observed person-days at risk as offset.
                value = np.log((exits + .5) / person_days)
                variance = 1. / (exits + .5)
                records.append(dict(cohort_id=str(cohort), source_id=str(g.source_id.iloc[0]),
                    source_feature=float(g.source_feature.iloc[0]),
                    calendar_day=float(np.average(days[start+1:stop+1])),
                    value=float(value), variance=max(float(variance), config.measurement_variance_floor),
                    dispersion_multiplier=0.))
            continue
        elif outcome == "unique":
            if "raw_unique_responders" not in g:
                continue
            g = g[g.accepted_exposure > 0]
            if g.empty:
                continue
            exposure = g.accepted_exposure.to_numpy(float) * g.response_report_fraction.to_numpy()
            total_exposure = exposure.sum()
            raw = float(g.raw_unique_responders.sum())
            audited = g.dropna(subset=["unique_audit_n", "unique_audit_human"])
            n, h = float(audited.unique_audit_n.sum()), float(audited.unique_audit_human.sum())
            ph, vp = _audit_fraction(n, h, config)
            human = raw * ph
            value = np.log((human + .5) / total_exposure)
            variance = (human + .5 + raw ** 2 * vp) / (human + .5) ** 2
            dispersion_multiplier = float(np.square(exposure).sum() / total_exposure ** 2)
            wt = exposure
        elif outcome == "response":
            g = g[(g.accepted_exposure > 0) & (g.response_report_fraction > 0)]
            if g.empty:
                continue
            exposure = g.accepted_exposure.to_numpy() * g.response_report_fraction.to_numpy()
            total_exposure = exposure.sum()
            raw = float(g.raw_events.sum())
            # Audits are pooled only inside this cohort/calendar bin. Missing
            # audit arrivals contribute no labels, rather than negative labels.
            audited = g.dropna(subset=["audit_n", "audit_human"])
            n, h = float(audited.audit_n.sum()), float(audited.audit_human.sum())
            ph, vp = _audit_fraction(n, h, config)
            human = raw * ph
            # Delta-method uncertainty includes audit fraction, event count and
            # independent daily NB cell frailty; the latter varies on the grid.
            # The same half-count smoothing must enter both log mean AND its
            # delta variance. Gamma(shape=human+.5) moment approximation gives
            # Var(log rate) approximately 1/(human+.5), hence variance2 at zero;
            # an unadjusted numerator would make zero bins spuriously precise.
            variance = (human + .5 + raw ** 2 * vp) / (human + .5) ** 2
            value = np.log((human + .5) / total_exposure)
            dispersion_multiplier = float(np.square(exposure).sum() / total_exposure ** 2)
            wt = exposure
        else:
            g = g[(g.payout_complete_fraction > 0) & g.net_human_revenue.notna() &
                  g.net_human_count.notna() & (g.net_human_count > 0)]
            if g.empty:
                continue
            n = float(g.net_human_count.sum())
            revenue = float(g.net_human_revenue.sum())
            if revenue < 0:
                raise ValueError("Net CPC approximation requires nonnegative net marks")
            # Zero-valued settled marks ARE observations. Half a dollar-cent
            # keeps the log transform finite, with deliberately broad variance.
            value = np.log((revenue + .005) / n)
            variance = config.net_mark_cv ** 2 / n
            if "net_human_revenue_sum_squares" in g and g.net_human_revenue_sum_squares.notna().all():
                second = float(g.net_human_revenue_sum_squares.sum()) / n
                mean = revenue / n
                variance = max(second - mean ** 2, 0) / max(n * mean ** 2, 1e-8)
            if revenue == 0:
                variance = max(variance, 4.)
            dispersion_multiplier = 0.
            wt = g.net_human_count.to_numpy()
        records.append(dict(cohort_id=str(cohort), source_id=str(g.source_id.iloc[0]),
            source_feature=float(g.source_feature.iloc[0]),
            calendar_day=float(np.average(g.calendar_day.to_numpy(), weights=wt)),
            value=float(value), variance=max(float(variance), config.measurement_variance_floor),
            dispersion_multiplier=dispersion_multiplier))
    return pd.DataFrame(records, columns=["cohort_id", "source_id", "source_feature",
        "calendar_day", "value", "variance", "dispersion_multiplier"])


@dataclass
class _Component:
    source_sd: float
    cohort_sd: float
    calendar_sd: float
    dispersion: float
    mean: np.ndarray
    precision_chol: np.ndarray
    log_weight: float


class _GaussianMixture:
    def __init__(self, obs, all_data, cutoff, mode, config, outcome):
        self.outcome, self.mode, self.config, self.cutoff = outcome, mode, config, cutoff
        self.sources = sorted(str(v) for v in all_data.source_id.unique())
        self.cohorts = sorted(str(v) for v in all_data.cohort_id.unique())
        self.source_cols, self.cohort_cols = {}, {}
        self.location = (config.retention_log_hazard if outcome == "retention" else
                         config.unique_response_log_location if outcome == "unique" else
                         config.response_log_location if outcome == "response" else config.payout_log_location)
        self.feature_col = None
        self.calendar_cols = []
        self.knots = np.array([0.], dtype=float)
        n = 0
        if mode == "target_only":
            self.cohort_cols = {c: i for i, c in enumerate(self.cohorts)}
            n = len(self.cohorts)
            self.global_col = None
        elif mode == "source_pooling":
            self.source_cols = {s: i for i, s in enumerate(self.sources)}
            n = len(self.sources)
            self.global_col = None
        else:
            self.global_col = 0
            n = 1
            if mode in ("static_hierarchy", "dynamic_hierarchy"):
                self.feature_col = n
                n += 1
                self.source_cols = {s: n + i for i, s in enumerate(self.sources)}
                n += len(self.sources)
                self.cohort_cols = {c: n + i for i, c in enumerate(self.cohorts)}
                n += len(self.cohorts)
                if mode == "dynamic_hierarchy" and cutoff > 0 and outcome != "retention":
                    # Fixed global knot grid. Inserting the cutoff as a knot
                    # would change the prior law of previously observed days.
                    # The last endpoint may be latent and later than cutoff.
                    endpoint = int(np.ceil(cutoff / config.calendar_bin_days)) * config.calendar_bin_days
                    self.knots = np.arange(0, endpoint + 1, config.calendar_bin_days, dtype=float)
                    self.calendar_cols = list(range(n, n + len(self.knots) - 1))
                    n += len(self.calendar_cols)
        # No-data independent modes still need a proper prior draw.
        self.dimension = max(n, 1)
        x = np.zeros((len(obs), self.dimension))
        for i, row in enumerate(obs.itertuples(index=False)):
            if self.global_col is not None:
                x[i, self.global_col] = 1.
            if self.feature_col is not None:
                x[i, self.feature_col] = row.source_feature
            if row.source_id in self.source_cols:
                x[i, self.source_cols[row.source_id]] = 1.
            if row.cohort_id in self.cohort_cols:
                x[i, self.cohort_cols[row.cohort_id]] = 1.
            if self.calendar_cols:
                x[i, self.calendar_cols] = self.calendar_basis([row.calendar_day])[0]
        self.obs_count = len(obs)
        self.components = []
        self.independent_weights = {}
        independent = mode in ("target_only", "source_pooling")
        block_cols = self.cohort_cols if mode == "target_only" else self.source_cols
        block_field = "cohort_id" if mode == "target_only" else "source_id"
        block_masks = {block: obs[block_field].to_numpy() == block for block in block_cols} if independent else {}
        block_log_weights = {block: [] for block in block_cols} if independent else {}
        hierarchy = mode in ("static_hierarchy", "dynamic_hierarchy")
        source_grid = list(zip(config.source_sds, config.effect_weights)) if hierarchy else [(0., 1.)]
        cohort_grid = list(zip(config.cohort_sds, config.effect_weights)) if hierarchy else [(0., 1.)]
        # Preserve the proper calendar-process prior at cutoff zero even
        # though no historical increments exist yet.  The sampled q still
        # controls future innovations in _states().
        calendar_grid = (list(zip(config.calendar_sds, config.calendar_weights))
                         if mode == "dynamic_hierarchy" and outcome != "retention" else [(0., 1.)])
        dispersion_grid = list(zip(config.count_dispersions, config.dispersion_weights)) if outcome == "response" else [(40., 1.)]
        m0 = np.zeros(self.dimension)
        if self.global_col is not None:
            m0[self.global_col] = self.location
        else:
            m0[:] = self.location
        y = obs.value.to_numpy(float) - x @ m0
        for (ss, ps), (cs, pc), (qs, pq), (r, pr) in product(source_grid, cohort_grid, calendar_grid, dispersion_grid):
            prior_var = np.full(self.dimension, config.location_sd ** 2)
            if hierarchy:
                prior_var[list(self.source_cols.values())] = ss ** 2
                prior_var[list(self.cohort_cols.values())] = cs ** 2
                prior_var[self.feature_col] = config.feature_sd ** 2
                if self.calendar_cols:
                    prior_var[self.calendar_cols] = qs ** 2 * np.diff(self.knots)
            variance = obs.variance.to_numpy(float) + obs.dispersion_multiplier.to_numpy(float) / r
            w = 1 / variance
            precision = np.diag(1 / prior_var) + (x.T * w) @ x
            fact, _ = cho_factor(precision, lower=True, check_finite=False)
            fact = np.tril(fact)
            b = x.T @ (w * y)
            delta = cho_solve((fact, True), b, check_finite=False)
            logdet = np.log(prior_var).sum() + 2 * np.log(np.diag(fact)).sum()
            logml = -.5 * (len(y) * np.log(2 * np.pi) + np.log(variance).sum() +
                            logdet + y @ (w * y) - b @ delta)
            self.components.append(_Component(ss, cs, qs, r, m0 + delta, fact,
                                              float(logml + np.log(ps * pc * pq * pr))))
            for block, mask in block_masks.items():
                col = block_cols[block]
                local_logml = -.5 * (mask.sum() * np.log(2 * np.pi) + np.log(variance[mask]).sum() +
                    np.log(prior_var[col]) + 2 * np.log(fact[col, col]) +
                    y[mask] @ (w[mask] * y[mask]) - b[col] * delta[col])
                block_log_weights[block].append(float(local_logml + np.log(pr)))
        lw = np.array([c.log_weight for c in self.components])
        self.weights = np.exp(lw - logsumexp(lw))
        for block, local_lw in block_log_weights.items():
            local_lw = np.asarray(local_lw)
            self.independent_weights[block] = np.exp(local_lw - logsumexp(local_lw))
        if self.independent_weights:
            # Diagnostic average, NOT a joint hyperposterior. Sampling below
            # uses each block's own properly normalized conditional evidence.
            self.weights = np.mean(list(self.independent_weights.values()), axis=0)

    def calendar_basis(self, days):
        days = np.asarray(days, float)
        if not self.calendar_cols:
            return np.empty((len(days), 0))
        return np.clip((days[:, None] - self.knots[:-1]) / np.diff(self.knots), 0., 1.)

    def sample(self, draws, rng):
        if self.mode in ("target_only", "source_pooling"):
            parameters = np.zeros((draws, self.dimension))
            scales = np.zeros((draws, 4))
            prior_r = rng.choice(self.config.count_dispersions, draws,
                                 p=self.config.dispersion_weights)
            scales[:, 3] = prior_r
            dispersion_by_block = {}
            cols = self.cohort_cols if self.mode == "target_only" else self.source_cols
            for block, col in cols.items():
                indices = rng.choice(len(self.components), draws, p=self.independent_weights[block])
                mean = np.array([self.components[idx].mean[col] for idx in indices])
                sd = np.array([1 / self.components[idx].precision_chol[col, col] for idx in indices])
                parameters[:, col] = mean + sd * rng.normal(size=draws)
                dispersion_by_block[block] = np.array([self.components[idx].dispersion for idx in indices])
            return parameters, scales, dispersion_by_block
        indices = rng.choice(len(self.components), size=draws, p=self.weights)
        parameters = np.zeros((draws, self.dimension))
        scales = np.zeros((draws, 4))
        for idx in np.unique(indices):
            mask = indices == idx
            c = self.components[idx]
            z = rng.normal(size=(self.dimension, mask.sum()))
            parameters[mask] = c.mean + solve_triangular(c.precision_chol.T, z,
                lower=False, check_finite=False).T
            scales[mask] = [c.source_sd, c.cohort_sd, c.calendar_sd, c.dispersion]
        return parameters, scales, {}

    def report(self):
        means = {}
        for field in ("source_sd", "cohort_sd", "calendar_sd", "dispersion"):
            means[field] = float(sum(w * getattr(c, field) for w, c in zip(self.weights, self.components)))
        return dict(measurement_bins=self.obs_count, parameters=self.dimension,
            grid_components=len(self.components), grid_probability_ess=float(1 / np.square(self.weights).sum()),
            maximum_grid_probability=float(self.weights.max()), hyperposterior_means=means,
            grid_weight_scope=("average of independent block hyperposteriors" if self.independent_weights else "joint hyperposterior"),
            posterior_global_log_mean=(float(sum(w * c.mean[self.global_col] for w, c in zip(
                self.weights, self.components))) if self.global_col is not None else None),
            posterior_feature_mean=(float(sum(w * c.mean[self.feature_col] for w, c in zip(
                self.weights, self.components))) if self.feature_col is not None else None))


class FittedModel:
    def __init__(self, data, cutoff, mode, seed, config):
        self.cutoff, self.mode, self.seed, self.config = int(cutoff), mode, int(seed), config
        self._data_json = data.to_json(orient="records", double_precision=15)
        self._data_columns = tuple(data.columns)
        self.data_hash = sha256(self._data_json.encode()).hexdigest()
        self.config_hash = sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()
        self.response = _GaussianMixture(_measurements(data, config, "response"), data,
                                         cutoff, mode, config, "response")
        self.payout = _GaussianMixture(_measurements(data, config, "payout"), data,
                                       cutoff, mode, config, "payout")
        self.retention = _GaussianMixture(_measurements(data, config, "retention"), data,
                                          cutoff, mode, config, "retention")
        self.unique_engagement = _GaussianMixture(_measurements(data, config, "unique"), data,
                                                   cutoff, mode, config, "unique")
        self.diagnostics = dict(method="conditional Gaussian posterior with integrated discrete hyperprior; approximate likelihood",
            mode=mode, cutoff=int(cutoff), input_hash=self.data_hash, config_hash=self.config_hash,
            response=self.response.report(), payout=self.payout.report(),
            contactability=self.retention.report(),
            unique_engagement=self.unique_engagement.report(),
            daily_rows=len(data), cohorts=int(data.cohort_id.nunique()), sources=int(data.source_id.nunique()),
            response_value_correlation="not learned; independent processes",
            past_pending_marks="approximate conditional audit/count imputation")

    def _states(self, mixture, params, scales, dates, scenario, rng):
        states = np.zeros((len(params), len(dates)))
        if mixture.mode != "dynamic_hierarchy":
            return states
        step = self.config.calendar_bin_days
        final_endpoint = int(np.ceil(dates[-1] / step)) * step
        knots = np.arange(0, final_endpoint + 1, step, dtype=float)
        increments = np.zeros((len(params), max(len(knots) - 1, 0)))
        old_count = len(mixture.calendar_cols)
        if old_count:
            increments[:, :old_count] = params[:, mixture.calendar_cols]
        if increments.shape[1] > old_count:
            # Draw the same weekly process used by historical inference,
            # including the latent endpoint spanning a non-knot cutoff.
            increments[:, old_count:] = rng.normal(size=(len(params), increments.shape[1] - old_count)) * scales[:, 2, None] * np.sqrt(step)
        if len(knots) > 1:
            basis = np.clip((dates[:, None] - knots[:-1]) / step, 0., 1.)
            states = increments @ basis.T
        future = dates > self.cutoff
        if not np.any(future):
            return states
        if scenario == "fixed_price" and mixture.outcome == "payout":
            states[:, future] = 0.  # declared calendar-zero payout anchor
            return states
        if scenario == "continue_recent_trend":
            start = max(0, self.cutoff - self.config.trend_window_days)
            cutoff_basis = np.clip((self.cutoff - knots[:-1]) / step, 0., 1.)
            start_basis = np.clip((start - knots[:-1]) / step, 0., 1.)
            end = increments @ cutoff_basis
            before = increments @ start_basis
            drift = (end - before) / max(self.cutoff - start, 1)
            states[:, future] += drift[:, None] * (dates[future] - self.cutoff)
        return states

    def _amplitudes(self, mixture, parameters, scales, targets, rng):
        result = np.empty((len(parameters), len(targets)))
        new_sources = {}
        new_cohorts = {}
        for i, target in enumerate(targets):
            source, cohort = str(target["source_id"]), str(target["cohort_id"])
            if mixture.mode == "target_only":
                result[:, i] = parameters[:, mixture.cohort_cols[cohort]] if cohort in mixture.cohort_cols else rng.normal(
                    mixture.location, self.config.location_sd, len(parameters))
                continue
            if mixture.mode == "source_pooling":
                if source not in new_sources:
                    new_sources[source] = parameters[:, mixture.source_cols[source]] if source in mixture.source_cols else rng.normal(
                        mixture.location, self.config.location_sd, len(parameters))
                result[:, i] = new_sources[source]
                continue
            amp = parameters[:, mixture.global_col].copy()
            if mixture.feature_col is not None:
                amp += parameters[:, mixture.feature_col] * float(target.get("source_feature", 0.))
                if source not in new_sources:
                    new_sources[source] = parameters[:, mixture.source_cols[source]] if source in mixture.source_cols else rng.normal(
                        size=len(parameters)) * scales[:, 0]
                if cohort not in new_cohorts:
                    new_cohorts[cohort] = parameters[:, mixture.cohort_cols[cohort]] if cohort in mixture.cohort_cols else rng.normal(
                        size=len(parameters)) * scales[:, 1]
                amp += new_sources[source] + new_cohorts[cohort]
            result[:, i] = amp
        return result

    def predict(self, targets: Sequence[Mapping[str, Any]], draws=256, horizon=425,
                scenario="hold_current", seed=None, cpc_shock=None):
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario {scenario!r}; choose {SCENARIOS}")
        if draws < 1 or horizon < 0 or horizon > 425:
            raise ValueError("draws must be positive; finite policy horizon must lie in [0,425]")
        targets = [dict(t) for t in targets]
        if not targets:
            raise ValueError("At least one forecast target is required")
        if cpc_shock is not None:
            if not isinstance(cpc_shock, Mapping) or "effective_day" not in cpc_shock or "drop_fraction" not in cpc_shock:
                raise ValueError("cpc_shock requires effective_day and drop_fraction")
            if int(cpc_shock["effective_day"]) <= self.cutoff:
                raise ValueError("A CPC shock must begin after the information cutoff; earlier clicks retain their booked value")
        ids = [str(t["cohort_id"]) for t in targets]
        if len(set(ids)) != len(ids):
            raise ValueError("Each cohort appears once in a joint forecast")
        rng = np.random.default_rng(self.seed if seed is None else seed)
        d = pd.DataFrame(json.loads(self._data_json), columns=self._data_columns)
        dates = np.arange(0, max(self.cutoff, max(int(t["birth_day"]) + horizon for t in targets)) + 1)
        rp, rs, response_block_r = self.response.sample(draws, rng)
        vp, vs, _ = self.payout.sample(draws, rng)
        tp, ts, _ = self.retention.sample(draws, rng)
        up, us, _ = self.unique_engagement.sample(draws, rng)
        response_state = self._states(self.response, rp, rs, dates, scenario, rng)
        cpc_state = self._states(self.payout, vp, vs, dates, scenario, rng)
        unique_state = self._states(self.unique_engagement, up, us, dates, scenario, rng)
        ra = self._amplitudes(self.response, rp, rs, targets, rng)
        va = self._amplitudes(self.payout, vp, vs, targets, rng)
        ta = self._amplitudes(self.retention, tp, ts, targets, rng)
        ua = self._amplitudes(self.unique_engagement, up, us, targets, rng)
        margins = np.empty((draws, len(targets), horizon + 1))
        revenues = np.empty_like(margins)
        send_cost_paths = np.empty_like(margins)
        counts = np.empty_like(margins)
        unique_counts = np.empty_like(margins)
        contactable_counts = np.empty_like(margins)
        expected_contactable_counts = np.empty_like(margins)
        expected_unique_counts = np.empty_like(margins)
        net_cpc = np.empty_like(margins)
        shock_factors = np.ones_like(margins)
        target_dispersion = np.empty((draws, len(targets)))
        ages = np.arange(horizon + 1)
        for j, target in enumerate(targets):
            birth = int(target["birth_day"])
            if birth < 0:
                raise ValueError("Calendar origin is day zero; negative births unsupported")
            calendar = birth + ages
            leads = float(target.get("leads", 100.))
            hazard = np.exp(np.clip(ta[:, j], -16., 2.))
            expected_exposure = leads * np.exp(-hazard[:, None] * ages[None, :])
            exposure = np.empty((draws, horizon + 1), dtype=float)
            exposure[:, 0] = leads
            p_stay = np.exp(-hazard)
            for age in range(1, horizon + 1):
                exposure[:, age] = rng.binomial(
                    np.maximum(np.rint(exposure[:, age - 1]).astype(int), 0), p_stay)
            past = d[d.cohort_id.astype(str) == str(target["cohort_id"])]
            observed_rows = past.dropna(subset=["accepted_exposure"]).sort_values("age")
            if not observed_rows.empty:
                for row in observed_rows.itertuples(index=False):
                    a = int(row.age)
                    if 0 <= a <= horizon:
                        exposure[:, a] = float(row.accepted_exposure)
                        expected_exposure[:, a] = float(row.accepted_exposure)
                last_age = int(observed_rows.age.iloc[-1])
                for age in range(last_age + 1, horizon + 1):
                    exposure[:, age] = rng.binomial(
                        np.maximum(np.rint(exposure[:, age - 1]).astype(int), 0), p_stay)
                    expected_exposure[:, age] = expected_exposure[:, age - 1] * p_stay
            contactable_counts[:, j] = exposure
            expected_contactable_counts[:, j] = expected_exposure
            activity_exposure = exposure.copy()
            activity_exposure[:, ages > self.config.send_last_age] = 0.
            mean_count = activity_exposure * np.exp(ra[:, j, None] + response_state[:, calendar])
            p_unique = np.clip(np.exp(ua[:, j, None] + unique_state[:, calendar]), 0., 1.)
            expected_unique_exposure = expected_exposure.copy()
            expected_unique_exposure[:, ages > self.config.send_last_age] = 0.
            expected_unique_counts[:, j] = expected_unique_exposure * p_unique
            unique_counts[:, j] = rng.binomial(
                np.maximum(np.rint(activity_exposure).astype(int), 0), p_unique)
            # Future NB cell frailty is drawn independently by send date;
            # common calendar and source/cohort amplitudes remain shared.
            block = str(target["cohort_id"] if self.mode == "target_only" else target["source_id"])
            if self.mode in ("target_only", "source_pooling") and block not in response_block_r:
                response_block_r[block] = rng.choice(self.config.count_dispersions, draws,
                                                     p=self.config.dispersion_weights)
            target_dispersion[:, j] = response_block_r.get(block, rs[:, 3])
            r = target_dispersion[:, j, None]
            frailty = rng.gamma(shape=r, scale=1/r, size=mean_count.shape)
            human = rng.poisson(mean_count * frailty).astype(float)
            mean_cpc = np.exp(va[:, j, None] + cpc_state[:, calendar])
            if "contracted_cpc_multiplier" in target:
                mean_cpc *= np.broadcast_to(np.asarray(target["contracted_cpc_multiplier"], float), (horizon + 1,))
            if cpc_shock is not None:
                effective = int(cpc_shock["effective_day"])
                drop = float(cpc_shock["drop_fraction"])
                ramp = max(0, int(cpc_shock.get("ramp_days", 0)))
                if not 0 <= drop < 1:
                    raise ValueError("CPC drop_fraction must lie in [0,1)")
                elapsed = calendar - effective
                factor = (1. - drop * np.clip(elapsed / ramp, 0., 1.)) if ramp else np.where(elapsed < 0, 1., 1. - drop)
                factor = np.where(elapsed < 0, 1., factor)
                mean_cpc *= factor[None, :]
                shock_factors[:, j, :] = factor[None, :]
            # Realized mature counts are commercial observations used only to
            # condition past predictions, not as another response likelihood.
            for row in past.itertuples(index=False):
                a = int(row.age)
                if not 0 <= a <= horizon:
                    continue
                if pd.notna(row.net_human_count) and row.payout_complete_fraction > 0:
                    human[:, a] = float(row.net_human_count)
                else:
                    n = 0. if pd.isna(row.audit_n) else float(row.audit_n)
                    h = 0. if pd.isna(row.audit_human) else float(row.audit_human)
                    audited_p = rng.beta(self.config.audit_beta_alpha + h,
                                         self.config.audit_beta_beta + n - h, draws)
                    det = self.config.audit_sensitivity + self.config.audit_specificity - 1
                    p = np.clip((audited_p - (1 - self.config.audit_specificity)) / det, 0, 1)
                    human[:, a] = rng.binomial(int(row.raw_events), p)
                    if row.response_report_fraction < 1:
                        human[:, a] += rng.poisson(mean_count[:, a] * (1 - row.response_report_fraction))
            shape = human / self.config.net_mark_cv ** 2
            rev = rng.gamma(shape=shape, scale=mean_cpc * self.config.net_mark_cv ** 2)
            for row in past.itertuples(index=False):
                a = int(row.age)
                if 0 <= a <= horizon and pd.notna(row.net_human_revenue) and row.payout_complete_fraction > 0:
                    rev[:, a] = float(row.net_human_revenue)
            if "send_cost_per_contactable" in target:
                send_cost = exposure * float(target["send_cost_per_contactable"])
            else:
                send_cost = np.broadcast_to(np.asarray(target.get("daily_send_cost", .025), float),
                                            (horizon + 1,)).copy()
            if send_cost.ndim == 2:
                send_cost[:, ages > self.config.send_last_age] = 0.
            else:
                send_cost[ages > self.config.send_last_age] = 0.
            acquisition = float(target.get("acquisition_cost", 0.))
            margins[:, j] = np.cumsum(rev - send_cost, axis=1) - acquisition
            send_cost_paths[:, j] = send_cost
            revenues[:, j], counts[:, j], net_cpc[:, j] = rev, human, mean_cpc
        diagnostic = dict(self.diagnostics, scenario=scenario, draws=int(draws), horizon=int(horizon),
            calendar_anchor=0, retention_model="partial-pooled exponential contactability hazard",
            portfolio_shared_calendar=True, scenario_is_conditional=True,
            cpc_shock=None if cpc_shock is None else dict(cpc_shock),
            acquisition_cost_semantics="total fixed historical or new-target acquisition cost",
            monetary_predictive_mean=float(revenues.sum(axis=2).mean()),
            top_one_percent_revenue_share=float(np.sort(revenues.sum(axis=(1, 2)))[-max(1, int(draws * .01)):].sum() /
                max(revenues.sum(), 1e-12)))
        return dict(margin_paths=margins, revenue_paths=revenues, human_count_paths=counts,
            send_cost_paths=send_cost_paths,
            unique_human_count_paths=unique_counts, contactable_count_paths=contactable_counts,
            expected_contactable_count_paths=expected_contactable_counts,
            expected_unique_response_paths=expected_unique_counts,
            net_human_cpc=net_cpc, cpc_shock_factor=shock_factors,target_ids=ids, calendar_dates=dates,
            calendar_response=response_state, calendar_cpc=cpc_state,
            calendar_response_factor=np.exp(response_state), calendar_cpc_factor=np.exp(cpc_state),
            response_log_amplitudes=ra, payout_log_amplitudes=va,
            response_count_dispersion_draws=target_dispersion,
            response_scale_draws=rs, payout_scale_draws=vs, diagnostics=diagnostic)


def fit(data, cutoff, mode="dynamic_hierarchy", seed=0, config=None):
    """Refit immutable as-of observations from the fixed declared hyperprior."""
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}; choose {MODES}")
    config = ModelConfig() if config is None else config
    if isinstance(config, Mapping):
        config = ModelConfig(**config)
    config.validate()
    if int(cutoff) != cutoff or cutoff < 0:
        raise ValueError("cutoff must be a nonnegative integer calendar day")
    return FittedModel(_snapshot(data, int(cutoff)), int(cutoff), mode, int(seed), config)
