#!/usr/bin/env python3
"""Render local, source-backed figures and RESULTS.md from the runner's CSVs.

The renderer performs no inference and never feeds evaluator truth to a model.
All displayed probabilities retain the unconditional never-under-policy mass.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cohort-lab-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd


BLUE = "#286AAB"
BLUE_LIGHT = "#DCEAF5"
BLUE_MID = "#9CBEDD"
INK = "#253442"
GREY = "#677782"
GRID = "#E5E9ED"
GOLD = "#C48627"
PINK = "#A25779"
PALETTE = [BLUE, GOLD, PINK, "#647B4B", "#775FA1"]
ORIGIN_STYLES = [("o", "-"), ("s", "--"), ("^", "-.")]
SCENARIO_STYLES = [("o", "-"), ("s", "--"), ("^", "-."), ("D", ":")]
QUANTILES = ["q025", "q10", "q25", "median", "q75", "q90", "q975"]


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 12, "axes.labelsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 10, "figure.titlesize": 18,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#B7C1C9", "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.facecolor": "white", "figure.facecolor": "white",
        "savefig.facecolor": "white", "axes.axisbelow": True,
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "lines.linewidth": 2.1,
    })


def _read(output: Path, filename: str, required: Iterable[str]) -> pd.DataFrame:
    path = output / filename
    if not path.exists():
        raise FileNotFoundError(f"Required runner artifact does not exist: {path}")
    data = pd.read_csv(path)
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"{filename} is missing columns: {sorted(missing)}")
    if data.empty:
        raise ValueError(f"{filename} has no rows to plot")
    return data


def _sources(data: pd.DataFrame) -> list[str]:
    found = set(data.source_id.astype(str))
    return [s for s in ("low", "middle", "high") if s in found] + sorted(found - {"low", "middle", "high"})


def _title_source(source: str) -> str:
    return source.replace("_", " ").capitalize() + " source"


def _unique(data: pd.DataFrame, keys: list[str], artifact: str) -> None:
    if data.duplicated(keys).any():
        raise ValueError(f"{artifact} has repeated {keys}; this demo renderer cannot silently average worlds/cohorts")


def _pct(value: float) -> str:
    # Do not round a small nonzero Monte Carlo mass to zero (or to certainty).
    decimals = 1 if 0 < value < .01 or .99 < value < 1 else 0
    return f"{100 * value:.{decimals}f}%"


def _dollars(value: float, decimals: int = 2) -> str:
    sign = "−" if value < 0 else ""
    return f"{sign}${abs(value):.{decimals}f}"


def _money_axis(ax: plt.Axes) -> None:
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: _dollars(y)))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.grid(axis="y", color=GRID, linewidth=.8)


def _money_x_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: _dollars(x)))
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.grid(axis="x", color=GRID, linewidth=.8)


def _probability_axis(ax: plt.Axes) -> None:
    ax.set_ylim(0, 1.035)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.grid(axis="y", color=GRID, linewidth=.8)


def _finish(fig: plt.Figure, directory: Path, name: str, title: str,
            subtitle: str, note: str, *, top: float = .78, bottom: float = .17,
            handles: list | None = None, labels: list[str] | None = None,
            legend_columns: int = 3, left: float = .08,
            wspace: float = .23) -> list[str]:
    """Reserve header/footer space explicitly; avoid cramped tight-layout guesses."""
    fig.suptitle(title, x=.06, y=.98, ha="left", weight="bold")
    fig.text(.06, .925, subtitle, ha="left", va="top", fontsize=11, color=GREY)
    if handles:
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.052, .885),
                   frameon=False, ncol=legend_columns, handlelength=2.4,
                   columnspacing=1.7)
    if _metadata(directory.parent).get("quick", False):
        note = "Quick wiring test only: low-draw illustration, not calibration evidence.\n" + note
        bottom += .035
    fig.text(.06, .025, note, ha="left", va="bottom", fontsize=9.5, color=GREY,
             linespacing=1.5)
    fig.subplots_adjust(left=left, right=.975, top=top, bottom=bottom,
                        wspace=wspace, hspace=.35)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in ("png", "svg"):
        path = directory / f"{name}.{extension}"
        fig.savefig(path, dpi=180, metadata={"Title": title, "Description": subtitle + " " + note})
        paths.append(str(path))
    plt.close(fig)
    return paths


def _interval_legend(mean: bool = False) -> tuple[list, list[str]]:
    handles = [Patch(facecolor=BLUE_LIGHT), Patch(facecolor=BLUE_MID),
               Line2D([], [], color=BLUE, marker="o" if mean else None),
               Line2D([], [], color=INK, linestyle="--")]
    labels = ["95% predictive interval", "80% predictive interval",
              "Predictive mean" if mean else "Predictive median", "Evaluator-only economic truth"]
    return handles, labels


def _horizon(summary: pd.DataFrame) -> int:
    horizons = sorted(summary.horizon.unique())
    return 180 if 180 in horizons else int(max(horizons))


def _limits(values: list[np.ndarray], include_zero: bool = True) -> tuple[float, float]:
    joined = np.concatenate([np.asarray(a, dtype=float).ravel() for a in values])
    joined = joined[np.isfinite(joined)]
    if include_zero:
        joined = np.append(joined, 0)
    lo, hi = float(joined.min()), float(joined.max())
    span = max(hi - lo, .04)
    return lo - span * .08, hi + span * .10


def _demo_figures(output: Path, figures: Path, summary: pd.DataFrame) -> list[str]:
    needed = ["source_id", "origin_age", "horizon", "n_leads", "mean", "true_margin", *QUANTILES,
              "probability_payback_by_horizon", "probability_later_than_180", "probability_never"]
    missing = set(needed) - set(summary.columns)
    if missing:
        raise ValueError(f"summary.csv is missing columns: {sorted(missing)}")
    _unique(summary, ["source_id", "origin_age", "horizon"], "summary.csv")
    trajectories = _read(output, "trajectories.csv.gz", ["source_id", "origin_age", "age", "n_leads", "truth", "observed", *QUANTILES])
    cdf = _read(output, "payback_cdf.csv.gz", ["source_id", "origin_age", "age", "cdf", "probability_never"])
    _unique(trajectories, ["source_id", "origin_age", "age"], "trajectories.csv.gz")
    _unique(cdf, ["source_id", "origin_age", "age"], "payback_cdf.csv.gz")
    sources = _sources(summary)
    origins = sorted(int(o) for o in summary.origin_age.unique())
    h = _horizon(summary)
    terminal = int(trajectories.age.max())
    records: list[str] = []

    # Discrete forecast origins: no interpolated interval boundary implies a trend
    # between fits or claims that a later interval must be narrower.
    selected = summary.loc[summary.horizon.eq(h)].copy()
    fig, axes = plt.subplots(1, len(sources), figsize=(max(12.8, 4.3 * len(sources)), 6.8), squeeze=False, sharey=True)
    ymin, ymax = _limits([selected.q025 / selected.n_leads, selected.q975 / selected.n_leads,
                         selected["mean"] / selected.n_leads,
                         selected.true_margin / selected.n_leads])
    positions = np.asarray(origins, dtype=float)
    for ax, source in zip(axes[0], sources):
        rows = selected.loc[selected.source_id.eq(source)].set_index("origin_age").reindex(origins)
        n = rows.n_leads.to_numpy(float)
        low95, high95 = rows.q025.to_numpy(float) / n, rows.q975.to_numpy(float) / n
        low80, high80 = rows.q10.to_numpy(float) / n, rows.q90.to_numpy(float) / n
        mean = rows["mean"].to_numpy(float) / n
        ax.vlines(positions, low95, high95, color=BLUE_LIGHT, linewidth=14, zorder=2)
        ax.vlines(positions, low80, high80, color=BLUE_MID, linewidth=7, zorder=3)
        ax.plot(positions, mean, "o-", color=BLUE, markersize=6, zorder=4)
        truth = rows.true_margin.to_numpy(float) / n
        ax.plot(positions, truth, "--", color=INK, linewidth=1.7, zorder=5)
        ax.axhline(0, color=GREY, linewidth=1)
        ax.set(title=_title_source(source), xlabel="Forecast origin (cohort age, days)",
               xticks=positions, xticklabels=[str(o) for o in origins], ylim=(ymin, ymax),
               xlim=(min(origins) - 3, max(origins) + 3))
        _money_axis(ax)
    axes[0, 0].set_ylabel(f"Economic contribution / acquired lead at day {h}")
    handles, labels = _interval_legend(mean=True)
    uncertainty_note = "Intervals are posterior predictive, conditional on the fitted model and finite sending policy. More data can shift or widen them.\nTruth is shown for evaluation only; it was unavailable to the forecasting fit."
    if ((selected["mean"] > selected.q975) | (selected["mean"] < selected.q025)).any():
        uncertainty_note += "\nA mean outside the central 95% interval is tail-sensitive; bounded model moments do not ensure a stable finite-draw mean. Inspect median/prior sensitivity."
    records += _finish(fig, figures, "uncertainty_by_origin",
        "How uncertainty updates as observations arrive", f"SYNTHETIC / FICTIONAL • Forecast target: realized day-{h} economic contribution per acquired lead",
        uncertainty_note,
        handles=handles, labels=labels, legend_columns=4)

    # Fan: the as-of reported ledger is neither full economic truth nor a model
    # target pinned down at the origin. Delay keeps the past uncertain too.
    fan_origins = [o for o in (3, 60) if o in origins]
    if not fan_origins:
        fan_origins = [origins[0], origins[-1]] if len(origins) > 1 else origins
    fan_data = trajectories.loc[trajectories.origin_age.isin(fan_origins)].copy()
    ymin, ymax = _limits([fan_data.q025 / fan_data.n_leads, fan_data.q975 / fan_data.n_leads,
                         fan_data.truth / fan_data.n_leads, fan_data.observed / fan_data.n_leads])
    fig, axes = plt.subplots(len(fan_origins), len(sources), figsize=(max(13.4, 4.5 * len(sources)), 5.0 * len(fan_origins) + 1.4),
                             squeeze=False, sharex=True, sharey=True)
    for row_i, origin in enumerate(fan_origins):
        for col_i, source in enumerate(sources):
            ax = axes[row_i, col_i]
            rows = fan_data.loc[fan_data.source_id.eq(source) & fan_data.origin_age.eq(origin)].sort_values("age")
            x, n = rows.age.to_numpy(float), rows.n_leads.to_numpy(float)
            ax.axvspan(0, origin, color="#F2F4F6", zorder=0)
            ax.fill_between(x, rows.q025.to_numpy(float) / n, rows.q975.to_numpy(float) / n, color=BLUE_LIGHT, zorder=1)
            ax.fill_between(x, rows.q10.to_numpy(float) / n, rows.q90.to_numpy(float) / n, color=BLUE_MID, zorder=2)
            ax.plot(x, rows["median"].to_numpy(float) / n, color=BLUE, linewidth=1.8, zorder=3)
            ax.plot(x, rows.truth.to_numpy(float) / n, color=INK, linestyle="--", linewidth=1.4, zorder=4)
            visible = rows.age.le(origin).to_numpy()
            ax.step(x[visible], (rows.observed.to_numpy(float) / n)[visible], where="post", color=GOLD, linewidth=2.3, zorder=5)
            ax.axvline(origin, color=GREY, linestyle=":", linewidth=1.2)
            ax.axhline(0, color=GREY, linewidth=.9)
            ax.set(title=f"{_title_source(source)} · origin day {origin}", xlim=(0, terminal), ylim=(ymin, ymax))
            ticks = sorted(set([0, 60, 180, 300, terminal]) & set(range(terminal + 1)))
            ax.set_xticks(ticks)
            _money_axis(ax)
            if row_i == len(fan_origins) - 1:
                ax.set_xlabel("Cohort age (days)")
            if col_i == 0:
                ax.set_ylabel("Economic contribution / acquired lead")
    handles, labels = _interval_legend()
    handles += [Line2D([], [], color=GOLD, linewidth=2.3), Line2D([], [], color=GREY, linestyle=":")]
    labels += ["Reported ledger visible at origin", "Forecast origin"]
    records += _finish(fig, figures, "posterior_predictive_fan",
        "Economic paths, seen early and later", f"SYNTHETIC / FICTIONAL • Full-policy paths through day {terminal} • Common dollar scale across all panels",
        "Gold uses only postings visible at the origin and stops there; it is not a forecast or complete economic history.\nShaded past remains uncertain because response, posting and revision delays separate the ledger from economic truth.\nThe dashed truth path is evaluator-only. Fan widths include modeled parameter and future-event uncertainty, not all possible model failures.",
        top=.80, bottom=.15, handles=handles, labels=labels, legend_columns=3)

    cdf_origins = [o for o in (3, 14, 60) if o in origins]
    if not cdf_origins:
        cdf_origins = origins
    fig, axes = plt.subplots(1, len(sources), figsize=(max(13.0, 4.4 * len(sources)), 7.1), squeeze=False, sharey=True)
    origin_handles = []
    origin_labels = []
    for j, origin in enumerate(cdf_origins):
        marker, line = ORIGIN_STYLES[j % len(ORIGIN_STYLES)]
        color = PALETTE[j % len(PALETTE)]
        origin_handles.append(Line2D([], [], color=color, linestyle=line, marker=marker))
        origin_labels.append(f"Origin day {origin}")
    for ax, source in zip(axes[0], sources):
        never_values = []
        for j, origin in enumerate(cdf_origins):
            rows = cdf.loc[cdf.source_id.eq(source) & cdf.origin_age.eq(origin)].sort_values("age")
            marker, line = ORIGIN_STYLES[j % len(ORIGIN_STYLES)]
            color = PALETTE[j % len(PALETTE)]
            ax.step(rows.age, rows.cdf, where="post", color=color, linestyle=line, linewidth=2)
            ax.plot([terminal], [rows.cdf.iloc[-1]], marker=marker, color=color, markersize=5)
            never_values.append(_pct(float(rows.probability_never.iloc[-1])))
            if not np.isclose(rows.cdf.iloc[-1] + rows.probability_never.iloc[-1], 1, atol=1e-7):
                raise ValueError("Terminal payback CDF and never mass do not reconcile to one")
        # Put terminal mass above the plotting region. In cold starts the same
        # source can have high and low CDFs, so no in-plot corner is always safe.
        ax.text(.02, 1.02, "P(never): " + " / ".join(never_values) + "\nAt origins " +
                " / ".join(str(o) for o in cdf_origins) + " days", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=9.5, linespacing=1.4)
        ax.axvline(180, color=GREY, linestyle=":", linewidth=1)
        ax.set(xlabel="First-payback age (days)", xlim=(0, terminal))
        ax.set_title(_title_source(source), y=1.15)
        ax.set_xticks([a for a in (0, 90, 180, 300, terminal) if a <= terminal])
        _probability_axis(ax)
    axes[0, 0].set_ylabel("P(first payback by age)")
    records += _finish(fig, figures, "unconditional_payback_cdf",
        "Payback includes a chance of never paying back", f"SYNTHETIC / FICTIONAL • Unconditional first-payback CDF • Finite policy ends at day {terminal}",
        "The CDF is not renormalized among eventual successes: its terminal height is 1 − P(never under this policy).\nFirst payback can later be reversed by costs or revisions; it is distinct from being profitable at day 180 or staying profitable.\n“Never” means no crossing through the finite policy endpoint, not an infinite-lifetime claim. 0%/100% draw estimates are not certainty.",
        handles=origin_handles, labels=origin_labels, legend_columns=min(4, len(cdf_origins)), top=.70)

    # These three mutually exclusive states are per draw, not a positive-at-H
    # probability. A source can pay back early and be negative at H180.
    h180 = summary.loc[summary.horizon.eq(180)].copy()
    if not h180.empty:
        fig, axes = plt.subplots(1, len(sources), figsize=(max(12.8, 4.3 * len(sources)), 6.9), squeeze=False, sharey=True)
        colors = [BLUE, GOLD, PINK]
        labels = ["First payback by day 180", f"First payback days 181–{terminal}", "Never under policy"]
        bar_positions = np.arange(len(origins))
        for ax, source in zip(axes[0], sources):
            rows = h180.loc[h180.source_id.eq(source)].set_index("origin_age").reindex(origins)
            parts = [rows.probability_payback_by_horizon.to_numpy(float), rows.probability_later_than_180.to_numpy(float),
                     rows.probability_never.to_numpy(float)]
            if not np.allclose(np.sum(parts, axis=0), 1, atol=1e-7):
                raise ValueError("Payback state probabilities do not reconcile to one")
            bottom = np.zeros(len(origins))
            for values, color in zip(parts, colors):
                ax.bar(bar_positions, values, bottom=bottom, width=.68, color=color, edgecolor="white", linewidth=.7)
                for x, value, baseline in zip(bar_positions, values, bottom):
                    if value >= .09:
                        ax.text(x, baseline + value / 2, _pct(value), ha="center", va="center", color="white", fontsize=10, weight="bold")
                bottom += values
            ax.set(title=_title_source(source), xlabel="Forecast origin (days)", xticks=bar_positions, xticklabels=origins)
            _probability_axis(ax)
        axes[0, 0].set_ylabel("Posterior-predictive probability")
        records += _finish(fig, figures, "payback_state_probabilities",
            "Early, later, or no payback under the policy", f"SYNTHETIC / FICTIONAL • Three mutually exclusive first-payback states through day {terminal}",
            "Each column sums to 100%, including draws that never pay back under the finite sending policy.\nThese are model-conditional probabilities. 0%/100% draw estimates are not certainty; first payback and horizon positivity are different events.",
            handles=[Patch(facecolor=c) for c in colors], labels=labels, legend_columns=3)
    return records


def _calibration_figures(output: Path, figures: Path, scores: pd.DataFrame) -> list[str]:
    metrics = _read(output, "metrics_by_origin.csv", ["origin_age", "metric", "mean", "lower95", "upper95", "worlds", "forecasts"])
    reliability = _read(output, "reliability.csv", ["probability_bin", "mean_prediction", "realized_fraction", "forecasts"])
    _unique(metrics, ["origin_age", "metric"], "metrics_by_origin.csv")
    h = _horizon(scores)
    selected = scores.loc[scores.horizon.eq(h)].copy()
    selected["signed_error_per_lead"] = (selected["mean"] - selected.true_margin) / selected.n_leads
    origins = sorted(int(o) for o in selected.origin_age.unique())
    worlds = selected.world_id.nunique()
    scenarios = list(dict.fromkeys(selected.scenario.astype(str)))
    records: list[str] = []

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.9), sharey=True)
    handles = [Line2D([], [], color=BLUE, marker="o"), Line2D([], [], color=INK, linestyle="--")]
    for ax, metric, target in zip(axes, ["coverage80", "coverage95"], [.80, .95]):
        rows = metrics.loc[metrics.metric.eq(metric)].sort_values("origin_age")
        if rows.empty:
            raise ValueError(f"metrics_by_origin.csv has no {metric} rows")
        x = rows.origin_age.to_numpy(float)
        ax.errorbar(x, rows["mean"], yerr=np.vstack([rows["mean"] - rows.lower95, rows.upper95 - rows["mean"]]),
                    fmt="o-", color=BLUE, capsize=5, linewidth=2, markersize=6)
        ax.axhline(target, color=INK, linestyle="--", linewidth=1.6)
        ax.set(title=f"{int(target * 100)}% predictive interval", xlabel="Forecast origin (cohort age, days)",
               xticks=x, xticklabels=rows.origin_age.astype(int).tolist(),
               xlim=(min(origins) - 3, max(origins) + 3))
        _probability_axis(ax)
        # Exact n remains visible rather than implying every row is independent.
        n_worlds = int(rows.worlds.min())
        ax.text(.04, .04, f"{n_worlds} independent {'world' if n_worlds == 1 else 'worlds'}; {int(rows.forecasts.min())} forecasts / origin",
                transform=ax.transAxes, color=GREY, fontsize=9.5)
    axes[0].set_ylabel("Empirical coverage of evaluator-only truth")
    records += _finish(fig, figures, "empirical_interval_coverage",
        "Do predictive intervals cover complete economic truth?", f"SYNTHETIC / FICTIONAL • Day-{h} prospective forecasts • Deliberately misspecified stress generators",
        "Error bars: 95% percentile-bootstrap intervals, resampling whole worlds rather than correlated forecasts.\nNominal 80%/95% lines are references, not validated guarantees. This is a small stress check, not Bayesian simulation-based calibration.",
        handles=handles, labels=["Empirical coverage + world-bootstrap 95% interval", "Nominal target"], legend_columns=2)

    fig, ax = plt.subplots(figsize=(8.8, 8.0))
    ax.plot([0, 1], [0, 1], color=INK, linestyle="--", linewidth=1.5)
    populated = reliability.loc[reliability.forecasts.gt(0)].dropna(subset=["mean_prediction", "realized_fraction"])
    for _, row in populated.iterrows():
        ax.scatter(row.mean_prediction, row.realized_fraction, s=72, color=BLUE, zorder=3)
        # Opposite offsets for alternating bins keep count labels distinct at edges.
        dx = -8 if row.mean_prediction > .84 else 8
        dy = -17 if row.realized_fraction > .90 else 9
        ax.annotate(f"n={int(row.forecasts)}", (row.mean_prediction, row.realized_fraction),
                    xytext=(dx, dy), textcoords="offset points", fontsize=10,
                    ha="right" if dx < 0 else "left", va="top" if dy < 0 else "bottom")
    ax.set(xlim=(-.025, 1.025), ylim=(-.025, 1.025),
           xlabel=f"Mean forecast P(economic contribution > 0 at day {h})",
           ylabel="Fraction actually profitable (evaluator truth)")
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xticks(np.linspace(0, 1, 6)); ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(color=GRID, linewidth=.8)
    empty = reliability.loc[reliability.forecasts.eq(0), "probability_bin"].astype(str).tolist()
    empty_note = " Empty bins: " + ", ".join(empty) + "." if empty else ""
    records += _finish(fig, figures, "profitability_reliability",
        "Probability reliability, descriptively", f"SYNTHETIC / FICTIONAL • Day-{h} profitability • {worlds} {'world' if worlds == 1 else 'worlds'}, {len(selected)} correlated forecasts",
        "Five fixed-width probability bins; labels count forecast rows, not independent worlds.\nRepeated sources and origins share a world. No IID binomial confidence intervals or calibration proof.\nThe diagonal is perfect reliability." + empty_note,
        top=.80, bottom=.18,
        handles=[Line2D([], [], color=BLUE, marker="o", linestyle=""), Line2D([], [], color=INK, linestyle="--")],
        labels=["Nonempty probability bin", "Perfect-reliability reference"], legend_columns=2)

    # Aggregate each world first. The resulting lines remain descriptive and have
    # no invented independent-row uncertainty or distribution-free guarantees.
    selected["miss95"] = 1 - selected.coverage95
    world_scores = selected.groupby(["scenario", "world_id", "origin_age"], observed=True)[
        ["crps_per_lead", "miss95", "signed_error_per_lead"]].mean().reset_index()
    averages = world_scores.groupby(["scenario", "origin_age"], observed=True)[
        ["crps_per_lead", "miss95", "signed_error_per_lead"]].mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 7.5))
    labels = []
    handles = []
    for j, scenario in enumerate(scenarios):
        color = PALETTE[j % len(PALETTE)]
        marker, line = SCENARIO_STYLES[j % len(SCENARIO_STYLES)]
        rows = averages.loc[averages.scenario.eq(scenario)].set_index("origin_age").reindex(origins)
        x = np.asarray(origins, dtype=float)
        for ax, column in zip(axes, ["crps_per_lead", "miss95", "signed_error_per_lead"]):
            ax.plot(x, rows[column], color=color, marker=marker, linestyle=line, markersize=5)
        handles.append(Line2D([], [], color=color, marker=marker, linestyle=line))
        labels.append(scenario.replace("_", " ").capitalize())
    for ax in axes:
        ax.set(xticks=origins, xticklabels=origins, xlabel="Forecast origin (days)",
               xlim=(min(origins) - 3, max(origins) + 3))
    axes[0].set(title="Distribution error (lower is better)", ylabel="CRPS / acquired lead")
    _money_axis(axes[0]); axes[0].set_ylim(bottom=0)
    axes[1].set(title="95% interval misses", ylabel="Fraction of forecasts outside interval")
    _probability_axis(axes[1]); axes[1].axhline(.05, color=GREY, linestyle=":", linewidth=1)
    axes[2].set(title="Signed contribution error", ylabel="(Predictive mean − truth) / acquired lead")
    _money_axis(axes[2]); axes[2].axhline(0, color=GREY, linewidth=1)
    records += _finish(fig, figures, "scenario_stress_scores",
        "Stress scenarios expose model failures", f"SYNTHETIC / FICTIONAL • Day-{h} outcomes • Equal weighting of world averages",
        "CRPS scores the entire predictive distribution. Miss rate is 1 − empirical 95% coverage; its dotted reference is 5%.\nPositive signed error means optimistic forecasts; averaging can hide opposite misses. Individual margins appear in scenario_margin_intervals.\nThese are descriptive stress comparisons with few seeds, not a statistically established model ranking.",
        handles=handles, labels=labels, legend_columns=2, top=.76, bottom=.19)

    # Keep actual intervals at their forecast grain: no averaged bounds passed
    # off as an interval around a scenario-average prediction.
    last_origin = max(origins)
    margin_rows = selected.loc[selected.origin_age.eq(last_origin)].copy()
    panels = len(scenarios)
    ncols = min(2, panels); nrows = math.ceil(panels / ncols)
    counts = margin_rows.groupby("scenario").size()
    fig, axes = plt.subplots(nrows, ncols, figsize=(14,
                             max(7.5, (max(counts) * .42 + 2.0) * nrows + 1.0)), squeeze=False, sharex=True)
    xmin, xmax = _limits([margin_rows.q025 / margin_rows.n_leads, margin_rows.q975 / margin_rows.n_leads,
                         margin_rows["mean"] / margin_rows.n_leads,
                         margin_rows.true_margin / margin_rows.n_leads])
    source_order = {s: i for i, s in enumerate(_sources(margin_rows))}
    for ax, scenario in zip(axes.flat, scenarios):
        rows = margin_rows.loc[margin_rows.scenario.eq(scenario)].copy()
        rows["source_order"] = rows.source_id.map(source_order)
        rows = rows.sort_values(["source_order", "seed", "cohort_id"])
        y = np.arange(len(rows))
        n = rows.n_leads.to_numpy(float)
        ax.hlines(y, rows.q025.to_numpy(float) / n, rows.q975.to_numpy(float) / n, color=BLUE_LIGHT, linewidth=9)
        ax.hlines(y, rows.q10.to_numpy(float) / n, rows.q90.to_numpy(float) / n, color=BLUE_MID, linewidth=5)
        ax.scatter(rows["mean"].to_numpy(float) / n, y, color=BLUE, s=26, zorder=4)
        ax.scatter(rows.true_margin.to_numpy(float) / n, y, color=INK, marker="D", s=23, zorder=5)
        ax.axvline(0, color=GREY, linewidth=1)
        labels_y = [f"{row.source_id} · seed {int(row.seed)}" for _, row in rows.iterrows()]
        ax.set(title=scenario.replace("_", " ").capitalize(), yticks=y, yticklabels=labels_y,
               xlim=(xmin, xmax), xlabel=f"Day-{h} economic contribution / acquired lead")
        ax.invert_yaxis()
        _money_x_axis(ax)
    for ax in list(axes.flat)[panels:]:
        ax.set_visible(False)
    handles, labels = _interval_legend(mean=True)
    handles[-1] = Line2D([], [], color=INK, marker="D", linestyle="")
    records += _finish(fig, figures, "scenario_margin_intervals",
        "Keep individual stress misses visible", f"SYNTHETIC / FICTIONAL • Origin day {last_origin} → day {h} • Each row is one source/cohort/world forecast",
        "Intervals are predictive at the individual-forecast grain, not uncertainty around scenario-average margins.\nTruth diamonds use evaluator-only complete economic outcomes. Common dollar scale; all selected seeds and sources are retained.",
        handles=handles, labels=labels, legend_columns=4, top=.80, bottom=.17,
        left=.15, wspace=.38)
    return records


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] + [
        "| " + " | ".join(row) + " |" for row in rows]


def _metadata(output: Path) -> dict:
    path = output / "run_metadata.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _write_results(output: Path, mode: str, data: pd.DataFrame, records: list[str]) -> None:
    meta = _metadata(output)
    quick = bool(meta.get("quick", False))
    h = _horizon(data)
    runs = meta.get("runs", [])
    lines = ["# Synthetic Bayesian paid-email cohort lab: results", "",
             "All sources, leads, observations, monetary amounts and outcomes are fictional. This report is generated from the CSV artifacts in this directory, not from hand-entered example numbers.", "",
             "## What this run establishes", ""]
    if quick:
        lines += ["**Quick wiring test only.** This run uses reduced cohorts/draws and fewer forecast origins. A single-world calibration smoke run cannot establish empirical calibration; its bootstrap interval is degenerate. Do not use these outputs as validation evidence.", ""]
    lines += [f"- Mode: `{mode}`; forecast target shown in headline figures: day {h}",
              f"- Origins present: {', '.join(str(int(v)) for v in sorted(data.origin_age.unique()))} days",
              f"- Worlds in these CSVs: {data.world_id.nunique() if 'world_id' in data else 'unavailable'}; sources: {', '.join(_sources(data))}"]
    if runs:
        lines += [f"- Predictive draws per forecast: {', '.join(str(int(v)) for v in sorted({r['draws'] for r in runs}))}",
                  f"- Measured runner wall time before rendering: {meta.get('wall_seconds', float('nan')):.1f} seconds"]
        draw_counts = sorted({int(r["draws"]) for r in runs})
        mcse = "; ".join(f"{d} draws: {0.5 / math.sqrt(d):.3f} probability units ({50 / math.sqrt(d):.1f} percentage points)" for d in draw_counts)
        lines += ["", "Probabilities are finite-draw Monte Carlo estimates. Observed 0% or 100% is not certainty. Under independent predictive sampling, worst-case probability Monte Carlo standard error is approximately " + mcse + ". Quantile estimates also have Monte Carlo noise; these errors do not include model misspecification."]
    lines += ["", "The point forecast is a posterior-predictive mean. The 80% and 95% intervals describe modeled realized economic contribution, not a confidence interval for a population-average effect or a parameter-only expected contribution. Early economic history can remain uncertain because responses, accounting postings and later revisions arrive on different clocks.", ""]
    if mode == "demo":
        selected = data.loc[data.horizon.eq(h)].sort_values(["source_id", "origin_age"])
        lines += [f"## Day-{h} forecasts (dollars per acquired lead)", "",
                  "Truth is available only to the evaluator after forecasting. It is printed here to make the synthetic illustration inspectable.", ""]
        rows = []
        h180 = data.loc[data.horizon.eq(180)].set_index(["source_id", "origin_age"])
        for _, row in selected.iterrows():
            n = row.n_leads
            p180 = h180.loc[(row.source_id, row.origin_age), "probability_payback_by_horizon"] if not h180.empty else np.nan
            rows.append([str(row.source_id), str(int(row.origin_age)), _dollars(row["mean"] / n, 3), _dollars(row["median"] / n, 3),
                         f"{_dollars(row.q10 / n, 3)} to {_dollars(row.q90 / n, 3)}",
                         f"{_dollars(row.q025 / n, 3)} to {_dollars(row.q975 / n, 3)}",
                         _dollars(row.true_margin / n, 3), _pct(row.probability_profitable),
                         _pct(p180) if np.isfinite(p180) else "N/A", _pct(row.probability_never)])
        lines += _markdown_table(["Source", "Origin", "Mean", "Median", "80% predictive interval", "95% predictive interval",
                                  "Truth", f"P(positive at {h})", "P(payback by 180)", "P(never under policy)"], rows)
        extreme_mean = selected.loc[(selected["mean"] > selected.q975) | (selected["mean"] < selected.q025)]
        if not extreme_mean.empty:
            lines += ["", f"**Tail-sensitive mean warning:** {len(extreme_mean)} forecast(s) have a predictive mean outside their central 95% predictive interval. This is possible for a skewed/heavy-tailed draw distribution. These means are retained in the chart scale and numeric table without trimming; a few extreme draws can dominate the average. Finite bounded model moments do not imply that a 500-draw mean is stable. The table also retains medians; the fan uses the median instead. Use medians/quantiles and inspect prior/variance sensitivity before interpreting a sparse-data mean as a stable expected-value estimate."]
        latest = selected.loc[selected.origin_age.eq(selected.origin_age.max())]
        lines += ["", "At the latest observed origin, interval widths are model-conditional. They may be smaller, larger or shifted relative to earlier fits; this renderer does not impose monotonically shrinking uncertainty.", "",
                  "The probabilities by day 180, later through the policy endpoint, and never under policy form mutually exclusive first-payback states. P(positive margin at the horizon) is a different event: a first crossing can be reversed by later costs or revisions.", ""]
        lines += ["Latest-origin first-payback state probabilities:", ""]
        state_rows = [[str(r.source_id), _pct(r.probability_payback_by_horizon),
                       _pct(r.probability_later_than_180), _pct(r.probability_never)]
                      for _, r in h180.reset_index().loc[lambda d: d.origin_age.eq(data.origin_age.max())].iterrows()]
        if state_rows:
            lines += _markdown_table(["Source", "By day 180", "Later through endpoint", "Never under policy"], state_rows)
        if latest.empty:
            raise ValueError("No latest-origin results were available")
    else:
        metrics = _read(output, "metrics_by_origin.csv", ["origin_age", "metric", "mean", "lower95", "upper95", "worlds", "forecasts"])
        lines += ["## Coverage and distribution scores", "",
                  "Coverage and CRPS below use day-180 prospective forecasts. Coverage uncertainty resamples whole worlds, not repeated sources/origins as independent observations. The scenarios deliberately differ from model assumptions. This is misspecification stress validation, not prior-generated Bayesian simulation-based calibration.", ""]
        rows = []
        for origin in sorted(metrics.origin_age.unique()):
            subset = metrics.loc[metrics.origin_age.eq(origin)].set_index("metric")
            def value(metric: str, probability: bool = False) -> str:
                r = subset.loc[metric]
                fmt = _pct if probability else lambda x: _dollars(x, 4)
                return f"{fmt(r['mean'])} [{fmt(r.lower95)}, {fmt(r.upper95)}]"
            rows.append([str(int(origin)), value("coverage80", True), value("coverage95", True),
                         value("crps_per_lead"), str(int(subset.worlds.iloc[0])), str(int(subset.forecasts.iloc[0]))])
        lines += _markdown_table(["Origin", "80% coverage [bootstrap 95%]", "95% coverage [bootstrap 95%]",
                                  "CRPS / lead [bootstrap 95%]", "Worlds", "Forecasts"], rows)
        selected = data.loc[data.horizon.eq(h)].copy()
        selected["signed_error_per_lead"] = (selected["mean"] - selected.true_margin) / selected.n_leads
        selected["miss95"] = 1 - selected.coverage95
        latest = selected.loc[selected.origin_age.eq(selected.origin_age.max())]
        world_values = latest.groupby(["scenario", "world_id"], observed=True)[["crps_per_lead", "miss95", "signed_error_per_lead"]].mean()
        scenario = world_values.groupby("scenario", observed=True).mean()
        lines += ["", f"## Scenario failures at origin {int(latest.origin_age.max())}, horizon {h}", "",
                  "These numbers average within each world first, then across worlds. Positive signed error is optimistic. Averages can hide cancellation, so individual forecast intervals and the largest misses are retained.", ""]
        rows = [[str(name), _dollars(r.crps_per_lead, 4), _pct(r.miss95), _dollars(r.signed_error_per_lead, 4)]
                for name, r in scenario.iterrows()]
        lines += _markdown_table(["Scenario", "CRPS / lead", "95% interval miss fraction", "Signed mean error / lead"], rows)
        worst = selected.sort_values("absolute_error_per_lead", ascending=False).head(8)
        lines += ["", "Largest individual day-target predictive-mean errors (no adverse seeds discarded):", ""]
        rows = [[str(r.scenario), str(int(r.seed)), str(r.source_id), str(int(r.origin_age)),
                 _dollars(r["mean"] / r.n_leads, 3), _dollars(r.true_margin / r.n_leads, 3),
                 _dollars(r.absolute_error_per_lead, 3), "covered" if r.coverage95 else "missed"]
                for _, r in worst.iterrows()]
        lines += _markdown_table(["Scenario", "Seed", "Source", "Origin", "Mean / lead", "Truth / lead",
                                  "Absolute error / lead", "95% interval"], rows)
        reliability = _read(output, "reliability.csv", ["probability_bin", "mean_prediction", "realized_fraction", "forecasts"])
        lines += ["", "## Descriptive reliability", "",
                  "Counts below are correlated forecast rows, not independent worlds; empty bins remain empty. No IID binomial uncertainty is asserted.", ""]
        rows = [[str(r.probability_bin), _pct(r.mean_prediction) if np.isfinite(r.mean_prediction) else "N/A",
                 _pct(r.realized_fraction) if np.isfinite(r.realized_fraction) else "N/A", str(int(r.forecasts))]
                for _, r in reliability.iterrows()]
        lines += _markdown_table(["Probability bin", "Mean predicted profitability", "Realized profitability", "Forecast count"], rows)

    baseline_path = output / "baseline_scores.csv"
    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path)
        if not baseline.empty and {"horizon", "model", "origin_age", "absolute_error_per_lead", "world_id"}.issubset(baseline.columns):
            comparable = baseline.loc[baseline.horizon.eq(h), ["world_id", "model", "origin_age", "absolute_error_per_lead"]]
            bayes = data.loc[data.horizon.eq(h), ["world_id", "model", "origin_age", "absolute_error_per_lead"]]
            comparison = pd.concat([comparable, bayes], ignore_index=True)
            # A point baseline has no predictive CRPS/coverage: do not award it
            # fabricated probabilistic metrics, or compare incompatible scores.
            mae = comparison.groupby(["world_id", "model", "origin_age"], observed=True).absolute_error_per_lead.mean().groupby(["model", "origin_age"]).mean()
            lines += ["", f"## Point accuracy comparison at day {h}", "",
                      "Mean absolute point error per lead, equal-weighting world means. Bayesian uses the predictive mean; the two baselines are point forecasts. This does not establish superiority, and point baselines do not have predictive-interval or CRPS scores.", ""]
            lines += _markdown_table(["Model", "Origin", "Mean absolute error / lead"],
                                     [[str(model), str(int(origin)), _dollars(value, 4)] for (model, origin), value in mae.items()])

    lines += ["", "## Reading and limitations", "",
              "- The terminal policy is finite: sends stop at day 365; delayed responses/accounting can settle through day 425. “Never” means no first-payback crossing within that policy, not an infinite-lifetime assertion",
              "- An unconditional payback CDF ends at one minus the never-under-policy mass. Do not normalize away unsuccessful draws or report a payback-age percentile that the CDF never reaches",
              "- Posterior-predictive bands account for modeled uncertainty only. Unmodeled calendar shocks, tail changes, attribution revisions, selection and value-dependent reporting can break them",
              "- Sources are fictional labels, not randomized treatment arms. These forecasts provide no causal A/B evidence and are not proof that acquisition or send-stopping decisions improve outcomes",
              "- The synthetic suite has few independent worlds. Repeated origins, horizons and source cohorts within a world are correlated; large row counts do not make a large independent sample",
              "- Truth enters scoring and these evaluation plots only after forecasting. The as-of reported ledger is kept distinct from complete economic truth",
              "", "## Figures", ""]
    for name in dict.fromkeys(Path(p).stem for p in records):
        lines += [f"- [{name.replace('_', ' ')}](figures/{name}.png) · [editable SVG](figures/{name}.svg)"]
    lines += ["", "All plotted figures have an explicit synthetic/fictional label. CSVs remain the numerical source of record; this report can be rebuilt without refitting."]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def render(output: Path, mode: str) -> list[str]:
    """Write PNG+SVG figures under output/figures and actual-CSV RESULTS.md.

    ``mode`` is either ``demo`` (one synthetic world) or ``calibration``.
    Quick runs with only origins 3 and 60 are handled without fabricated ages.
    Returns the rendered figure paths; does not modify source CSVs.
    """
    output = Path(output)
    if mode not in {"demo", "calibration"}:
        raise ValueError("mode must be 'demo' or 'calibration'")
    _style()
    if mode == "demo":
        data = _read(output, "summary.csv", ["source_id", "origin_age", "horizon", "n_leads"])
        records = _demo_figures(output, output / "figures", data)
    else:
        data = _read(output, "scores.csv", ["source_id", "world_id", "scenario", "seed", "cohort_id", "origin_age", "horizon", "n_leads",
                                                 "mean", "true_margin", "crps_per_lead", "absolute_error_per_lead", "coverage80", "coverage95", *QUANTILES])
        records = _calibration_figures(output, output / "figures", data)
    _write_results(output, mode, data, records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Directory containing runner CSV artifacts")
    parser.add_argument("--mode", choices=["demo", "calibration"], help="Defaults to run_metadata.json mode, then demo")
    args = parser.parse_args()
    mode = args.mode or _metadata(args.output).get("mode", "demo")
    records = render(args.output, mode)
    print(f"Rendered {len(records) // 2} figures (PNG + SVG) and {args.output / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
