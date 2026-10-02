"""Render source-bound figures from sealed synthetic dynamic-lab outputs.

This module never simulates or refits. The output table alongside each figure
contains every plotted row, including derived quantities. PNG and SVG variants
are rendered separately so that mobile panels remain legible.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import html
import io
import json
import os
from pathlib import Path
import re
import shutil
import textwrap
from typing import Callable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dynamic-cohort-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, PercentFormatter, MaxNLocator
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
LAB = Path(__file__).resolve().parent
BLUE = "#2d679d"
AMBER = "#b17721"
INK = "#283641"
MUTED = "#687580"
PALE_BLUE = "#dce7f0"
PALE_AMBER = "#ede2cf"
GRID = "#e4e9ed"
MODEL_LABELS = {
    "dynamic": "Dynamic hierarchy",
    "dynamic_hierarchical": "Dynamic hierarchy",
    "static": "Static hierarchy",
    "static_hierarchical": "Static hierarchy",
    "source_only": "Source only",
    "target_only": "Target only",
    "global": "Complete pooling",
    "global_complete_pooling": "Complete pooling",
    "global_pooling": "Global pooling",
    "source_pooling": "Source pooling",
    "static_hierarchy": "Static hierarchy",
    "dynamic_hierarchy": "Dynamic hierarchy",
}


@dataclass(frozen=True)
class Source:
    path: str
    sha256: str
    rows: int
    origin_path: str | None = None

    def as_dict(self):
        record = {"path": self.path, "sha256": self.sha256, "rows": self.rows}
        if self.origin_path is not None: record["origin_path"] = self.origin_path
        return record


class Renderer:
    def __init__(self, artifacts: Path, output: Path):
        self.artifacts = artifacts.resolve()
        self.output = output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "tables").mkdir(exist_ok=True)
        self.records = []
        plt.rcParams.update({
            "font.family": "DejaVu Sans", "font.size": 11.5,
            "axes.labelsize": 11.5, "axes.titlesize": 12.5,
            "axes.edgecolor": "#b8c2ca", "axes.labelcolor": INK,
            "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
            "axes.spines.top": False, "axes.spines.right": False,
            "figure.facecolor": "white", "axes.facecolor": "white",
            "savefig.facecolor": "white", "svg.fonttype": "none",
        })

    def read(self, name: str, required=()) -> tuple[pd.DataFrame, Source]:
        path = self.artifacts / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing sealed output: {path}")
        payload = path.read_bytes()
        data = pd.read_csv(io.BytesIO(payload))
        missing = sorted(set(required) - set(data.columns))
        if missing:
            raise ValueError(f"{name} lacks required columns: {missing}")
        return data, Source(str(path.relative_to(ROOT)), hashlib.sha256(payload).hexdigest(), len(data))

    def emit(self, name: str, title: str, caption: str, alt: str,
             sources: list[Source], selection: str, rows: pd.DataFrame,
             plot: Callable[[bool], plt.Figure], interval_meaning: str,
             denominator: str):
        if rows.empty:
            raise ValueError(f"Cannot render {name}: no source rows selected")
        table = self.output / "tables" / f"{name}.csv"
        rows.to_csv(table, index=False)
        files = []
        for mobile in (False, True):
            stem = name + ("-mobile" if mobile else "")
            fig = plot(mobile)
            for extension in ("png", "svg"):
                path = self.output / f"{stem}.{extension}"
                fig.savefig(path, dpi=190, bbox_inches="tight", pad_inches=.14)
                if extension == "svg":
                    self._accessible_svg(path, title, alt + " " + caption)
                files.append(str(path.relative_to(ROOT)))
            plt.close(fig)
        self.records.append({
            "id": name, "title": title, "caption": caption, "alt": alt,
            "synthetic_only": True, "files": files,
            "file_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in files},
            "table": str(table.relative_to(ROOT)),
            "table_sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
            "source_files": [s.as_dict() for s in sources],
            "selection": selection, "interval_meaning": interval_meaning,
            "denominator": denominator,
        })

    @staticmethod
    def _accessible_svg(path: Path, title: str, description: str):
        text = path.read_text()
        start = text.index("<svg ")
        identifier = path.stem
        text = text[:start] + text[start:].replace(
            "<svg ", f'<svg role="img" aria-labelledby="{identifier}-title {identifier}-desc" ', 1)
        end = text.index(">", start) + 1
        text = text[:end] + (
            f'<title id="{identifier}-title">{html.escape(title)}</title>'
            f'<desc id="{identifier}-desc">{html.escape(description)}</desc>'
        ) + text[end:]
        path.write_text(text)

    def finish(self):
        run_manifest_path = self.artifacts / "run_manifest.json"
        sealed_outputs = None
        if run_manifest_path.is_file():
            sealed_outputs = json.loads(run_manifest_path.read_text()).get("output_csv_sha256", {})
        for chart in self.records:
            for source in chart["source_files"]:
                current = ROOT / source["path"]
                if not current.is_file() or hashlib.sha256(current.read_bytes()).hexdigest() != source["sha256"]:
                    raise ValueError(f"Source changed during rendering: {source['path']}; rerender after reseal")
                if sealed_outputs is not None and current.parent == self.artifacts:
                    if sealed_outputs.get(current.name) != source["sha256"]:
                        raise ValueError(f"CSV differs from the final run manifest: {current.name}; await reseal")
        manifest = {"synthetic_only": True, "renderer_refits_model": False,
                    "input_versions_verified_at_finish": True,
                    "charts": self.records}
        if run_manifest_path.is_file():
            manifest["run_manifest_path"] = str(run_manifest_path.relative_to(ROOT))
            manifest["run_manifest_sha256"] = hashlib.sha256(run_manifest_path.read_bytes()).hexdigest()
        (self.output / "chart_provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
        sections = ["# Synthetic dynamic-cohort figures", "",
                    "All marks come from the sealed fitted or evaluator outputs named below. "
                    "These are synthetic demonstrations, not observed commercial outcomes.", ""]
        for row in self.records:
            sections.extend([f"## {row['title']}", "", row["caption"], "",
                             f"Alt text: {row['alt']}", "",
                             f"Intervals: {row['interval_meaning']}", "",
                             f"Denominator: {row['denominator']}", "",
                             f"Numeric table: [{Path(row['table']).name}](tables/{Path(row['table']).name})", ""])
            for companion in row.get("companion_tables", []):
                sections.extend([f"Companion numeric table: [{Path(companion['path']).name}](tables/{Path(companion['path']).name})", ""])
        (self.output / "CAPTIONS.md").write_text("\n".join(sections))
        mobile_checks = []
        for svg in sorted(self.output.glob("*-mobile.svg")):
            contents = svg.read_text()
            width = float(re.search(r'viewBox="[\d.]+ [\d.]+ ([\d.]+)', contents).group(1))
            sizes = [float(v) for v in re.findall(r'font-size:\s*([\d.]+)px', contents)]
            # Matplotlib versions use either longhand font-size or font shorthand.
            sizes += [float(v) for v in re.findall(r'font:\s*(?:[\w-]+\s+)*?([\d.]+)px', contents)]
            if not sizes: raise ValueError(f"Cannot verify mobile text sizes in {svg.name}")
            minimum = min(sizes) * 390 / width
            mobile_checks.append({"file": svg.name, "viewbox_width": width,
                                  "min_text_svg_units": min(sizes), "viewport_css_px": 390,
                                  "min_effective_text_css_px": minimum, "pass": minimum >= 12})
        (self.output / "mobile_text_checks.json").write_text(json.dumps(mobile_checks, indent=2) + "\n")
        failed = [r["file"] for r in mobile_checks if not r["pass"]]
        if failed: raise ValueError(f"Mobile text below 12px at390px: {failed}")
        print(f"Rendered {len(self.records)} source-bound figures in {self.output}")


def setup(ax, grid_axis="y"):
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=GRID, linewidth=.8)
    ax.tick_params(length=0, pad=7)


def frame(title, nrows=1, mobile=False, heights=None):
    width = 4.5 if mobile else 7.3
    height = (6.0 if mobile else 4.4) * nrows + .85
    fig, axes = plt.subplots(nrows, 1, figsize=(width, height), squeeze=False,
                             gridspec_kw={"height_ratios": heights or [1] * nrows})
    top = min(.82, .70 + .06 * (nrows - 2)) if mobile else min(.85, .76 + .06 * (nrows - 2))
    fig.subplots_adjust(left=.20 if mobile else .15, right=.97, top=top,
                        bottom=.09 if nrows == 1 else .05, hspace=1.1 if mobile else .70)
    fig.text(.02, .985, "SYNTHETIC EXTENSION · APPROXIMATE POSTERIOR", fontsize=12.4 if mobile else 9.1, color=MUTED, va="top")
    fig.text(.02, .955, textwrap.fill(title, 42 if mobile else 75),
             fontsize=13 if mobile else 15, color=INK, weight="bold", va="top")
    for ax in axes[:, 0]:
        ax._mobile = mobile
        if mobile:
            ax.tick_params(labelsize=12.4)
            ax.xaxis.label.set_fontsize(12.4)
            ax.yaxis.label.set_fontsize(12.4)
        setup(ax)
    return fig, list(axes[:, 0])


def panel_title(ax, title):
    mobile = getattr(ax, "_mobile", False)
    ax.set_title(textwrap.fill(title, 31 if mobile else 65), loc="left", pad=11,
                 y=1.65 if mobile else 1.29,
                 fontsize=12.4 if mobile else 11.3)


def calendar_axis(ax, values):
    low, high = float(min(values)), float(max(values))
    padding = .016 * (high - low)
    ax.set_xlim(low - padding, high + padding)
    interior = np.linspace(low, high, 5)[1:-1]
    ax.set_xticks(np.unique(np.r_[low, np.rint(interior), high]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.set_xlabel("Calendar day since launch")


def dollars(ax, axis="y", decimals=2):
    formatter = FuncFormatter(lambda value, _: f"${value:,.{decimals}f}")
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(formatter)


def legend(ax, mobile=False, ncol=2):
    handles, labels = ax.get_legend_handles_labels()
    if mobile: labels = [textwrap.fill(label, 25) for label in labels]
    ax.legend(handles, labels, frameon=False, fontsize=12.4 if mobile else 9.3, ncol=1 if mobile else ncol,
              loc="lower left", bbox_to_anchor=(0, 1.01), borderaxespad=0,
              handlelength=2.2, columnspacing=1.2)


def finite(data, fields):
    for field in fields:
        if not np.isfinite(data[field].to_numpy(dtype=float)).all():
            raise ValueError(f"Nonfinite plotted values in {field}")


def require_unique(data, keys, description):
    if data.duplicated(keys).any():
        raise ValueError(f"Ambiguous {description}: duplicate rows for {keys}")


def interval_band(ax, data, x, center="median", color=BLUE, label="Posterior median"):
    finite(data, [x, center, "q025", "q975"])
    if ((data.q025 > data[center]) | (data[center] > data.q975)).any():
        raise ValueError("Median must lie inside the 95% interval")
    ax.fill_between(data[x], data.q025, data.q975, color=color, alpha=.17)
    ax.plot(data[x], data[center], color=color, lw=2, label=label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=LAB / "artifacts")
    parser.add_argument("--output", type=Path, default=LAB / "artifacts" / "figures")
    args = parser.parse_args()
    renderer = Renderer(args.artifacts, args.output)
    render_all(renderer)
    renderer.finish()


def demo_slice(data, world=None, scenario=None, select_policy=True):
    """Select one declared illustration world, never average independent worlds."""
    if "world" not in data or "scenario" not in data:
        raise ValueError("Illustration outputs must identify world and scenario")
    scenarios = list(data.scenario.drop_duplicates())
    if scenario is None:
        scenario = "human_decline_bot_rise" if "human_decline_bot_rise" in scenarios else next(
            (s for s in scenarios if any(k in str(s).lower() for k in ("squeeze", "deterior", "stress", "decline"))), scenarios[0])
    part = data[data.scenario == scenario]
    if world is None:
        world = sorted(part.world.unique())[0]
    part = part[part.world == world].copy()
    if select_policy: part = policy_slice(part)
    if part.empty:
        raise ValueError(f"No illustration rows: scenario={scenario}; world={world}")
    return part, world, scenario


def policy_slice(data):
    if "future_scenario" not in data: return data
    policies = list(data.future_scenario.drop_duplicates())
    if len(policies) <= 1: return data
    selected = next((p for p in ["hold_current", "continue_recent_trend", "fixed_price"] if p in policies), None)
    if selected is None: raise ValueError(f"No declared policy selection among {policies}")
    return data[data.future_scenario == selected].copy()


def known_source(data):
    candidates = [s for s in data.source_id.drop_duplicates()
                  if not any(k in str(s).lower() for k in ("cheap", "new", "good", "bad"))]
    if not candidates:
        candidates = list(data.source_id.drop_duplicates())
    return next((s for s in candidates if str(s).lower() in ("middle", "medium", "standard", "established")),
                sorted(candidates)[0])


def dynamic_model(data):
    candidates = list(data.model.drop_duplicates())
    result = next((m for m in candidates if "dynamic" in str(m).lower()), None)
    if result is None:
        raise ValueError(f"No fitted dynamic-model rows among {candidates}")
    return result


def add_panel(data, panel):
    result = data.copy()
    result.insert(0, "panel", panel)
    return result


def economics(renderer, world=None, scenario=None):
    data, src = renderer.read("calendar_economics.csv", ["world", "scenario", "calendar_day",
         "price_per_send", "human_cpc", "cpl_quote", "new_purchases", "acquisition_total"])
    complete = data
    data, world, scenario = demo_slice(data, world, scenario)
    source_id = known_source(data) if "source_id" in data else None
    if source_id is not None:
        data = data[data.source_id == source_id].copy()
    data = data.sort_values("calendar_day")
    require_unique(data, ["calendar_day"], "calendar economics")
    finite(data, ["calendar_day", "price_per_send", "human_cpc", "cpl_quote", "new_purchases", "acquisition_total"])
    purchases = data[data.new_purchases > 0].copy()
    purchases["paid_cpl"] = purchases.acquisition_total / purchases.new_purchases
    fixed = pd.DataFrame()
    if "fixed_price" in complete.scenario.unique() and scenario != "fixed_price":
        fixed, _, _ = demo_slice(complete, scenario="fixed_price")
        if source_id is not None: fixed = fixed[fixed.source_id == source_id].copy()
        fixed = fixed.sort_values("calendar_day")
        require_unique(fixed, ["calendar_day"], "fixed-price reference")
    table = pd.concat([add_panel(data, "commercial_inputs"), add_panel(purchases, "actual_purchase_costs"),
                       add_panel(fixed, "fixed_price_reference")], ignore_index=True)
    title = "Calendar prices and arriving purchases"
    def plot(mobile):
        fig, (cpc, cpl, arrivals) = frame(title, 3, mobile)
        panel_title(cpc, "Underlying human CPC: fixed versus changing")
        cpc.plot(data.calendar_day, data.human_cpc, color=AMBER, ls="--", lw=2.2, label="Changing net-CPC opportunity (truth)")
        if not fixed.empty:
            cpc.plot(fixed.calendar_day, fixed.human_cpc, color=INK, ls=":", lw=1.8, label="Fixed-price opportunity (truth)")
        cpc.set_ylabel("USD / qualified human event"); cpc.set_ylim(bottom=0); dollars(cpc)
        legend(cpc, mobile)
        panel_title(cpl, "Current acquisition quote and paid batch price")
        cpl.plot(data.calendar_day, data.cpl_quote, color=BLUE, lw=2.1, label="Quote for a new purchase")
        if not fixed.empty:
            cpl.plot(fixed.calendar_day, fixed.cpl_quote, color=INK, ls=":", lw=1.7, label="Fixed-scenario acquisition quote")
        cpl.plot(purchases.calendar_day, purchases.paid_cpl, color=AMBER, marker="o", ms=5,
                 lw=0, label="Immutable cost / acquired lead")
        cpl.set_ylabel("USD / acquired lead"); cpl.set_ylim(bottom=0); dollars(cpl); legend(cpl, mobile)
        panel_title(arrivals, "New cohorts continue to arrive")
        arrivals.bar(purchases.calendar_day, purchases.new_purchases, width=2.7, color=BLUE)
        arrivals.set_ylabel("Leads purchased"); arrivals.yaxis.set_major_locator(MaxNLocator(integer=True))
        for ax in (cpc, cpl, arrivals): calendar_axis(ax, data.calendar_day)
        return fig
    renderer.emit("01-calendar-economics", title,
        f"The synthetic rich illustration ({world}, {scenario}) and fixed-price reference use the same demo seed; " +
        (f"established source {source_id}. " if source_id is not None else "") +
        "The net-human CPC opportunity is evaluator-only truth, not an observed contract rate supplied to inference; "
        "the model learns it from mature settled monetary marks. Send cost and the current acquisition quote are observed inputs. "
        "Purchase markers divide the batch's recorded acquisition cost by its purchased leads; "
        "a later quote applies to new purchases and never rewrites old acquisition costs.",
        "Three calendar panels compare fixed and changing evaluator-only human CPC, acquisition quotes with actual purchase prices, and weekly arriving cohorts.",
        [src], f"world={world}; scenario={scenario}; source_id={source_id}; plus fixed_price same demo seed; all calendar days and positive purchase rows", table, plot,
        "No uncertainty interval: net-CPC is evaluator-only truth; quotes, send prices and immutable purchases are simulated known inputs.",
        "Revenue per qualified human event; send cost per eligible send; acquisition quote and cost per purchased lead.")
    return world, scenario


def measurement(renderer, world, scenario):
    data, src = renderer.read("measurement.csv", ["world", "scenario", "calendar_day", "true_human_events",
             "reported_events", "audited_bot_fraction", "audit_n"])
    data, _, _ = demo_slice(data, world, scenario)
    if "origin_day" in data:
        origin = data.origin_day.max(); data = data[data.origin_day == origin].copy()
    else: origin = None
    if "source_id" in data:
        source_id = known_source(data); data = data[data.source_id == source_id].copy()
    else: source_id = None
    data = data.sort_values("calendar_day")
    require_unique(data, ["calendar_day"], "measurement evidence")
    data["audit_fraction_plotted"] = data.audited_bot_fraction.where(data.audit_n > 0)
    title = "Human response and audited bot evidence"
    fit_fields = ["fitted_human_median", "fitted_human_q025", "fitted_human_q975"]
    fit = all(f in data for f in fit_fields)
    source_scope = f"source {source_id}" if source_id is not None else "all active sources and cohorts"
    information_scope = f"as-of calendar origin {int(origin)}" if origin is not None else "information available on each plotted calendar day"
    def plot(mobile):
        fig, (human, bots) = frame(title, 2, mobile)
        panel_title(human, "Reported traffic is not human response")
        human.plot(data.calendar_day, data.reported_events, color=AMBER, ls="--", lw=1.8, label="Events by report day, including bots")
        human.plot(data.calendar_day, data.true_human_events, color=INK, ls=":", lw=1.7, label="Human truth by occurrence day")
        if fit:
            valid = data.dropna(subset=fit_fields)
            human.fill_between(valid.calendar_day, valid.fitted_human_q025, valid.fitted_human_q975, color=BLUE, alpha=.17)
            human.plot(valid.calendar_day, valid.fitted_human_median, color=BLUE, lw=2.1, label="Fitted human median + 95% band")
        human.set_ylabel("Events / calendar day"); human.set_ylim(bottom=0); legend(human, mobile)
        panel_title(bots, "Audited bots among audited reported events")
        valid = data[data.audit_n > 0].dropna(subset=["audited_bot_fraction"])
        bots.plot(valid.calendar_day, valid.audited_bot_fraction, color=BLUE, lw=1.7, marker="o", ms=3.5)
        if "audit_bot_lower95" in valid and "audit_bot_upper95" in valid:
            bots.fill_between(valid.calendar_day, valid.audit_bot_lower95, valid.audit_bot_upper95, color=BLUE, alpha=.17)
        bots.set_ylim(0, 1); bots.yaxis.set_major_formatter(PercentFormatter(1)); bots.set_ylabel("Fraction of audited events")
        for ax in (human, bots): calendar_axis(ax, data.calendar_day)
        return fig
    renderer.emit("02-human-bot-evidence", title,
        f"One synthetic world ({world}, {scenario}), aggregating {source_scope}; {information_scope}. "
        "Raw reported activity includes bots and uses report-calendar day, with a two-day report lag. "
        "The dotted true-human series uses occurrence-calendar day and is evaluator-only, "
        "not a fitting input. Audit fractions use trailing14-day audit reports available on that day, "
        "Beta-smoothed and corrected for the known sensitivity/specificity; audit_n is their denominator. "
        "Rows with no available audit are omitted from the fraction plot and marked missing in audit_fraction_plotted. "
        "Audit instrument error and representative-selection assumptions remain part of the approximate model.",
        "Reported-event counts are compared with evaluator-only true human events, above the measured fraction of audited events labeled bot.",
        [src], f"world={world}; scenario={scenario}; {source_scope}; {information_scope}; event-count lines unsmoothed; audit fractions use available trailing14-day reports", data, plot,
        "95% fitted human interval where provided; audit interval only if explicit audit bounds exist. No invented bounds or smoothed interval endpoints.",
        "Daily response events, which may repeat per person; bot fraction uses audit_n reported events, not acquired leads.")


def calendar_states(renderer, world, scenario):
    data, src = renderer.read("calendar_states.csv", ["world", "scenario", "origin_day", "calendar_day",
               "state", "true_value", "median", "q025", "q975"])
    data, _, _ = demo_slice(data, world, scenario)
    origin = data.origin_day.max(); data = data[data.origin_day == origin].copy()
    require_unique(data, ["state", "calendar_day"], "calendar state")
    lo = min(data.q025.min(), data.true_value.min(), 0)
    hi = max(data.q975.max(), data.true_value.max(), 0)
    padding = .04 * (hi - lo)
    title = "One shared response and payout calendar"
    def plot(mobile):
        fig, axes = frame(title, 2, mobile)
        for ax, state, label in zip(axes, ["response", "value"], ["Human-response state", "Net-payment opportunity state"]):
            part = data[data.state == state].sort_values("calendar_day")
            if part.empty: raise ValueError(f"Missing fitted calendar state {state}")
            panel_title(ax, label)
            interval_band(ax, part, "calendar_day", label="Median + 95% state band")
            ax.plot(part.calendar_day, part.true_value, color=AMBER, ls="--", lw=1.8, label="Evaluator-only state")
            ax.axvline(origin, color=INK, ls=":", lw=1, label=f"Fitting cutoff: day {int(origin)}")
            ax.axhline(0, color=MUTED, lw=.7)
            ax.set_ylim(lo - padding, hi + padding)
            ax.set_ylabel("Log multiplier (anchor = 0)"); calendar_axis(ax, part.calendar_day); legend(ax, mobile)
        return fig
    renderer.emit("03-shared-calendar-states", title,
        f"One synthetic world ({world}, {scenario}); refit at calendar day {origin}. "
        "This bounded fixture uses same-day send/response cells; it does not fit a response-delay process. "
        "Report, audit and mature monetary visibility clocks remain separate. "
        "Within each posterior draw, one response path and one value path are shared across all cohorts; "
        "the two processes are conditionally independent and their covariance is not learned. "
        "Fixed age curves, anchored states and exchangeable cohort effects constrain the age–period–cohort decomposition; "
        "these drivers are assumption-dependent. The dashed truth is evaluator-only.",
        "Two panels show fitted shared response and payment calendar states with 95% intervals and evaluator-only states, separated by the last fitting origin.",
        [src], f"world={world}; scenario={scenario}; origin_day={origin}; states=response,value", data, plot,
        "95% approximate posterior state interval through the fitting cutoff and conditional future-state predictive band afterward; not a contribution interval or empirical calibration guarantee.",
        "Log multiplicative state relative to the anchored reference date; one shared state per calendar day.")


def cohort_forecasts(renderer, world, scenario):
    predictions, pred_src = renderer.read("cohort_predictions.csv", ["world", "scenario", "model", "origin_day", "source_id",
                  "cohort_id", "birth_day", "target_age", "n_leads", "median", "q025", "q975", "true_margin"])
    all_predictions = predictions.copy()
    predictions, _, _ = demo_slice(predictions, world, scenario)
    source_id = known_source(predictions)
    predictions = predictions[(predictions.source_id == source_id) & (predictions.target_age == 180)].copy()
    predictions["origin_age"] = predictions.origin_day - predictions.birth_day
    if "target_role" in predictions:
        newest = predictions[predictions.target_role == "newest"].copy()
    else:
        newest = predictions[predictions.origin_age == 7].copy()
    if newest.empty:
        raise ValueError("Common-age figure needs explicit newest/day7 forecast rows")
    if newest.origin_age.nunique() != 1:
        raise ValueError("Common-age figure cannot mix target ages at forecast")
    origin_age = int(newest.origin_age.iloc[0]); dyn = dynamic_model(newest)
    compare = next((m for m in newest.model.unique() if "static" in str(m).lower()), None)
    model_ids = [dyn] + ([compare] if compare is not None else [])
    newest = newest[newest.model.isin(model_ids)].copy()
    for key in ["median", "q025", "q975", "true_margin"]:
        newest[key + "_per_lead"] = newest[key] / newest.n_leads
    require_unique(newest, ["model", "origin_day"], "common-age cohort predictions")
    cdf, cdf_src = renderer.read("payback_cdf.csv", ["world", "scenario", "model", "origin_day", "source_id", "cohort_id", "age", "cdf"])
    cdf, _, _ = demo_slice(cdf, world, scenario, select_policy=False)
    last = newest[newest.model == dyn].sort_values("origin_day").iloc[-1]
    cdf = cdf[(cdf.source_id == source_id) & (cdf.cohort_id == last.cohort_id) & (cdf.origin_day == last.origin_day)
               & (cdf.model == dyn)].copy()
    if cdf.empty or cdf.age.max() != 425:
        raise ValueError("First-payback CDF must include the selected target through age425")
    denominator_keys = [key for key in ["world", "scenario", "model", "origin_day", "source_id", "cohort_id", "future_scenario"] if key in cdf]
    if "draws" not in all_predictions:
        raise ValueError("Payback figure needs the actual draw denominator from sealed summary rows")
    denominators = all_predictions[denominator_keys + ["draws"]].drop_duplicates()
    require_unique(denominators, denominator_keys, "all-path payback denominator")
    cdf = cdf.merge(denominators, on=denominator_keys, how="left", validate="many_to_one")
    if cdf.draws.isna().any(): raise ValueError("Payback denominator missing for selected paths")
    crossing_counts = cdf.cdf * cdf.draws
    if not np.allclose(crossing_counts, np.rint(crossing_counts), atol=1e-8, rtol=0):
        raise ValueError("CDF does not correspond to the sealed predictive draw denominator")
    cdf["cumulative_first_crossing_paths"] = np.rint(crossing_counts).astype(int)
    for model, part in cdf.groupby("future_scenario" if "future_scenario" in cdf else "model"):
        part = part.sort_values("age")
        if (part.cdf.diff().dropna() < -1e-10).any() or not part.cdf.between(0, 1).all():
            raise ValueError(f"Invalid unconditional payback CDF for {model}")
        if "p_no_payback425" in part and not np.isclose(part.cdf.iloc[-1] + part.p_no_payback425.iloc[-1], 1):
            raise ValueError(f"Payback CDF must retain noncrossing paths for {model}")
    trajectories, trajectory_src = renderer.read("trajectories.csv", ["world", "scenario", "model", "origin_day", "source_id", "cohort_id", "age",
              "n_leads", "median", "q025", "q975"])
    if "truth" not in trajectories and "true_margin" in trajectories:
        trajectories = trajectories.rename(columns={"true_margin": "truth"})
    if "truth" not in trajectories: raise ValueError("Contribution path needs evaluator-only true margin")
    trajectories, _, _ = demo_slice(trajectories, world, scenario, select_policy=False)
    trajectory = trajectories[(trajectories.source_id == source_id) & (trajectories.cohort_id == last.cohort_id)
                       & (trajectories.origin_day == last.origin_day) & (trajectories.model == dyn)].copy().sort_values("age")
    if trajectory.empty: raise ValueError("Missing selected dynamic contribution trajectory")
    for key in ["median", "q025", "q975", "truth"]:
        trajectory[key + "_per_lead"] = trajectory[key] / trajectory.n_leads
    available_policies = list(trajectory.future_scenario.drop_duplicates()) if "future_scenario" in trajectory else ["hold_current"]
    policies = [p for p in ["hold_current", "continue_recent_trend", "fixed_price"] if p in available_policies]
    policies += [p for p in available_policies if p not in policies]
    policy_styles = {"hold_current": (BLUE, "-", "Current-state forecast"),
                     "continue_recent_trend": (AMBER, "--", "Recent-state extrapolation"),
                     "fixed_price": (INK, ":", "Early-reference fixed CPC")}
    table = pd.concat([add_panel(newest, "same_age_day180_forecasts"), add_panel(trajectory, "selected_contribution_path"),
                       add_panel(cdf, "unconditional_first_payback")], ignore_index=True)
    title = "Later cohorts face the later calendar"
    def plot(mobile):
        fig, (cross, path, payback) = frame(title, 3, mobile)
        panel_title(cross, f"New cohorts at age {origin_age}: day-180 contribution, 95% intervals")
        colors = {dyn: BLUE, compare: AMBER}
        for idx, model in enumerate(model_ids):
            part = newest[newest.model == model].sort_values("origin_day")
            x = part.origin_day + (idx - .5) * 1.2
            cross.vlines(x, part.q025_per_lead, part.q975_per_lead, color=colors[model], lw=3, alpha=.62)
            cross.plot(x, part.median_per_lead, "o" if idx == 0 else "s", color=colors[model], ms=5.5,
                       label=MODEL_LABELS.get(model, model))
        actual = newest[newest.model == dyn].sort_values("origin_day")
        cross.plot(actual.origin_day, actual.true_margin_per_lead, color=INK, ls=":", marker="x", lw=1.3, label="Outcome")
        cross.axhline(0, color=INK, lw=.8); cross.set_ylabel("USD / acquired lead"); dollars(cross); legend(cross, mobile)
        cross.set_xticks(sorted(newest.origin_day.unique())); cross.set_xlabel("Forecast origin (calendar day)")
        panel_title(path, f"Newest cohort at calendar day {int(last.origin_day)}")
        main_path = trajectory[trajectory.future_scenario == "hold_current"] if "future_scenario" in trajectory else trajectory
        path.fill_between(main_path.age, main_path.q025_per_lead, main_path.q975_per_lead, color=PALE_BLUE)
        for policy in policies:
            part = trajectory[trajectory.future_scenario == policy] if "future_scenario" in trajectory else trajectory
            color, style, policy_label = policy_styles.get(policy, (AMBER, "--", str(policy)))
            path.plot(part.age, part.median_per_lead, color=color, ls=style, lw=2, label=policy_label)
        path.plot(main_path.age, main_path.truth_per_lead, color="#7f8991", ls="-.", lw=1.5, label="Evaluator-only outcome")
        path.axhline(0, color=INK, lw=.8); path.axvline(origin_age, color=MUTED, ls=":", lw=.8)
        path.set_ylabel("USD / acquired lead"); dollars(path); legend(path, mobile)
        panel_title(payback, "First payback, including paths that never cross")
        for policy in policies:
            part = cdf[cdf.future_scenario == policy].sort_values("age") if "future_scenario" in cdf else cdf.sort_values("age")
            if part.empty: continue
            color, style, policy_label = policy_styles.get(policy, (AMBER, "--", str(policy)))
            payback.plot(part.age, part.cdf, color=color, lw=2.1, ls=style, label=policy_label)
        payback.set_ylim(0, 1.04); payback.set_ylabel("Share of all predictive paths")
        payback.yaxis.set_major_formatter(PercentFormatter(1)); legend(payback, mobile)
        for ax in (path, payback):
            ax.set_xlim(0, 425); ax.set_xticks([0, 90, 180, 300, 425]); ax.set_xlabel("Cohort age (days)")
        return fig
    renderer.emit("04-cohort-contribution-payback", title,
        f"One synthetic world ({world}, {scenario}), source {source_id}. Each top-panel forecast uses a new cohort at age {origin_age} "
        "and the same target age180; its actual birth date determines future calendar exposure. "
        f"The lower panels select cohort {last.cohort_id} at calendar origin {int(last.origin_day)} and compare "
        "conditional future states: no deterministic drift in the fitted response/value LOG states, with future innovations; "
        "extrapolate each posterior draw's recent slope in BOTH response and value, with future innovations; "
        "or fix the future CPC state to the learned day0 reference while retaining the response process. "
        "Sampled recent slopes may point up or down; these assumptions do not reveal future truth. "
        "The pale band belongs to hold_current only; the other lines are separate conditional predictive medians, not interval bounds. "
        "The first-payback CDF retains every predictive path, including paths without a crossing by age425. "
        "First crossing, positive contribution at age180 and continued positivity are separate events.",
        "At a common cohort age, day180 contribution forecasts are compared across calendar origins, followed by one selected contribution path and an unconditional first-payback curve ending at age425.",
        [pred_src, trajectory_src, cdf_src], f"world={world}; scenario={scenario}; source={source_id}; newest age={origin_age}; target180; top=hold_current at all cutoffs; selected cohort={last.cohort_id}, origin={last.origin_day}; lower=all future policies",
        table, plot, "95% approximate posterior-predictive contribution intervals; first-payback probability over all draws through age425.",
        f"Contribution / original acquired leads (n_leads from each row); first-payback CDF / all {int(last.draws)} predictive paths, never only crossing paths. Draw denominator and cumulative crossing count are exported per CDF row.")


def source_transfer(renderer, world, scenario):
    data, src = renderer.read("transfer.csv", ["world", "scenario", "stage", "new_source_kind", "origin_day", "source_id",
                      "target_age", "n_leads", "median", "q025", "q975", "true_margin"])
    # New-source good/bad transfer lives in two paired scenario worlds, rather
    # than the rich calendar illustration selected for figures one through four.
    data = policy_slice(data)
    selected = []
    for kind, part in data.groupby("new_source_kind", sort=False):
        selected.append(part[part.world == sorted(part.world.unique())[0]])
    data = pd.concat(selected, ignore_index=True)
    if "model" in data:
        model = dynamic_model(data); data = data[data.model == model].copy()
    data = data[data.target_age == 180].copy()
    for key in ["median", "q025", "q975", "true_margin"]:
        data[key + "_per_lead"] = data[key] / data.n_leads
    require_unique(data, ["new_source_kind", "stage", "origin_day"], "new-source transfer")
    if set(data.stage.unique()) != {"prior", "posterior"}:
        raise ValueError("Transfer comparison must retain both prior and posterior rows")
    kinds = list(data.new_source_kind.drop_duplicates())
    if len(kinds) != 2: raise ValueError("Transfer comparison needs the two equally cheap good/bad sources")
    lower = min(data.q025_per_lead.min(), data.true_margin_per_lead.min(), 0)
    upper = max(data.q975_per_lead.max(), data.true_margin_per_lead.max(), 0)
    span = upper - lower; lower -= .04 * span; upper += .04 * span
    posterior = data[data.stage == "posterior"]
    update_lower = min(posterior.q025_per_lead.min(), posterior.true_margin_per_lead.min(), 0)
    update_upper = max(posterior.q975_per_lead.max(), posterior.true_margin_per_lead.max(), 0)
    update_span = update_upper - update_lower
    update_lower -= .06 * update_span; update_upper += .06 * update_span
    quotes, quote_src = renderer.read("quote_sensitivity.csv", ["world", "scenario", "new_source_kind", "origin_day", "birth_day",
        "n_leads", "offered_cpl", "mean_margin180", "median_margin180", "q025_margin180", "q975_margin180",
        "p_payback180", "p_payback425", "p_no_payback425", "median_first_payback", "draws"])
    require_unique(quotes, ["world", "origin_day", "birth_day", "offered_cpl"], "prospective quote sensitivity")
    if not (quotes.birth_day > quotes.origin_day).all():
        raise ValueError("Quote sensitivity must apply only to prospective purchases")
    if not np.allclose(quotes.p_payback425 + quotes.p_no_payback425, 1):
        raise ValueError("Quote sensitivity must retain every noncrossing path")
    for _, part in quotes.groupby(["world", "origin_day", "birth_day"]):
        part = part.sort_values("offered_cpl")
        for metric in ["mean_margin180", "median_margin180", "q025_margin180", "q975_margin180"]:
            if not np.allclose(np.diff(part[metric]), -np.diff(part.offered_cpl) * part.n_leads.iloc[0]):
                raise ValueError(f"Quote-only query changed predicted quality in {metric}")
        if (np.diff(part.p_payback425) > 1e-12).any() or (np.diff(part.p_payback180) > 1e-12).any():
            raise ValueError("More expensive identical-path acquisition cannot hasten first payback")
    for metric in ["mean_margin180", "median_margin180", "q025_margin180", "q975_margin180"]:
        quotes[metric + "_per_lead"] = quotes[metric] / quotes.n_leads
    quote_table = renderer.output / "tables" / "05-offered-price-sensitivity.csv"
    quotes.to_csv(quote_table, index=False)
    title = "Equal cheap quotes do not imply equal quality"
    def plot(mobile):
        fig, axes = frame(title, 3, mobile)
        comparisons = [(data[data.stage == "prior"].sort_values("new_source_kind"),
                        "Before purchase: transferred prediction", lower, upper, True)]
        comparisons += [(posterior[posterior.new_source_kind == kind].sort_values("origin_day"),
                         f"{str(kind).title()} source updates: shared expanded scale", update_lower, update_upper, False)
                        for kind in kinds]
        for ax, (part, heading, left, right, prior) in zip(axes, comparisons):
            panel_title(ax, heading)
            y = np.arange(len(part))
            color = AMBER if prior else BLUE
            for idx, row in enumerate(part.itertuples()):
                ax.hlines(y[idx], row.q025_per_lead, row.q975_per_lead, color=color, lw=5, alpha=.58)
                ax.plot(row.median_per_lead, y[idx], marker="o", color=color, ms=7)
                ax.plot(row.true_margin_per_lead, y[idx], marker="x", color=INK, ms=8, mew=1.8)
            labels = [(str(row.new_source_kind).title() + " world" if prior else "After observations") +
                      f"\nCalendar day {int(row.origin_day)}" for row in part.itertuples()]
            ax.set_yticks(y, labels, fontsize=12.4 if mobile else 10.2); ax.invert_yaxis(); ax.set_xlim(left, right)
            ax.axvline(0, color=INK, lw=.9); ax.set_xlabel("Day-180 contribution\nUSD / acquired lead" if mobile else "Day-180 contribution / acquired lead"); dollars(ax, "x")
            ax.grid(False); ax.grid(axis="x", color=GRID, lw=.8)
            ax.legend([Line2D([], [], color=color, marker="o"), Line2D([], [], color=INK, marker="x", ls="")],
                      ["Transferred predictive" if prior else "Updated predictive", "Evaluator-only outcome"],
                      frameon=False, fontsize=12.4 if mobile else 8.8,
                      ncol=1 if mobile else 2, loc="lower left", bbox_to_anchor=(0, 1.02))
        fig.subplots_adjust(left=.36 if mobile else .26, hspace=1.4 if mobile else .90)
        return fig
    renderer.emit("05-new-source-transfer", title,
        "Paired synthetic cheap_good and cheap_bad worlds with the same demo seed; two late-arriving cheap sources, "
        "both evaluated at age180. The first pane preserves the full transferred interval on common prior axes. "
        "The two update panes use the SAME expanded contribution axis, clearly labeled, to show their narrower intervals. "
        "The before-purchase distribution integrates the fitted global hierarchy "
        "with fresh source and cohort effects; it is not the posterior of an old source. "
        "The updated predictive uses that source's subsequently available observation likelihood. "
        "Known cheap acquisition quotes enter economics, not an assumed quality feature. "
        "Outcome crosses are evaluator-only; table rows retain information counts when exported. "
        "The companion table queries ten offered-price cases for a NEW cohort born at day126, viewed at day119. "
        "It holds the same posterior outcome draws fixed while changing only that prospective batch's acquisition cost; "
        "it does not refit source quality or rewrite existing purchase costs. Missing median first-payback means "
        "the unconditional median does not cross by age425.",
        "One pane shows the full before-purchase distributions for good and bad worlds; two panes compare updates on the same explicitly expanded contribution scale, with evaluator-only outcomes as crosses.",
        [src, quote_src], f"first demo world per cheap_good/cheap_bad kind; dynamic; target_age180; prior and all posterior origins; both new-source kinds; companion query origin119,new birth126,all offered prices", data, plot,
        "95% approximate posterior-predictive intervals; transferred prediction includes fresh source/cohort uncertainty; observations update via the likelihood.",
        "Contribution / original acquired leads for each prospective batch; source information counts remain in the numeric table.")
    renderer.records[-1]["companion_tables"] = [{
        "path": str(quote_table.relative_to(ROOT)),
        "sha256": hashlib.sha256(quote_table.read_bytes()).hexdigest(),
        "rows": len(quotes), "scope": "Prospective acquisition cost query with unchanged posterior outcome draws",
    }]


def heldout_comparison(renderer):
    data, src = renderer.read("metrics.csv", ["scenario", "model", "metric", "mean", "lower95", "upper95", "n_worlds", "n_forecasts"])
    metric_aliases = {"coverage95": "coverage95", "coverage_95": "coverage95", "crps": "crps", "mean_crps": "crps", "crps_per_lead": "crps",
                      "decision_loss": "decision_loss", "acquisition_decision_loss": "decision_loss", "mean_decision_loss": "decision_loss",
                      "decision_loss_per_lead": "decision_loss"}
    data = policy_slice(data[data.metric.isin(metric_aliases)].copy())
    data["display_metric"] = data.metric.map(metric_aliases)
    if not {"coverage95", "crps", "decision_loss"}.issubset(data.display_metric.unique()):
        raise ValueError("Held-out figure needs coverage95, CRPS and decision loss")
    for bound in ["mean", "lower95", "upper95"]:
        data[bound + "_display"] = data[bound]
        monetary = data.display_metric.isin(["crps", "decision_loss"])
        per_lead = data["units"].isin(["USD/lead", "USD per lead", "USD_per_lead", "dollars_per_lead", "per_lead"]) if "units" in data else pd.Series(False, index=data.index)
        batch_money = monetary & ~per_lead
        if batch_money.any():
            if "n_leads" not in data: raise ValueError("Monetary metrics require n_leads or explicit units=USD/lead")
            data.loc[batch_money, bound + "_display"] = data.loc[batch_money, bound] / data.loc[batch_money, "n_leads"]
    frozen = ROOT.parent / "bayesian-email-cohort-lab" / "artifacts" / "calibration" / "metrics_by_origin.csv"
    if not frozen.exists(): raise FileNotFoundError("Frozen original-v1 failed coverage output must remain visible")
    baseline = pd.read_csv(frozen); baseline = baseline[baseline.metric == "coverage95"].sort_values("origin_age").copy()
    baseline["display_metric"] = "original_v1_coverage95"
    # Carry the bounded aggregate baseline bytes into the figure bundle so the
    # unchanged failure evidence is still inspectable after portable packaging.
    frozen_copy = renderer.output / "sources" / "original_v1_metrics_by_origin.csv"
    frozen_copy.parent.mkdir(exist_ok=True)
    shutil.copyfile(frozen, frozen_copy)
    base_src = Source(str(frozen_copy.relative_to(ROOT)), hashlib.sha256(frozen.read_bytes()).hexdigest(),
                      len(pd.read_csv(frozen)), str(frozen.relative_to(ROOT.parent)))
    all_metrics = data.copy()
    scenarios = [s for s in ["fixed_price", "all_scenarios"] if s in data.scenario.unique()]
    if not scenarios: scenarios = list(data.scenario.drop_duplicates())[:2]
    data = data[data.scenario.isin(scenarios)].copy()
    order = ["target_only", "global", "global_complete_pooling", "global_pooling", "source_only", "source_pooling", "static", "static_hierarchical", "static_hierarchy", "dynamic", "dynamic_hierarchical", "dynamic_hierarchy"]
    models = sorted(data.model.unique(), key=lambda m: order.index(m) if m in order else 99)
    title = "Held-out predictive and decision comparison"
    def plot(mobile):
        fig, axes = frame(title, 4, mobile, heights=[1.1, 1.1, 1.1, .78])
        markers = ["o", "s", "^", "D", "v"]
        for ax, metric, label in zip(axes[:3], ["coverage95", "crps", "decision_loss"],
                         ["95% interval coverage", "CRPS: lower is better", "Acquisition decision loss: lower is better"]):
            panel_title(ax, label)
            for j, scenario in enumerate(scenarios):
                part = data[(data.scenario == scenario) & (data.display_metric == metric)]
                for i, model in enumerate(models):
                    row = part[part.model == model]
                    if len(row) != 1: raise ValueError(f"Metric rows not unique: {scenario}/{model}/{metric}")
                    row = row.iloc[0]; y = i + (j - (len(scenarios) - 1) / 2) * .19
                    color = BLUE if "dynamic" in str(model).lower() else AMBER
                    ax.errorbar(row.mean_display, y, xerr=[[row.mean_display-row.lower95_display], [row.upper95_display-row.mean_display]],
                                fmt=markers[j % len(markers)], color=color, ms=5.5, lw=1.2, capsize=3)
            model_labels = [MODEL_LABELS.get(m, m) for m in models]
            if mobile: model_labels = [label.replace(" ", "\n", 1) for label in model_labels]
            ax.set_yticks(range(len(models)), model_labels, fontsize=12.4 if mobile else 10.5)
            ax.invert_yaxis(); ax.grid(False); ax.grid(axis="x", color=GRID, lw=.8)
            if metric == "coverage95":
                ax.set_xlim(0, 1.04); ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.axvline(.95, color=INK, ls="--", lw=1)
                ax.set_xlabel("Covered outcomes /\nheld-out forecasts" if mobile else "Covered outcomes / held-out forecasts")
            else:
                ax.set_xlim(left=0); dollars(ax, "x"); ax.set_xlabel("USD / acquired lead")
            legend_handles = [Line2D([], [], color=MUTED, marker=markers[j % len(markers)], ls="") for j in range(len(scenarios))]
            legend_labels = [textwrap.fill("Fixed-price control" if s == "fixed_price" else "All six scenarios", 25 if mobile else 60) for s in scenarios]
            if metric == "coverage95":
                legend_handles.append(Line2D([], [], color=INK, ls="--"))
                legend_labels.append("Nominal 95% coverage")
            ax.legend(legend_handles, legend_labels, frameon=False, fontsize=12.4 if mobile else 9.0, ncol=1 if mobile else 2,
                      loc="lower left", bbox_to_anchor=(0, 1.02))
        ax = axes[-1]; panel_title(ax, "Frozen original v1: a separate failed stress test")
        original_x = np.arange(len(baseline))
        ax.errorbar(original_x, baseline["mean"], yerr=[baseline["mean"]-baseline.lower95, baseline.upper95-baseline["mean"]],
                    fmt="s-", color=AMBER, lw=1.7, ms=5, capsize=3)
        ax.axhline(.95, color=INK, ls="--", lw=1); ax.set_ylim(0, 1.04); ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xticks(original_x, baseline.origin_age); ax.set_xlabel("Original snapshot (age in days)"); ax.set_ylabel("Coverage")
        fig.subplots_adjust(left=.39 if mobile else .28, hspace=1.1 if mobile else .82)
        return fig
    per_scenario_worlds = sorted(all_metrics[all_metrics.scenario != "all_scenarios"].n_worlds.drop_duplicates())
    overall_worlds = sorted(all_metrics[all_metrics.scenario == "all_scenarios"].n_worlds.drop_duplicates())
    world_scope = (f"{','.join(str(int(v)) for v in per_scenario_worlds)} independent worlds per scenario; "
                   f"{','.join(str(int(v)) for v in overall_worlds)} across all six declared scenarios. ")
    renderer.emit("06-heldout-comparison", title,
        "Independent held-out synthetic worlds: " + world_scope +
        "These are separate from the one-world illustration. The visible score markers show the fixed-price control "
        "and an aggregate of all six declared scenarios; the numeric table retains every scenario-specific score. "
        "All five model alternatives use the same information clocks and target economics. "
        "Coverage measures nominal 95% contribution intervals; CRPS scores the full forecast; decision loss scores the declared acquisition policy. "
        "World-level uncertainty retains within-world dependence. The bottom panel preserves the original v1's frozen 12-world stress-coverage failure "
        "as a different experiment, without combining it with new scores. Limited evaluation cannot establish general calibration.",
        "Three horizontal comparisons show 95% coverage, CRPS and acquisition decision loss for held-out worlds, above a distinct panel showing the original v1 coverage failures across forecast ages.",
        [src, base_src], "all independent-world metrics for coverage95,CRPS,decision_loss; original frozen stress coverage displayed separately",
        pd.concat([add_panel(all_metrics, "extension_heldout_metrics"), add_panel(baseline, "frozen_original_v1")], ignore_index=True), plot,
        "95% evaluator uncertainty bounds clustered by independent world; original panel uses its frozen whole-world bootstrap bounds. Nominal predictive level is 95%.",
        "Coverage / n_forecasts; n_worlds independent evaluation clusters; CRPS and decision loss / acquired leads, with source counts retained in table.")


def render_all(artifacts: Path | Renderer, output: Path | None = None):
    """Render a CSV directory directly, or use an already configured Renderer."""
    managed = not isinstance(artifacts, Renderer)
    renderer = Renderer(Path(artifacts), output or Path(artifacts) / "figures") if managed else artifacts
    world, scenario = economics(renderer)
    measurement(renderer, world, scenario)
    calendar_states(renderer, world, scenario)
    cohort_forecasts(renderer, world, scenario)
    source_transfer(renderer, world, scenario)
    heldout_comparison(renderer)
    if managed: renderer.finish()
    return renderer


if __name__ == "__main__":
    main()
