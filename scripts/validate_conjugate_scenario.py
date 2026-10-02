#!/usr/bin/env python3
"""Exact component calibration in a declared, well-specified Gamma-Poisson world.

This does not validate the main model's monetary/telemetry/operational modules.
Its known-shape response posterior is cross-checked against the package API.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import scipy
from scipy.stats import binomtest, gamma, nbinom, kstest


@dataclass(frozen=True)
class ControlledConfig:
    configuration_version: str = "controlled-gamma-poisson/1.0.0"
    seed: int = 2026100201
    worlds: int = 2000
    original_leads: int = 100
    accepted_per_day: int = 100
    horizon: int = 180
    origins: tuple = (3, 7, 14, 30, 60)
    cohort_shape: float = 8.
    source_mean: float = .12
    fast_weight: float = .8
    fast_timescale: float = 1.
    tail_timescale: float = 60.
    value_per_click: float = .125
    acquisition_per_lead: float = .15
    send_cost_per_exposure: float = .00025
    module_crosscheck_draws: int = 4000

    def __post_init__(self):
        if self.worlds < 1 or self.accepted_per_day < 1 or self.original_leads < 1:
            raise ValueError("worlds and fixed exposure counts must be positive")
        if any(a < 0 or a >= self.horizon for a in self.origins):
            raise ValueError("all origins must be before the horizon")
        if self.cohort_shape <= 0 or self.source_mean <= 0 or self.value_per_click <= 0:
            raise ValueError("Gamma parameters and fixed monetary value must be positive")


def age_curve(ages, config):
    ages = np.asarray(ages)
    return config.fast_weight * np.exp(-ages / config.fast_timescale) + (1 - config.fast_weight) * np.exp(-ages / config.tail_timescale)


def posterior_parameters(observed_count, observed_exposure, config):
    """The same conditional Gamma update used by bayes_cohort.model."""
    return config.cohort_shape + np.asarray(observed_count), config.cohort_shape / config.source_mean + observed_exposure


def predictive_distribution(shape, rate, future_exposure):
    """Gamma-Poisson mixture: SciPy NB counts failures with p=rate/(rate+L)."""
    return nbinom(shape, rate / (rate + future_exposure))


def evaluate_controlled(config=None):
    config = config or ControlledConfig()
    rng = np.random.default_rng(config.seed)
    ages = np.arange(config.horizon + 1)
    effective_exposure = config.accepted_per_day * age_curve(ages, config)
    theta = rng.gamma(config.cohort_shape, config.source_mean / config.cohort_shape, config.worlds)
    counts = rng.poisson(theta[:, None] * effective_exposure[None, :])
    cumulative = counts.cumsum(axis=1)
    total_count = cumulative[:, -1]
    cost = config.original_leads * config.acquisition_per_lead + config.accepted_per_day * (config.horizon + 1) * config.send_cost_per_exposure
    true_margin = total_count * config.value_per_click - cost
    forecasts, summaries, reliability = [], [], []
    for origin in config.origins:
        observed = cumulative[:, origin]
        shape, rate = posterior_parameters(observed, effective_exposure[:origin + 1].sum(), config)
        future_exposure = effective_exposure[origin + 1:].sum()
        posterior = gamma(shape, scale=1 / rate)
        predictive = predictive_distribution(shape, rate, future_exposure)
        future_count = total_count - observed
        threshold = int(np.floor(cost / config.value_per_click)) - observed
        profitable_probability = predictive.sf(threshold)
        actual_profitable = true_margin > 0
        lower80 = (observed + predictive.ppf(.1)) * config.value_per_click - cost
        upper80 = (observed + predictive.ppf(.9)) * config.value_per_click - cost
        lower95 = (observed + predictive.ppf(.025)) * config.value_per_click - cost
        upper95 = (observed + predictive.ppf(.975)) * config.value_per_click - cost
        exact_mass80 = predictive.cdf(predictive.ppf(.9)) - predictive.cdf(predictive.ppf(.1) - 1)
        exact_mass95 = predictive.cdf(predictive.ppf(.975)) - predictive.cdf(predictive.ppf(.025) - 1)
        theta_lower, theta_upper = posterior.ppf(.025), posterior.ppf(.975)
        rank = posterior.cdf(theta)
        cdf_before = predictive.cdf(future_count - 1)
        pit = cdf_before + rng.random(config.worlds) * (predictive.cdf(future_count) - cdf_before)
        cover80 = (true_margin >= lower80) & (true_margin <= upper80)
        cover95 = (true_margin >= lower95) & (true_margin <= upper95)
        cover_theta = (theta >= theta_lower) & (theta <= theta_upper)
        rows = pd.DataFrame({
            "world_id": np.arange(config.worlds), "origin_age": origin, "horizon": config.horizon,
            "true_theta": theta, "observed_count": observed, "future_count": future_count,
            "posterior_shape": shape, "posterior_rate": rate,
            "posterior_theta_mean": shape / rate, "posterior_theta_lower95": theta_lower,
            "posterior_theta_upper95": theta_upper, "posterior_theta_rank": rank,
            "predictive_randomized_pit": pit, "true_margin": true_margin,
            "predictive_margin_mean": (observed + predictive.mean()) * config.value_per_click - cost,
            "lower80": lower80, "upper80": upper80, "lower95": lower95, "upper95": upper95,
            "coverage80": cover80.astype(int), "coverage95": cover95.astype(int),
            "exact_interval_mass80": exact_mass80, "exact_interval_mass95": exact_mass95,
            "theta_coverage95": cover_theta.astype(int), "probability_profitable": profitable_probability,
            "actual_profitable": actual_profitable.astype(int), "brier": (profitable_probability - actual_profitable) ** 2,
        })
        forecasts.append(rows)
        ci80 = binomtest(int(cover80.sum()), config.worlds).proportion_ci(method="wilson")
        ci95 = binomtest(int(cover95.sum()), config.worlds).proportion_ci(method="wilson")
        summaries.append({
            "origin_age": origin, "independent_worlds": config.worlds, "horizon": config.horizon,
            "coverage80": cover80.mean(), "coverage80_wilson_lower": ci80.low, "coverage80_wilson_upper": ci80.high,
            "coverage95": cover95.mean(), "coverage95_wilson_lower": ci95.low, "coverage95_wilson_upper": ci95.high,
            "expected_discrete_coverage80": exact_mass80.mean(), "expected_discrete_coverage95": exact_mass95.mean(),
            "coverage80_error_standard_errors": float((cover80.mean() - exact_mass80.mean()) / (np.sqrt(np.sum(exact_mass80 * (1 - exact_mass80))) / config.worlds)),
            "coverage95_error_standard_errors": float((cover95.mean() - exact_mass95.mean()) / (np.sqrt(np.sum(exact_mass95 * (1 - exact_mass95))) / config.worlds)),
            "theta_coverage95": cover_theta.mean(), "mean_width80": np.mean(upper80 - lower80),
            "mean_width95": np.mean(upper95 - lower95), "mean_theta_width95": np.mean(theta_upper - theta_lower),
            "brier": rows.brier.mean(), "prior_only_brier": np.mean((actual_profitable.mean() - actual_profitable) ** 2),
            "posterior_rank_mean": rank.mean(), "posterior_rank_variance": rank.var(),
            "posterior_rank_ks_p": kstest(rank, "uniform").pvalue,
            "predictive_pit_mean": pit.mean(), "predictive_pit_ks_p": kstest(pit, "uniform").pvalue,
        })
        bin_ids = np.minimum((profitable_probability * 10).astype(int), 9)
        for bin_id in range(10):
            selected = bin_ids == bin_id
            if selected.any():
                reliability.append({"origin_age": origin, "probability_bin": bin_id, "worlds": int(selected.sum()),
                                    "predicted_probability": float(profitable_probability[selected].mean()),
                                    "observed_frequency": float(actual_profitable[selected].mean())})
    world_draws = pd.DataFrame(counts, columns=[f"count_age_{age}" for age in ages])
    world_draws.insert(0, "true_theta", theta)
    world_draws.insert(0, "world_id", np.arange(config.worlds))
    return {"config": config, "forecasts": pd.concat(forecasts, ignore_index=True), "summary": pd.DataFrame(summaries),
            "reliability": pd.DataFrame(reliability), "world_draws": world_draws,
            "total_known_cost": float(cost), "counts": counts, "theta": theta}


def module_crosscheck(config, daily_counts, origin=30):
    """Check the actual package posterior sampler on a valid observed Snapshot.

    Other model modules are intentionally not scored by this component check.
    """
    from bayes_cohort import ModelConfig, forecast_cohort
    from cohort_lab.accounting import Snapshot, Policy
    from cohort_lab.generator import LEAD_COLUMNS, EXPOSURE_COLUMNS, EVENT_COLUMNS, LEDGER_COLUMNS, EXIT_COLUMNS
    # Generator column constants are public schemas only; no generator output
    # or truth supplies any parameters to this fixture or to the package.
    leads = pd.DataFrame([(f"u{i}", "known", "controlled", 0, config.acquisition_per_lead, "USD", 0, .5, "fixed") for i in range(config.original_leads)], columns=LEAD_COLUMNS)
    exposure = pd.DataFrame([("controlled", "known", age, age, 0, config.accepted_per_day, config.accepted_per_day,
                             config.accepted_per_day, 0, age, "observed", "fixed") for age in range(origin + 1)], columns=EXPOSURE_COLUMNS)
    events, ledger = [], [("a", "", "controlled", "known", 0, -config.original_leads * config.acquisition_per_lead, 0, 0, "acquisition", "", "", "USD")]
    for age in range(origin + 1):
        for number in range(int(daily_counts[age])):
            event_id = f"e{age}-{number}"
            events.append((f"r{event_id}", event_id, "u0", "controlled", "known", f"m{age}", 0, age, age, age, "human", 0, "fixed"))
            ledger.append((f"l{event_id}", event_id, "controlled", "known", 0, config.value_per_click, age, age, "revenue", "", "", "USD"))
    snapshot = Snapshot(origin, leads, exposure, pd.DataFrame(events, columns=EVENT_COLUMNS),
                        pd.DataFrame(ledger, columns=LEDGER_COLUMNS), pd.DataFrame(columns=EXIT_COLUMNS))
    frozen = ModelConfig(source_means=(config.source_mean,), fast_weights=(config.fast_weight,),
                         fast_timescales=(config.fast_timescale,), tail_timescales=(config.tail_timescale,),
                         cohort_shape=config.cohort_shape, response_delay=((0, 1.),), event_report_delay=((0, 1.),),
                         settlement_delay=((0, 1.),), ledger_report_delay=((0, 1.),), forecast_invoice_revisions=False)
    result = forecast_cohort(snapshot, Policy(last_send_age=origin, terminal_age=origin + 30), "controlled",
                             config=frozen, draws=config.module_crosscheck_draws, seed=config.seed + 1)
    amplitude = result["parameter_draws"]["cohort_amplitude"]
    observed = int(np.sum(daily_counts[:origin + 1]))
    shape, rate = posterior_parameters(observed, config.accepted_per_day * age_curve(np.arange(origin + 1), config).sum(), config)
    exact_mean, exact_variance = float(shape / rate), float(shape / rate ** 2)
    mean_error_se = float(abs(amplitude.mean() - exact_mean) / np.sqrt(exact_variance / len(amplitude)))
    variance_relative_error = float(abs(amplitude.var() / exact_variance - 1))
    return {"origin_age": origin, "observed_count": observed, "posterior_shape": float(shape), "posterior_rate": float(rate),
            "exact_mean": exact_mean, "sample_mean": float(amplitude.mean()), "exact_variance": exact_variance,
            "sample_variance": float(amplitude.var()), "draws": len(amplitude), "mean_error_standard_errors": mean_error_se,
            "variance_relative_error": variance_relative_error,
            "passed": bool(mean_error_se < 5 and variance_relative_error < .10),
            "scope": "main package known-shape conditional Gamma rate posterior only"}


def _stress_comparison(path, horizon):
    if not path.exists():
        return pd.DataFrame()
    rows = pd.read_csv(path)
    rows = rows.loc[rows.horizon.eq(horizon) & rows.evaluation_role.eq("prospective")]
    grouped = rows.groupby("scenario", as_index=False).agg(coverage80=("coverage80", "mean"), coverage95=("coverage95", "mean"),
                                                          forecast_cases=("coverage95", "size"), independent_worlds=("world_id", "nunique"))
    return grouped


def _plots(result, stress, destination):
    import os
    cache = Path("/tmp/bayesian-conjugate-matplotlib")
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
    summary = result["summary"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for level, color in ((80, "#2563eb"), (95, "#c7552e")):
        axes[0].plot(summary.origin_age, summary[f"coverage{level}"], "o-", color=color, label=f"{level}% interval")
        axes[0].axhline(level / 100, color=color, linestyle="--", alpha=.55)
        axes[0].fill_between(summary.origin_age, summary[f"coverage{level}_wilson_lower"], summary[f"coverage{level}_wilson_upper"], color=color, alpha=.12)
    axes[0].set(title="Well-specified rate component", xlabel="Observed cohort age (days)", ylabel="Predictive margin interval coverage", ylim=(.65, 1.))
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(loc="lower right")
    if len(stress):
        locations = np.arange(len(stress))
        axes[1].bar(locations - .16, stress.coverage80, .32, color="#2563eb", label="80% interval")
        axes[1].bar(locations + .16, stress.coverage95, .32, color="#c7552e", label="95% interval")
        axes[1].axhline(.8, color="#2563eb", linestyle="--", alpha=.55)
        axes[1].axhline(.95, color="#c7552e", linestyle="--", alpha=.55)
        labels = [name.replace("calendar_attribution_shock", "Calendar\nshock").replace("collapse_tail", "Collapse\ntail").replace("late_reactivation", "Late\nreactivation").replace("stationary", "Stationary") for name in stress.scenario]
        axes[1].set_xticks(locations, labels)
        axes[1].set(title="Full modular pipeline: misspecified worlds", ylim=(0, 1.))
        axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    else:
        axes[1].text(.5, .5, "Full-world stress scores unavailable", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_axis_off()
    figure.suptitle("Correct conjugate updating does not validate a misspecified full pipeline", fontsize=14)
    figure.text(.06, .015, f"Controlled: {result['config'].worlds:,} independent worlds; exact NB quantiles, discrete intervals conservative.  "
                f"Stress: unchanged, 3 worlds/scenario; repeated forecasts correlate.  Horizon {result['config'].horizon} days.", fontsize=9)
    figure.tight_layout(rect=(0, .055, 1, .94))
    for extension in ("png", "svg"):
        figure.savefig(destination / f"controlled_vs_full_world_coverage.{extension}", dpi=170, bbox_inches="tight")
    plt.close(figure)
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    axes[0].plot(summary.origin_age, summary.mean_theta_width95, "o-", color="#2563eb")
    axes[0].set(title="Parameter uncertainty updates", xlabel="Observed age", ylabel="Mean 95% amplitude interval width")
    axes[1].plot(summary.origin_age, summary.mean_width95, "o-", color="#c7552e", label="95%")
    axes[1].plot(summary.origin_age, summary.mean_width80, "o-", color="#2563eb", label="80%")
    axes[1].set(title="Predictive uncertainty updates", xlabel="Observed age", ylabel="Mean monetary interval width ($)")
    axes[1].legend()
    first = result["forecasts"].loc[result["forecasts"].origin_age.eq(result["config"].origins[0])]
    axes[2].hist(first.posterior_theta_rank, bins=np.linspace(0, 1, 11), density=True, color="#568a72", alpha=.8)
    axes[2].axhline(1, color="#444444", linestyle="--")
    axes[2].set(title=f"Posterior ranks at age {result['config'].origins[0]}", xlabel="Gamma posterior CDF at true amplitude", ylabel="Density")
    figure.suptitle("Known curve, fixed value and exposure: component-level validation only", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, .93))
    for extension in ("png", "svg"):
        figure.savefig(destination / f"controlled_updating_and_ranks.{extension}", dpi=170, bbox_inches="tight")
    plt.close(figure)


def run(output, stress_path, config=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result = evaluate_controlled(config)
    config = result["config"]
    crosscheck = module_crosscheck(config, result["counts"][0])
    if not crosscheck["passed"]:
        raise AssertionError(f"Main module conjugate cross-check failed: {crosscheck}")
    stress = _stress_comparison(Path(stress_path), config.horizon)
    result["summary"].to_csv(output / "metrics_by_origin.csv", index=False)
    result["forecasts"].to_csv(output / "exact_forecasts.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    result["world_draws"].to_csv(output / "all_prespecified_worlds.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    result["reliability"].to_csv(output / "profitability_reliability.csv", index=False)
    stress.to_csv(output / "paired_full_world_stress.csv", index=False)
    sources = [Path(__file__), ROOT / "bayes_cohort/model.py", ROOT / "bayes_cohort/config.py"]
    metadata = {
        "configuration": asdict(config), "numpy": np.__version__, "scipy": scipy.__version__, "python": platform.python_version(),
        "all_worlds_retained": True, "independent_worlds": config.worlds, "correlated_origin_cases": config.worlds * len(config.origins),
        "predictive_quantiles": "exact negative-binomial PPF; discrete central intervals can be conservative",
        "scope": "Known-shape Gamma-Poisson rate component with fixed value/cost/exposure only; not full modular pipeline validation",
        "module_crosscheck": crosscheck, "total_known_horizon_cost": result["total_known_cost"],
        "paired_stress_scores": str(Path(stress_path).relative_to(ROOT)) if Path(stress_path).is_relative_to(ROOT) else str(stress_path),
        "paired_stress_scores_sha256": hashlib.sha256(Path(stress_path).read_bytes()).hexdigest() if Path(stress_path).exists() else None,
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "notes": ["Seed and configuration are declared before examining results; no worlds are selected, discarded, or reweighted",
                  "Ranks and randomized PIT are assessed separately by origin; pooling origins does not create more independent worlds",
                  "No full-world failure is repaired, tuned, excluded or relabeled by this controlled check"],
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    _plots(result, stress, output)
    summary = result["summary"]
    text = ["# Controlled component validation", "", metadata["scope"] + ".", "",
            f"All {config.worlds:,} prespecified worlds retained; seed {config.seed}; origins {list(config.origins)}; horizon {config.horizon}.", "",
            "Exact predictive intervals include future Poisson variation and posterior amplitude uncertainty. Their monetary transform uses a fixed positive value and known costs.", "",
            "| Origin | 80% coverage | 95% coverage | Amplitude 95% coverage | Mean 95% monetary width | Brier |", "|---:|---:|---:|---:|---:|---:|"]
    for row in summary.itertuples():
        text.append(f"| {row.origin_age} | {row.coverage80:.3f} | {row.coverage95:.3f} | {row.theta_coverage95:.3f} | {row.mean_width95:.2f} | {row.brier:.3f} |")
    text.extend(["", f"Main-package posterior-sampler cross-check: mean error {crosscheck['mean_error_standard_errors']:.2f} standard errors; variance relative error {crosscheck['variance_relative_error']:.3%}.", "",
                 "Posterior mean/variance agree with the identical analytic Gamma update on a Snapshot fixture. The validation is component-level; fixed marks, no reporting delays, no revisions, and known exposure remove the main pipeline's additional modeling risks.", "",
                 "## Unchanged full-world stress comparison", ""])
    if len(stress):
        for row in stress.itertuples():
            text.append(f"- {row.scenario}: 80% coverage {row.coverage80:.3f}; 95% coverage {row.coverage95:.3f}, at the same horizon, across {row.forecast_cases} correlated cases in {row.independent_worlds} worlds")
    else:
        text.append("Full-world stress scores were unavailable for pairing.")
    text.extend(["", "The contrast establishes that conjugate updating can work under matching assumptions while the full modular model remains undercovered on the richer synthetic worlds. It is not evidence that full-pipeline profitability probabilities are trustworthy.", "",
                 "PNG and SVG charts: controlled_vs_full_world_coverage and controlled_updating_and_ranks. Exact forecast and world files retain every prespecified case."])
    (output / "RESULTS.md").write_text("\n".join(text) + "\n")
    return result, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/controlled_scenario")
    parser.add_argument("--stress-scores", type=Path, default=ROOT / "artifacts/calibration/scores.csv")
    args = parser.parse_args()
    result, metadata = run(args.output, args.stress_scores)
    print(result["summary"].to_string(index=False))
    print(json.dumps(metadata["module_crosscheck"], indent=2))
    print(f"Artifacts: {args.output}")


if __name__ == "__main__":
    main()
