"""Frozen, declared model priors and synthetic measurement-clock assumptions.

These constants are not read from generator parameters or latent truth. The
delay clocks intentionally describe the demo's published synthetic instrument.
Response, cohort, mark and operational parameters are learned from snapshots.
"""
from dataclasses import asdict, dataclass, fields
from collections.abc import Mapping
import numpy as np


@dataclass(frozen=True)
class ModelConfig:
    source_means: tuple = (.008, .02, .05, .12, .30, .70, 1.5)
    fast_weights: tuple = (.20, .50, .80, .95)
    fast_timescales: tuple = (.4, 1., 3.)
    tail_timescales: tuple = (20., 60., 180., 400.)
    log_source_mean_prior_location: float = -2.302585092994046
    log_source_mean_prior_scale: float = 1.2
    cohort_shape: float = 8.
    hazard_alpha: float = 1.
    hazard_beta: float = 499.
    acceptance_alpha: float = 19.
    acceptance_beta: float = 1.
    positive_alpha: float = 2.
    positive_beta: float = 2.
    log_mark_prior_mean: float = -2.302585092994046
    log_mark_prior_precision: float = .5
    log_mark_prior_shape: float = 3.
    log_mark_prior_scale: float = 1.
    max_log_mark_variance: float = 4.
    revision_alpha: float = 1.
    revision_beta: float = 19.
    mark_maturity_days: int = 30
    response_delay: tuple = ((0, .80), (1, .13), (3, .04), (10, .02), (30, .01))
    event_report_delay: tuple = ((0, .85), (1, .10), (2, .05))
    settlement_delay: tuple = ((0, .50), (2, .30), (7, .15), (14, .05))
    ledger_report_delay: tuple = ((0, .80), (1, .15), (3, .05))
    revision_delay_min: int = 8
    revision_delay_max: int = 25
    revision_report_delay: int = 3
    revision_fraction_min: float = .25
    revision_fraction_max: float = 1.
    invoice_economic_age: int = 21
    invoice_report_age: int = 24
    invoice_fraction_mean: float = .01
    invoice_fraction_sd: float = .025
    forecast_invoice_revisions: bool = True

    def __post_init__(self):
        vector_names = ("source_means", "fast_weights", "fast_timescales", "tail_timescales")
        kernel_names = ("response_delay", "event_report_delay", "settlement_delay", "ledger_report_delay")
        for name in vector_names:
            object.__setattr__(self, name, tuple(float(x) for x in getattr(self, name)))
        for name in kernel_names:
            object.__setattr__(self, name, tuple((int(d), float(p)) for d, p in getattr(self, name)))
            values = getattr(self, name)
            if not values or any(d < 0 or p < 0 for d, p in values) or not np.isclose(sum(p for _, p in values), 1):
                raise ValueError(f"{name} must be a nonnegative normalized delay distribution")
        for name in ("source_means", "fast_timescales", "tail_timescales"):
            if not getattr(self, name) or min(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must contain positive values")
        if not self.fast_weights or any(w <= 0 or w >= 1 for w in self.fast_weights):
            raise ValueError("fast_weights must lie strictly between zero and one")
        for name in ("log_source_mean_prior_scale", "cohort_shape", "hazard_alpha", "hazard_beta", "acceptance_alpha", "acceptance_beta", "positive_alpha", "positive_beta", "log_mark_prior_precision", "log_mark_prior_shape", "log_mark_prior_scale", "max_log_mark_variance", "revision_alpha", "revision_beta"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.mark_maturity_days < max(d for d, _ in self.settlement_delay) + max(d for d, _ in self.ledger_report_delay):
            raise ValueError("mark maturity must cover the declared base-revenue observation support")
        revision_support = max(self.revision_delay_max, max(d for d, _ in self.settlement_delay) + 1) + self.revision_report_delay
        if self.mark_maturity_days < revision_support:
            raise ValueError("mark maturity must cover the declared revision observation support")
        if self.revision_delay_min < 0 or self.revision_delay_max < self.revision_delay_min:
            raise ValueError("invalid revision delay range")
        if not 0 <= self.revision_fraction_min <= self.revision_fraction_max <= 1:
            raise ValueError("revision fractions must lie between zero and one")
        if self.invoice_report_age < self.invoice_economic_age or self.invoice_fraction_sd < 0:
            raise ValueError("invalid invoice uncertainty configuration")


def resolve_config(config=None):
    if config is None:
        return ModelConfig()
    if isinstance(config, ModelConfig):
        return config
    if isinstance(config, Mapping):
        unknown = set(config) - {f.name for f in fields(ModelConfig)}
        if unknown:
            raise ValueError(f"Unknown frozen model configuration keys: {sorted(unknown)}")
        return ModelConfig(**config)
    raise TypeError("config must be ModelConfig, a configuration mapping, or None")


def default_config_dict():
    """A JSON-serializable export, independent of generator configuration."""
    return asdict(ModelConfig())
