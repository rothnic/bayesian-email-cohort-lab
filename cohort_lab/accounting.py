"""Schema owner: immutable observations and common finite-policy accounting."""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import pandas as pd

SCHEMA_VERSION = "cohort-lab/0.1.0"
ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Policy:
    policy_id: str = "daily_to_365_response30_settlement30"
    last_send_age: int = 365
    max_response_delay: int = 30
    max_accounting_delay: int = 30
    terminal_age: int = 425
    send_cost: float = 0.00025

@dataclass(frozen=True)
class Snapshot:
    cutoff: int
    leads: pd.DataFrame
    exposure_daily: pd.DataFrame
    engagement_events: pd.DataFrame
    ledger: pd.DataFrame
    operational_exits: pd.DataFrame

@dataclass(frozen=True)
class World:
    scenario: str
    seed: int
    leads: pd.DataFrame
    exposure_daily: pd.DataFrame
    engagement_events: pd.DataFrame
    ledger: pd.DataFrame
    operational_exits: pd.DataFrame

    def as_of(self, cutoff: int) -> Snapshot:
        def visible(frame, time):
            return frame.loc[frame[time].le(cutoff) & frame.observed_at.notna() & frame.observed_at.le(cutoff)].copy()
        events = visible(self.engagement_events, "occurred_at")
        events = events.sort_values(["event_id", "revision_sequence", "observed_at"]).drop_duplicates("event_id", keep="last")
        exposure = visible(self.exposure_daily, "date")
        unavailable = exposure.telemetry_status.ne("observed")
        exposure.loc[unavailable, ["attempted", "accepted", "eligible", "exits"]] = np.nan
        return Snapshot(int(cutoff), self.leads.loc[self.leads.acquired_at.le(cutoff)].copy(), exposure, events,
                        visible(self.ledger, "economic_at"), visible(self.operational_exits, "occurred_at"))

@dataclass(frozen=True)
class GeneratorTruth:
    scenario: str
    seed: int
    complete_ledger: pd.DataFrame
    final_human_events: pd.DataFrame
    latent_leads: pd.DataFrame
    calendar_shocks: pd.DataFrame
    parameters: dict

    def cohort_path(self, leads: pd.DataFrame, cohort_id: str, policy: Policy) -> np.ndarray:
        acquired = int(leads.loc[leads.cohort_id.eq(cohort_id), "acquired_at"].iloc[0])
        rows = self.complete_ledger.loc[self.complete_ledger.cohort_id.eq(cohort_id)]
        return ledger_path(rows, acquired, policy.terminal_age)

def load_parameters(path=None):
    return json.loads(Path(path or ROOT / "parameters.json").read_text())

def ledger_path(ledger: pd.DataFrame, acquired_at: int, terminal_age: int) -> np.ndarray:
    increments = np.zeros(terminal_age + 1)
    if len(ledger):
        ages = ledger.economic_at.to_numpy(dtype=int) - acquired_at
        valid = (ages >= 0) & (ages <= terminal_age)
        np.add.at(increments, ages[valid], ledger.amount.to_numpy()[valid])
    return increments.cumsum()

def summarize_path(path, horizons=(30,90,180,425)):
    path = np.asarray(path, dtype=float)
    crossing = np.flatnonzero(path >= 0)
    first = int(crossing[0]) if len(crossing) else None
    state = "never_under_policy" if first is None else ("by_180" if first <= 180 else "later_than_180")
    return [{"horizon":int(h),"margin":float(path[h]),"first_payback_age":first,"payback_state":state,
             "positive_at_horizon":bool(path[h] > 0),
             "remains_positive_after_payback":bool(first is not None and first <= h and np.all(path[first:h+1] >= 0))}
            for h in horizons]

def cohort_metadata(snapshot: Snapshot) -> pd.DataFrame:
    return snapshot.leads.groupby(["cohort_id","source_id","acquired_at"], as_index=False).agg(
        n_leads=("lead_id","size"), acquisition_cost=("acquisition_cost","sum"))
