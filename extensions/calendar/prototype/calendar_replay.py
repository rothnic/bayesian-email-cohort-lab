"""Synthetic, immutable as-of contract proof. No statistical model is fitted."""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import argparse
import json
import math
import random
from typing import Protocol

Scalar = int | float | str | None


@dataclass(frozen=True)
class Record:
    record_id: str
    kind: str
    source: str
    cohort: str
    effective_at: int
    observed_at: int | None
    values: tuple[tuple[str, Scalar], ...]
    replaces: str | None = None
    revision: int = 0

    def get(self, name: str) -> Scalar:
        return dict(self.values)[name]

    def canonical(self):
        return (self.record_id, self.kind, self.source, self.cohort,
                self.effective_at, self.observed_at, self.values,
                self.replaces, self.revision)


def record(record_id, kind, source, cohort, effective_at, observed_at, *,
           replaces=None, revision=0, **values):
    if kind == 'payment' and replaces is not None:
        raise ValueError('Monetary corrections must use separate signed entries')
    if any(not isinstance(v, (int, float, str, type(None))) for v in values.values()):
        raise TypeError('Record payloads must contain immutable JSON scalars')
    if any(isinstance(v, float) and not math.isfinite(v) for v in values.values()):
        raise ValueError('Record payloads must be finite')
    return Record(record_id, kind, source, cohort, effective_at, observed_at,
                  tuple(sorted(values.items())), replaces, revision)


@dataclass(frozen=True)
class Snapshot:
    cutoff: int
    records: tuple[Record, ...]

    def canonical_bytes(self):
        return json.dumps({'cutoff': self.cutoff, 'records': [r.canonical() for r in self.records]},
                          sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

    @property
    def digest(self):
        return sha256(self.canonical_bytes()).hexdigest()

    def kinds(self, kind):
        return tuple(r for r in self.records if r.kind == kind)


def as_of(stream: tuple[Record, ...], cutoff: int) -> Snapshot:
    """Replace measurement versions; retain separately signed monetary corrections."""
    if len({r.record_id for r in stream}) != len(stream):
        raise ValueError('Immutable record IDs must be unique')
    visible = {r.record_id:r for r in stream if r.observed_at is not None
               and r.observed_at <= cutoff and r.effective_at <= cutoff}
    def root(r):
        seen=set()
        while r.replaces is not None:
            if r.record_id in seen or r.replaces not in visible:
                raise ValueError('Revision chains require visible ancestors and no cycles')
            seen.add(r.record_id)
            parent=visible[r.replaces]
            if (r.kind,r.source,r.cohort,r.effective_at)!=(parent.kind,parent.source,parent.cohort,parent.effective_at):
                raise ValueError('A revision must address the same measurement cell')
            if r.revision <= parent.revision or r.observed_at < parent.observed_at:
                raise ValueError('Revision versions and report dates must advance')
            r=parent
        return r.record_id
    latest = {}
    for r in visible.values():
        key = (r.kind, root(r))
        old = latest.get(key)
        if old is None or (r.revision, r.observed_at, r.record_id) > (old.revision, old.observed_at, old.record_id):
            latest[key] = r
    return Snapshot(cutoff, tuple(sorted(latest.values(), key=lambda r:r.record_id)))


def scoped(snapshot: Snapshot, target: str, scope: str) -> Snapshot:
    cohort = next(r for r in snapshot.kinds('cohort') if r.cohort == target)
    if scope not in {'target_only', 'same_source', 'all_sources'}:
        raise ValueError('Unknown information scope')
    if scope == 'all_sources':
        return snapshot
    chosen = {r.cohort for r in snapshot.kinds('cohort')
              if r.cohort == target or (scope == 'same_source' and r.source == cohort.source)}
    # Common quote/contract facts stay in all scopes; observations are scoped.
    return Snapshot(snapshot.cutoff, tuple(r for r in snapshot.records
                    if r.cohort in chosen or r.kind in {'quote', 'contract'}))


def fixture(seed=19, scenario='squeeze', last_acquisition=84, through=112):
    """Small aggregate cells; no individual leads or copied dataset exports.

    Audit labels are delayed, imperfect sample measurements, never simulator truth.
    The fixture tests bookkeeping only; its deterministic expectations are not a
    Bayesian likelihood or held-out performance experiment.
    """
    if scenario not in {'stationary', 'squeeze', 'cheap_good', 'cheap_bad', 'bot_rise'}:
        raise ValueError('Unknown scenario')
    rng = random.Random(seed)
    records = []
    sources = [('established_a', 0, .08, .20), ('established_b', 0, .05, .16)]
    if scenario in {'cheap_good', 'cheap_bad'}:
        sources.append(('new_cheap', 42, .09 if scenario == 'cheap_good' else .018, .07))
    for source, start, quality, original_price in sources:
        for birth in range(start, last_acquisition + 1, 7):
            price = round(original_price * (1 + (.003 * birth if scenario == 'squeeze' else 0)), 6)
            qid = f'quote:{source}:{birth}'
            records.append(record(qid, 'quote', source, '', birth, birth, price_per_lead=price))
            cohort = f'{source}:b{birth}'
            records.append(record('cohort:'+cohort, 'cohort', source, cohort, birth, birth,
                                  acquired_at=birth, n_leads=100, acquisition_cost=100*price, quote_id=qid))
            for day in range(birth, through + 1):
                age = day-birth
                age_factor = .75*math.exp(-age/3) + .25*math.exp(-age/70)
                cal = math.exp(-.012*max(day-35, 0)) if scenario == 'squeeze' else 1.
                humans = max(0, round(100 * quality * age_factor * cal + rng.random()))
                repeats = humans + (2 if humans and day % 11 == 0 else 0)
                bots = 1 + (day//7 if scenario == 'bot_rise' else int(day % 17 == 0))
                # Missing labels and missing telemetry are represented explicitly.
                observed = None if day % 23 == 0 else day + (2 if day % 5 == 0 else 1)
                cell = f'cell:{cohort}:{day}'
                records.append(record(cell, 'response_cell', source, cohort, day, observed,
                                      age=age, accepted_exposure=100, raw_events=repeats+bots,
                                      classifier_human_events=repeats + int(bots>4),
                                      distinct_reported_humans=humans, classifier_version='v1'))
                if day % 7 == 0:
                    # A known-probability audit is a separate delayed observation.
                    records.append(record(f'audit:{cohort}:{day}', 'audit', source, cohort, day, day+10,
                                          sampling_probability=.5, sampled_events=max(1,(repeats+bots)//2),
                                          human_labels=max(0,repeats//2-int(day % 21 == 0)),
                                          label_sensitivity=.95, label_specificity=.98))
                if day % 13 == 0 and observed is not None:
                    records.append(record(cell+':r1', 'response_cell', source, cohort, day, day+14,
                                          replaces=cell, revision=1, age=age, accepted_exposure=100,
                                          raw_events=repeats+bots, classifier_human_events=repeats,
                                          distinct_reported_humans=humans, classifier_version='audit-corrected'))
    # Contracted value is a dated fact only after announcement. No future schedule is smuggled in.
    for day in range(0, through+1, 7):
        value = .11*math.exp(-.009*max(day-35,0)) if scenario == 'squeeze' else .11
        records.append(record(f'contract:{day}', 'contract', '', '', day, day,
                              human_net_value=value))
    return tuple(records)


class Forecaster(Protocol):
    def predict(self, snapshot: Snapshot, target: str, prior_bytes: bytes, seed: int): ...


class DynamicHierarchy:
    def predict(self, snapshot: Snapshot, target: str, prior_bytes: bytes, seed: int):
        raise NotImplementedError('Design only: actual posterior inference is not implemented')


def calendar_paths(seed: int, draws: int, horizon: int):
    """Illustrative shared scenario paths; not posterior or fitted forecasts."""
    rng = random.Random(seed)
    result=[]
    for _ in range(draws):
        response=value=0.; path=[]
        for _ in range(horizon):
            common = rng.gauss(0, .01)
            response += common
            value += .6*common + rng.gauss(0,.008)
            path.append((response,value))
        result.append(tuple(path))
    return tuple(result)


def joint_scenarios(cohorts, paths):
    # The exact same draw objects feed every cohort; do not resample here.
    return tuple(tuple((cohort, path) for cohort in cohorts) for path in paths)


def contribution_path(acquisition_cost, human_net_receipts, send_costs):
    if len(human_net_receipts) != len(send_costs):
        raise ValueError('Aligned economic paths required')
    balance=-acquisition_cost; path=[]
    for rev,cost in zip(human_net_receipts,send_costs):
        balance += rev-cost
        path.append(balance)
    return tuple(path)


def first_payback(path):
    return next((age for age,value in enumerate(path) if value >= 0), math.inf)


def replay_manifest(stream, cutoffs, target):
    fixed_prior = b'{"stage":"input_contract_only","version":1}'
    prior_hash = sha256(fixed_prior).hexdigest()
    rows=[]
    for cutoff in cutoffs:
        snap=as_of(stream,cutoff)
        if target not in {r.cohort for r in snap.kinds('cohort')}:
            continue
        for scope in ('target_only','same_source','all_sources'):
            chosen=scoped(snap,target,scope)
            rows.append(dict(calendar_cutoff=cutoff,target=target,scope=scope,
                             snapshot_sha256=chosen.digest,prior_sha256=prior_hash,
                             cohorts=len(chosen.kinds('cohort')),response_cells=len(chosen.kinds('response_cell')),
                             audit_batches=len(chosen.kinds('audit')),
                             forecast_status='not_implemented',evaluation_role='input_census_only'))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('replay_input_manifest.json'))
    args=p.parse_args()
    rows=replay_manifest(fixture(scenario='cheap_bad'), [3,7,14,30,42,60,84,112], 'established_a:b0')
    args.output.write_text(json.dumps({'synthetic':True,'model_fitted':False,'rows':rows},indent=2)+'\n')
    print(f'Wrote {len(rows)} input-census rows; no posterior forecasts or model performance results')


if __name__ == '__main__':main()
