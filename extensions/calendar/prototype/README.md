# Calendar input-contract prototype

Run with Python 3.10 or later, standard library only:

```
cd prototype
python -m unittest -v test_calendar_replay
python calendar_replay.py --output replay_input_manifest.json
```

This is a synthetic aggregate fixture and an executable input/accounting contract. It fits no posterior, produces no profitability forecast and establishes no model-quality result. `DynamicHierarchy.predict` deliberately raises `NotImplementedError` until real inference exists. `replay_input_manifest.json` contains only input census counts and content hashes.

The separate `replay_v1_ablation.py` uses the existing research repository's NumPy/pandas/SciPy dependencies to run twenty real unchanged-v1 forecasts in one development world. Run all sixteen tests with `python -m unittest -v` when those dependencies and the research repository are available. See `../REPLAY_CHECKPOINT.md` for reproduction and limits. The standard-library fixture above and the fitted-v1 replay are separate artifacts; neither implements the proposed dynamic hierarchy.

The fixture creates continuing weekly cohorts; a later cheap source can be good or bad. It separates age, calendar date, observation time, raw activity, reported human qualification, delayed imperfect audits and dated acquisition quotes. The squeeze scenario lowers human activity and known net-payment opportunity while raising later purchase quotes. Source quality is simulator configuration, never a fitted conclusion. The illustrative future calendar draws are shared scenario inputs, not posterior paths.

Immutable tuples prevent edits through a nominally frozen snapshot. Missing observations are omitted, measurement revisions replace their prior version, and monetary corrections are separate signed entries. A cohort's actual acquisition cost stays fixed even when later prices rise. A selected current contract applies only after its effective date; advance contracted schedules would need a separate explicitly known-future input type in the model API.

These are newly authored small aggregate fixtures. No individual-lead table, existing row-level ledger or denied export is written or repackaged. The original research code and all its results remain unchanged. The design and independent review in the parent folder specify the next inference work and its acceptance criteria.
