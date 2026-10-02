"""Deterministic leakage, finite-policy, accounting and replay tests."""
from dataclasses import replace
import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from cohort_lab.accounting import Snapshot, World, Policy, load_parameters, summarize_path
from cohort_lab.baselines import MODELS, clear_fit_cache, forecast_cohort, forecast_records
from cohort_lab.evaluation import evaluate_world, allocate_budget
from cohort_lab.generator import (generate_world, LEAD_COLUMNS, EXPOSURE_COLUMNS,
                                  EVENT_COLUMNS, LEDGER_COLUMNS, EXIT_COLUMNS)


def empty(columns):
    return pd.DataFrame(columns=columns)


def simple_snapshot(cost=1.0):
    leads = pd.DataFrame([
        ("t1", "s", "target", 0, cost, "USD", 0, .5, "v1"),
        ("t2", "s", "target", 0, cost, "USD", 1, .5, "v1"),
    ], columns=LEAD_COLUMNS)
    ledger = pd.DataFrame([
        ("a", "", "target", "s", 0, -2 * cost, 0, 0, "acquisition", "", "", "USD"),
    ], columns=LEDGER_COLUMNS)
    exposure = pd.DataFrame([
        ("target", "s", 0, 0, 0, 1, 1, 1, 0, 0, "observed", "c"),
        ("target", "s", 0, 0, 1, 1, 1, 1, 0, 0, "observed", "c"),
    ], columns=EXPOSURE_COLUMNS)
    return Snapshot(0, leads, exposure, empty(EVENT_COLUMNS), ledger, empty(EXIT_COLUMNS))


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parameters = load_parameters()
        cls.world, cls.truth, cls.policy = generate_world(11, "collapse_tail", cls.parameters, small=True)

    def tearDown(self):
        clear_fit_cache()

    def test_full_point_forecasts_and_probability_contract(self):
        snapshot = self.world.as_of(31)
        costs = []
        for model in MODELS:
            result = forecast_cohort(snapshot, self.policy, "high-w04", model, self.parameters)
            costs.append(result["future_send_cost"])
            self.assertEqual(len(result["margin_path"]), 426)
            self.assertTrue(np.isfinite(result["margin_path"]).all())
            self.assertEqual(result["origin_age"], 3)
            records = forecast_records(result, "fictional", 11)
            self.assertEqual([r["horizon"] for r in records], [30, 90, 180, 425])
            for record in records:
                self.assertEqual(record["draw_id"], 0)
                self.assertEqual(record["forecast_type"], "point")
                self.assertIsNone(record["probability_profitable"])
                self.assertEqual(record["evaluation_role"], "prospective")
        self.assertEqual(costs[0], costs[1])

    def test_zero_history_no_truth_prior_and_inactivity_not_exit(self):
        snapshot = simple_snapshot()
        policy = Policy(last_send_age=2, max_response_delay=2, max_accounting_delay=2,
                        terminal_age=6, send_cost=.01)
        for model in MODELS:
            result = forecast_cohort(snapshot, policy, "target", model, {})
            np.testing.assert_allclose(result["margin_path"], [-2, -2.02, -2.04, -2.04, -2.04, -2.04, -2.04])
            self.assertAlmostEqual(result["future_send_cost"], .04)
            self.assertAlmostEqual(result["incremental_continuation_value"], -.04)
            self.assertEqual(forecast_records(result, "fictional", 0, [6])[0]["payback_state"], "never_under_policy")

    def test_no_mutation_of_observations(self):
        snapshot = self.world.as_of(35)
        before = {name: getattr(snapshot, name).copy(deep=True) for name in
                  ("leads", "exposure_daily", "engagement_events", "ledger", "operational_exits")}
        for model in MODELS:
            forecast_cohort(snapshot, self.policy, "middle-w04", model, self.parameters)
        for name, frame in before.items():
            assert_frame_equal(frame, getattr(snapshot, name))

    def test_future_perturbations_cannot_change_earlier_forecast(self):
        cutoff = 35
        original = self.world.as_of(cutoff)
        leads = self.world.leads.copy(deep=True)
        leads.loc[leads.acquired_at.gt(cutoff), "acquisition_cost"] = 1e8
        ledger = self.world.ledger.copy(deep=True)
        invisible = ledger.economic_at.gt(cutoff) | ledger.observed_at.gt(cutoff) | ledger.observed_at.isna()
        ledger.loc[invisible, "amount"] = -1e10
        exposure = self.world.exposure_daily.copy(deep=True)
        invisible = exposure.date.gt(cutoff) | exposure.observed_at.gt(cutoff) | exposure.observed_at.isna()
        exposure.loc[invisible, ["attempted", "accepted", "eligible", "exits"]] = 1e9
        events = self.world.engagement_events.copy(deep=True)
        invisible = events.occurred_at.gt(cutoff) | events.observed_at.gt(cutoff) | events.observed_at.isna()
        events.loc[invisible, "qualification"] = "bot"
        events.loc[invisible, "revision_sequence"] = 100000
        exits = self.world.operational_exits.copy(deep=True)
        invisible = exits.occurred_at.gt(cutoff) | exits.observed_at.gt(cutoff) | exits.observed_at.isna()
        exits.loc[invisible, "reason"] = "future_changed"
        changed = replace(self.world, leads=leads, exposure_daily=exposure, engagement_events=events, ledger=ledger, operational_exits=exits).as_of(cutoff)
        for name in ("leads", "exposure_daily", "engagement_events", "ledger", "operational_exits"):
            assert_frame_equal(getattr(original, name), getattr(changed, name))
        for model in MODELS:
            a = forecast_cohort(original, self.policy, "high-w04", model, self.parameters)
            b = forecast_cohort(changed, self.policy, "high-w04", model, self.parameters)
            np.testing.assert_array_equal(a["margin_path"], b["margin_path"])
            self.assertEqual(a["stop_sending_value"], b["stop_sending_value"])

    def test_world_and_truth_are_not_model_inputs(self):
        for forbidden in (self.world, self.truth):
            with self.assertRaises(TypeError):
                forecast_cohort(forbidden, self.policy, "high-w04", MODELS[0], self.parameters)

    def test_pending_known_click_settlement_after_last_send_is_preserved(self):
        leads = pd.DataFrame([
            ("h", "s", "history", 0, 1., "USD", 0, .5, "v1"),
            ("t", "s", "target", 5, 1., "USD", 0, .5, "v1"),
        ], columns=LEAD_COLUMNS)
        exposure = pd.DataFrame([
            ("history", "s", 0, 0, 0, 1, 1, 1, 0, 0, "observed", "c"),
            ("target", "s", 5, 0, 0, 1, 1, 1, 0, 5, "observed", "c"),
        ], columns=EXPOSURE_COLUMNS)
        events = pd.DataFrame([
            ("r0", "eh", "h", "history", "s", "mh", 0, 0, 0, 0, "human", 0, "v1"),
            ("r1", "et", "t", "target", "s", "mt", 0, 5, 5, 5, "human", 0, "v1"),
        ], columns=EVENT_COLUMNS)
        ledger = pd.DataFrame([
            ("ah", "", "history", "s", 0, -1., 0, 0, "acquisition", "", "", "USD"),
            ("rh", "eh", "history", "s", 0, 2., 2, 2, "revenue", "", "", "USD"),
            ("at", "", "target", "s", 0, -1., 5, 5, "acquisition", "", "", "USD"),
        ], columns=LEDGER_COLUMNS)
        snapshot = Snapshot(5, leads, exposure, events, ledger, empty(EXIT_COLUMNS))
        policy = Policy(last_send_age=0, max_response_delay=1, max_accounting_delay=2, terminal_age=3)
        for model in MODELS:
            result = forecast_cohort(snapshot, policy, "target", model, {})
            np.testing.assert_allclose(result["margin_path"], [-1, -1, 1, 1])
            self.assertEqual(result["incremental_continuation_value"], 0)
            self.assertEqual(result["pending_prior_send_revenue"], 2)

    def test_stop_sending_ignores_sunk_acquisition_cost(self):
        policy = Policy(last_send_age=2, terminal_age=6, send_cost=.01)
        for model in MODELS:
            cheap = forecast_cohort(simple_snapshot(1), policy, "target", model, {})
            expensive = forecast_cohort(simple_snapshot(1000), policy, "target", model, {})
            self.assertEqual(cheap["stop_sending_value"], expensive["stop_sending_value"])
            self.assertEqual(cheap["future_send_cost"], expensive["future_send_cost"])

    def test_missing_telemetry_is_not_changed_into_zero(self):
        snapshot = simple_snapshot()
        exposure = snapshot.exposure_daily.copy()
        exposure[["attempted", "accepted", "eligible", "exits"]] = np.nan
        missing = replace(snapshot, exposure_daily=exposure)
        policy = Policy(last_send_age=2, terminal_age=6, send_cost=.01)
        for model in MODELS:
            result = forecast_cohort(missing, policy, "target", model, {})
            self.assertAlmostEqual(result["future_send_cost"], .04)
            self.assertTrue(missing.exposure_daily.accepted.isna().all())
            self.assertTrue(any("Missing target past exposure" in x for x in result["assumptions"]))

    def test_retrospective_horizons_are_labelled_or_omitted(self):
        snapshot = self.world.as_of(88)
        result = forecast_cohort(snapshot, self.policy, "high-w04", "observable_survival", self.parameters)
        self.assertEqual(result["origin_age"], 60)
        rows = forecast_records(result, "fictional", 0)
        self.assertNotIn(30, [r["horizon"] for r in rows])
        rows = forecast_records(result, "fictional", 0, include_retrospective=True)
        self.assertEqual(rows[0]["evaluation_role"], "retrospective")

    def test_never_and_later_180_are_distinct(self):
        never = np.full(426, -1.)
        late = never.copy(); late[200:] = 1.
        self.assertEqual(summarize_path(never, [425])[0]["payback_state"], "never_under_policy")
        self.assertEqual(summarize_path(late, [425])[0]["payback_state"], "later_than_180")
        self.assertEqual(summarize_path(late, [425])[0]["first_payback_age"], 200)

    def test_budget_capacities_and_losing_sources(self):
        quantities = allocate_budget({"a": 1., "b": .5, "c": -.2}, {"a": 1., "b": 1., "c": 1.}, {"a": 2, "b": 3, "c": 9}, 4.5)
        self.assertEqual(quantities, {"a": 2., "b": 2.5, "c": 0.})
        self.assertLessEqual(sum(quantities.values()), 4.5)

    def test_replay_strictly_prospective_and_false_stop_accounting(self):
        result = evaluate_world(self.world, self.truth, self.policy, self.parameters, origin_ages=[30, 60], horizons=[30, 90, 425])
        scores = result["margin_scores"]
        self.assertTrue((scores.horizon > scores.origin_age).all())
        self.assertTrue((result["forecasts"].evaluation_role == "at_origin_nowcast").any())
        self.assertEqual(result["summary"]["n_target_cohorts"], 9)
        self.assertEqual(result["summary"]["decision_settings"], {
            "normal_batch": 500.0, "small_batch_fraction": .25,
            "small_positive_margin_threshold": .01})
        self.assertEqual(result["summary"]["allocation_settings"], {
            "budget": 100.0, "default_source_capacity": 500.0,
            "source_capacity_overrides": {}})
        self.assertIn("calibration", result["summary"]["uncertainty_status"])
        self.assertTrue(result["forecasts"].probability_profitable.isna().all())
        decisions = result["decisions"]
        np.testing.assert_allclose(decisions.decision_regret,
                                   decisions.oracle_realized_margin - decisions.chosen_realized_margin)
        np.testing.assert_allclose(decisions.always_continue_realized_margin - decisions.chosen_realized_margin,
                                   decisions.forgone_profitable_margin - decisions.losses_avoided_vs_always_continue)
        for _, allocation in result["allocations"].iterrows():
            self.assertLessEqual(allocation.acquisition_spend, allocation.budget + 1e-9)
            self.assertGreaterEqual(allocation.allocation_regret, 0)
            for source, quantity in allocation.chosen_quantities.items():
                self.assertLessEqual(quantity, allocation.capacity_per_source[source] + 1e-9)


if __name__ == "__main__":
    unittest.main()
