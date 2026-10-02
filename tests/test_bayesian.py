"""Focused model leakage, conjugacy, accounting and predictive tests."""
from dataclasses import replace
import json
import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from cohort_lab.accounting import Snapshot, Policy
from cohort_lab.generator import generate_world, LEAD_COLUMNS, EXPOSURE_COLUMNS, EVENT_COLUMNS, LEDGER_COLUMNS, EXIT_COLUMNS
from bayes_cohort import ModelConfig, clear_fit_cache, forecast_cohort, forecast_records, payback_summary, default_config_dict


def empty(columns):
    return pd.DataFrame(columns=columns)


def one_day_snapshot(accepted=20, missing=False, cost=1.):
    leads = pd.DataFrame([(f"u{i}", "s", "target", 0, cost, "USD", 0, .5, "v1") for i in range(accepted)], columns=LEAD_COLUMNS)
    exposure = pd.DataFrame([("target", "s", 0, 0, 0, accepted, accepted, accepted, 0, 0, "observed", "c")], columns=EXPOSURE_COLUMNS)
    if missing:
        exposure[["attempted", "accepted", "eligible", "exits"]] = np.nan
        exposure["telemetry_status"] = "missing"
    ledger = pd.DataFrame([("a", "", "target", "s", 0, -accepted * cost, 0, 0, "acquisition", "", "", "USD")], columns=LEDGER_COLUMNS)
    return Snapshot(0, leads, exposure, empty(EVENT_COLUMNS), ledger, empty(EXIT_COLUMNS))


def simple_config(**updates):
    config = ModelConfig(source_means=(.1,), fast_weights=(.5,), fast_timescales=(1.,), tail_timescales=(20.,),
                         response_delay=((0, 1.),), event_report_delay=((0, 1.),), settlement_delay=((0, 1.),),
                         ledger_report_delay=((0, 1.),), forecast_invoice_revisions=False)
    return replace(config, **updates)


class BayesianTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world, cls.truth, cls.policy = generate_world(11, "collapse_tail", small=True)

    def tearDown(self):
        clear_fit_cache()

    def test_public_inputs_and_json_diagnostics(self):
        for invalid in (self.world, self.truth):
            with self.assertRaises(TypeError):
                forecast_cohort(invalid, self.policy, "high-w05", draws=10)
        result = forecast_cohort(self.world.as_of(38), self.policy, "high-w05", draws=100, seed=4)
        self.assertEqual(result["margin_paths"].shape, (100, 426))
        self.assertTrue(np.isfinite(result["margin_paths"]).all())
        self.assertGreater(result["margin_paths"][:, -1].std(), 0)
        json.dumps(result["diagnostics"], allow_nan=False)
        json.dumps(default_config_dict(), allow_nan=False)
        self.assertAlmostEqual(sum(result["payback_state_probabilities"].values()), 1)

    def test_seed_replay_and_no_mutation(self):
        snapshot = self.world.as_of(42)
        names = ("leads", "exposure_daily", "engagement_events", "ledger", "operational_exits")
        before = {name: getattr(snapshot, name).copy(deep=True) for name in names}
        a = forecast_cohort(snapshot, self.policy, "middle-w05", draws=80, seed=4)
        b = forecast_cohort(snapshot, self.policy, "middle-w05", draws=80, seed=4)
        c = forecast_cohort(snapshot, self.policy, "middle-w05", draws=80, seed=5)
        np.testing.assert_array_equal(a["margin_paths"], b["margin_paths"])
        self.assertFalse(np.array_equal(a["margin_paths"], c["margin_paths"]))
        for name in names:
            assert_frame_equal(before[name], getattr(snapshot, name))

    def test_future_perturbations_do_not_leak(self):
        cutoff = 42
        ledger = self.world.ledger.copy(deep=True)
        invisible = ledger.economic_at.gt(cutoff) | ledger.observed_at.gt(cutoff) | ledger.observed_at.isna()
        ledger.loc[invisible, "amount"] = -1e8
        events = self.world.engagement_events.copy(deep=True)
        invisible = events.occurred_at.gt(cutoff) | events.observed_at.gt(cutoff) | events.observed_at.isna()
        events.loc[invisible, "qualification"] = "bot"
        exposure = self.world.exposure_daily.copy(deep=True)
        invisible = exposure.date.gt(cutoff) | exposure.observed_at.gt(cutoff) | exposure.observed_at.isna()
        exposure.loc[invisible, ["attempted", "accepted", "eligible", "exits"]] = 1e9
        changed = replace(self.world, ledger=ledger, engagement_events=events, exposure_daily=exposure)
        a = forecast_cohort(self.world.as_of(cutoff), self.policy, "middle-w05", draws=80, seed=4)
        b = forecast_cohort(changed.as_of(cutoff), self.policy, "middle-w05", draws=80, seed=4)
        np.testing.assert_array_equal(a["margin_paths"], b["margin_paths"])

    def test_delay_thinning_matches_gamma_posterior(self):
        snapshot = one_day_snapshot()
        policy = Policy(last_send_age=0, terminal_age=60)
        complete = forecast_cohort(snapshot, policy, "target", config=simple_config(), draws=4000, seed=2)
        delayed = forecast_cohort(snapshot, policy, "target", config=simple_config(response_delay=((30, 1.),)), draws=4000, seed=2)
        # Gamma shape=8, rate=8/.1 + accepted*q. An absent future report
        # contributes q=0, rather than twenty falsely observed zero responses.
        self.assertAlmostEqual(complete["parameter_draws"]["cohort_amplitude"].mean(), 8 / 100, delta=.002)
        self.assertAlmostEqual(delayed["parameter_draws"]["cohort_amplitude"].mean(), 8 / 80, delta=.002)

    def test_missing_telemetry_is_not_a_zero_denominator(self):
        policy = Policy(last_send_age=0, terminal_age=60)
        missing = one_day_snapshot(missing=True)
        result = forecast_cohort(missing, policy, "target", config=simple_config(), draws=4000, seed=2)
        self.assertEqual(result["diagnostics"]["source_observed_exposure_cells"], 0)
        self.assertEqual(result["diagnostics"]["missing_target_past_exposure_cells"], 1)
        self.assertAlmostEqual(result["parameter_draws"]["cohort_amplitude"].mean(), .1, delta=.002)
        self.assertTrue(missing.exposure_daily.accepted.isna().all())

    def test_partial_arm_telemetry_does_not_masquerade_as_complete(self):
        snapshot = one_day_snapshot(accepted=2)
        extra = snapshot.exposure_daily.copy()
        extra["arm"] = 1
        extra[["attempted", "accepted", "eligible", "exits"]] = np.nan
        snapshot = replace(snapshot, exposure_daily=pd.concat([snapshot.exposure_daily, extra], ignore_index=True))
        result = forecast_cohort(snapshot, Policy(last_send_age=0, terminal_age=60), "target", config=simple_config(), draws=20)
        self.assertEqual(result["diagnostics"]["source_observed_exposure_cells"], 0)

    def test_mature_only_marks_and_age_updated_uncertainty(self):
        counts, variation, distributions = [], [], []
        for age in (3, 7, 14, 30, 60):
            result = forecast_cohort(self.world.as_of(35 + age), self.policy, "middle-w05", draws=500, seed=7)
            counts.append(result["diagnostics"]["target_observed_click_count"])
            amplitude = result["parameter_draws"]["cohort_amplitude"]
            variation.append(amplitude.std() / amplitude.mean())
            distributions.append(result["diagnostics"]["tail_timescale_probabilities"])
            self.assertGreater(result["margin_paths"][:, -1].std(), 0)
        self.assertEqual(counts, sorted(counts))
        self.assertLess(variation[-1], variation[0])
        self.assertNotEqual(distributions[0], distributions[-1])
        fresh = forecast_cohort(self.world.as_of(3), self.policy, "middle-w00", draws=100, seed=7)
        self.assertTrue(fresh["diagnostics"]["mark_prior_only"])
        self.assertEqual(fresh["diagnostics"]["mature_mark_count"], 0)
        self.assertGreater(fresh["margin_paths"][:, -1].std(), 0)

    def test_within_source_partial_pooling_changes_target_posterior(self):
        snapshot = self.world.as_of(38)
        cohort = "middle-w05"
        isolated = replace(snapshot, **{name: getattr(snapshot, name).loc[getattr(snapshot, name).cohort_id.eq(cohort)].copy() for name in
                                       ("leads", "exposure_daily", "engagement_events", "ledger", "operational_exits")})
        full = forecast_cohort(snapshot, self.policy, cohort, draws=500, seed=5)
        alone = forecast_cohort(isolated, self.policy, cohort, draws=500, seed=5)
        self.assertGreater(full["diagnostics"]["source_observed_click_count"], alone["diagnostics"]["source_observed_click_count"])
        self.assertNotEqual(full["diagnostics"]["tail_timescale_probabilities"], alone["diagnostics"]["tail_timescale_probabilities"])
        self.assertFalse(np.array_equal(full["parameter_draws"]["cohort_amplitude"], alone["parameter_draws"]["cohort_amplitude"]))

    def test_complete_policy_reconciles_all_signed_ledger_entries(self):
        snapshot = self.world.as_of(35 + self.policy.terminal_age)
        result = forecast_cohort(snapshot, self.policy, "high-w05", draws=60, seed=2)
        np.testing.assert_array_equal(result["margin_paths"], np.tile(result["observed_margin_path"], (60, 1)))
        np.testing.assert_allclose(result["observed_margin_path"], self.truth.cohort_path(self.world.leads, "high-w05", self.policy))
        self.assertEqual(result["future_send_cost"].sum(), 0)
        self.assertEqual(result["incremental_continuation_value"].sum(), 0)

    def test_pending_observed_click_is_settled_after_sending_ends(self):
        snapshot = one_day_snapshot(accepted=1)
        event = pd.DataFrame([("r", "e", "u0", "target", "s", "m", 0, 0, 0, 0, "human", 0, "v1")], columns=EVENT_COLUMNS)
        snapshot = replace(snapshot, engagement_events=event)
        config = simple_config(settlement_delay=((2, 1.),))
        result = forecast_cohort(snapshot, Policy(last_send_age=0, terminal_age=30), "target", config=config, draws=500, seed=8)
        self.assertEqual(result["diagnostics"]["pending_observed_clicks_with_unreported_base"], 1)
        self.assertEqual(result["future_send_cost"].sum(), 0)
        self.assertGreater(result["pending_prior_send_revenue"].mean(), 0)
        self.assertGreater((result["margin_paths"][:, 2] > result["margin_paths"][:, 1]).sum(), 0)
        np.testing.assert_array_equal(result["margin_paths"][:, 0], np.full(500, -1.))

    def test_outstanding_signed_revision_can_reverse_payback(self):
        snapshot = one_day_snapshot(accepted=1, cost=1.8)
        event = pd.DataFrame([("r", "e", "u0", "target", "s", "m", 0, 0, 0, 0, "human", 0, "v1")], columns=EVENT_COLUMNS)
        base = pd.DataFrame([("b", "e", "target", "s", 0, 2., 0, 0, "revenue", "", "", "USD")], columns=LEDGER_COLUMNS)
        snapshot = replace(snapshot, engagement_events=event, ledger=pd.concat([snapshot.ledger, base], ignore_index=True))
        config = simple_config(revision_alpha=1e9, revision_beta=1.)
        result = forecast_cohort(snapshot, Policy(last_send_age=0, terminal_age=30), "target", config=config, draws=100, seed=2)
        np.testing.assert_allclose(result["margin_paths"][:, 0], .2)
        self.assertTrue((result["margin_paths"][:, -1] < 0).all())
        self.assertEqual(result["payback_state_probabilities"]["by_180"], 1.)
        rows = forecast_records(result, "test", 0, horizons=(30,))
        self.assertTrue(all(not row["remains_positive_after_payback"] for row in rows))

    def test_invoice_adjustment_is_uncertain_until_reported(self):
        snapshot = one_day_snapshot(accepted=2)
        result = forecast_cohort(snapshot, Policy(last_send_age=0, terminal_age=60), "target", config=replace(simple_config(), forecast_invoice_revisions=True), draws=100, seed=2)
        self.assertEqual(result["diagnostics"]["unobserved_acquisition_invoice_arms"], 1)
        self.assertGreater(result["margin_paths"][:, 21].std(), 0)
        np.testing.assert_array_equal(result["margin_paths"][:, 0], np.full(100, -2.))

    def test_unconditional_payback_keeps_never_atom(self):
        paths = np.full((3, 426), -1.)
        paths[0, 10:] = 1
        paths[1, 200:] = 1
        result = payback_summary(paths)
        self.assertEqual(result["first_payback_ages"].tolist(), [10, 200, -1])
        self.assertEqual(result["probabilities"], {"by_180": 1 / 3, "later_than_180": 1 / 3, "never_under_policy": 1 / 3})

    def test_records_retro_roles_and_probabilities(self):
        forecast = forecast_cohort(self.world.as_of(95), self.policy, "middle-w05", draws=30, seed=9)
        rows = forecast_records(forecast, "test", 9)
        self.assertNotIn(30, {row["horizon"] for row in rows})
        retro = forecast_records(forecast, "test", 9, include_retrospective=True)
        self.assertEqual(retro[0]["evaluation_role"], "retrospective")
        self.assertTrue(all(row["forecast_type"] == "posterior_predictive" and 0 <= row["probability_profitable"] <= 1 for row in retro))

    def test_frozen_prior_sensitivity_changes_forecasts(self):
        snapshot = self.world.as_of(38)
        short = forecast_cohort(snapshot, self.policy, "middle-w05", config=ModelConfig(tail_timescales=(20.,)), draws=100, seed=8)
        long = forecast_cohort(snapshot, self.policy, "middle-w05", config=ModelConfig(tail_timescales=(400.,)), draws=100, seed=8)
        self.assertNotEqual(short["margin_paths"][:, -1].mean(), long["margin_paths"][:, -1].mean())
        with self.assertRaises(ValueError):
            forecast_cohort(snapshot, self.policy, "middle-w05", config={"latent_quality": 1.})
        with self.assertRaises(ValueError):
            ModelConfig(response_delay=((0, .3),))
        with self.assertRaises(ValueError):
            ModelConfig(mark_maturity_days=1)

    def test_bounded_log_variance_has_finite_predictive_value_moments(self):
        snapshot = one_day_snapshot(accepted=2)
        policy = Policy(last_send_age=0, terminal_age=60)
        config = replace(simple_config(), max_log_mark_variance=.4)
        result = forecast_cohort(snapshot, policy, "target", config=config, draws=1000, seed=1)
        variance = result["parameter_draws"]["log_mark_sigma"] ** 2
        self.assertTrue((variance <= .4).all())
        self.assertTrue((variance > 0).all())
        self.assertGreater(result["diagnostics"]["untruncated_log_variance_posterior_mass_below_bound"], 0)
        self.assertLess(result["diagnostics"]["untruncated_log_variance_posterior_mass_below_bound"], 1)
        self.assertTrue(np.isfinite(result["parameter_draws"]["log_mark_mean"]).all())
        self.assertEqual(result["diagnostics"]["max_log_mark_variance"], .4)
        self.assertEqual(result["diagnostics"]["grid_boundary_mass"]["source_mean"], {"lower": 1., "upper": 1.})
        with self.assertRaises(ValueError):
            ModelConfig(max_log_mark_variance=0.)


if __name__ == "__main__":
    unittest.main()
