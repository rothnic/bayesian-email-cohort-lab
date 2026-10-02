"""Independent integration checks against direct probability/accounting definitions.

No production data, external service, or paid inference is needed. Artifact
checks skip when a checkout deliberately omits the generated public examples.
"""
from dataclasses import replace
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from scipy.integrate import quad
from scipy.stats import gamma, poisson

from bayes_cohort import ModelConfig, clear_fit_cache, forecast_cohort, payback_summary
from bayes_cohort.model import _fit
from cohort_lab.accounting import Policy, Snapshot, World, ledger_path
from cohort_lab.generator import (
    EVENT_COLUMNS, EXIT_COLUMNS, EXPOSURE_COLUMNS, LEAD_COLUMNS, LEDGER_COLUMNS,
)
from run_bayesian_demo import calibration_report, crps, first_payback, summarize_draws

ROOT = Path(__file__).resolve().parents[1]


def empty(columns):
    return pd.DataFrame(columns=columns)


def tiny_config(**updates):
    return replace(ModelConfig(
        source_means=(.1,), fast_weights=(.5,), fast_timescales=(1.,),
        tail_timescales=(20.,), response_delay=((0, 1.),),
        event_report_delay=((0, 1.),), settlement_delay=((0, 1.),),
        ledger_report_delay=((0, 1.),), forecast_invoice_revisions=False,
    ), **updates)


def tiny_snapshot(cutoff=0, cost=1., cohorts=("target",)):
    leads, exposure, ledger = [], [], []
    for cohort in cohorts:
        for lead in range(4):
            leads.append((f"{cohort}-u{lead}", "s", cohort, 0, cost, "USD", 0, .5, "v1"))
        exposure.append((cohort, "s", 0, 0, 0, 4, 4, 4, 0, 0, "observed", "c"))
        ledger.append((f"a-{cohort}", "", cohort, "s", 0, -4 * cost, 0, 0, "acquisition", "", "", "USD"))
    return Snapshot(cutoff, pd.DataFrame(leads, columns=LEAD_COLUMNS),
                    pd.DataFrame(exposure, columns=EXPOSURE_COLUMNS),
                    empty(EVENT_COLUMNS), pd.DataFrame(ledger, columns=LEDGER_COLUMNS),
                    empty(EXIT_COLUMNS))


def event(event_id, cohort="target", send=0, occurred=0, observed=0):
    return (f"r-{event_id}", event_id, f"{cohort}-u0", cohort, "s", f"m-{event_id}",
            0, send, occurred, observed, "human", 0, "v1")


class ScoringDefinitionTests(unittest.TestCase):
    def test_sorted_crps_matches_quadratic_empirical_definition(self):
        rng = np.random.default_rng(219)
        samples = [np.array([4.]), np.array([-3., 3.]), np.array([0., 0., 0., 2.])]
        samples += [rng.normal(size=n) for n in (7, 31, 100)]
        for draws in samples:
            for truth in (-7., 0., 2.5, 9.):
                expected = np.abs(draws - truth).mean() - .5 * np.abs(draws[:, None] - draws[None, :]).mean()
                self.assertAlmostEqual(crps(draws, truth), expected, places=12)
                self.assertGreaterEqual(crps(draws, truth), -1e-12)

    def test_crps_translation_and_scale_units(self):
        draws = np.array([-7., -1., 0., 2., 2., 4.])
        expected = crps(draws, 1.5)
        self.assertAlmostEqual(crps(draws + 137., 138.5), expected, places=12)
        self.assertAlmostEqual(crps(draws * 23., 1.5 * 23.), expected * 23., places=12)

    def test_zero_crossing_reversal_and_never_are_distinct(self):
        paths = np.full((6, 426), -1.)
        paths[1, 0] = 0.  # Initial break-even can be reversed immediately.
        paths[2, 180:] = 1.
        paths[3, 181:] = 1.
        paths[4, 425] = 1.
        paths[5, 7:9] = 0.  # Crossing at exactly zero is payback, not profit.
        expected = np.array([np.inf, 0., 180., 181., 425., 7.])
        np.testing.assert_array_equal(first_payback(paths), expected)
        summary = payback_summary(paths)
        np.testing.assert_array_equal(summary["first_payback_ages"], [-1, 0, 180, 181, 425, 7])
        self.assertEqual(summary["probabilities"], {
            "by_180": 3 / 6, "later_than_180": 2 / 6, "never_under_policy": 1 / 6,
        })
        truth = np.full(426, -1.)
        truth[180] = 0.
        rows = summarize_draws(paths, truth, [180, 425], n_leads=2)
        self.assertEqual(rows[0]["probability_profitable"], 1 / 6)
        self.assertEqual(rows[0]["probability_payback_by_horizon"], 3 / 6)
        self.assertEqual(rows[0]["probability_remains_positive"], 1 / 6)
        self.assertEqual(rows[0]["brier"], (1 / 6) ** 2)
        self.assertEqual(rows[0]["payback_brier"], (3 / 6 - 1) ** 2)
        self.assertAlmostEqual(rows[1]["probability_payback_by_horizon"] + rows[1]["probability_never"], 1.)

    def test_coverage_is_inclusive_and_profit_is_strict(self):
        paths = np.zeros((5, 3))
        rows = summarize_draws(paths, np.zeros(3), [2], n_leads=5)
        self.assertEqual(rows[0]["coverage80"], 1.)
        self.assertEqual(rows[0]["coverage95"], 1.)
        self.assertEqual(rows[0]["width80"], 0.)
        self.assertEqual(rows[0]["probability_profitable"], 0.)
        self.assertEqual(rows[0]["probability_payback_by_horizon"], 1.)
        self.assertEqual(rows[0]["brier"], 0.)
        self.assertEqual(rows[0]["payback_brier"], 0.)


class IndependentPosteriorTests(unittest.TestCase):
    def tearDown(self):
        clear_fit_cache()

    def test_source_grid_weights_match_numerical_gamma_poisson_integration(self):
        snapshot = tiny_snapshot(cutoff=2, cohorts=("history", "target"))
        extra = snapshot.exposure_daily.copy()
        extra["age"] = 1
        extra["date"] = 1
        extra["observed_at"] = 1
        extra["accepted"] = 3
        snapshot = replace(snapshot, exposure_daily=pd.concat([snapshot.exposure_daily, extra], ignore_index=True),
                           engagement_events=pd.DataFrame([
                               event("h0", "history"), event("h1", "history", send=1, occurred=1, observed=1),
                               event("t0"), event("t1"), event("t2", send=1, occurred=1, observed=1),
                           ], columns=EVENT_COLUMNS))
        config = tiny_config(source_means=(.1, .5), fast_weights=(.3, .8),
                             fast_timescales=(1., 2.), tail_timescales=(10., 30.),
                             cohort_shape=1.6, response_delay=((0, .5), (3, .5)))
        fit = _fit(snapshot, Policy(last_send_age=1, terminal_age=60), config)
        expected = []
        for mean, weight, fast, tail in fit["grid"]:
            curve = weight * np.exp(-np.array([0., 1.]) / fast) + (1 - weight) * np.exp(-np.array([0., 1.]) / tail)
            intensity = np.array([4., 3.]) * curve * .5
            prior = np.exp(-.5 * ((np.log(mean) - config.log_source_mean_prior_location) / config.log_source_mean_prior_scale) ** 2)
            marginal = prior
            for counts in (np.array([1, 1]), np.array([2, 1])):
                integral, error = quad(
                    lambda theta: gamma.pdf(theta, a=config.cohort_shape, scale=mean / config.cohort_shape)
                    * np.prod(poisson.pmf(counts, theta * intensity)),
                    0, np.inf, epsabs=1e-12, epsrel=1e-10,
                )
                self.assertLess(error, 1e-9)
                marginal *= integral
            expected.append(marginal)
        expected = np.asarray(expected) / np.sum(expected)
        np.testing.assert_allclose(fit["sources"]["s"]["weights"], expected, rtol=2e-9, atol=1e-12)
        self.assertAlmostEqual(fit["sources"]["s"]["weights"].sum(), 1.)

    def test_impossible_observed_counts_have_no_silent_posterior(self):
        snapshot = replace(tiny_snapshot(), engagement_events=pd.DataFrame([event("impossible")], columns=EVENT_COLUMNS))
        policy = Policy(last_send_age=0, terminal_age=60)
        with self.assertRaisesRegex(ValueError, "Positive response counts"):
            forecast_cohort(snapshot, policy, "target", config=tiny_config(response_delay=((30, 1.),)), draws=12)
        exposure = snapshot.exposure_daily.copy()
        exposure["accepted"] = 0
        snapshot = replace(snapshot, exposure_daily=exposure)
        with self.assertRaisesRegex(ValueError, "Positive response counts"):
            forecast_cohort(snapshot, policy, "target", config=tiny_config(), draws=12)

    def test_missing_mature_receipts_are_not_observed_zero_marks(self):
        snapshot = tiny_snapshot(cutoff=40)
        events = pd.DataFrame([event("zero"), event("positive"), event("missing")], columns=EVENT_COLUMNS)
        receipts = pd.DataFrame([
            ("b0", "zero", "target", "s", 0, 0., 0, 0, "revenue", "", "", "USD"),
            ("b1", "positive", "target", "s", 0, 2., 0, 0, "revenue", "", "", "USD"),
        ], columns=LEDGER_COLUMNS)
        snapshot = replace(snapshot, engagement_events=events, ledger=pd.concat([snapshot.ledger, receipts], ignore_index=True))
        fit = _fit(snapshot, Policy(last_send_age=0, terminal_age=60), tiny_config())
        source = fit["sources"]["s"]
        self.assertEqual(source["mature_marks"], 2)
        self.assertEqual(source["missing_mature_base_rows"], 1)
        self.assertEqual(source["positive"], (3., 3.))
        self.assertEqual(source["mature_positive_marks"], 1)

    def test_missing_exposure_and_measured_zero_have_different_information(self):
        measured = tiny_snapshot()
        exposure = measured.exposure_daily.copy()
        exposure[["attempted", "accepted", "eligible", "exits"]] = np.nan
        exposure["telemetry_status"] = "missing"
        missing = replace(measured, exposure_daily=exposure)
        policy = Policy(last_send_age=0, terminal_age=60)
        config = tiny_config()
        observed_fit = _fit(measured, policy, config)
        missing_fit = _fit(missing, policy, config)
        np.testing.assert_array_equal(observed_fit["sources"]["s"]["cohorts"]["target"][1], [4.])
        np.testing.assert_array_equal(missing_fit["sources"]["s"]["cohorts"]["target"][1], [0.])
        self.assertEqual(missing_fit["sources"]["s"]["observed_exposure_cells"], 0)
        self.assertTrue(missing.exposure_daily.accepted.isna().all())

    def test_forecasting_never_reads_generator_parameter_files(self):
        with patch("cohort_lab.accounting.load_parameters", side_effect=AssertionError("Unexpected generator parameter read")), \
             patch("pathlib.Path.read_text", side_effect=AssertionError("Unexpected file read during forecasting")):
            result = forecast_cohort(tiny_snapshot(), Policy(last_send_age=0, terminal_age=60),
                                     "target", config=tiny_config(), draws=12, seed=43)
        self.assertEqual(result["margin_paths"].shape, (12, 61))

    def test_future_leads_and_unobserved_revisions_cannot_change_asof_result(self):
        snapshot = tiny_snapshot()
        revision = pd.DataFrame([
            ("future", "e", "target", "s", 0, -100., 0, 4, "revenue_revision", "revision-e", "a-target", "USD"),
        ], columns=LEDGER_COLUMNS)
        hidden = pd.DataFrame([
            ("unavailable", "e2", "target", "s", 0, 1e9, 0, np.nan, "revenue", "", "", "USD"),
        ], columns=LEDGER_COLUMNS)
        future_leads = snapshot.leads.copy()
        future_leads["lead_id"] += "-future"
        future_leads["cohort_id"] = "future"
        future_leads["acquired_at"] = 20
        world = World("fictional", 7, pd.concat([snapshot.leads, future_leads], ignore_index=True),
                      snapshot.exposure_daily, snapshot.engagement_events,
                      pd.concat([snapshot.ledger, revision, hidden], ignore_index=True), snapshot.operational_exits)
        changed_ledger = world.ledger.copy()
        changed_ledger.loc[changed_ledger.entry_id.ne("a-target"), "amount"] *= -1000
        changed_leads = world.leads.copy()
        changed_leads.loc[changed_leads.acquired_at.gt(0), "acquisition_cost"] = 1e12
        changed = replace(world, ledger=changed_ledger, leads=changed_leads)
        a, b = world.as_of(0), changed.as_of(0)
        for name in ("leads", "exposure_daily", "engagement_events", "ledger", "operational_exits"):
            assert_frame_equal(getattr(a, name), getattr(b, name))
        args = dict(policy=Policy(last_send_age=0, terminal_age=60), cohort_id="target", config=tiny_config(), draws=30, seed=1)
        np.testing.assert_array_equal(forecast_cohort(a, **args)["margin_paths"], forecast_cohort(b, **args)["margin_paths"])

    def test_sunk_acquisition_changes_full_margin_not_incremental_continuation(self):
        config = tiny_config()
        policy = Policy(last_send_age=3, terminal_age=60)
        cheap = forecast_cohort(tiny_snapshot(cost=1.), policy, "target", config=config, draws=100, seed=14)
        expensive = forecast_cohort(tiny_snapshot(cost=100.), policy, "target", config=config, draws=100, seed=14)
        np.testing.assert_allclose(expensive["margin_paths"], cheap["margin_paths"] - 396.)
        np.testing.assert_array_equal(cheap["incremental_continuation_value"], expensive["incremental_continuation_value"])
        np.testing.assert_array_equal(cheap["pending_prior_send_revenue"], expensive["pending_prior_send_revenue"])
        np.testing.assert_array_equal(cheap["stop_sending_value"], -cheap["incremental_continuation_value"])

    def test_unreported_receipt_is_added_at_economic_date_not_report_date(self):
        snapshot = replace(tiny_snapshot(cutoff=1), engagement_events=pd.DataFrame([event("pending")], columns=EVENT_COLUMNS))
        config = tiny_config(ledger_report_delay=((3, 1.),), positive_alpha=1e9, positive_beta=1.,
                             revision_alpha=1e-9, revision_beta=1.)
        result = forecast_cohort(snapshot, Policy(last_send_age=0, terminal_age=60), "target", config=config, draws=64, seed=21)
        np.testing.assert_array_equal(result["observed_margin_path"], np.full(61, -4.))
        self.assertTrue((result["margin_paths"][:, 0] > -4.).all())
        np.testing.assert_array_equal(result["margin_paths"][:, 0], result["margin_paths"][:, 3])
        self.assertEqual(result["future_send_cost"].sum(), 0.)
        self.assertEqual(result["incremental_continuation_value"].sum(), 0.)
        self.assertTrue((result["pending_prior_send_revenue"] > 0.).all())

    def test_visible_signed_entries_and_revisions_are_preserved_once(self):
        snapshot = tiny_snapshot(cutoff=40)
        entries = pd.DataFrame([
            ("b0", "click", "target", "s", 0, 2., 0, 0, "revenue", "", "", "USD"),
            ("r0", "click", "target", "s", 0, -.75, 8, 11, "revenue_revision", "rev-click", "b0", "USD"),
            ("invoice", "", "target", "s", 0, .2, 21, 24, "acquisition_revision", "inv", "a-target", "USD"),
        ], columns=LEDGER_COLUMNS)
        snapshot = replace(snapshot, ledger=pd.concat([snapshot.ledger, entries], ignore_index=True))
        before = snapshot.ledger.copy(deep=True)
        result = forecast_cohort(snapshot, Policy(last_send_age=0, terminal_age=60), "target", config=tiny_config(), draws=20, seed=9)
        expected = ledger_path(snapshot.ledger, 0, 60)
        np.testing.assert_array_equal(result["margin_paths"], np.tile(expected, (20, 1)))
        self.assertAlmostEqual(expected[-1], -2.55)
        assert_frame_equal(snapshot.ledger, before)


class EvaluationIntegrationTests(unittest.TestCase):
    def test_calibration_uncertainty_resamples_whole_worlds(self):
        # Two independent worlds, each with three perfectly dependent forecasts.
        # A row bootstrap would generate intermediate 1/6-based coverage means;
        # the whole-world bootstrap can produce only 0, 1/2, or 1.
        metrics = ["absolute_error_per_lead", "crps_per_lead", "brier", "payback_brier",
                   "coverage80", "coverage95", "width80", "width95"]
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            for world_id, outcome in (("a-1", 0.), ("b-2", 1.)):
                directory = output / world_id
                directory.mkdir()
                rows = [dict(world_id=world_id, scenario=world_id[0], seed=i, origin_age=3,
                             horizon=180, probability_profitable=.5, true_margin=outcome,
                             **{metric: outcome for metric in metrics}) for i in range(3)]
                pd.DataFrame(rows).to_csv(directory / "scores.csv", index=False)
                pd.DataFrame([dict(model="fixed_aggregate", absolute_error_per_lead=outcome)]).to_csv(directory / "baseline_scores.csv", index=False)
            calibration_report(output)
            result = pd.read_csv(output / "metrics_by_origin.csv")
            self.assertTrue(result.worlds.eq(2).all())
            self.assertTrue(result.forecasts.eq(6).all())
            self.assertTrue(result["mean"].eq(.5).all())
            self.assertTrue(result.lower95.eq(0.).all())
            self.assertTrue(result.upper95.eq(1.).all())

    def test_checked_in_scores_have_independent_world_ids_and_prospective_horizons(self):
        output = ROOT / "artifacts" / "calibration"
        if not (output / "scores.csv").exists():
            self.skipTest("Generated calibration artifacts are not present")
        scores = pd.read_csv(output / "scores.csv")
        meta = json.loads((output / "run_metadata.json").read_text())
        self.assertEqual(scores.world_id.nunique(), 12)
        self.assertEqual(len({run["seed"] for run in meta["runs"]}), 12)
        self.assertTrue((scores.horizon > scores.origin_age).all())
        self.assertFalse(scores.duplicated(["world_id", "cohort_id", "origin_age", "horizon"]).any())
        self.assertTrue(meta["synthetic_only"])
        self.assertFalse(meta["quick"])

    def test_public_draws_and_cdf_use_same_unconditional_denominator(self):
        for mode in ("demo", "cold_start"):
            output = ROOT / "artifacts" / mode
            if not (output / "posterior_predictive_draws.csv.gz").exists():
                continue
            draws = pd.read_csv(output / "posterior_predictive_draws.csv.gz")
            cdf = pd.read_csv(output / "payback_cdf.csv.gz")
            summary = pd.read_csv(output / "summary.csv")
            for (cohort, origin), group in draws.groupby(["cohort_id", "origin_age"]):
                first = group.drop_duplicates("draw_id").first_payback_age
                self.assertEqual(len(first), int(group.draw_count.iloc[0]))
                curve = cdf.loc[cdf.cohort_id.eq(cohort) & cdf.origin_age.eq(origin)].sort_values("age")
                expected = [first.le(age).mean() for age in curve.age]
                np.testing.assert_allclose(curve.cdf, expected, atol=1e-12, rtol=0)
                self.assertAlmostEqual(curve.cdf.iloc[-1] + first.isna().mean(), 1.)
                for row in summary.loc[summary.cohort_id.eq(cohort) & summary.origin_age.eq(origin)].itertuples():
                    outcome = group.loc[group.horizon.eq(row.horizon), "margin"].to_numpy()
                    self.assertAlmostEqual(getattr(row, "mean"), outcome.mean(), places=10)
                    self.assertAlmostEqual(row.probability_profitable, np.mean(outcome > 0), places=12)
                    self.assertAlmostEqual(row.crps, crps(outcome, row.true_margin), places=10)

    def test_public_observation_schema_is_fictional_and_excludes_latent_truth(self):
        output = ROOT / "artifacts" / "demo" / "synthetic_observations"
        if not output.exists():
            self.skipTest("Generated demo observations are not present")
        leads = pd.read_csv(output / "leads.csv.gz")
        self.assertEqual(leads.columns.tolist(), LEAD_COLUMNS)
        self.assertTrue(leads.lead_id.str.fullmatch(r"u\d+").all())
        self.assertEqual(set(leads.source_id), {"low", "middle", "high"})
        events = pd.read_csv(output / "engagement_events.csv.gz")
        self.assertEqual(events.columns.tolist(), EVENT_COLUMNS)
        self.assertTrue(events.event_id.str.fullmatch(r"e\d+").all())
        self.assertFalse({"persistent_quality", "dormant_returner", "cohort_quality"} & set(leads.columns))
        self.assertFalse({"email", "name", "phone", "address"} & set(leads.columns))


if __name__ == "__main__":
    unittest.main()
