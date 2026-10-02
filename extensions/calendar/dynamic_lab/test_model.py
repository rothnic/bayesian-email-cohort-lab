"""Reproducible numerical/contract checks; these are NOT calibration evidence.

Run from extension root:
OPENBLAS_NUM_THREADS=1 python -m unittest -v dynamic_lab.test_model
"""
import unittest
from dataclasses import replace, asdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.linalg import cho_solve

from dynamic_lab.model import fit, ModelConfig, MODES, REQUIRED, age_shape, _measurements, SCENARIOS as POLICIES
from dynamic_lab.simulation import generate, SCENARIOS


def prospective():
    return dict(cohort_id="new_cheap-b070", source_id="new_cheap", birth_day=70,
                leads=100, acquisition_cost=8., daily_send_cost=.025, source_feature=0.)


def prior_tail_diagnostics():
    """Source-free analytical prior expectation and fixed-seed MC sensitivity.

    Reproduce without inspecting held-out outcomes:
    OPENBLAS_NUM_THREADS=1 python -m dynamic_lab.test_model --prior-tail-check dynamic_lab/prior_tail_summary.json
    """
    empty = pd.DataFrame(columns=list(REQUIRED) + ["source_feature"])
    target = dict(prospective(), birth_day=0)
    base = ModelConfig()
    rows = []
    for label, factor in (("smaller_scales", .5), ("default", 1.), ("larger_scales", 1.35)):
        config = replace(base,
            source_sds=tuple(factor * x for x in base.source_sds),
            cohort_sds=tuple(factor * x for x in base.cohort_sds),
            calendar_sds=tuple(factor * x for x in base.calendar_sds))
        ages = np.arange(366)
        source_moment = sum(w * np.exp(s * s / 2) for w, s in zip(config.effect_weights, config.source_sds))
        cohort_moment = sum(w * np.exp(s * s / 2) for w, s in zip(config.effect_weights, config.cohort_sds))
        # Weekly piecewise-linear process, not a daily Brownian interpolation:
        # Var(x_a | q)=q²*step*(floor(a/step)+(a/step-floor(a/step))²).
        step = config.calendar_bin_days
        bins, fraction = np.divmod(ages, step)
        variance_time = step * bins + np.square(fraction) / step
        calendar_moment = sum(w * np.exp(q * q * variance_time / 2) for w, q in zip(config.calendar_weights, config.calendar_sds))
        analytical = 100 * np.exp(config.response_log_location + config.payout_log_location +
            config.location_sd ** 2) * (source_moment * cohort_moment) ** 2 * np.sum(
                age_shape(ages) * calendar_moment ** 2)
        fitted = fit(empty, 1, config=config, seed=1777)
        for budget in ((512, 2048, 8192) if label == "default" else (8192,)):
            p = fitted.predict([target], draws=budget, scenario="hold_current")
            totals = p["revenue_paths"].sum(axis=(1, 2))
            mean, mcse = float(totals.mean()), float(totals.std(ddof=1) / np.sqrt(budget))
            rows.append(dict(scale_sensitivity=label, scale_multiplier=factor,
                draws=budget, seed=1777, cutoff=1, horizon=425, last_send_age=365,
                accepted_exposure=100, observed_rows=0, scenario="hold_current",
                mean_lifetime_revenue=mean, analytic_mean_lifetime_revenue=float(analytical),
                monte_carlo_standard_error=mcse, relative_monte_carlo_standard_error=mcse / mean,
                largest_one_percent_draw_revenue_share=p["diagnostics"]["top_one_percent_revenue_share"],
                config=asdict(config), config_hash=fitted.config_hash))
    return dict(evidence="Prior/component check, not held-out calibration or empirical Bayes tuning",
        mean_derivation="100*exp(mu_R+mu_V+location_sd^2)*(Eexp(source_sd^2/2)*Eexp(cohort_sd^2/2))^2*sum_age[f(age)*(Eexp(calendar_sd^2*(step*floor(age/step)+(age mod step)^2/step)/2))^2]",
        rows=rows)


class FittedModelChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world, _ = generate(913, "cheap_good")
        cls.frame = cls.world.as_of(69).frame()
        cls.fitted = fit(cls.frame, 69, seed=5)

    def test_sixty_finite_scenario_cutoff_mode_fits(self):
        completed = 0
        for scenario in SCENARIOS:
            world, _ = generate(912, scenario)
            for cutoff in (69, 119):
                available = world.targets(cutoff)
                targets = [available[0], available[-1]]
                for mode in MODES:
                    with self.subTest(scenario=scenario, cutoff=cutoff, mode=mode):
                        p = fit(world.as_of(cutoff), cutoff, mode, seed=45).predict(targets, draws=32)
                        self.assertEqual(p["margin_paths"].shape, (32, 2, 426))
                        self.assertTrue(np.isfinite(p["margin_paths"]).all())
                        completed += 1
        self.assertEqual(completed, 60)

    def test_target_and_source_distributions_ignore_other_blocks(self):
        d = self.world.as_of(119).frame()
        for mode, field, columns in (("target_only", "cohort_id", "cohort_cols"),
                                     ("source_pooling", "source_id", "source_cols")):
            block = str(d[field].iloc[0])
            full = fit(d, 119, mode)
            local = fit(d[d[field] == block], 119, mode)
            for outcome in ("response", "payout"):
                a, b = getattr(full, outcome), getattr(local, outcome)
                ia, ib = getattr(a, columns)[block], getattr(b, columns)[block]
                np.testing.assert_allclose(a.independent_weights[block], b.independent_weights[block], atol=1e-12)
                np.testing.assert_allclose([c.mean[ia] for c in a.components],
                                           [c.mean[ib] for c in b.components], atol=1e-12)
                np.testing.assert_allclose([c.precision_chol[ia, ia] for c in a.components],
                                           [c.precision_chol[ib, ib] for c in b.components], atol=1e-12)
        # Invariance is of the parameter distribution. Different-sized calls
        # consume different RNG streams; seed-matched arrays need not match.

    def test_zero_count_bins_keep_half_count_uncertainty(self):
        row = self.frame.iloc[[0]].copy()
        row["raw_events"] = 0
        row["audit_n"] = 0
        row["audit_human"] = 0
        config = ModelConfig()
        measurements = _measurements(row, config, "response")
        self.assertEqual(len(measurements), 1)
        self.assertAlmostEqual(float(measurements.variance.iloc[0]), 2.)
        self.assertTrue(np.isfinite(measurements.value.iloc[0]))
        # More accepted exposure changes the rate offset but cannot turn a
        # single zero-event bin into a nearly exact log-rate measurement.
        row["accepted_exposure"] = 100000
        self.assertAlmostEqual(float(_measurements(row, config, "response").variance.iloc[0]), 2.)

    def test_immutable_copy_and_late_instrument_clocks(self):
        d = self.frame.copy(deep=True)
        fitted = fit(d, 69, seed=5)
        before = fitted.predict([prospective()], draws=128)["margin_paths"]
        identity = fitted.data_hash
        d["raw_events"] = 0
        self.assertEqual(fitted.data_hash, identity)
        np.testing.assert_array_equal(fitted.predict([prospective()], draws=128)["margin_paths"], before)
        raw = self.world.observations.copy(deep=True)
        raw.loc[raw.audit_observed_at > 69, "audit_human"] = 99999
        raw.loc[raw.payout_observed_at > 69, "net_human_revenue"] = 1e12
        raw["hidden_evaluator_quality"] = 1e99
        self.assertEqual(fit(raw, 69).data_hash, fit(self.world.observations, 69).data_hash)

    def test_shared_calendar_reconstructs_joint_target_values(self):
        first = prospective()
        second = dict(first, cohort_id="other-fresh-cohort", birth_day=77)
        p = self.fitted.predict([first, second], draws=128)
        for j, target in enumerate((first, second)):
            days = target["birth_day"] + np.arange(426)
            expected = np.exp(p["payout_log_amplitudes"][:, j, None] + p["calendar_cpc"][:, days])
            np.testing.assert_allclose(p["net_human_cpc"][:, j], expected)
        self.assertTrue(p["diagnostics"]["portfolio_shared_calendar"])

    def test_conditional_scenarios_and_fixed_early_value_reference(self):
        for policy in POLICIES:
            p = self.fitted.predict([prospective()], draws=64, scenario=policy)
            self.assertTrue(np.isfinite(p["margin_paths"]).all())
            if policy == "fixed_price":
                np.testing.assert_array_equal(p["calendar_cpc"][:, 70:], 0)

    def test_common_random_accounting_monotonicity(self):
        target = prospective()
        baseline = self.fitted.predict([target], draws=512)
        cheaper = self.fitted.predict([dict(target, acquisition_cost=4.)], draws=512)
        np.testing.assert_allclose(cheaper["margin_paths"] - baseline["margin_paths"], 4.)
        better_value = self.fitted.predict([dict(target, contracted_cpc_multiplier=2.)], draws=512)
        np.testing.assert_array_equal(better_value["human_count_paths"], baseline["human_count_paths"])
        self.assertTrue((better_value["margin_paths"] >= baseline["margin_paths"] - 1e-10).all())
        def first_cross(paths):
            hit = paths[:, 0] >= 0
            return np.where(hit.any(axis=1), hit.argmax(axis=1), np.inf)
        self.assertTrue((first_cross(cheaper["margin_paths"]) <= first_cross(baseline["margin_paths"])).all())
        self.assertTrue((first_cross(better_value["margin_paths"]) <= first_cross(baseline["margin_paths"])).all())

    def test_unseen_source_integrates_full_prior_variation(self):
        empty = pd.DataFrame(columns=list(REQUIRED) + ["source_feature"])
        p = fit(empty, 0, seed=76).predict([dict(prospective(), birth_day=1)], draws=12000, horizon=0)
        config = ModelConfig()
        expected = config.location_sd ** 2 + sum(w * s ** 2 for w, s in zip(config.effect_weights, config.source_sds)) + sum(
            w * s ** 2 for w, s in zip(config.effect_weights, config.cohort_sds))
        observed = np.var(p["response_log_amplitudes"][:, 0])
        self.assertLess(abs(observed - expected), .08, (observed, expected))
        self.assertGreater(observed, config.location_sd ** 2 + .25)

    def test_cutoff_zero_retains_future_calendar_hyperprior(self):
        empty = pd.DataFrame(columns=list(REQUIRED) + ["source_feature"])
        target = dict(prospective(), birth_day=0)
        f = fit(empty, 0, seed=719)
        p = f.predict([target], draws=6000, horizon=200)
        config = ModelConfig()
        bins, rem = divmod(200, config.calendar_bin_days)
        variance_time = config.calendar_bin_days * bins + rem ** 2 / config.calendar_bin_days
        expected = variance_time * sum(w * s ** 2 for w, s in zip(config.calendar_weights, config.calendar_sds))
        self.assertEqual(f.response.report()["grid_components"], 54)
        for name in ("calendar_response", "calendar_cpc"):
            np.testing.assert_array_equal(p[name][:, 0], 0)
            observed = np.var(p[name][:, 200])
            self.assertLess(abs(observed - expected), .008, (name, observed, expected))
            self.assertGreater(observed, .04)

    def test_fixed_calendar_grid_prior_covariance_across_cutoffs(self):
        empty = pd.DataFrame(columns=list(REQUIRED) + ["source_feature"])
        days = np.array([3., 6., 9., 10.])
        reference = None
        for cutoff in (10, 14, 15, 21):
            model = fit(empty, cutoff)
            mixture = model.response
            component = mixture.components[0]
            basis = mixture.calendar_basis(days)
            covariance = (basis * (component.calendar_sd ** 2 * np.diff(mixture.knots))) @ basis.T
            if reference is None:
                reference = covariance
            else:
                np.testing.assert_allclose(covariance, reference, atol=1e-15)
        q = ModelConfig().calendar_sds[0]
        self.assertAlmostEqual(reference[2, 2], q ** 2 * (7 + 4 / 7))
        self.assertAlmostEqual(reference[2, 3], q ** 2 * (7 + 6 / 7))

    def test_same_observations_keep_past_calendar_posterior_across_cutoffs(self):
        d = self.frame[self.frame.calendar_day <= 9].copy()
        # Exactly the same available measurements, despite later refit cutoffs.
        d["observed_at"] = d.calendar_day
        d["audit_observed_at"] = d.calendar_day
        d["payout_observed_at"] = d.calendar_day
        reference = None
        for cutoff in (10, 14, 21):
            mixture = fit(d, cutoff).response
            rows = []
            for c in mixture.components:
                b = np.zeros(mixture.dimension)
                b[mixture.calendar_cols] = mixture.calendar_basis([9])[0]
                mean = b @ c.mean
                var = b @ cho_solve((c.precision_chol, True), b)
                rows.append([mean, var])
            result = (mixture.weights, np.asarray(rows))
            if reference is None:
                reference = result
            else:
                np.testing.assert_allclose(result[0], reference[0], atol=1e-10)
                np.testing.assert_allclose(result[1], reference[1], atol=1e-10)

    def test_cheap_good_and_bad_have_identical_prelaunch_predictions(self):
        good, _ = generate(724, "cheap_good")
        bad, _ = generate(724, "cheap_bad")
        a, b = fit(good.as_of(69), 69, seed=13), fit(bad.as_of(69), 69, seed=13)
        self.assertEqual(a.data_hash, b.data_hash)
        np.testing.assert_array_equal(a.predict([prospective()], draws=128)["margin_paths"],
                                      b.predict([prospective()], draws=128)["margin_paths"])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--prior-tail-check":
        output = Path(sys.argv[2] if len(sys.argv) > 2 else "dynamic_lab/prior_tail_summary.json")
        result = prior_tail_diagnostics()
        output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        for row in result["rows"]:
            print(row["scale_sensitivity"], row["draws"], round(row["mean_lifetime_revenue"], 3),
                  round(row["analytic_mean_lifetime_revenue"], 3),
                  round(row["monte_carlo_standard_error"], 3),
                  round(row["largest_one_percent_draw_revenue_share"], 3))
    else:
        unittest.main()
