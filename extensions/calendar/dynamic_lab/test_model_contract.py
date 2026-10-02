"""Independent statistical/API boundary checks; not predictive calibration."""
import unittest

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular

from dynamic_lab.model import fit, ModelConfig, REQUIRED
from dynamic_lab.evaluate import summarize


def visible_rows():
    return pd.DataFrame([
        dict(cohort_id="c", source_id="s", birth_day=0, calendar_day=day,
             age=day, accepted_exposure=100, raw_events=10, audit_n=5,
             audit_human=4, net_human_revenue=amount, net_human_count=5,
             response_report_fraction=1., payout_complete_fraction=1.,
             observed_at=day, payout_observed_at=day)
        for day, amount in [(0, 1.), (1, 2.)]
    ])


class IndependentModelContractTests(unittest.TestCase):
    def test_partial_settlement_is_rejected_not_fixed_as_complete(self):
        rows = visible_rows()
        rows.loc[0, "payout_complete_fraction"] = .5
        with self.assertRaises(ValueError):
            fit(rows, 1)

    def test_negative_daily_receipt_cannot_hide_in_positive_week(self):
        rows = visible_rows()
        rows.loc[0, "net_human_revenue"] = -1.
        self.assertGreater(rows.net_human_revenue.sum(), 0)
        with self.assertRaises(ValueError):
            fit(rows, 1)

    def test_future_monetary_values_are_masked_before_validation(self):
        rows = visible_rows()
        rows["payout_observed_at"] = 10
        altered = rows.copy(deep=True)
        altered["net_human_revenue"] = -1e20
        altered["net_human_count"] = 1e20
        altered["payout_complete_fraction"] = .5
        self.assertEqual(fit(rows, 1).data_hash, fit(altered, 1).data_hash)

    def test_calendar_prior_for_existing_date_is_cutoff_invariant(self):
        empty = pd.DataFrame(columns=REQUIRED)
        for day in (9, 10):
            variances = []
            for cutoff in (10, 14, 21):
                mixture = fit(empty, cutoff).response
                component = mixture.components[0]
                basis = mixture.calendar_basis([day])[0]
                # Compare the same finite hyperprior component's induced
                # variance, including a fit with an extra unobserved endpoint.
                variances.append(sum(
                    coefficient ** 2 / component.precision_chol[col, col] ** 2
                    for coefficient, col in zip(basis, mixture.calendar_cols)
                ))
            np.testing.assert_allclose(variances, variances[0], atol=1e-14, rtol=0)

    def test_same_data_retains_posterior_past_states_at_later_cutoff(self):
        def moments(mixture):
            basis = np.zeros((2, mixture.dimension))
            basis[:, mixture.calendar_cols] = mixture.calendar_basis([9, 10])
            means, seconds = [], []
            for component in mixture.components:
                mean = basis @ component.mean
                transformed = solve_triangular(component.precision_chol,
                    basis.T, lower=True)
                covariance = transformed.T @ transformed
                means.append(mean)
                seconds.append(covariance + np.outer(mean, mean))
            mean = mixture.weights @ np.asarray(means)
            second = np.tensordot(mixture.weights, np.asarray(seconds), axes=1)
            return mean, second - np.outer(mean, mean)
        fits = [fit(visible_rows(), cutoff) for cutoff in (10, 14, 21)]
        self.assertEqual(len({f.data_hash for f in fits}), 1)
        for outcome in ("response", "payout"):
            expected = moments(getattr(fits[0], outcome))
            for fitted in fits[1:]:
                current = moments(getattr(fitted, outcome))
                np.testing.assert_allclose(current[0], expected[0], atol=1e-12, rtol=0)
                np.testing.assert_allclose(current[1], expected[1], atol=1e-12, rtol=0)

    def test_new_source_draw_is_shared_without_erasing_cohort_variance(self):
        empty = pd.DataFrame(columns=REQUIRED)
        targets = [
            dict(cohort_id="a", source_id="fresh", birth_day=1, leads=10,
                 acquisition_cost=1, daily_send_cost=0),
            dict(cohort_id="b", source_id="fresh", birth_day=1, leads=10,
                 acquisition_cost=1, daily_send_cost=0),
            dict(cohort_id="c", source_id="another", birth_day=1, leads=10,
                 acquisition_cost=1, daily_send_cost=0),
        ]
        result = fit(empty, 0, seed=98471).predict(targets, draws=14000, horizon=0)
        config = ModelConfig()
        source_var = sum(w * s ** 2 for w, s in zip(config.effect_weights, config.source_sds))
        cohort_var = sum(w * s ** 2 for w, s in zip(config.effect_weights, config.cohort_sds))
        for name in ("response_log_amplitudes", "payout_log_amplitudes"):
            covariance = np.cov(result[name], rowvar=False)
            self.assertAlmostEqual(covariance[0, 1], config.location_sd ** 2 + source_var, delta=.06)
            self.assertAlmostEqual(covariance[0, 2], config.location_sd ** 2, delta=.06)
            self.assertAlmostEqual(covariance[0, 0] - covariance[0, 1], cohort_var, delta=.04)

    def test_first_payback_keeps_failures_and_reversed_crossings(self):
        paths = np.full((3, 426), -1.)
        paths[0, 1] = 0.  # first crossing, later reversed
        paths[2, 400:] = 1.  # late crossing
        target = dict(cohort_id="c", source_id="s", birth_day=0,
                      leads=10, acquisition_cost=1)
        row = summarize(paths, target, dict(origin_day=0))
        self.assertAlmostEqual(row["p_payback425"], 2 / 3)
        self.assertAlmostEqual(row["p_no_payback425"], 1 / 3)
        self.assertEqual(row["p_positive180"], 0)


if __name__ == "__main__":
    unittest.main()
