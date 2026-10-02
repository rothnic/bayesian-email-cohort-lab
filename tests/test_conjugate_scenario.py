"""Component calibration is deliberately separate from full-world model tests."""
from dataclasses import replace
import unittest
import numpy as np
from scipy.stats import nbinom
from scripts.validate_conjugate_scenario import ControlledConfig, age_curve, evaluate_controlled, module_crosscheck, posterior_parameters, predictive_distribution


class ControlledConjugateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = evaluate_controlled()

    def test_analytic_update_and_negative_binomial_moments(self):
        config = ControlledConfig()
        shape, rate = posterior_parameters(10, 100, config)
        self.assertEqual(shape, 18)
        self.assertAlmostEqual(rate, 8 / .12 + 100)
        distribution = predictive_distribution(shape, rate, 50.)
        self.assertAlmostEqual(distribution.mean(), shape / rate * 50)
        self.assertAlmostEqual(distribution.var(), shape / rate * 50 + shape / rate ** 2 * 50 ** 2)

    def test_all_fixed_seed_worlds_retained_and_reproducible(self):
        result = self.result
        self.assertEqual(len(result["world_draws"]), 2000)
        self.assertEqual(len(result["forecasts"]), 10000)
        self.assertEqual(result["forecasts"].groupby("origin_age").world_id.nunique().tolist(), [2000] * 5)
        small = replace(result["config"], worlds=3)
        again = evaluate_controlled(small)
        # A different array shape changes RNG consumption after the Gamma draw;
        # compare full equal-size replays rather than implying prefix stability.
        replay = evaluate_controlled(small)
        np.testing.assert_array_equal(again["counts"], replay["counts"])
        np.testing.assert_array_equal(again["theta"], replay["theta"])

    def test_predictive_coverage_is_near_nominal_under_matching_model(self):
        summary = self.result["summary"]
        # Integer quantiles need not attain exactly nominal probability mass.
        # Compare realized coverage to the actual conditional NB interval mass,
        # rather than tuning a nominal upper bound to a particular realized run.
        self.assertTrue((summary.expected_discrete_coverage80 >= .8).all())
        self.assertTrue((summary.expected_discrete_coverage95 >= .95).all())
        self.assertTrue((summary.coverage80_error_standard_errors.abs() < 5).all())
        self.assertTrue((summary.coverage95_error_standard_errors.abs() < 5).all())
        self.assertTrue(summary.coverage95.between(.93, .98).all())
        self.assertTrue(summary.theta_coverage95.between(.925, .975).all())
        self.assertTrue(summary.posterior_rank_mean.between(.47, .53).all())
        self.assertTrue(summary.predictive_pit_mean.between(.47, .53).all())
        self.assertTrue(summary.posterior_rank_variance.between(.075, .092).all())

    def test_parameter_and_predictive_uncertainty_update(self):
        summary = self.result["summary"]
        self.assertTrue((np.diff(summary.mean_theta_width95) < 0).all())
        self.assertTrue((np.diff(summary.mean_width95) < 0).all())
        self.assertTrue((np.diff(summary.brier) < 0).all())

    def test_exact_monetary_threshold_and_interval_transform(self):
        row = self.result["forecasts"].iloc[123]
        config = self.result["config"]
        future_exposure = config.accepted_per_day * age_curve(np.arange(row.origin_age + 1, config.horizon + 1), config).sum()
        distribution = nbinom(row.posterior_shape, row.posterior_rate / (row.posterior_rate + future_exposure))
        expected = distribution.sf(int(np.floor(self.result["total_known_cost"] / config.value_per_click)) - row.observed_count)
        self.assertAlmostEqual(row.probability_profitable, expected)
        self.assertAlmostEqual(row.lower95, (row.observed_count + distribution.ppf(.025)) * config.value_per_click - self.result["total_known_cost"])

    def test_same_rate_component_in_main_module(self):
        result = module_crosscheck(self.result["config"], self.result["counts"][0])
        self.assertTrue(result["passed"], result)
        self.assertLess(result["mean_error_standard_errors"], 5)
        self.assertLess(result["variance_relative_error"], .10)


if __name__ == "__main__":
    unittest.main()
