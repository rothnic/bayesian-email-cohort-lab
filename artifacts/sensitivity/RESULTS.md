# Tail and revenue-variance prior sensitivity

All inputs are fictional. The same data are fitted with four prespecified discrete tail supports and log-variance caps of 1, 4 (default) and 9; each setting changes only the named assumption. These are different models, not a search for the most favorable result. Finite Monte Carlo errors remain.

Posterior predictive day-180 contribution per acquired lead (fictional USD):

| Source | Origin | Prior setting | Mean | 95% predictive interval | P(profit) |
|---|---:|---|---:|---|---:|
| low | 3 | default | -0.150 | [-0.175, -0.121] | 0.0% |
| low | 3 | short_tail | -0.160 | [-0.186, -0.116] | 0.0% |
| low | 3 | long_tail | -0.135 | [-0.164, -0.101] | 0.0% |
| low | 3 | denser_grid | -0.154 | [-0.182, -0.121] | 0.0% |
| low | 3 | variance_cap_1 | -0.150 | [-0.175, -0.121] | 0.0% |
| low | 3 | variance_cap_9 | -0.150 | [-0.175, -0.121] | 0.0% |
| middle | 3 | default | 0.134 | [0.059, 0.214] | 100.0% |
| middle | 3 | short_tail | 0.103 | [0.026, 0.256] | 99.7% |
| middle | 3 | long_tail | 0.209 | [0.119, 0.308] | 100.0% |
| middle | 3 | denser_grid | 0.128 | [0.046, 0.213] | 99.8% |
| middle | 3 | variance_cap_1 | 0.134 | [0.059, 0.214] | 100.0% |
| middle | 3 | variance_cap_9 | 0.134 | [0.059, 0.214] | 100.0% |
| high | 3 | default | 1.461 | [1.241, 1.700] | 100.0% |
| high | 3 | short_tail | 1.810 | [1.537, 2.107] | 100.0% |
| high | 3 | long_tail | 1.810 | [1.537, 2.107] | 100.0% |
| high | 3 | denser_grid | 1.460 | [1.236, 1.715] | 100.0% |
| high | 3 | variance_cap_1 | 1.461 | [1.241, 1.700] | 100.0% |
| high | 3 | variance_cap_9 | 1.461 | [1.241, 1.700] | 100.0% |
| low | 60 | default | -0.143 | [-0.151, -0.132] | 0.0% |
| low | 60 | short_tail | -0.133 | [-0.143, -0.122] | 0.0% |
| low | 60 | long_tail | -0.133 | [-0.143, -0.122] | 0.0% |
| low | 60 | denser_grid | -0.143 | [-0.152, -0.131] | 0.0% |
| low | 60 | variance_cap_1 | -0.143 | [-0.151, -0.132] | 0.0% |
| low | 60 | variance_cap_9 | -0.143 | [-0.151, -0.132] | 0.0% |
| middle | 60 | default | 0.175 | [0.150, 0.202] | 100.0% |
| middle | 60 | short_tail | 0.217 | [0.184, 0.253] | 100.0% |
| middle | 60 | long_tail | 0.217 | [0.184, 0.253] | 100.0% |
| middle | 60 | denser_grid | 0.235 | [0.203, 0.272] | 100.0% |
| middle | 60 | variance_cap_1 | 0.175 | [0.150, 0.202] | 100.0% |
| middle | 60 | variance_cap_9 | 0.175 | [0.150, 0.202] | 100.0% |
| high | 60 | default | 1.568 | [1.494, 1.643] | 100.0% |
| high | 60 | short_tail | 1.768 | [1.668, 1.856] | 100.0% |
| high | 60 | long_tail | 1.768 | [1.668, 1.856] | 100.0% |
| high | 60 | denser_grid | 1.568 | [1.494, 1.643] | 100.0% |
| high | 60 | variance_cap_1 | 1.568 | [1.494, 1.643] | 100.0% |
| high | 60 | variance_cap_9 | 1.568 | [1.494, 1.643] | 100.0% |

A large sensitivity to prior support means that tail assumptions remain decision-relevant. A small difference here does not validate omitted reactivation, calendar shocks or informative missingness. See `tail_sensitivity.csv` for the terminal-horizon results and evaluator-only truth.

Reproduce: `python run_sensitivity.py`

## Cold-start monetary-prior sensitivity

The mature-history comparisons above barely respond to variance caps because observed mature marks dominate that part of the prior. The first cohort at age 3 has no mature marks. Here are all three prespecified caps on that same cold-start snapshot, with 2,000 draws per setting. These heavy-tailed sample means and their empirical MCSE can still be unstable; neither is a reliable bound on unseen extremes.

| Source | Variance cap | Mean / lead | Median / lead | 95% interval | P(profit) |
|---|---:|---:|---:|---|---:|
| low | 1 | 0.023 | -0.128 | [-0.231, 1.185] | 28.6% |
| low | 4 | 0.106 | -0.124 | [-0.231, 1.652] | 29.4% |
| low | 9 | 0.127 | -0.124 | [-0.231, 1.655] | 29.5% |
| middle | 1 | 0.090 | -0.101 | [-0.212, 1.400] | 32.6% |
| middle | 4 | 0.202 | -0.100 | [-0.213, 1.872] | 33.3% |
| middle | 9 | 0.255 | -0.100 | [-0.213, 1.904] | 33.3% |
| high | 1 | 0.296 | 0.047 | [-0.167, 2.507] | 57.7% |
| high | 4 | 0.443 | 0.051 | [-0.167, 3.324] | 58.1% |
| high | 9 | 0.981 | 0.051 | [-0.167, 3.383] | 58.1% |

Analytic positive-mark prior moments below integrate the bounded variance mixture, including uncertainty in the log mean. They describe a single **positive** click payment before mature data, not total cohort profit. These quadrature results make the prior tail risk visible without relying on a few Monte Carlo draws.

| Variance cap | Prior mean positive payment | Prior standard deviation |
|---|---:|---:|
| 1 | 0.1914 | 0.489 |
| 4 | 0.3145 | 284 |
| 9 | 5.1939 | 1.86e+08 |

All moments are finite, yet a large standard deviation can make finite-draw economic means unreliable. This is why a finite mathematical expectation alone is not an adequate prior-predictive check. The default prior is retained to make its cold-start fragility inspectable; it is not recommended as a production monetary prior.
