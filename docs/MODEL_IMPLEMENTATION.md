# Implemented Bayesian model

This is a **small, declared, modular Bayesian model for public synthetic data**.
It does not implement unrestricted lifetime value, unique-person retention,
latent alive/dead engagement states, correlated calendar shocks, or a fully
joint model of every telemetry and ledger record. The rate module has an exact
finite-grid posterior. Conditional conjugate parameter draws and simulated
future outcomes produce complete economic margin paths. Held-out coverage can
fail, especially in the deliberately misspecified scenarios.

## Public API

```python
from bayes_cohort import forecast_cohort, ModelConfig, default_config_dict
forecast = forecast_cohort(snapshot, policy, "middle-w07", draws=500, seed=11)
paths = forecast["margin_paths"]  # draws × (terminal_age + 1)
```

Only the public `Snapshot` and `Policy` classes are accepted. Neither a `World`
nor the separately held generator truth is a model input. The package does not
import the generator or read its parameter file. Configuration is frozen in
`bayes_cohort/config.py`; `default_config_dict()` exports it for run provenance.
A caller may supply a `ModelConfig` or a mapping with recognized model keys.
Unknown keys fail rather than being silently accepted.

The return value includes:

- `margin_paths`, the arithmetic Monte Carlo mean `margin_path`, and the
  separately preserved `observed_margin_path`
- Posterior `parameter_draws`, `first_payback_ages` (`-1` means no crossing), and
  unconditional `payback_state_probabilities`
- Draw arrays `future_send_cost`, `future_send_revenue`,
  `pending_prior_send_revenue`, `incremental_continuation_value`, and
  `stop_sending_value`
- JSON-serializable `diagnostics`, including source-grid boundary mass, posterior
  quantiles, mature mark counts, missing-observation flags, and assumptions

`forecast_records` serializes draw-level records and labels strictly prospective,
at-origin nowcast, and retrospective horizons. Retrospective horizons are omitted
by default. `clear_fit_cache` releases the bounded, immutable-snapshot fit cache.

## Response hierarchy and exact grid integration

For source `s`, a declared discrete grid covers initial response mean `m_s`, fast
weight `w_s`, fast timescale `t_f`, and tail timescale `t_l`. The default grid has
336 combinations. Each source has its own posterior over that grid, with a
proper discrete log-normal-shaped prior on `m_s` and equal prior shape weights.
The declared age curve is:

`f_s(a) = w_s exp(-a/t_f) + (1-w_s) exp(-a/t_l)`

A cohort response amplitude has the source-shared Gamma prior:

`theta_c | m_s ~ Gamma(k, rate=k/m_s)`, with default `k=8`

For observed accepted exposures `E_ca`, latest currently qualified-human
response counts `Y_ca` follow a Poisson rate `theta_c E_ca f_s(a) q_ca`. Here
`q_ca` is the probability that a response **and its telemetry report** would
be visible by the cutoff under the declared measurement clock. Missing exposure
cells, including partially missing arm cells, are excluded entirely. Counts are
response events, including repeated responses, rather than distinct leads.

For a grid point, define `L_c = sum_a E_ca f_s(a) q_ca` and
`N_c = sum_a Y_ca`. The amplitude integrates out analytically. Up to factors
constant across grid points, its source likelihood factor is:

`Gamma(k+N_c)/Gamma(k) × (k/m_s)^k / (k/m_s+L_c)^(k+N_c)
 × product_a f_s(a)^Y_ca`

The product over all visible same-source cohorts, combined with the grid prior,
gives normalized source posterior weights. After drawing a source grid point,
the target amplitude is drawn from:

`theta_c | grid,data ~ Gamma(k+N_c, rate=k/m_s+L_c)`

This is actual partial pooling: other source cohorts update the shared source
mean/curve, and the target's own counts update its Gamma amplitude. No fitted
generator effects, latent qualities, final human classifications, or future
calendar states enter the model. Grid boundary mass is reported because a
concentrated discrete posterior can be sensitive to its finite support.

## Declared measurement clocks

Response, event-reporting, settlement, and ledger-reporting delay distributions
are **known synthetic instrument assumptions**, stored separately from generator
parameters. Ordinary revision delay/fraction support and acquisition-invoice
reporting clocks are also declared synthetic mechanism assumptions. This is not
an empirical delay estimator. It avoids fitting a naive delay histogram from
right-truncated current reports, but it makes the synthetic result easier than a
real deployment with unknown reporting delays. A real deployment would need
external validation or an explicit delay-inference module.

For past sends, the Poisson process is split into visible and unreported events
using its actual clock CDF, without an arbitrary completion floor. Unknown
responses are sampled conditional on not-yet-visible telemetry. Their base
receipts are added only if also unreported. Already-visible base receipts are
retained once and carry their own future revision prediction. Small permanently
unavailable telemetry and temporary bot/human misclassification are omitted
risks, explicitly disclosed in the forecast.

## Operational exposure and costs

Source-level exit hazards and acceptance probabilities have declared Beta
priors, updated only with measured exposure cells. Exit trials use
`eligible + exits`; acceptance trials use `attempted`. Future daily exits and
accepted sends are binomial draws conditional on those drawn probabilities.
The latest measured target eligibility anchors the future path. Missing past
target cells are stochastically imputed for prediction, without converting a
missing cell into a likelihood observation or rewriting the input table.

The finite policy fixes the last send age. Each future attempted send incurs the
declared policy cost. Dormancy/inactivity never causes an operational exit.
Acquisition is sunk for the incremental continuation diagnostic, which includes
only genuinely future sends' receipts, revisions, and costs. Pending outcomes
from old sends remain in complete margin paths, but are excluded from the
incremental continuation value.

## Zero-plus-lognormal values and signed revisions

Only qualified human events at least 30 days past occurrence and linked to an
actual base-revenue row contribute to mark learning. The maturity window must
cover all declared base receipt and ordinary revision observation support.
Missing mature receipts are flagged as unavailable, never assigned a zero.

The probability of a positive base mark has a Beta posterior. Positive log
amounts update a Normal–inverse-Gamma model with declared prior mean,
precision, shape, and scale. Crucially, the inverse-Gamma **prior support is
restricted to `0 < sigma² <= max_log_mark_variance`**, default `4`. The conjugate
update retains that same support. Draws use the exact truncated inverse-Gamma
CDF/PPF followed by the conditional Normal draw of the log mean. No realized
amount cap is imposed.

Without this variance bound, the log predictive is a Student-t mixture whose
exponential moment is infinite. A finite sample mean would then not be an
honest estimate of finite expected positive revenue. With the declared bound,
the log mean's conditional Normal variance and predictive variance are bounded,
so positive-value moments are finite. The variance bound is a substantive prior
assumption, not a numerical workaround; its effect must be included in
sensitivity checks. Diagnostics report the untruncated posterior mass below the
bound, so an influential restriction is visible.

Ordinary negative revision frequency has a Beta posterior from mature positive
marks. Revision fraction and delay distributions are declared synthetic
mechanisms. For an already-observed positive receipt without a revision report,
future revision probability conditions on that absence:

`P(future revision | no report) = p(1-F_report)/(1-p F_report)`

The future revision clock is then sampled conditional on remaining unreported.
Observed signed revisions are retained exactly and are not regenerated. The
implemented model permits one ordinary revision per event. It does not capture
the additional systematic attribution reversals in the shock scenario. An
observed base with missing event telemetry uses an explicitly approximate
settlement-clock back-imputation of occurrence for this revision module.

## Acquisition invoices and economic nowcasting

Visible acquisition and sending charges, base revenue, and signed corrections
are immutable ledger components. Before the declared final-invoice report age,
an unobserved acquisition adjustment is sampled per arm using a declared Normal
fraction prior and appended at its economic age. Missing invoice records after
the declared reporting support are flagged as incomplete data, not described as
settled. No unsupported long reporting tail is silently invented.

Unreported receipts can already have an economic date before the cutoff. They
are added at that economic date rather than shifted to a future reporting date.
Therefore predictive complete economic margin paths need not all pass through
the currently visible ledger balance. `observed_margin_path` remains the exact
visible accounting path. At the complete terminal snapshot, every draw
reconciles exactly to the full observed signed ledger.

## Payback and model limitations

First crossing is computed independently for every full path over the declared
finite policy. The unconditional distribution retains the `never_under_policy`
atom alongside `by_180` and `later_than_180`; non-crossing paths are not dropped
from its denominator. First crossing is distinct from remaining positive after
crossing, since sending costs and signed revisions can reverse payback.

This is a modular posterior predictive calculation. The response likelihood,
complete-window mark likelihood, and operational likelihood use selected
observable data; this is not a claim to an exact joint posterior conditioned on
every ledger-only and misclassified event. The specifically disclosed
ledger-only revision back-imputation and missing-exposure imputation are
approximations. Lead-level heterogeneity, repeat-click clustering, content
effects, calendar shocks, and reactivation are not learned. Coverage, proper
scores, probability reliability, and grid/prior sensitivity must be reported on
held-out worlds. Narrow posterior intervals alone are not evidence of accuracy.

## Focused checks

`python -m unittest discover -s tests -p test_bayesian.py -v` checks Snapshot-only
inputs, deterministic replay, future perturbation isolation, no input mutation,
an analytic Gamma posterior under delay thinning, missing/partial telemetry,
real within-source pooling, updated uncertainty across ages 3/7/14/30/60,
mature-only marks, bounded log variance, pending receipts after sending stops,
signed payback reversal, uncertain invoices, unconditional non-payback mass,
prior sensitivity, record roles, and complete terminal ledger reconciliation.
