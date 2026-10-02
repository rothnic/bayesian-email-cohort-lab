# Research choices and next experiments

## Why model economics directly?

A paid lead can remain contactable while rarely clicking. A dormant lead can return. A small number of valuable clicks can dominate revenue, and reports can arrive well after the click. A single “retention” curve conflates these quantities.

The demo therefore separates operational eligibility, accepted exposure, qualified click events, signed revenue marks and acquisition/sending costs. Its target is contribution and payback under an explicit finite sending policy, not an unobservable permanent-death label.

## Why a finite-grid Bayesian model?

The implemented model provides a transparent, dependency-light way to inspect posterior updating. Discrete candidate age curves describe a rapid early component plus a slower tail. Hierarchical Gamma amplitudes let cohorts differ while sharing information within a source. Conditional conjugacy makes the calculation and predictive simulation inspectable without a probabilistic-programming service or MCMC setup. The exact implemented math and numerical assumptions are in [MODEL_IMPLEMENTATION.md](MODEL_IMPLEMENTATION.md).

This is one bounded research model. The grid and its priors constrain what it can learn. An apparently narrow interval does not include uncertainty about every omitted mechanism. The sample generator deliberately includes cohort/lead heterogeneity and shocks that the compact inference model does not fully represent.

## Why not start with latent death?

Classic customer-base models remain useful comparison candidates, but their assumptions must match the observation opportunity. The BG/NBD formulation assumes Poisson transactions while active and dropout after a transaction. Acquisition is not itself a purchase or a click, so silently treating it as one changes that model. See the original [Fader, Hardie and Lee paper](https://www.brucehardie.com/papers/018/fader_et_al_mksc_05.pdf).

A hidden active/dormant/dead state model might eventually improve predictions. It should first beat the simpler exposure/count/mark models on mature, time-ordered holdouts. No recent clicks does not establish a permanent exit. A hidden-state backtest must filter on then-available information, rather than use future-aware smoothed states.

## Delays and missingness

Right-truncated observations are not zero outcomes. The current Bayesian implementation conditions on declared synthetic observation-delay assumptions; their uncertainty is not learned from a mature external validation sample. A production version needs explicit reporting/qualification delay models and sensitivity to value-dependent missingness. Bayesian nowcasting provides a methodological example of separating event and report times, without implying this application shares an epidemic model: [McGough et al., 2020](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007735).

## What the evaluation establishes

Out-of-model stress tests can reveal predictive failures, but they are not formal simulation-based calibration (SBC). SBC draws model parameters from the prior, data from that same model, and checks posterior rank behavior to test an inference implementation; it does not establish real-world adequacy. See [Talts et al.](https://arxiv.org/abs/1804.06788). The checked-in suite instead asks how this chosen approximation behaves under four fictional, partially misspecified worlds.

Posterior predictive checks compare relevant observed summaries with replicated data. Checking only means can miss overdispersion or tails, so future checks should include click-count dispersion, cohort heterogeneity, zero proportions, delays and large monetary marks. The [Stan User's Guide](https://mc-stan.org/docs/stan-users-guide/posterior-predictive-checks.html) explains this distinction and why broad priors need prior-predictive checks.

## Next experiments, in order

1. Expand independent worlds and stress unknown delays, invoice changes and value-dependent missingness; retain all failures
2. Compare tail priors and a denser grid; inspect boundary mass and numerical sensitivity
3. Add an explicitly overdispersed daily-count likelihood and learned shared calendar/content components, with constrained age/cohort/calendar effects
4. Predict several cohorts jointly with one shared future calendar shock per date; do not pretend shared risk diversifies away
5. Add block-bootstrap predictive intervals to the inexpensive baselines before claiming uncertainty superiority
6. Implement prior-generated SBC and posterior predictive checks for the full observation model
7. Add next-batch decision loss and expected value of information; then implement and stress the A/B procedures in [CAUSAL_EXPERIMENTS.md](CAUSAL_EXPERIMENTS.md)

Each addition should earn its complexity through more reliable decisions or predictions on prespecified holdouts. A simpler baseline is allowed to win.
