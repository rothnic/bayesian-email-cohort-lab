# Forecasting profitability is not estimating causal lift

The generator persists a 50/50 randomized `arm` assignment for each acquired lead. The current scenarios have **no simulated treatment effect**, and the implemented model pools the arms. There is no implemented A/B winner, sequential stopping rule or claim of causal improvement.

## The estimand comes first

For a future layout experiment, a useful intention-to-treat estimand is expected contribution per assigned lead at a fixed horizon under layout B minus that under A. Count every randomized eligible lead, including non-openers, non-clickers and unsubscribers. Comparing revenue only among clickers conditions on a post-treatment event and can change the population being compared.

Keep paid acquisition quality, content, date, eligibility and measurement definitions comparable. Randomization identifies a causal contrast only under appropriate implementation, follow-up and interference assumptions; a posterior probability does not repair broken assignment or missing outcomes. Repeated observations from one lead are not independent randomized units.

## Build the experiment into the generator and replay

Before proposing a stopping policy, add scenarios with:

- A true null, modest positive and harmful effects
- Early novelty that fades or reverses
- Effects on unsubscribes, not just clicks
- Delayed revenue and negative attribution revisions
- Differential reporting or event qualification
- Shared sender reputation or inbox-placement effects, where individual-level no-interference can fail

Then specify the policy *before* evaluating it: randomization unit, horizon, minimum maturity, loss for a false winner, loss for a missed opportunity, budget and review schedule. Replay the whole adaptive procedure, including repeated looks and data delays. Evaluate long-run false selections, regret and total cost under the actual simulation scenarios. Do not justify optional stopping simply by calling an analysis Bayesian.

## Keep three decisions separate

1. **Stop acquiring:** changes future purchases of leads; existing cohorts remain under their original sending policy
2. **Stop sending:** compare avoidable future sending costs with future revenue lost; acquisition cost is already sunk
3. **Buy information:** compare the expected benefit of a test with its acquisition, sending and delay costs

A 90% probability of profitability alone cannot choose among these. Magnitude and downside matter. A new cohort must receive a newly drawn cohort effect, not inherit the favorable fitted effect of the historical cohort that prompted the decision.

This repository's baseline acquisition replay is illustrative constant-per-lead counterfactual bookkeeping. It is not a validated causal model of new-cohort performance, source saturation or spillovers.
