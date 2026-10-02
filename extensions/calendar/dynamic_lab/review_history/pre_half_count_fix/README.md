# Superseded initial approximation run

These files preserve the first completed run before correcting the zero-response-bin variance. The original formula used (estimated_humans + raw_events² Var(human_fraction))/(estimated_humans+.5)² and gave zero response sampling variance to an all-zero bin. The subsequent fixed model uses (estimated_humans+.5 + raw_events² Var(human_fraction))/(estimated_humans+.5)². No priors or world seeds were tuned.

A separate evaluation coverage gap was fixed at the same checkpoint: these initial tables scored only established-source targets, including within cheap-source worlds. Final tables also directly score the unseen/new cheap sources. Therefore aggregate before/after scores are not a controlled estimate of the likelihood correction's benefit. The retained code is historical text, not the active model.

The initial dynamic 95% coverage was57.87% across216 correlated forecasts from18 independent worlds; this remains evidence of the provisional approximation's failure, not the final evaluation statistic. The new zero-count test and derivation are the direct justification for the correction.
