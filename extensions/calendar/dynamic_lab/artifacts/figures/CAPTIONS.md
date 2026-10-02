# Synthetic dynamic-cohort figures

All marks come from the sealed fitted or evaluator outputs named below. These are synthetic demonstrations, not observed commercial outcomes.

## Calendar prices and arriving purchases

The synthetic rich illustration (demo-human_decline_bot_rise-884000, human_decline_bot_rise) and fixed-price reference use the same demo seed; established source established_a. The net-human CPC opportunity is evaluator-only truth, not an observed contract rate supplied to inference; the model learns it from mature settled monetary marks. Send cost and the current acquisition quote are observed inputs. Purchase markers divide the batch's recorded acquisition cost by its purchased leads; a later quote applies to new purchases and never rewrites old acquisition costs.

Alt text: Three calendar panels compare fixed and changing evaluator-only human CPC, acquisition quotes with actual purchase prices, and weekly arriving cohorts.

Intervals: No uncertainty interval: net-CPC is evaluator-only truth; quotes, send prices and immutable purchases are simulated known inputs.

Denominator: Revenue per qualified human event; send cost per eligible send; acquisition quote and cost per purchased lead.

Numeric table: [01-calendar-economics.csv](tables/01-calendar-economics.csv)

## Human response and audited bot evidence

One synthetic world (demo-human_decline_bot_rise-884000, human_decline_bot_rise), aggregating all active sources and cohorts; information available on each plotted calendar day. Raw reported activity includes bots and uses report-calendar day, with a two-day report lag. The dotted true-human series uses occurrence-calendar day and is evaluator-only, not a fitting input. Audit fractions use trailing14-day audit reports available on that day, Beta-smoothed and corrected for the known sensitivity/specificity; audit_n is their denominator. Rows with no available audit are omitted from the fraction plot and marked missing in audit_fraction_plotted. Audit instrument error and representative-selection assumptions remain part of the approximate model.

Alt text: Reported-event counts are compared with evaluator-only true human events, above the measured fraction of audited events labeled bot.

Intervals: 95% fitted human interval where provided; audit interval only if explicit audit bounds exist. No invented bounds or smoothed interval endpoints.

Denominator: Daily response events, which may repeat per person; bot fraction uses audit_n reported events, not acquired leads.

Numeric table: [02-human-bot-evidence.csv](tables/02-human-bot-evidence.csv)

## One shared response and payout calendar

One synthetic world (demo-human_decline_bot_rise-884000, human_decline_bot_rise); refit at calendar day 119. This bounded fixture uses same-day send/response cells; it does not fit a response-delay process. Report, audit and mature monetary visibility clocks remain separate. Within each posterior draw, one response path and one value path are shared across all cohorts; the two processes are conditionally independent and their covariance is not learned. Fixed age curves, anchored states and exchangeable cohort effects constrain the age–period–cohort decomposition; these drivers are assumption-dependent. The dashed truth is evaluator-only.

Alt text: Two panels show fitted shared response and payment calendar states with 95% intervals and evaluator-only states, separated by the last fitting origin.

Intervals: 95% approximate posterior state interval through the fitting cutoff and conditional future-state predictive band afterward; not a contribution interval or empirical calibration guarantee.

Denominator: Log multiplicative state relative to the anchored reference date; one shared state per calendar day.

Numeric table: [03-shared-calendar-states.csv](tables/03-shared-calendar-states.csv)

## Later cohorts face the later calendar

One synthetic world (demo-human_decline_bot_rise-884000, human_decline_bot_rise), source established_a. Each top-panel forecast uses a new cohort at age 7 and the same target age180; its actual birth date determines future calendar exposure. The lower panels select cohort established_a-b112 at calendar origin 119 and compare conditional future states: no deterministic drift in the fitted response/value LOG states, with future innovations; extrapolate each posterior draw's recent slope in BOTH response and value, with future innovations; or fix the future CPC state to the learned day0 reference while retaining the response process. Sampled recent slopes may point up or down; these assumptions do not reveal future truth. The pale band belongs to hold_current only; the other lines are separate conditional predictive medians, not interval bounds. The first-payback CDF retains every predictive path, including paths without a crossing by age425. First crossing, positive contribution at age180 and continued positivity are separate events.

Alt text: At a common cohort age, day180 contribution forecasts are compared across calendar origins, followed by one selected contribution path and an unconditional first-payback curve ending at age425.

Intervals: 95% approximate posterior-predictive contribution intervals; first-payback probability over all draws through age425.

Denominator: Contribution / original acquired leads (n_leads from each row); first-payback CDF / all 256 predictive paths, never only crossing paths. Draw denominator and cumulative crossing count are exported per CDF row.

Numeric table: [04-cohort-contribution-payback.csv](tables/04-cohort-contribution-payback.csv)

## Equal cheap quotes do not imply equal quality

Paired synthetic cheap_good and cheap_bad worlds with the same demo seed; two late-arriving cheap sources, both evaluated at age180. The first pane preserves the full transferred interval on common prior axes. The two update panes use the SAME expanded contribution axis, clearly labeled, to show their narrower intervals. The before-purchase distribution integrates the fitted global hierarchy with fresh source and cohort effects; it is not the posterior of an old source. The updated predictive uses that source's subsequently available observation likelihood. Known cheap acquisition quotes enter economics, not an assumed quality feature. Outcome crosses are evaluator-only; table rows retain information counts when exported. The companion table queries ten offered-price cases for a NEW cohort born at day126, viewed at day119. It holds the same posterior outcome draws fixed while changing only that prospective batch's acquisition cost; it does not refit source quality or rewrite existing purchase costs. Missing median first-payback means the unconditional median does not cross by age425.

Alt text: One pane shows the full before-purchase distributions for good and bad worlds; two panes compare updates on the same explicitly expanded contribution scale, with evaluator-only outcomes as crosses.

Intervals: 95% approximate posterior-predictive intervals; transferred prediction includes fresh source/cohort uncertainty; observations update via the likelihood.

Denominator: Contribution / original acquired leads for each prospective batch; source information counts remain in the numeric table.

Numeric table: [05-new-source-transfer.csv](tables/05-new-source-transfer.csv)

Companion numeric table: [05-offered-price-sensitivity.csv](tables/05-offered-price-sensitivity.csv)

## Held-out predictive and decision comparison

Independent held-out synthetic worlds: 3 independent worlds per scenario; 18 across all six declared scenarios. These are separate from the one-world illustration. The visible score markers show the fixed-price control and an aggregate of all six declared scenarios; the numeric table retains every scenario-specific score. All five model alternatives use the same information clocks and target economics. Coverage measures nominal 95% contribution intervals; CRPS scores the full forecast; decision loss scores the declared acquisition policy. World-level uncertainty retains within-world dependence. The bottom panel preserves the original v1's frozen 12-world stress-coverage failure as a different experiment, without combining it with new scores. Limited evaluation cannot establish general calibration.

Alt text: Three horizontal comparisons show 95% coverage, CRPS and acquisition decision loss for held-out worlds, above a distinct panel showing the original v1 coverage failures across forecast ages.

Intervals: 95% evaluator uncertainty bounds clustered by independent world; original panel uses its frozen whole-world bootstrap bounds. Nominal predictive level is 95%.

Denominator: Coverage / n_forecasts; n_worlds independent evaluation clusters; CRPS and decision loss / acquired leads, with source counts retained in table.

Numeric table: [06-heldout-comparison.csv](tables/06-heldout-comparison.csv)
