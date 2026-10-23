# Article draft: when does a subscriber pay for itself?

`article.md` is the review draft; `article.mdx` uses portable plain-text math fences. `index.html` is a responsive local article preview. All eleven figures have desktop/mobile SVG and PNG versions, plain-language captions, alt text and numerical tables. `claims.json` binds narrative quantities to saved synthetic outputs and hashes; `chart_provenance.json` binds chart tables to their source data.

The article follows people and cohorts, shows how aggregate retention and distinct-responder summaries are calculated, and fixes a single cohort and day-180 margin target while evidence accumulates. A separate chart updates the posterior predictive first break-even age as the same cohort matures, and shows the probability of no break-even by day 425. It also separates CPC deterioration, acquisition-price inflation, human/bot quality change and new-source transfer. A paired future-CPC what-if changes price while holding the same subscriber/activity draws fixed. Held-out probability reliability and a sequential threshold replay show why posterior cutoffs still need empirical checks.

Rebuild the article after reproducing the extension outputs:

```sh
python article/build_article.py --extension extensions/calendar
```

The builder binds content from existing outputs; it does not refit a model. `article.template.md` holds editorial text and explicit computed placeholders. The script writes Markdown, MDX, HTML and a self-contained review preview. Mobile figure text is checked at a 390-pixel viewport. Full browser rendering remains unverified here.

All subscribers, prices and outcomes are fictional. The article preserves undercoverage and other limitations instead of presenting the model as validated. It is a draft only; it has not been published on nickroth.com.
