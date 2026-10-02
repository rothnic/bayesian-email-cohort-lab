# Article draft: changing cohort economics

The main draft is `article.md`; `article.mdx` uses plain-text math fences for portable MDX import. `index.html` is the responsive local preview. `standalone.html` embeds every figure, table and source download needed to open the review copy as one file. External repository links are citations only; there are no external runtime fonts, scripts or analytics.

Six figures each have desktop/mobile SVG and PNG versions, descriptive text, captions and numeric tables. `tables/offered-cost.csv` gives the complete future-batch price sensitivity. `claims.json` binds substantive numerical claims to exact saved output rows and hashes. `chart_provenance.json` binds figure values to their extension sources.

The draft covers the implemented extension with changing net human CPC, rising acquisition quotes, audited bot activity, global/source/cohort learning and late cheap sources. It preserves the original v1 failures separately. The approximate likelihood, favorable known measurement assumptions, poor nominal coverage, conditional future-price scenarios and finite-draw limitations are stated in the article.

To rebuild within the public research repository:

```
python article/build_article.py --extension extensions/calendar
```

Install the research repository's existing requirements first. The script reads saved scientific output; it does not fit or select a model. `article.template.md` holds the editorial text and explicit numerical placeholders. `style.css` is self-contained. The six chart families are regenerated separately from `extensions/calendar` with `make figures`.

All article values, selected table rows, PNG decoding, SVG accessibility metadata, local links and standalone dependencies were checked. Dedicated mobile chart assets were visually inspected and their text measures at least 13.86 CSS pixels at a 390-pixel viewport. Actual browser rendering of the full HTML remains unverified because the available rendering routes were blocked. No browser screenshots are claimed or included. This is a draft, with no live nickroth.com deployment.
