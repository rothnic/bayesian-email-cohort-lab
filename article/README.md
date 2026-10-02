# Article draft and local preview

Working title: **When does an email subscriber pay for itself?**

This is a draft for nickroth.com, not a deployed website. The package includes:

- `article.md` and `article.mdx`: the same editorial draft; MDX uses a plain-text math fence for broad compatibility
- `index.html`: a standalone responsive preview with no remote scripts, fonts or analytics
- `assets/`: five main figures plus an optional reactivation diagnostic, each as desktop/mobile SVG and PNG
- `essential_values.csv` and `.json`: readable numeric values behind the article tables
- `engagement_values.csv`: the descriptive reported-engagement series
- `chart_provenance.json` and `ARTICLE_SOURCES.md`: exact input hashes, selectors and factual source bindings
- `render_charts.py` and `build_preview.py`: rebuild article figures and HTML from existing research outputs without fitting models
- `ASSET_MANIFEST.json`: exact files, sizes and SHA-256 hashes for this article bundle

## Preview

Open `index.html` locally, or serve this folder with `python -m http.server 8771`. On a phone-width viewport the charts use dedicated narrower SVG variants. Each figure also has a readable numeric table and a full-size SVG link. SVG files have title, description and ARIA metadata.

For integration into the research repository, put this folder at `article/`. The rebuild scripts locate the adjacent repository artifacts. They also support the sibling-directory layout used for the standalone draft. Scientific Python dependencies are the research repository's existing requirements.

## Review status and publication gates

- Numeric claims were independently checked against the frozen CSV artifacts and the 60-test implementation
- Main figure values, alt text, local links, mobile chart variants and numeric tables were inspected
- The source repository commit is `8e3443fe287d8e3ee29156d628df931b35fad9d5`
- Browser-level desktop and phone screenshots still require the publisher's supported environment: this authoring environment blocked loopback browser access and local Chromium sockets; no browser security protections were weakened
- Before publishing to nickroth.com, inspect the static preview at desktop and 390px widths, confirm all five value-table disclosures and full-size links work, and approve the editorial draft
- No live website deployment is included or authorized by this package

Every modeled data point is fictional. The opening recollection about a prior DealNews project is explicitly presented as unverified motivation, not a historical result.
