# giulioisac.github.io

Personal site. Hand-built static HTML, served by GitHub Pages (`.nojekyll`, so
what is committed is what is served).

## Writing a post

Prose lives in Markdown, not in the HTML:

```
content/<slug>.md      prose + front matter
figures/<name>.html    hand-tuned inline SVG figures
templates/post.html    page chrome (topbar, MathJax, footer, analytics)
```

Edit the Markdown, then rebuild:

```
python build.py            # regenerate <slug>.html and the index cards
python build.py --check    # exit 1 if the committed HTML is stale
```

`<slug>.html` is a build artifact. Editing it directly is wasted work, because
the next build overwrites it. Commit the sources and the generated HTML
together.

### Front matter

```yaml
---
title: Migration Networks from Neutral Allele Frequencies
description: shown in the <meta> tag and to search engines
blurb: shown on the index card, if it should differ from description
tag: Notes / Population Genetics
order: 3
math: true
unlisted: true
script: assets/gene-flow.js
---
```

`order` sets the position in the writing list on the home page. `math` pulls in
MathJax. `script` is an optional per-post JavaScript file, used for the live
figures. `unlisted: true` still builds the page, but keeps it off the home page
and adds `noindex, nofollow`, so a draft can sit at its URL without being linked
or indexed. Unlinked is not private: anyone with the URL can still read it.

### Figures

A figure is a block of hand-tuned HTML in `figures/`, referenced from the
Markdown by name on its own line:

```
{{figure: gene-flow-figRelax}}
```

which splices in `figures/gene-flow-figRelax.html`. Keeping them out of the
prose is the point: the SVG is long and the text should stay readable.

### Notes

- LaTeX survives the Markdown pass untouched. `\( ... \)` and `\[ ... \]` are
  stashed behind tokens before conversion and restored afterwards, so
  underscores and backslashes are safe.
- External links get `target="_blank" rel="noopener"` added automatically.
  Write them as ordinary Markdown links.
- The stylesheet cache-buster (`assets/style.css?v=...`) is a hash of
  `assets/style.css`, stamped onto every page by the build. Never edit it by
  hand: after changing the CSS, just rebuild.
