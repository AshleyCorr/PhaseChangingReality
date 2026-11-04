"""Build the standalone web page for sidenerdapps.com from the saved write-up. Usage:

    python scripts/make_site_page.py

Input: `docs/public/did-space-snap-open_v1.html`, the plain-language write-up exactly as published.
Output: `docs/public/site/physics/index.html` (served at sidenerdapps.com/physics/), a complete HTML document with its own head, sharing-preview
tags, public links in place of the private companion ones, and a no-JavaScript note inside each
chart. Every picture on the page is inline SVG, so nothing is fetched except two Google fonts, and
the page renders with fallback faces without them. Change `URL` below if the published path differs.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "public" / "did-space-snap-open_v1.html"
OUT = ROOT / "docs" / "public" / "site" / "physics" / "index.html"

REPO = "https://github.com/EmilySmithCreate/RecreationalPhysics"
URL = "https://sidenerdapps.com/physics/"
TITLE = "Did Space Snap Open?"
DESC = ("A hobbyist's test of an idea about where space came from: a curled-up arrangement opening "
        "out sharply and releasing the energy that became everything in it, tested in a close cousin "
        "of a published computer model, with the predictions written down first.")

ICON = ("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E"
        "%3Ctext y='26' font-size='26'%3E%E2%9C%A6%3C/text%3E%3C/svg%3E")

NOJS = ('<noscript><p class="nojs">This chart is drawn by JavaScript, which is switched off. '
        'The figures behind it are in &ldquo;See the numbers&rdquo; just below.</p></noscript>')


def build(body):
    # the private companion links become public ones
    body = body.replace(
        '      <span class="btn soon" aria-disabled="true">Full methods and results (paper) &mdash; link coming soon</span>\n'
        '      <a class="btn" href="https://claude.ai/artifact/TTC9knKGJFkR43X2JjsjZ5">The long version of this page &rarr;</a>\n',
        '      <a class="btn" href="' + REPO + '">The code and every result file &rarr;</a>\n'
        '      <span class="btn soon" aria-disabled="true">Full methods and results (paper) &mdash; coming soon</span>\n')
    body = body.replace(
        '<p><b>More detail.</b> Full methods and results will be in the paper (link coming soon). The long '
        'version of this page covers every step in 31 sections: '
        '<a href="https://claude.ai/artifact/TTC9knKGJFkR43X2JjsjZ5">read it here</a>. '
        'Code, configurations and every result file: <code>EmilySmithCreate/RecreationalPhysics</code>.</p>',
        '<p><b>More detail.</b> The code, the configurations and every result file, including the ones with '
        'unwelcome answers: <a href="' + REPO + '">' + REPO.replace("https://", "") + '</a>. Full methods and '
        'results will be in a paper, and a longer version of this write-up covers every step in 31 sections; '
        'both are available on request in the meantime.</p>')

    head = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="description" content="{DESC}">
<meta name="author" content="Emily Smith">
<link rel="canonical" href="{URL}">
<meta property="og:type" content="article">
<meta property="og:title" content="{TITLE}">
<meta property="og:description" content="{DESC}">
<meta property="og:url" content="{URL}">
<meta property="og:site_name" content="Side Nerd Apps">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{TITLE}">
<meta name="twitter:description" content="{DESC}">
<link rel="icon" href="{ICON}">
<style>
  *, *::before, *::after {{ box-sizing: border-box; }}
  html {{ -webkit-text-size-adjust: 100%; }}
  body {{ margin: 0; }}
  img {{ max-width: 100%; }}
  .nojs {{ margin: 0; padding: 1rem; font: 400 .9rem/1.5 system-ui, sans-serif;
          color: #6f747e; border: 1px dashed currentColor; border-radius: 8px; }}
</style>
"""

    # the write-up opens with <title>, the font <link> and its own <style>: all of that is head matter
    cut = body.index("</style>") + len("</style>")
    doc = head + body[:cut] + "\n</head>\n<body>\n" + body[cut:].lstrip("\n") + "\n</body>\n</html>\n"

    # the five data charts are drawn by JavaScript; say so where they would be, and point at the tables
    doc, charts = re.subn(r'(<div class="scroll" id="c-[a-z]+"></div>)',
                          lambda m: m.group(1)[:-6] + NOJS + "</div>", doc)
    return doc, charts


def main():
    doc, charts = build(SRC.read_text(encoding="utf-8"))
    assert "claude.ai/artifact" not in doc, "a private companion link survived"
    assert "<img" not in doc, "an external image survived; every picture should be inline SVG"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(doc, encoding="utf-8")
    print("written %s (%.0f KB), no-JavaScript notes in %d charts" % (OUT, len(doc) / 1024, charts))


if __name__ == "__main__":
    main()
