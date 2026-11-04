"""Build a standalone web page from the whole-account write-up. NOT PUBLISHED: /physics now serves the
plain-language page (scripts/make_site_page.py, the owner's decision of 24 September), and this page's URL
is undecided. It also needs revising before it is shown (matter as the melt, the waiting-time figure, the
curvature claim for the curled torus, Part V). Usage:

    python scripts/make_physics_page.py

Input: `docs/public/the-loop-and-the-floor_v1.html`, the page exactly as published as an artifact
(it opens with <title>, a font <link> and its own <style>, and has no <html>/<head>/<body> of its own).
Output: `docs/public/site/physics/index.html`, a complete HTML document with its own head, sharing
tags, a canonical URL and public links in place of repository paths. The page has no JavaScript and
every picture is inline SVG, so nothing is fetched except two Google fonts, and the page renders with
fallback faces without them. Copy the output file into the SideNerdMarketing repository at the path
that serves /physics/ and let its publishing action pick it up.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "public" / "the-loop-and-the-floor_v1.html"
OUT = ROOT / "docs" / "public" / "site" / "account" / "index.html"

REPO = "https://github.com/EmilySmithCreate/RecreationalPhysics"
URL = "https://sidenerdapps.com/physics/account/"  # placeholder; not published
TITLE = "Did Space Snap Open?"
DESC = ("The whole account, told without hedges: space opened out of something curled by a push small "
        "enough to count, gravity as a slope in the number of ways, black holes folding space back into "
        "what it came from, why the world is quantum, and why you would find yourself on the floor.")

ICON = ("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E"
        "%3Ctext y='26' font-size='26'%3E%E2%9C%A6%3C/text%3E%3C/svg%3E")


def build(body):
    # repository paths become public links
    body = body.replace(
        "<code>github.com/EmilySmithCreate/RecreationalPhysics</code>",
        '<a href="' + REPO + '">github.com/EmilySmithCreate/RecreationalPhysics</a>')
    body = body.replace(
        "the tests and open questions are in the repository</p>",
        'the tests and open questions are <a href="' + REPO + '">in the repository</a></p>')

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
</style>
"""
    cut = body.index("</style>") + len("</style>")
    return head + body[:cut] + "\n</head>\n<body>\n" + body[cut:].lstrip("\n") + "\n</body>\n</html>\n"


def main():
    doc = build(SRC.read_text(encoding="utf-8"))
    assert "claude.ai/artifact" not in doc, "a private companion link survived"
    assert "<img" not in doc, "an external image survived; every picture should be inline SVG"
    assert "<script" not in doc, "the page is meant to carry no JavaScript"
    assert doc.count("<title>") == 1 and doc.count("<style>") == 2
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(doc, encoding="utf-8")
    print("written %s (%.0f KB)" % (OUT, len(doc) / 1024))


if __name__ == "__main__":
    main()
