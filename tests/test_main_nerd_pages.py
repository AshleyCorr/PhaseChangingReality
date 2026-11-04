"""The web pages of the blog are generated from the papers' LaTeX: the converter must turn the mathematics it
claims to understand into the right HTML, refuse what it does not know, and every page must link to pages and
anchors that exist, for a post with both editions as pages and for a post with the plain-language edition only."""
import collections
import html
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_main_nerd_pages as build  # noqa: E402
import main_nerd_latex as tex  # noqa: E402
import main_nerd_site as site  # noqa: E402
from main_nerd_posts import POST_01, POST_02, POSTS, Post  # noqa: E402

SIZES = collections.defaultdict(lambda: (640, 360))        # the tests render no figure


@pytest.mark.parametrize("source, expected", [
    (r"\lambda = 1.25", "<i>λ</i> = 1.25"),
    (r"4(\lambda-1)", "4(<i>λ</i> − 1)"),
    (r"4\times L", "4 × <i>L</i>"),
    (r"N=64,96,144,192", "<i>N</i> = 64,96,144,192"),
    (r"e^{-12/g}", "<i>e</i><sup>−12/<i>g</i></sup>"),
    (r"\Delta E^\ddagger = 32 - 16\lambda", "Δ<i>E</i><sup>‡</sup> = 32 − 16<i>λ</i>"),
    (r"f_{200}\ge0.75", "<i>f</i><sub>200</sub> ≥ 0.75"),
    (r"\lambda>1", "<i>λ</i> &gt; 1"),
    (r"-0.7", "−0.7"),
    (r"1\tfrac12", "1½"),
])
def test_simple_mathematics_becomes_plain_html(source, expected):
    assert tex.simple_math(source) == expected


@pytest.mark.parametrize("source", [r"\sum_e (S_e-2)_+", r"\binom{M+C-1}{C-1}", r"\widehat\tau", r"\frac{3}{2N}"])
def test_other_mathematics_is_left_for_the_typesetter(source):
    assert tex.simple_math(source) is None
    assert tex.math_inline(source).startswith(r"\(")


def test_inline_text():
    conv = tex.Converter({"a": 1, "b": 2}, {"fig:x": ("3", "#fig-3")})
    out = conv.inline(r"\emph{One} and \textbf{two}~\cite{b,a}, Figure~\ref{fig:x}; 5\% of ``it'' -- done.")
    assert "<em>One</em>" in out and "<strong>two</strong>" in out
    assert '[<a class="cite" href="#src-1">1</a>, <a class="cite" href="#src-2">2</a>]' in out
    assert '<a href="#fig-3">3</a>' in out and "5%" in out and "“it”" in out and "–" in out
    with pytest.raises(tex.Unknown):
        conv.inline(r"\somethingelse{x}")


def test_table_rows_split_outside_mathematics():
    conv = tex.Converter({}, {})
    out = conv.table(r"\toprule A & B \\ \midrule $x & y$ & 2 \\ \bottomrule")
    assert out.count("<th>") == 2 and out.count("<td>") == 2


def reading_text(page):
    """What a reader sees of a page: its body without scripts, code and the mathematics left for the typesetter."""
    body = re.sub(r"<script.*?</script>|<pre>.*?</pre>|\\\(.*?\\\)|\\\[.*?\\\]", " ", page[page.index("<body"):],
                  flags=re.S)
    return html.unescape(re.sub(r"<[^>]+>", " ", body))


def nav(page):
    """The links of a page's navigation between posts, as (address, words)."""
    found = re.search(r'<div class="postnav">(.*?)</div>', page, flags=re.S)
    return re.findall(r'href="([^"]+)">([^<]+)</a>', found.group(1)) if found else []


# ------------------------------------------------------------------ post 01: both editions, and a companion
@pytest.fixture(scope="module")
def pages():
    plain, technical = build.convert(POST_01, SIZES)
    return (site.page_plain(POST_01, plain, technical), site.page_technical(POST_01, technical, plain), plain,
            technical)


def test_the_pages_hold_the_whole_of_both_papers(pages):
    plain_html, tech_html, plain, technical = pages
    assert [c["num"] for c in plain["chapters"]] == list(range(1, 11)) and plain["n_figures"] == 23
    assert plain_html.count('<p class="trail"><strong>The experiment:</strong>') == 6     # chapters 3 to 8
    assert [s["roman"] for s in technical["sections"]] == ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", ""]
    for page in (plain_html, tech_html):
        assert not re.search(r"\\[a-zA-Z]+|\$", reading_text(page))    # no LaTeX left in the reading text


def test_every_cross_link_lands_on_an_anchor(pages):
    plain_html, tech_html, _plain, _technical = pages
    ids = {"plain": set(re.findall(r'\sid="([^"]+)"', plain_html)),
           "tech": set(re.findall(r'\sid="([^"]+)"', tech_html))}
    for name, page in (("plain", plain_html), ("tech", tech_html)):
        for href in re.findall(r'href="([^"]+)"', page):
            if href.startswith("#"):
                assert href[1:] in ids[name], href
            elif href.startswith(site.TECH + "#"):
                assert href.split("#")[1] in ids["tech"], href
            elif href.startswith(site.BASE + "#"):
                assert href.split("#")[1] in ids["plain"], href
    # every chapter points across, and every numbered section points back
    assert plain_html.count('class="xlink"') == 10 and tech_html.count('class="xlink"') == 9
    assert re.search(r'PL, Chs\.\u00a0<a href="/mainnerd/01thecurledtorusburpsPL/#ch2"', tech_html)


def test_the_frame_keeps_its_promises(pages):
    plain_html, tech_html, _plain, _technical = pages
    # every chapter carries one searched question as a heading, and its answer never claims the paper settles it
    assert plain_html.count('<div class="ask"><h3 id="ch') == 10
    assert set(site.CHAPTER_QUESTIONS) == set(range(1, 11))
    for question, answer in site.BIG_QUESTIONS:
        assert question.endswith("?") and html.escape(question) in plain_html
    joined = " ".join(a for _q, a in site.BIG_QUESTIONS)
    assert "Nobody knows" in joined and "not yet reviewed by a physicist" in joined and "toy" in joined
    # advertisements are labelled with the one word, the jokes carry the logo, and one asks for a reader
    for page in (plain_html, tech_html):
        labels = re.findall(r'<span class="ad-label">([^<]*)</span>', page)
        assert labels and set(labels) == {"ADVERTISEMENT"}
        assert "Volunteer to be a reader" in page and "mailto:" + site.AUTHOR_EMAIL in page
    # two of the plain page's four show Ither's phones in the logo's place: one in the sidebar, a pair mid-article
    assert plain_html.count('class="ad-logo"') == 2 and tech_html.count('class="ad-logo"') == 1
    phones = re.findall(r'<span class="ad-phone (\w+)"><img src="([^"]*)" alt="([^"]*)"', plain_html)
    assert [which for which, _src, _alt in phones] == ["talk", "talk", "report"] and "ad-phone" not in tech_html
    assert all(src == site.HERO and alt.startswith("A phone showing") and "Ither" in alt for _w, src, alt in phones)
    for which in ("talk", "report"):                       # each phone has its cut-out rule in the stylesheet
        assert ".ad-phone.%s img{" % which in site.CSS
    # the guestbook exists where the buttons point, and the PDFs are served beside the pages
    assert 'id="guestbook"' in plain_html and site.BASE + "#guestbook" in tech_html
    assert site.PDF_PLAIN in plain_html and site.PDF_TECH in tech_html
    assert "github.com" not in site.PDF_PLAIN + site.PDF_TECH


def test_links_back_to_side_nerd_carry_the_campaign(pages):
    plain_html, tech_html, _plain, _technical = pages
    links = [u for u in re.findall(r'href="(https://sidenerdapps\.com/[^"]*)"', plain_html + tech_html)
             if "/mainnerd" not in u.split("?")[0]]               # the pages' own addresses are not adverts
    assert len(links) >= 6
    for link in links:
        assert "utm_source=mainnerd" in link and "utm_medium=owned-media" in link
        assert "utm_campaign=physics-01-curled-torus-burps" in link and "utm_content=" in link


def test_post_navigation_always_has_somewhere_to_go(pages):
    plain_html, tech_html, _plain, _technical = pages
    soon_html, index_html = site.page_coming_soon([POST_01]), site.page_index([POST_01])
    for page in (plain_html, tech_html):                              # a post: the list, and on to the next post
        assert nav(page) == [(site.BLOG, "all posts"), (site.SOON, "next post »")]
    assert nav(soon_html) == [(site.BASE, "« previous post"), (site.BLOG, "all posts")]
    assert nav(index_html) == [] and "Web Ring" not in plain_html + tech_html + soon_html + index_html
    # the placeholder says plainly that nothing is there, stays out of search results and is not tagged as a paper
    assert "nothing to read here yet" in soon_html and 'content="noindex, follow"' in soon_html
    assert "citation_" not in soon_html and "citation_" not in index_html
    assert "citation_title" in plain_html and "citation_title" in tech_html
    # the cow belongs to the torus paper: on both editions, not on the blog's front page
    assert "sleepy-cow.gif" in plain_html and "sleepy-cow.gif" in tech_html and "sleepy-cow.gif" not in index_html


def test_the_interactive_companion_is_hosted_whole_and_unchanged(pages):
    plain_html, tech_html, _plain, _technical = pages
    source = build.COMPANION.read_text(encoding="utf-8")
    page = site.page_knobs(POST_01, source)
    # one document, one title, and every part of the companion carried over as written
    assert page.startswith("<!DOCTYPE html>") and page.count("<title>") == 1 and page.count("</head>") == 1
    style = re.search(r"<style>.*?</style>", source, flags=re.S).group(0)
    script = source[source.index("<script>"):].strip()
    markup = source[source.index("</style>") + len("</style>"):source.index("<script>")].strip()
    assert style in page and script in page and markup in page
    assert page.index(style) < page.index("</head>") < page.index(markup) < page.index(script)
    for link in re.findall(r"<link\b[^>]*>", source):                 # its fonts, now in the head
        assert link in page[:page.index("</head>")]
    # the blog's own style sheet stays off this page: the two would restyle each other
    assert site.STYLESHEET not in page and 'class="mnb"' in page and "Main Nerd" in page
    assert not re.search(r"<(p|h1|h2|header|section|table)\b[^>]*mnb", page)   # the bar uses no element the tool styles
    # the editions link to the blog's copy, not to the artifact, and the companion links back
    assert site.KNOBS in plain_html and site.ARTIFACT not in plain_html + tech_html + page
    assert 'href="%s"' % site.BASE in page and 'href="%s"' % site.TECH in page and 'href="%s"' % site.BLOG in page


def test_a_post_is_data_and_post_01_keeps_its_old_names():
    assert [p.number for p in POSTS] == sorted(p.number for p in POSTS) and POSTS[0] is POST_01
    assert len({p.slug for p in POSTS}) == len(POSTS)
    assert (site.SLUG, site.BASE, site.TECH) == ("01thecurledtorusburps", "/mainnerd/01thecurledtorusburpsPL/",
                                                 "/mainnerd/01thecurledtorusburps/")
    assert build.CHAPTER_TO_SECTION is POST_01.chapter_to_section and site.FAQ is POST_01.faq
    # post 01's figures keep their bare names in the record of sizes; a later post's carry its slug
    assert build.size_key(POST_01, "plain-fig-01.svg") == "plain-fig-01.svg"
    assert build.size_key(POST_02, "plain-fig-01.svg") == POST_02.slug + "/plain-fig-01.svg"
    recorded = build.recorded_sizes(POST_01)
    assert "plain-fig-23.svg" in recorded and "tech-fig1.svg" in recorded and not any("/" in k for k in recorded)


def test_figure_sizes_are_recorded_post_by_post(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "FIGURE_SIZES", tmp_path / "sizes.json")
    assert build.recorded_sizes(POST_01) == {}                           # nothing rendered yet
    build.write_sizes({POST_01.slug: {"plain-fig-01.svg": (657, 161)}})
    assert json.loads(build.FIGURE_SIZES.read_text()) == {"plain-fig-01.svg": [657, 161]}
    # a post rendered later is added after it, and rendering one post again leaves the other's sizes alone
    build.write_sizes({POST_02.slug: {"plain-fig-01.svg": (657, 154)}})
    build.write_sizes({POST_01.slug: {"plain-fig-01.svg": (640, 150), "tech-fig1.svg": (900, 400)}})
    assert list(json.loads(build.FIGURE_SIZES.read_text())) == ["plain-fig-01.svg", "tech-fig1.svg",
                                                               POST_02.slug + "/plain-fig-01.svg"]
    assert build.recorded_sizes(POST_01) == {"plain-fig-01.svg": (640, 150), "tech-fig1.svg": (900, 400)}
    assert build.recorded_sizes(POST_02) == {"plain-fig-01.svg": (657, 154)}


# ------------------------------------------------------------------ a post with the plain-language edition only
# The stub is a throwaway post pointed at a small file in the macro vocabulary of the plain editions. It is tested
# with no frame text, as its configuration names it (both PDFs offered) and as it stands (neither PDF exists, and no
# share image), and once with every block of frame text written. Every plain-only post whose source is in the
# repository is tested too, as it stands today.
STUB = Post(number=2, slug="02astubpost", title="A stub post", published="2026-10-13", paper_dir="tests/fixtures",
            plain_source="main_nerd_stub_plain.tex", pdf_plain_name="a-stub-post-plain-language.pdf",
            pdf_tech_name="a-stub-post-technical.pdf", opening_words="This file is a fixture",
            cite_key="fixture2026stubpost")
WRITTEN = replace(STUB, subtitle="A subtitle", page_title="A stub post, with a title for its page",
                  description="What a search result would show.", summary="<p><b>In one paragraph.</b> A summary.</p>",
                  chapter_questions={2: ("A question for the second chapter?", "A bridge to it.")},
                  big_questions=[("A big question?", "An answer to it.")],
                  faq=[("A quick question?", "A quick answer.")], key_terms=["a term", "another term"])
PLAIN_ONLY = [pytest.param((STUB, False), id="stub"), pytest.param((STUB, True), id="stub-as-it-stands"),
              pytest.param((WRITTEN, False), id="stub-with-its-frame-text")]
PLAIN_ONLY += [pytest.param((p, True), id="post%02d" % p.number) for p in POSTS
               if not p.has_technical and build.source_of(p).exists()]


def blog_of(posts):
    converted = {p.slug: build.convert(p, SIZES) for p in posts}
    return converted, build.blog_pages(posts, converted)


@pytest.fixture(scope="module", params=PLAIN_ONLY)
def blog(request):
    """A blog of post 01 and one plain-only post after it: (that post, its converted text, every page by path)."""
    post, as_it_stands = request.param
    if as_it_stands:
        post = build.as_it_stands(post)
    converted, pages = blog_of([POST_01, post])
    return post, converted[post.slug][0], pages


def test_the_stub_post_converts():
    plain = build.convert_plain(STUB, SIZES)
    assert [c["num"] for c in plain["chapters"]] == [1, 2] and plain["n_figures"] == 1 and not plain["notes"]
    assert [c["kicker"] for c in plain["chapters"]] == ["THE FIRST PART", "THE SECOND PART"]
    first, second = (c["html"] for c in plain["chapters"])
    opening = plain["opening"]                              # the title block above it is the page's own header
    assert opening.startswith("<p>This file is a fixture") and "A FIXTURE FOR THE TESTS" not in opening
    assert '<figure class="fig" id="fig-1"><img src="%splain-fig-01.svg"' % STUB.assets in first
    assert '<a href="#fig-1">1</a>' in first and '<span class="tag tag-exact">EXACT</span>' in first
    assert '<h3 id="ch1-a-sub-heading">' in first and '<h3 id="ch2-a-sub-heading">' in second
    assert first.count('<aside class="takeaway">') == 1 and second.count('<aside class="takeaway">') == 1
    assert '<p class="why">' in second and second.count("<li>") == 2 and second.count("<td>") == 4
    # chapters that exist are linked, a chapter that does not is not, and neither is a section of a paper without a page
    assert 'Chapter\u00a0<a href="#ch2">2</a>' in first and 'Chapter\u00a0<a href="#ch1">1</a>' in second
    assert "Chapter\u00a07 here" in second and "Section\u00a0II of" in first
    # the back matter starts at the closing heading and holds the glossary and the three sources
    back = plain["back"]
    assert back.startswith('<h2 id="continuing">Continuing into the technical paper</h2>')
    assert site._terms(back) == [("N", "Number of dots."), ("Link", "A line between two dots.")]
    assert back.count('<li id="src-') == 3 and "J. Fixt. <strong>1</strong>" in back     # a control space at a line end
    # the technical paper is cited by a link to its PDF; with no PDF, to its entry among the sources
    assert '[<a class="cite" href="%s">3</a>]' % STUB.pdf_tech in plain["opening"]
    without = build.convert_plain(replace(STUB, pdf_tech_name=""), SIZES)
    assert '[<a class="cite" href="#src-3">3</a>]' in without["opening"]
    # a post that names no opening words starts at its second paragraph, after the title block
    assert build.convert_plain(replace(STUB, opening_words=""), SIZES) == plain


def test_a_source_that_lacks_what_a_page_is_built_around_stops_the_build():
    with pytest.raises(SystemExit, match="does not have the words"):
        build.convert_plain(replace(STUB, opening_words="words that are not in the file"), SIZES)
    with pytest.raises(SystemExit, match="figure 1 has not been rendered"):
        build.convert_plain(STUB, {})


def test_a_post_without_its_source_is_left_out_with_one_note(capsys):
    unwritten = replace(STUB, plain_source="not_written_yet.tex")
    ready = build.posts_to_build([POST_01, unwritten])
    notes = [line for line in capsys.readouterr().out.splitlines() if "is not built" in line]
    assert [p.number for p in ready] == [1] and len(notes) == 1
    assert notes[0] == "NOTE: post 02 is not built: tests/fixtures/not_written_yet.tex is not there yet"
    with pytest.raises(SystemExit):
        build.posts_to_build([unwritten])
    # alone, post 01 is the blog it was: its "next post" is the placeholder, and it is the whole list
    _converted, pages = blog_of(ready)
    assert sorted(pages) == ["01thecurledtorusburps/index.html", "01thecurledtorusburpsKnobs/index.html",
                             "01thecurledtorusburpsPL/index.html", "assets/main-nerd.css", "comingsoon/index.html",
                             "index.html"]
    assert nav(pages["01thecurledtorusburpsPL/index.html"])[-1] == (site.SOON, "next post »")
    assert pages["index.html"].count('class="post-card"') == 1


def test_a_plain_only_post_converts(blog):
    post, plain, pages = blog
    page = pages[post.slug + "PL/index.html"]
    assert page.startswith("<!DOCTYPE html>") and page.count("<h1>") == 1 and "<h1>%s</h1>" % post.title in page
    assert plain["chapters"] and page.count('<section class="chapter" id="ch') == len(plain["chapters"])
    assert page.count('<figure class="fig"') == plain["n_figures"]
    assert not re.search(r"\\[a-zA-Z]+|\$", reading_text(page))        # no LaTeX left in the reading text
    assert 'id="continuing"' in page and 'id="guestbook"' in page and 'id="how-to-cite"' in page
    # its pictures are its own, and the advertisements are the blog's
    assert all(src.startswith(post.assets) for src in re.findall(r'<img src="(/mainnerd/[^"]*)"', page))
    for ad in (site.AD_TOP, site.AD_MID, site.AD_SIDE, site.AD_FOOT, site.ad_reader(post)):
        assert page.count(ad) == 1


def test_a_plain_only_post_has_no_page_it_does_not_have(blog):
    post, _plain, pages = blog
    assert post.slug + "/index.html" not in pages and post.slug + "Knobs/index.html" not in pages
    for name, page in pages.items():
        for href in re.findall(r'href="([^"#]*)', page):
            assert href not in (post.tech, post.knobs), (name, href)
    page = pages[post.slug + "PL/index.html"]
    assert 'class="xlink"' not in page and "sleepy-cow.gif" not in page and "Turn the knobs" not in page
    # "the technical edition" is its PDF where that is offered; otherwise the page does not offer one
    across = re.findall(r'href="([^"]*)" target="mn-technical"', page)
    assert set(across) == ({post.pdf_tech} if post.pdf_tech else set())
    assert ("Read the technical edition (PDF) »" in page) == bool(post.pdf_tech)
    assert ("This edition as a PDF" in page) == ("citation_pdf_url" in page) == bool(post.pdf_plain)
    # a citation leads to its entry among the sources; one of the technical paper may lead to that PDF instead
    for href in re.findall(r'<a class="cite" href="([^"]+)">', page):
        assert href.startswith("#src-") or (post.pdf_tech and href == post.pdf_tech), href


def test_an_empty_block_is_left_off_the_page(blog):
    post, plain, pages = blog
    page = pages[post.slug + "PL/index.html"]
    for wanted, marks in ((post.subtitle, ['<p class="subtitle">']), (post.chapter_questions, ['<div class="ask">']),
                          (post.big_questions, ['id="big-questions"', 'href="#big-questions"']),
                          (post.faq, ['id="quick-answers"', 'href="#quick-answers"']),
                          (post.key_terms, ['id="key-terms"']),
                          (post.description, ['<meta name="description"', '<meta property="og:description"'])):
        for mark in marks:
            assert (mark in page) == bool(wanted), mark
    # the opening section starts with the summary, or straight away with the paper's own first words
    assert '<section class="opening">\n' + (post.summary or plain["opening"][:40]) in page
    assert not re.search(r"<(h[1-6]|p|section|ul|ol)\b[^>]*>\s*</\1>", page)       # nothing empty is left standing
    # what the answer engines read is valid, and says nothing with an empty value
    (ld,) = re.findall(r'<script type="application/ld\+json">(.*?)</script>', page, flags=re.S)
    graph = json.loads(ld)["@graph"]
    kinds = [node["@type"] for node in graph]
    assert kinds[:2] == ["ScholarlyArticle", "BreadcrumbList"]
    assert ("FAQPage" in kinds) == bool(post.big_questions or post.faq) and "DefinedTermSet" in kinds
    article = graph[0]
    assert all(value not in ("", [], None) for value in article.values())
    assert article["headline"] == post.headline and article["datePublished"] == post.published
    assert article["dateModified"] == (post.updated or post.published)
    assert ("isBasedOn" in article) == bool(post.pdf_tech) and ("keywords" in article) == bool(post.key_terms)
    # the front page lists it the same way: no empty subtitle, no button without a file behind it
    card = re.search(r'<article class="post-card">.*?%s.*?</article>' % re.escape(post.title), pages["index.html"],
                     flags=re.S).group(0)
    assert ("<p><i>%s.</i> By Emily Smith.</p>" % post.subtitle if post.subtitle else "<p>By Emily Smith.</p>") in card
    assert "Read the technical edition" not in card
    assert ("Plain PDF" in card) == bool(post.pdf_plain) and ("Technical PDF" in card) == bool(post.pdf_tech)


def test_every_link_on_the_blog_lands(blog):
    post, _plain, pages = blog
    posts = [POST_01, post]
    ids = {name: set(re.findall(r'\sid="([^"]+)"', page)) for name, page in pages.items()}
    files = {site.SHARED + site.STYLESHEET} | {pdf for p in posts for pdf in (p.pdf_plain, p.pdf_tech) if pdf}
    for name, page in pages.items():
        for href in re.findall(r'href="([^"]+)"', page):
            if href.startswith("#"):
                assert href[1:] in ids[name], (name, href)
            elif href.startswith(site.BLOG):
                address, _, anchor = href.partition("#")
                if address in files:
                    continue
                target = address[len(site.BLOG):] + "index.html"
                assert address.endswith("/") and target in pages, (name, href)
                assert not anchor or anchor in ids[target], (name, href)
        # a page shows only pictures of a post that is up, and is shared with a picture that is there
        for src in re.findall(r'<img src="(/mainnerd/[^"]*)"', page):
            assert any(src.startswith(p.assets) for p in posts), (name, src)
        for shared in re.findall(r'<meta property="og:image" content="([^"]*)"', page):
            if not post.share_image:                       # as it stands, it has none of its own: post 01's
                assert shared == site.SITE + POST_01.assets + POST_01.share_image, (name, shared)
    assert (build.assets_dir(POST_01) / POST_01.share_image).exists()


def test_navigation_between_posts(blog):
    post, _plain, pages = blog
    first, second = (pages[p.slug + "PL/index.html"] for p in (POST_01, post))
    assert nav(first) == nav(pages[POST_01.slug + "/index.html"]) == [(site.BLOG, "all posts"),
                                                                    (post.base, "next post »")]
    assert nav(second) == [(POST_01.base, "« previous post"), (site.BLOG, "all posts"), (site.SOON, "next post »")]
    assert nav(pages["comingsoon/index.html"]) == [(post.base, "« previous post"), (site.BLOG, "all posts")]
    # the front page lists every post that is up, newest first, and the placeholder leads back to the newest
    listed = re.findall(r'<article class="post-card">.*?<h3><a href="([^"]+)">', pages["index.html"], flags=re.S)
    assert listed == [post.base, POST_01.base] and nav(pages["index.html"]) == []
    assert "Post No. %02d in plain language" % post.number in pages["comingsoon/index.html"]
    assert "technical edition »" not in pages["comingsoon/index.html"]
    # each page's ticker announces its own post; the front page and the placeholder announce the newest
    for page, announced in ((first, POST_01), (second, post), (pages["index.html"], post),
                            (pages["comingsoon/index.html"], post)):
        assert html.escape(site.ticker(announced)) in page


def test_a_second_post_changes_only_where_post_01_leads_next(blog):
    post, _plain, pages = blog
    _converted, alone = blog_of([POST_01])
    for name in ("01thecurledtorusburpsPL/index.html", "01thecurledtorusburps/index.html",
                 "01thecurledtorusburpsKnobs/index.html", "assets/main-nerd.css"):
        before, after = alone[name].split("\n"), pages[name].split("\n")
        changed = [(a, b) for a, b in zip(before, after) if a != b]
        assert len(before) == len(after) and len(changed) == (0 if "Knobs" in name or name.endswith(".css") else 1)
        for a, b in changed:
            assert 'class="postnav"' in a and b == a.replace('href="%s">next post' % site.SOON,
                                                             'href="%s">next post' % post.base)
