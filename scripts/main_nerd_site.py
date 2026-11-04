"""The page shell for the posts of the Side Nerd Blog, sidenerdapps.com/mainnerd/: each edition of a paper is a
page, at /mainnerd/<slug>PL/ (plain language) and, where the post has one, /mainnerd/<slug>/ (technical), drawn as
a 1990s home page in Side Nerd's colors, with a few advertisements (for Side Nerd, the author's company, which
sponsors the pages, and one that asks for a scientist to read the work), a guestbook that runs on e-mail, and the
blocks search engines and AI assistants read (summary, question headings, quick answers, glossary, how to cite,
JSON-LD).

Everything about a paper on these pages comes from the papers' own text (scripts/build_main_nerd_pages.py);
what is written here is the frame around it, the same for every post. What differs from post to post (title,
addresses, the questions and answers, the mascot) is data, in scripts/main_nerd_posts.py, and each page function
takes the post it draws. A block whose text is empty is left off the page, with the links that would lead to it.

Links are root-absolute because the site serves a folder with and without a trailing slash, and a relative
link breaks without one. A post's folder name is its slug; change it in main_nerd_posts.py and nowhere else.
"""
import html
import json
import re
from urllib.parse import quote

from main_nerd_posts import BLOG, POST_01, POSTS, REPO, SHARED

SITE = "https://sidenerdapps.com"
STYLESHEET = "main-nerd.css"
SOON = BLOG + "comingsoon/"                        # where "next post" goes while the next post is not up yet
LOGO = "https://sidenerd-marketing-assets.s3.us-east-1.amazonaws.com/sidenerdapps/logo-original.png"
AUTHOR_EMAIL = "emilysmithcreate@gmail.com"                      # the address printed on the technical paper
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December")

# Post 01 under the names it had when it was the only post.
SLUG, TITLE, SUBTITLE, PUBLISHED = POST_01.slug, POST_01.title, POST_01.subtitle, POST_01.published
BASE, TECH, KNOBS, ASSETS = POST_01.base, POST_01.tech, POST_01.knobs, POST_01.assets
PAPER_DIR = POST_01.paper_url
PDF_PLAIN_NAME, PDF_TECH_NAME = POST_01.pdf_plain_name, POST_01.pdf_tech_name
PDF_PLAIN, PDF_TECH = POST_01.pdf_plain, POST_01.pdf_tech
ARTIFACT, LADDER = POST_01.artifact, POST_01.knobs
SUMMARY, CHAPTER_QUESTIONS, BIG_QUESTIONS = POST_01.summary, POST_01.chapter_questions, POST_01.big_questions
FAQ, KEY_TERMS = POST_01.faq, POST_01.key_terms


def long_date(iso):
    """'2026-10-10' as it is written on the pages: 10 October 2026."""
    year, month, day = (int(x) for x in iso.split("-"))
    return "%d %s %d" % (day, MONTHS[month - 1], year)


def utm(slot, path="/"):
    """A link back to Side Nerd, tagged the way the site's other links are (lowercase, hyphenated). The campaign
    is the blog's one campaign, named when post 01 was its only post, and is the same on every page."""
    return ("%s%s?utm_source=mainnerd&utm_medium=owned-media&utm_campaign=physics-01-curled-torus-burps"
            "&utm_content=%s" % (SITE, path, slot))


def mailto(subject):
    return "mailto:%s?subject=%s" % (AUTHOR_EMAIL, quote(subject))


# ------------------------------------------------------------------ the advertisements
# Ither, Side Nerd's volunteer-hours product, in the two advertisements that show a phone. The site has one
# Ither picture: three tilted phones side by side, 1000 x 500 (on /capture-work and textither.com). The style
# sheet cuts one phone out of it and stands it upright (.ad-phone), so nothing is copied into this repository.
HERO = "https://sidenerd-marketing-assets.s3.us-east-1.amazonaws.com/sidenerdapps/ither-3phones-hero.png"
PHONE_ALT = {
    "talk": "A phone showing a text conversation with Ither: a volunteer texts the hours they gave, and the replies "
            "log each entry and send a link to a report.",
    "report": "A phone showing the Ither Volunteer Activity Report: 61 volunteer hours, 20 entries, 9 organizations, "
              "and a bar chart of hours by month.",
}


def phone(which):
    return ('<span class="ad-phone %s"><img src="%s" alt="%s" width="1000" height="500" loading="lazy"></span>'
            % (which, HERO, html.escape(PHONE_ALT[which])))


PHONE_TALK = phone("talk")                                              # the sidebar panel: the conversation
PHONE_PAIR = ('<span class="ad-pair">%s<span class="ad-arrow" aria-hidden="true">»</span>%s</span>'
              % (phone("talk"), phone("report")))                       # mid-article: text in, report out


def ad(slot, kind, headline, body, button, picture=None):
    """An advertisement for Side Nerd, with its logo or, where `picture` is given, phone screenshots of a product
    in the logo's place. The humor is the 1990s banner; the product is real."""
    if picture is None:
        picture = '<img class="ad-logo" src="%s" alt="Side Nerd" width="72" height="72" loading="lazy">' % LOGO
    return ('<aside class="ad ad-%s" aria-label="Advertisement">'
            '<span class="ad-label">ADVERTISEMENT</span>'
            '<a href="%s" rel="noopener">%s<span class="ad-copy"><span class="ad-head">%s</span>'
            '<span class="ad-body">%s</span><span class="ad-btn">%s</span></span></a></aside>'
            % (kind, html.escape(utm(slot)), picture, headline, body, button))


AD_TOP = ad("top-banner", "banner", "TIRED OF LEARNING NEW SOFTWARE?",
            "So is everyone you work with. Side Nerd turns a plain text message into a finished record. "
            "If they can text, they can use it.", "CLICK HERE!!")
AD_MID = ad("mid-article", "box", "4,000 TUBES. ZERO FORMS FILLED OUT.",
            "The simulations on this page ran without a single login, portal or training session. Your team "
            "deserves the same. Side Nerd: messy input in, structured data out.", "Stop asking people to learn "
            "your software »", picture=PHONE_PAIR)
AD_SIDE = ad("sidebar", "tower", "SOFTWARE PEOPLE ACTUALLY USE",
             "No app. No portal. No login. If they can text, they can use it. Day one, no training.", "SEE HOW",
             picture=PHONE_TALK)
AD_FOOT = ad("footer-banner", "banner", "THIS PHYSICS IS BROUGHT TO YOU BY CONVERSATIONAL FORMS",
             "Side Nerd pays the computing bills. It makes software for people who will not use software: "
             "they text, the record writes itself.", "Book a demo »")
AD_TECH = ad("technical-footer", "banner", "YOU READ A TECHNICAL PAPER.",
             "Your users will not read a manual. Side Nerd: they text what happened, and the "
             "structured record appears where it belongs.", "See how »")


def ad_reader(post):
    """The work has had no scientific reader yet. The e-mail the advertisement opens names the post."""
    return ('<aside class="ad ad-wanted" aria-label="Advertisement">'
            '<span class="ad-label">ADVERTISEMENT</span>'
            '<a href="%s"><span class="ad-copy">'
            '<span class="ad-head">REAL PHYSICIST, MATHEMATICIAN OR RELEVANT SCIENTIST?</span>'
            '<span class="ad-body">Volunteer to be a reader! No physicist has reviewed this work yet, and you '
            'could be the first. Compensation: sincere thanks, and first rights to say what is wrong with it.'
            '</span><span class="ad-btn">E-MAIL THE AUTHOR »</span></span></a></aside>'
            % html.escape(mailto("Volunteer reader: " + post.title)))


TICKER_ALWAYS = ("No physicist has reviewed this page (volunteers wanted)", "Sign the guestbook")


def ticker(post):
    """The line that runs across the top of a page: the post, its own lines, then the blog's. A post's pages
    announce that post; the front page and the placeholder announce the newest one."""
    return "★ " + " ★ ".join(("NEW POST! " + post.title,) + tuple(post.ticker) + TICKER_ALWAYS) + " ★"


# ------------------------------------------------------------------ the posts that are up
# The page functions take `posts`: the posts that are up, oldest first. The build passes the ones it could build.
def neighbors(post, posts):
    """(the post before, the post after) among the posts that are up; None where there is none."""
    k = [p.number for p in posts].index(post.number)
    return (posts[k - 1] if k else None), (posts[k + 1] if k + 1 < len(posts) else None)


def share_image(post):
    """The picture shown when a page is shared: the post's own, or the first post's where it has none."""
    own = post if post.share_image else POSTS[0]
    return SITE + own.assets + own.share_image


def own_companions(page, posts):
    """Links to a companion's first home, the artifact, lead to the blog's own copy of it."""
    for p in posts:
        if p.has_companion and p.artifact:
            page = page.replace(p.artifact, p.knobs)
    return page


# ------------------------------------------------------------------ what the answer engines read
def cite_block(post, url, title, note):
    year, month, _day = post.published.split("-")
    bib = ("@misc{%s,\n  author = {Smith, Emily},\n  title  = {%s},\n  year   = {%s},\n"
           "  month  = %s,\n  note   = {%s},\n  url    = {%s}\n}"
           % (post.cite_key, title, year, MONTHS[int(month) - 1][:3].lower(), note, url))
    return ('<section class="cite-box" id="how-to-cite"><h2>How to cite this page</h2>'
            '<p>Emily Smith, <i>%s</i> (%s), %s. <a href="%s">%s</a></p>'
            '<pre>%s</pre></section>' % (html.escape(title), html.escape(note), long_date(post.published), url, url,
                                         html.escape(bib)))


def json_ld(post, kind, url, headline, description, extra=None, faq=None, terms=None):
    article = {
        "@type": "ScholarlyArticle", "@id": url + "#article", "headline": headline, "description": description,
        "url": url, "mainEntityOfPage": url, "inLanguage": "en", "datePublished": post.published,
        "dateModified": post.updated or post.published,
        "author": {"@type": "Person", "name": "Emily Smith", "affiliation": "Independent researcher",
                   "url": SITE + "/"},
        "publisher": {"@type": "Organization", "name": "Side Nerd Apps", "url": SITE + "/"},
        "isPartOf": {"@type": "Blog", "name": "The Side Nerd Blog", "url": SITE + BLOG},
        "keywords": ", ".join(post.key_terms), "image": share_image(post),
        "isAccessibleForFree": True, "sameAs": [post.paper_url],
        "about": [{"@type": "Thing", "name": t} for t in post.key_terms[:9]],
    }
    article.update(extra or {})
    article = {k: v for k, v in article.items() if v not in ("", [])}      # nothing is said with an empty value
    graph = [article, {
        "@type": "BreadcrumbList", "itemListElement": [
            {"@type": "ListItem", "position": 1, "name": "Side Nerd Apps", "item": SITE + "/"},
            {"@type": "ListItem", "position": 2, "name": "The Side Nerd Blog", "item": SITE + BLOG},
            {"@type": "ListItem", "position": 3,
             "name": post.title + (" (technical edition)" if kind == "technical" else " (plain-language edition)"),
             "item": url}]}]
    if faq:
        graph.append({"@type": "FAQPage", "@id": url + "#faq", "mainEntity": [
            {"@type": "Question", "name": q, "acceptedAnswer": {"@type": "Answer", "text": a}} for q, a in faq]})
    if terms:
        graph.append({"@type": "DefinedTermSet", "@id": url + "#translations",
                      "name": "Terms used in " + post.title,
                      "hasDefinedTerm": [{"@type": "DefinedTerm", "name": n, "description": d,
                                          "inDefinedTermSet": url + "#translations"} for n, d in terms]})
    return ('<script type="application/ld+json">%s</script>'
            % json.dumps({"@context": "https://schema.org", "@graph": graph}, ensure_ascii=False, indent=1))


def head(post, title, description, url, ld, mathjax, pdf="", paper=True):
    """`post` gives the share picture and, on a paper's own pages, the citation tags; `pdf` is that edition's PDF
    where it has one. `paper=False` is for pages that are not a paper (the front page, the placeholder, the
    companion): no citation tags. A page with no description yet goes without the tags that would carry it."""
    image = share_image(post)

    def only(wanted, tag):
        return tag if wanted else None

    parts = [
        '<!DOCTYPE html>', '<html lang="en">', '<head>', '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<title>%s | Side Nerd Apps</title>' % html.escape(title),
        only(description, '<meta name="description" content="%s">' % html.escape(description)),
        '<meta name="author" content="Emily Smith">',
        '<meta name="robots" content="index, follow, max-image-preview:large">',
        '<link rel="canonical" href="%s">' % url,
        '<meta property="og:type" content="article">', '<meta property="og:url" content="%s">' % url,
        '<meta property="og:title" content="%s">' % html.escape(title),
        only(description, '<meta property="og:description" content="%s">' % html.escape(description)),
        '<meta property="og:image" content="%s">' % image,
        '<meta property="og:site_name" content="Side Nerd Apps">',
        '<meta name="twitter:card" content="summary_large_image">',
        '<meta name="twitter:title" content="%s">' % html.escape(title),
        only(description, '<meta name="twitter:description" content="%s">' % html.escape(description)),
        '<meta name="twitter:image" content="%s">' % image,
        '<meta name="theme-color" content="#34b1e8">',
        only(paper, '<meta name="citation_title" content="%s">' % html.escape(title)),
        only(paper, '<meta name="citation_author" content="Smith, Emily">'),
        only(paper, '<meta name="citation_publication_date" content="%s">' % post.published.replace("-", "/")),
        only(paper and pdf, '<meta name="citation_pdf_url" content="%s">' % (SITE + pdf)),
        '<link rel="icon" type="image/png" href="https://sidenerd-marketing-assets.s3.us-east-1.amazonaws.com/'
        'sidenerdapps/sidenerd-icon.png">',
        '<link rel="preconnect" href="https://fonts.googleapis.com">',
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>',
        '<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&family=Space+Mono:wght@400;700'
        '&display=swap" rel="stylesheet">',
        '<link rel="stylesheet" href="%s%s">' % (SHARED, STYLESHEET),
        ld,
    ]
    parts = [p for p in parts if p is not None]
    if mathjax:
        parts.append("<script>window.MathJax={tex:{inlineMath:[['\\\\(','\\\\)']],displayMath:[['\\\\[','\\\\]']],"
                     "tags:'none'},options:{skipHtmlTags:['script','noscript','style','textarea','pre','code']}};"
                     "</script>")
        parts.append('<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-chtml.js"></script>')
    parts.append('</head>')
    return "\n".join(parts)


def mascot(post, caption=""):
    """The post's mascot under its own caption, or under the one given; nothing for a post without a mascot."""
    m = post.mascot
    if not m:
        return ""
    return ('\n <figure class="mascot">\n  <img src="%s%s" width="%d" height="%d" alt="%s">\n'
            '  <figcaption>%s</figcaption>\n </figure>'
            % (post.assets, m["file"], m["width"], m["height"], m["alt"], caption or m["caption"]))


def masthead(post, edition, other=None, answers="#quick-answers", guestbook="#guestbook"):
    """`other` is the button to the post's other edition, as (label, address, window name). It is left out where
    there is no other edition to open, and so is the button to the quick answers where `answers` is empty."""
    other = '<a class="bevel" href="%s" target="%s">%s</a>\n   ' % (other[1], other[2], other[0]) if other else ""
    answers = '<a class="bevel" href="%s">Quick answers</a> ' % answers if answers else ""
    subtitle = '\n  <p class="subtitle">%s</p>' % post.subtitle if post.subtitle else ""
    return '''
<div class="ticker" aria-hidden="true"><div class="ticker-run">%(ticker)s %(ticker)s</div></div>
<header class="masthead">
 <div class="mast-text">
  <p class="welcome">~ Welcome to <a href="%(blog)s">the Side Nerd Blog</a>, Main Nerd ~</p>
  <h1>%(title)s</h1>%(subtitle)s
  <p class="byline">Post No. %(number)02d · %(date)s · by <b>Emily Smith</b>, independent researcher · filed under Physics · %(edition)s</p>
  <p class="jump">%(other)s%(answers)s<a class="bevel" href="%(guestbook)s">Sign the guestbook</a>
   <a class="bevel" href="#how-to-cite">Cite</a></p>
  <p class="sponsor">Sponsored by <a href="%(sponsor)s">Side Nerd Apps</a></p>
 </div>%(mascot)s
</header>''' % dict(ticker=html.escape(ticker(post)), title=post.title, subtitle=subtitle, number=post.number,
                    date=long_date(post.published) + (" · updated %s" % long_date(post.updated) if post.updated else ""),
                    edition=edition, other=other, answers=answers,
                    guestbook=guestbook, sponsor=html.escape(utm("masthead-sponsor")), blog=BLOG,
                    mascot=mascot(post))


def badges():
    return '''
<div class="badges" aria-hidden="true">
 <span class="badge">Best viewed with<br><b>Netscape Navigator 3.0</b></span>
 <span class="badge">800 × 600<br><b>256 colors</b></span>
 <span class="badge">Made with<br><b>Notepad</b></span>
 <span class="badge">Y2K<br><b>compliant</b></span>
</div>'''


GUESTBOOK_SCRIPT = r'''<script>
document.getElementById('gb-form').addEventListener('submit', function (e) {
  e.preventDefault();
  var f = e.target;
  function v(n) { return (f.elements[n].value || '').trim(); }
  var body = 'Name: ' + v('name') + '\nE-mail: ' + v('email') + '\nSurfing in from: ' + v('from') +
    '\nTell me when the next paper is posted: ' + (f.elements.updates.checked ? 'yes' : 'no') + '\n\n' + v('entry');
  window.location.href = 'mailto:ADDRESS?subject=' + encodeURIComponent(SUBJECT) +
    '&body=' + encodeURIComponent(body);
});
</script>'''


def guestbook(post):
    """The same guestbook on every post; the e-mail it opens names the post it was signed on."""
    subject = "Guestbook: " + post.title
    quoted = "'%s'" % subject.replace("\\", "\\\\").replace("'", "\\'")       # as the script writes a string
    return ('''
<section class="guestbook" id="guestbook"><h2>Sign the guestbook</h2>
<p>Leave a note, a question or a correction. The guestbook runs on e-mail, as in 1996: the button opens your mail
program with your entry filled in, and you press send.</p>
<form class="gb" id="gb-form" action="%(mailto)s" method="get">
 <label>Name <input name="name" autocomplete="name"></label>
 <label>E-mail <input name="email" type="email" autocomplete="email"></label>
 <label>Surfing in from <input name="from" placeholder="city, planet or lab"></label>
 <label class="wide">Your entry <textarea name="entry" rows="4"></textarea></label>
 <label class="check"><input type="checkbox" name="updates" value="yes"> E-mail me when the next paper is posted</label>
 <button class="bevel" type="submit">Sign it!</button>
</form>
<p class="small-ink">Your address is used only to reply to you and, if you tick the box, to tell you about the next
paper. Or skip the form and write to <a href="%(mailto)s">%(address)s</a>.</p>
''' % dict(mailto=html.escape(mailto(subject)), address=AUTHOR_EMAIL)
            + GUESTBOOK_SCRIPT.replace("ADDRESS", AUTHOR_EMAIL).replace("SUBJECT", quoted) + "\n</section>")


def post_nav(kind, post, posts):
    """Navigation between posts, in the footer. A post links back to the one before it and on to the next one;
    while that is not up, to the placeholder. On the placeholder, `post` is the newest post."""
    links = ['<a href="%s">all posts</a>' % BLOG]
    if kind in ("plain", "technical"):
        before, after = neighbors(post, posts)
        if before:
            links.insert(0, '<a href="%s">« previous post</a>' % before.base)
        links.append('<a href="%s">next post »</a>' % (after.base if after else SOON))
    elif kind == "soon":
        links.insert(0, '<a href="%s">« previous post</a>' % post.base)
    else:                                              # the front page is the list of posts
        return ""
    return '<div class="postnav">%s</div>' % " | ".join(links)


def footer(kind, post, posts):
    """`post` is the post the page belongs to; on the front page and the placeholder, the newest one."""
    return '''
<footer class="foot">
 %(nav)s
 <p class="sponsor-foot"><a href="%(sponsor)s"><img src="%(logo)s" alt="" width="44" height="44" loading="lazy">
  Sponsored by Side Nerd Apps</a></p>
 <p><a class="bevel" href="%(base)s#guestbook">Sign the guestbook</a>
  <a href="%(repo)s/issues">report a mistake</a> ·
  <a href="%(paper)s">source, data and supplement</a> ·
  <a href="%(side)s">Side Nerd</a>, which hosts this blog and has no opinion on quantum gravity</p>
 %(badges)s
 <p class="small">Text and figures © 2026 Emily Smith. No physicist has reviewed this work. A toy model shows what is
  possible within a class of models, nothing about the real universe. The advertisements are real.</p>
</footer>''' % dict(base=post.base, repo=REPO, paper=post.paper_url, side=html.escape(utm("footer-text-" + kind)),
                    sponsor=html.escape(utm("footer-sponsor-" + kind)), logo=LOGO, badges=badges(),
                    nav=post_nav(kind, post, posts))


# ------------------------------------------------------------------ the plain-language page
def page_plain(post, plain, technical=None, posts=None):
    """`technical` is the converted technical edition, for a post that has one as a page of its own; `posts` are
    the posts that are up, oldest first (this post alone if they are not given)."""
    posts = posts or [post]
    url = SITE + post.base
    terms = _terms(plain["back"])
    ld = json_ld(post, "plain", url, post.headline, post.description,
                 extra={"alternativeHeadline": "Plain-language edition",
                        "isBasedOn": SITE + post.technical_href if post.technical_href else "", "hasPart": [
                            {"@type": "WebPageElement", "name": c["title_text"], "url": url + "#ch%d" % c["num"]}
                            for c in _with_text(plain["chapters"])]},
                 faq=post.big_questions + post.faq, terms=terms)
    nav = ['<nav class="side" aria-label="Chapters"><p class="side-head">THIS POST</p><ol>']
    for c in plain["chapters"]:
        nav.append('<li><a href="#ch%d">%s</a></li>' % (c["num"], _title_case(c["kicker"])))
    nav.append('</ol><p class="side-head">ELSEWHERE</p><ul>%s</ul>' % _listed([
        (post.has_technical, '<a href="%s" target="mn-technical">Technical edition</a>' % post.tech),
        (post.pdf_tech and not post.has_technical, '<a href="%s">Technical edition (PDF)</a>' % post.pdf_tech),
        (post.has_companion, '<a href="%s">Turn the knobs (interactive)</a>' % post.knobs),
        (post.pdf_plain, '<a href="%s">This edition as a PDF</a>' % post.pdf_plain),
        (post.big_questions, '<a href="#big-questions">The big questions</a>'),
        (_has_id(plain, "translations"), '<a href="#translations">Glossary</a>'),
        (_has_id(plain, "sources-head"), '<a href="#sources-head">Sources</a>'),
        (True, '<a href="#guestbook">Guestbook</a>')]))
    nav.append(AD_SIDE)
    nav.append('</nav>')

    body = [head(post, post.page_title or post.title + " (plain-language edition)", post.description, url, ld,
                 plain["mathjax"], post.pdf_plain),
            '<body class="plain">', '<a class="skip" href="#ch%d">Skip to the paper</a>' % plain["chapters"][0]["num"],
            masthead(post, "plain-language edition", _technical_button(post),
                     answers="#quick-answers" if post.faq else ""),
            AD_TOP, '<div class="frame">', "\n".join(nav), '<main class="paper">',
            '<section class="opening">']
    body += [part for part in (post.summary, plain["opening"]) if part]
    body.append('</section>')
    mid = _mid_ad_chapter(post, plain["chapters"])
    for c in plain["chapters"]:
        body.append('<section class="chapter" id="ch%d">' % c["num"])
        body.append('<p class="kicker">CHAPTER %d · %s</p><h2>%s</h2>' % (c["num"], c["kicker"], c["title"]))
        if c["num"] in post.chapter_questions:
            question, bridge = post.chapter_questions[c["num"]]
            body.append('<div class="ask"><h3 id="ch%d-question">%s</h3><p>%s</p></div>'
                        % (c["num"], html.escape(question), html.escape(bridge)))
        if post.has_technical:
            sec = post.chapter_to_section[c["num"]]
            body.append('<p class="xlink">Side by side: <a href="%s#%s" target="mn-technical">the same ground in '
                        'the technical edition, %s »</a></p>' % (post.tech, sec, _section_label(technical, sec)))
        body.append(c["html"])
        body.append('</section>')
        if c["num"] == mid:
            body.append(AD_MID)
    body.append(ad_reader(post))
    body.append('<section class="back">%s</section>' % plain["back"])
    if plain["notes"]:
        body.append('<section class="notes" id="notes"><h2>Notes</h2><ol>%s</ol></section>' % plain["notes"])
    if post.big_questions:
        body.append('<section class="faq" id="big-questions"><h2>The big questions, and what this work can '
                    'honestly say</h2>')
        for q, a in post.big_questions:
            body.append('<h3>%s</h3><p>%s</p>' % (html.escape(q), html.escape(a)))
        body.append('</section>')
    if post.faq:
        body.append('<section class="faq" id="quick-answers"><h2>Quick answers</h2>')
        for q, a in post.faq:
            body.append('<h3>%s</h3><p>%s</p>' % (html.escape(q), html.escape(a)))
        body.append('</section>')
    if post.key_terms:
        body.append('<section class="terms" id="key-terms"><h2>What this page is about</h2><p>%s.</p></section>'
                    % ", ".join(post.key_terms))
    body.append(guestbook(post))
    body.append(cite_block(post, url, post.headline, "plain-language edition"))
    body.append('</main></div>')
    body.append(AD_FOOT)
    body.append(footer("plain", post, posts))
    body.append('</body></html>')
    return own_companions("\n".join(body) + "\n", posts)


def _listed(links):
    """The (wanted, link) pairs as list items; a link is left out where there is nothing for it to open."""
    return "".join("<li>%s</li>" % link for wanted, link in links if wanted)


def _has_id(plain, ident):
    return any('id="%s"' % ident in part
               for part in [plain["opening"], plain["back"]] + [c["html"] for c in plain["chapters"]])


def _technical_button(post):
    """The masthead's button from the plain edition to the technical one: its page, or its PDF where the post
    has no technical page, or no button while there is neither."""
    if post.has_technical:
        return "Read the technical edition »", post.tech, "mn-technical"
    if post.pdf_tech:
        return "Read the technical edition (PDF) »", post.pdf_tech, "mn-technical"
    return None


def _mid_ad_chapter(post, chapters):
    """The chapter the mid-article advertisement follows: the post's choice, or one a third of the way in."""
    return post.mid_ad_after or chapters[(len(chapters) - 1) // 3]["num"]


def _with_text(chapters):
    return [dict(c, title_text=re.sub(r"<[^>]+>", "", c["title"])) for c in chapters]


def _title_case(kicker):
    text = re.sub(r"<[^>]+>", "", kicker).lower()
    return text[:1].upper() + text[1:]


def _section_label(technical, sec):
    s = next(x for x in technical["sections"] if x["id"] == sec)
    name = re.sub(r"<[^>]+>", "", s["title"])
    return ("Sec. %s, %s" % (s["roman"], name)) if s["roman"] else name


def _terms(back_html):
    """(term, definition) pairs from the translations table of the plain edition, as plain text."""
    m = re.search(r'<table id="translations".*?</table>', back_html, flags=re.S)
    out = []
    if m:
        for row in re.findall(r"<tr><td>(.*?)</td><td>(.*?)</td></tr>", m.group(0), flags=re.S):
            clean = [html.unescape(re.sub(r"<[^>]+>", "", x)).replace("\\(", "").replace("\\)", "").strip()
                     for x in row]
            out.append((clean[0], clean[1]))
    return out


# ------------------------------------------------------------------ the technical page
def page_technical(post, technical, plain, posts=None):
    posts = posts or [post]
    url = SITE + post.tech
    meta = technical["meta"]
    abstract_text = html.unescape(re.sub(r"<[^>]+>", "", technical["abstract"]))
    ld = json_ld(post, "technical", url, re.sub(r"<[^>]+>", "", meta["title"]), post.technical_description,
                 extra={"abstract": abstract_text, "alternativeHeadline": "Technical edition",
                        "hasPart": [{"@type": "WebPageElement", "name": re.sub(r"<[^>]+>", "", s["title"]),
                                     "url": url + "#" + s["id"]} for s in technical["sections"]]})
    kickers = {c["num"]: _title_case(c["kicker"]) for c in plain["chapters"]}
    nav = ['<nav class="side" aria-label="Sections"><p class="side-head">THIS POST</p><ol class="roman">']
    for s in technical["sections"]:
        nav.append('<li><a href="#%s">%s%s</a></li>' % (s["id"], (s["roman"] + ". ") if s["roman"] else "",
                                                       re.sub(r"<[^>]+>", "", s["title"])))
    nav.append('</ol><p class="side-head">ELSEWHERE</p><ul>%s</ul></nav>' % _listed([
        (True, '<a href="%s" target="mn-plain">Plain-language edition</a>' % post.base),
        (post.has_companion, '<a href="%s">Turn the knobs (interactive)</a>' % post.knobs),
        (post.pdf_tech, '<a href="%s">This paper as a PDF</a>' % post.pdf_tech),
        (True, '<a href="%s">Supplement, data and code</a>' % post.paper_url),
        (True, '<a href="#references-head">References</a>'),
        (True, '<a href="%s#guestbook" target="mn-plain">Guestbook</a>' % post.base)]))

    body = [head(post, re.sub(r"<[^>]+>", "", meta["title"]) + " (technical edition)", post.technical_description,
                 url, ld, technical["mathjax"], post.pdf_tech), '<body class="technical">',
            '<a class="skip" href="#%s">Skip to the paper</a>' % technical["sections"][0]["id"],
            masthead(post, "technical edition", ("« Read the plain-language edition", post.base, "mn-plain"),
                     answers=post.base + "#quick-answers" if post.faq else "", guestbook=post.base + "#guestbook"),
            ad_reader(post),
            '<div class="frame">', "\n".join(nav), '<main class="paper">',
            '<section class="abstract" id="abstract"><h2 class="full-title">%s</h2>'
            '<p class="affil">%s · %s · %s</p><p><b>Abstract.</b> %s</p></section>'
            % (meta["title"], meta["author"], meta["affiliation"], meta["date"], technical["abstract"])]
    for s in technical["sections"]:
        body.append('<section class="chapter" id="%s">' % s["id"])
        body.append('<h2>%s%s</h2>' % ((s["roman"] + ". ") if s["roman"] else "", s["title"]))
        chapters = post.section_to_chapters.get(s["id"], [])
        if chapters:
            links = " and ".join('<a href="%s#ch%d" target="mn-plain">Chapter %d, %s</a>'
                                 % (post.base, n, n, html.escape(kickers[n])) for n in chapters)
            body.append('<p class="xlink">Side by side: in the plain-language edition this is %s »</p>' % links)
        body.append(s["html"])
        body.append('</section>')
    body.append(cite_block(post, url, re.sub(r"<[^>]+>", "", meta["title"]), "technical edition"))
    body.append('</main></div>')
    body.append(AD_TECH)
    body.append(footer("technical", post, posts))
    body.append('</body></html>')
    return own_companions("\n".join(body) + "\n", posts)


# ------------------------------------------------------------------ the blog's front page
def page_index(posts):
    """The list of posts, newest first. `posts` are the posts that are up, oldest first."""
    newest, url = posts[-1], SITE + BLOG
    description = ("The Side Nerd Blog: physics papers by Emily Smith, each in a plain-language edition and a "
                   "technical edition, hosted by Side Nerd Apps.")
    postings = []
    for p in reversed(posts):
        postings.append(_posting(p, p.headline, p.base))
        if p.has_technical:
            postings.append(_posting(p, p.title + " (technical edition)", p.tech))
    ld = ('<script type="application/ld+json">%s</script>' % json.dumps({
        "@context": "https://schema.org", "@type": "Blog", "@id": url + "#blog", "name": "The Side Nerd Blog",
        "url": url, "description": description, "inLanguage": "en",
        "publisher": {"@type": "Organization", "name": "Side Nerd Apps", "url": SITE + "/"},
        "author": {"@type": "Person", "name": "Emily Smith"},
        "blogPost": postings,
    }, ensure_ascii=False, indent=1))
    body = [head(newest, "The Side Nerd Blog", description, url, ld, False, paper=False), '<body class="index">',
            '''
<div class="ticker" aria-hidden="true"><div class="ticker-run">%(ticker)s %(ticker)s</div></div>
<header class="masthead">
 <div class="mast-text">
  <p class="welcome">~ Welcome, Main Nerd ~</p>
  <h1>The Side Nerd Blog</h1>
  <p class="subtitle">Recreational physics, in plain language and in full</p>
  <p class="sponsor">Sponsored by <a href="%(sponsor)s">Side Nerd Apps</a></p>
 </div>
</header>''' % dict(ticker=html.escape(ticker(newest)), sponsor=html.escape(utm("index-sponsor"))),
            AD_TOP, '<div class="frame index-frame"><main class="paper">',
            '<h2>Posts</h2>']
    body += [_post_card(p) for p in reversed(posts)]
    body += [ad_reader(newest), '</main></div>', footer("index", newest, posts), '</body></html>']
    return "\n".join(body) + "\n"


def _posting(post, headline, address):
    return {"@type": "BlogPosting", "headline": headline, "url": SITE + address, "datePublished": post.published,
            "dateModified": post.updated or post.published,
            "author": {"@type": "Person", "name": "Emily Smith"}}


def _post_card(post):
    """One post in the front page's list, with a button for each way there is to read it."""
    buttons = [(post.base, "Read it in plain language »"),
               (post.tech if post.has_technical else "", "Read the technical edition »"),
               (post.pdf_plain, "Plain PDF"), (post.pdf_tech, "Technical PDF")]
    by = "<i>%s.</i> By Emily Smith." % post.subtitle if post.subtitle else "By Emily Smith."
    return ('<article class="post-card"><p class="kicker">POST No. %02d · %s · PHYSICS</p>'
            '<h3><a href="%s">%s</a></h3><p>%s</p>%s'
            '<p class="jump">%s</p></article>'
            % (post.number, long_date(post.published).upper(), post.base, post.title, by,
               post.summary.replace("<b>In one paragraph.</b> ", ""),
               " ".join('<a class="bevel" href="%s">%s</a>' % button for button in buttons if button[0])))


# ------------------------------------------------------------------ the placeholder for a post that is not up yet
def page_coming_soon(posts):
    """Where the newest post's "next post" lands while the next post does not exist. Kept out of search results.
    `posts` are the posts that are up, oldest first; the page leads back to the newest of them."""
    newest, url = posts[-1], SITE + SOON
    description = "The next post on the Side Nerd Blog is not up yet."
    top = head(newest, "The next post is not up yet", description, url, "", False, paper=False)
    top = top.replace("index, follow, max-image-preview:large", "noindex, follow")
    buttons = ['<a class="bevel" href="%s">Post No. %02d in plain language »</a>' % (newest.base, newest.number)]
    if newest.has_technical:
        buttons.append('<a class="bevel" href="%s">Post No. %02d, technical edition »</a>'
                       % (newest.tech, newest.number))
    buttons.append('<a class="bevel" href="%s#guestbook">Sign the guestbook to hear when it is up</a>' % newest.base)
    body = [top, '<body class="index">',
            '''
<div class="ticker" aria-hidden="true"><div class="ticker-run">%(ticker)s %(ticker)s</div></div>
<header class="masthead">
 <div class="mast-text">
  <p class="welcome">~ <a href="%(blog)s">The Side Nerd Blog</a>, Main Nerd ~</p>
  <h1>Not up yet</h1>
  <p class="subtitle">The next post is still curled up</p>
  <p class="sponsor">Sponsored by <a href="%(sponsor)s">Side Nerd Apps</a></p>
 </div>%(mascot)s
</header>''' % dict(ticker=html.escape(ticker(newest)), blog=BLOG, sponsor=html.escape(utm("comingsoon-sponsor")),
                    mascot=next((mascot(p, "Stable for now.") for p in posts if p.mascot), "")),
            '<div class="frame index-frame"><main class="paper">',
            '<h2>This post is not up yet</h2>'
            '<p>You have reached the end of what has been posted. The next post is being written; '
            'there is nothing to read here yet.</p>'
            '<p class="jump">%s</p>' % " ".join(buttons),
            '</main></div>', footer("soon", newest, posts), '</body></html>']
    return "\n".join(body) + "\n"


# ------------------------------------------------------------------ the interactive companion
# The blog's bar above and below the companion. The companion keeps its own style sheet, which styles bare elements
# (body, header, h1, p, table), so the bar is made only of div, span, a and img under one prefix, and the two
# leave each other alone.
KNOBS_BAR_CSS = """
html{-webkit-text-size-adjust:100%}body{margin:0}img{max-width:100%}[hidden]{display:none!important}
.mnb{margin:0 -16px;padding:10px 16px;background:#192933;color:#e2e8f0;border:4px ridge #34b1e8;text-align:center;
  font-family:"DM Sans",Arial,Helvetica,sans-serif;font-size:14px;line-height:1.6}
.mnb a{color:#5bc4f0}
.mnb-name{display:block;font-family:"Comic Sans MS","Comic Sans",cursive;font-size:17px;color:#5bc4f0}
.mnb-what{display:block;margin:2px 0 6px;color:#cbd5e1}
.mnb .mnb-btn{display:inline-block;margin:3px;padding:3px 12px;background:#c9d3da;color:#0d1419;font-weight:700;
  font-size:13.5px;text-decoration:none;border:3px outset #f1f5f9}
.mnb .mnb-btn:active{border-style:inset}
.mnb-sponsor{display:block;margin-top:6px;font-family:"Space Mono","Courier New",monospace;font-size:12.5px;
  letter-spacing:.06em;color:#cbd5e1}
.mnb-small{display:block;margin-top:8px;font-size:12.5px;color:#cbd5e1}
.mnb-foot{margin-top:28px}
@media print{.mnb{display:none}}
"""


def page_knobs(post, source):
    """The post's companion as a page of the blog. `source` is the text of its source file (for post 01,
    docs/public/curling_ladder_tube.html): a title, font links, a style sheet, the markup and one script, written
    as an artifact's body. Its parts are moved to where a whole document wants them and nothing in them is
    changed."""
    title = re.search(r"<title>(.*?)</title>", source, flags=re.S)
    links = re.findall(r"<link\b[^>]*>", source)
    style = re.search(r"<style>.*?</style>", source, flags=re.S)
    assert title and style and source.count("<style>") == 1 and source.count("<script>") == 1
    body = source[style.end():].strip()
    assert "<link" not in body and "<title" not in body and body.rstrip().endswith("</script>")
    name = html.unescape(title.group(1))
    url = SITE + post.knobs
    top = head(post, name + ": turn the knobs", post.companion_description, url, "", False, paper=False)
    top = top.replace('<link rel="stylesheet" href="%s%s">\n' % (SHARED, STYLESHEET), "")   # not the blog's style sheet
    top = top.replace("</head>", "\n".join(links) + "\n<style>" + KNOBS_BAR_CSS + "</style>\n" + style.group(0)
                      + "\n</head>")
    technical = '<a class="mnb-btn" href="%s">Read the technical edition</a>' % post.tech if post.has_technical else ""
    bar = ('<div class="mnb">'
           '<span class="mnb-name">~ <a href="%(blog)s">The Side Nerd Blog</a>, Main Nerd ~</span>'
           '<span class="mnb-what">The interactive companion to Post No. %(number)02d, <i>%(paper)s</i></span>'
           '<a class="mnb-btn" href="%(base)s">« Read it in plain language</a>%(technical)s'
           '<span class="mnb-sponsor">Sponsored by <a href="%(sponsor)s">Side Nerd Apps</a></span></div>'
           % dict(blog=BLOG, number=post.number, paper=post.title, base=post.base, technical=technical,
                  sponsor=html.escape(utm("knobs-sponsor"))))
    foot = ('<div class="mnb mnb-foot">'
            '<a class="mnb-btn" href="%(base)s">« Back to the post</a>'
            '<a class="mnb-btn" href="%(blog)s">All posts</a>'
            '<a class="mnb-btn" href="%(base)s#guestbook">Sign the guestbook</a>'
            '<span class="mnb-sponsor">Sponsored by <a href="%(sponsor)s">Side Nerd Apps</a></span>'
            '<span class="mnb-small">Text and figures © 2026 Emily Smith. No physicist has reviewed this work. A toy '
            'model shows what is possible within a class of models, nothing about the real universe. '
            '<a href="%(repo)s/issues">Report a mistake</a>.</span></div>'
            % dict(blog=BLOG, base=post.base, repo=REPO, sponsor=html.escape(utm("knobs-footer-sponsor"))))
    cut = body.index("<script>")
    return "\n".join([top, "<body>", bar, body[:cut].rstrip(), foot, body[cut:], "</body></html>"]) + "\n"


# ------------------------------------------------------------------ the stylesheet
CSS = r'''
/* The Side Nerd Blog (sidenerdapps.com/mainnerd): a 1996 home page in Side Nerd's colors. One stylesheet. */
:root{
  --midnight:#0d1419; --navy:#192933; --slate:#2d3748; --aqua:#34b1e8; --aqua-light:#5bc4f0; --aqua-ink:#0b6795;
  --paper:#ffffff; --canvas:#f6f7f8; --ink:#16181d; --muted:#6b7280; --rule:#d8dadf; --text-on-dark:#e2e8f0;
  --exact:#176b70; --meas:#2f5f9e; --pub:#5a5a5a; --ours:#9a5b00; --claim:#8a2d5a;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;color:var(--text-on-dark);font-family:"Times New Roman",Times,serif;font-size:18px;line-height:1.55;
  background-color:var(--midnight);
  background-image:radial-gradient(1px 1px at 12px 18px,#5bc4f0 50%,transparent 51%),
    radial-gradient(1px 1px at 70px 96px,#e2e8f0 50%,transparent 51%),
    radial-gradient(1.5px 1.5px at 140px 40px,#94a3b8 50%,transparent 51%),
    radial-gradient(1px 1px at 190px 150px,#34b1e8 50%,transparent 51%),
    radial-gradient(1px 1px at 40px 170px,#e2e8f0 50%,transparent 51%);
  background-size:220px 200px}
a{color:var(--aqua-ink)}
.masthead a,.foot a,.side a{color:var(--aqua-light)}
a:visited{color:#5a4a9a}
.side a:visited,.foot a:visited{color:#b9a6ee}
.skip{position:absolute;left:-999px}.skip:focus{left:8px;top:8px;background:#fff;color:#000;padding:6px;z-index:9}

/* ticker: the marquee, done politely */
.ticker{overflow:hidden;white-space:nowrap;background:var(--navy);border-bottom:2px ridge var(--aqua);
  font-family:"Space Mono","Courier New",monospace;font-size:13px;color:var(--aqua-light);padding:4px 0}
.ticker-run{display:inline-block;padding-left:100%;animation:run 60s linear infinite}
.ticker:hover .ticker-run{animation-play-state:paused}
@keyframes run{to{transform:translateX(-100%)}}
@media (prefers-reduced-motion:reduce){.ticker-run{animation:none;padding-left:8px}}

.masthead{max-width:1180px;margin:18px auto 10px;padding:14px 18px;display:flex;gap:22px;align-items:center;
  border:4px ridge var(--aqua);background:var(--navy)}
.mast-text{flex:1;min-width:0;text-align:center}
.welcome{margin:0;font-family:"Comic Sans MS","Comic Sans",cursive;color:var(--aqua-light);font-size:17px}
.masthead h1{margin:6px 0 4px;font-size:clamp(34px,6vw,62px);line-height:1.05;color:#fff;letter-spacing:.5px;
  text-shadow:3px 3px 0 var(--aqua-ink),-1px -1px 0 var(--aqua)}
.subtitle{margin:0 0 8px;font-style:italic;font-size:21px;color:var(--text-on-dark)}
.byline{margin:0 0 10px;font-size:16px;color:#cbd5e1}
.jump{margin:0}
.bevel{display:inline-block;margin:3px;padding:3px 12px;background:#c9d3da;color:#0d1419 !important;
  border:2px outset #f1f5f9;font-family:Arial,Helvetica,sans-serif;font-size:14px;font-weight:bold;text-decoration:none;cursor:pointer}
.bevel:active{border-style:inset}
.mascot{margin:0;flex:0 0 170px;text-align:center}
.mascot img{width:170px;height:auto;background:#fff;border:3px inset #c9d3da}
.mascot figcaption{font-size:12px;line-height:1.3;color:#cbd5e1;margin-top:4px}

.frame{max-width:1180px;margin:0 auto;display:grid;grid-template-columns:230px minmax(0,1fr);gap:16px;padding:0 10px}
.index-frame{grid-template-columns:minmax(0,1fr);max-width:900px}
.post-card{border:3px ridge var(--aqua);background:var(--canvas);padding:14px 18px;margin:1em 0}
.post-card h3{margin:.2em 0;font-size:28px}
.side{align-self:start;position:sticky;top:8px;max-height:calc(100vh - 16px);overflow:auto;
  background:var(--navy);border:3px ridge var(--aqua);padding:10px 12px;font-family:Arial,Helvetica,sans-serif;font-size:14px}
.side-head{margin:8px 0 4px;font-family:"Space Mono","Courier New",monospace;font-size:12px;letter-spacing:.08em;color:var(--aqua)}
.side ol,.side ul{margin:0 0 8px;padding-left:22px}
.side ol.roman{list-style:none;padding-left:4px}
.side li{margin:3px 0;line-height:1.3}

.paper{background:var(--paper);color:var(--ink);border:4px ridge #c9d3da;padding:26px clamp(16px,4vw,46px) 34px;min-width:0}
.paper h2{font-family:Arial,Helvetica,sans-serif;font-size:30px;line-height:1.15;margin:.3em 0 .5em;color:#0d1419}
.paper h3{font-family:Arial,Helvetica,sans-serif;font-size:21px;margin:1.5em 0 .3em;color:#0d1419}
.paper p{margin:.7em 0}
.chapter{border-top:6px double var(--aqua);margin-top:2.6em;padding-top:1.1em}
.kicker{margin:0;font-family:"Space Mono","Courier New",monospace;font-size:13px;letter-spacing:.08em;color:var(--aqua-ink);font-weight:bold}
.ask{margin:.2em 0 .9em;padding:8px 14px;background:var(--canvas);border-left:6px solid var(--aqua-ink)}
.ask h3{margin:0;font-size:18px;font-style:italic;color:var(--aqua-ink)}
.ask h3::before{content:"People ask: ";font-style:normal;font-weight:normal;color:var(--muted);font-size:14px}
.ask p{margin:.25em 0 0;font-size:16.5px}
.xlink{font-family:Arial,Helvetica,sans-serif;font-size:14px;background:var(--canvas);border:1px dashed var(--aqua-ink);padding:5px 10px}
.takeaway{background:#e8f6fc;border-left:6px solid var(--aqua);padding:10px 14px;margin:1.3em 0}
.why{font-size:16px;color:#374151;border-left:3px dotted var(--aqua-ink);padding-left:12px}
.trail{font-family:Arial,Helvetica,sans-serif;font-size:13px;letter-spacing:.03em;color:#6b7280;margin:.2em 0 1em}
.trail strong{color:var(--aqua-ink)}
.pull{text-align:center;font-family:Arial,Helvetica,sans-serif;font-weight:bold;font-size:25px;line-height:1.25;margin:1.2em 0;color:#0d1419}
.m{white-space:nowrap}
.m i,.paper i.v{font-family:"Times New Roman",Times,serif}
.tag{display:inline-block;font-family:Arial,Helvetica,sans-serif;font-size:10.5px;font-weight:bold;letter-spacing:.04em;
  color:#fff;padding:1px 5px;vertical-align:1px;white-space:nowrap}
.tag-exact{background:var(--exact)}.tag-meas{background:var(--meas)}.tag-pub{background:var(--pub)}
.tag-ours{background:var(--ours)}.tag-claim{background:var(--claim)}
.fig{margin:1.5em 0;text-align:center}
.fig img{max-width:100%;height:auto;background:#fff}
.fig figcaption{font-size:15.5px;line-height:1.4;text-align:left;color:#374151;margin-top:6px}
.tablewrap{overflow-x:auto;margin:1.2em 0}
table{border-collapse:collapse;margin:0 auto;min-width:min(100%,560px);font-family:Arial,Helvetica,sans-serif;font-size:14.5px;line-height:1.35}
caption{caption-side:top;text-align:left;font-family:"Times New Roman",Times,serif;font-size:15.5px;color:#374151;padding-bottom:6px}
th,td{border:1px solid var(--rule);padding:5px 9px;text-align:left;vertical-align:top}
th{background:#dbeef8}
tr:nth-child(even) td{background:var(--canvas)}
table.glossary td:first-child{font-weight:bold;white-space:nowrap}
.eq{display:block;overflow-x:auto;margin:.4em 0}
.runin{font-style:italic;font-weight:bold}
.cite{text-decoration:none}
sup.fn a{text-decoration:none}
.sources,.notes ol{font-size:16px}
.sources li{margin:.45em 0}
.abstract .full-title{font-size:27px}
.affil{font-family:Arial,Helvetica,sans-serif;font-size:14px;color:var(--muted)}
.faq h3{margin-top:1.1em}
.cite-box pre{background:var(--canvas);border:2px inset #c9d3da;padding:10px;overflow-x:auto;font-size:13px}
.terms p{font-size:16px;color:#374151}

/* the guestbook */
.guestbook{margin-top:2.4em;padding:14px 18px 18px;border:4px ridge var(--aqua);background:var(--canvas)}
.guestbook h2{margin-top:0}
.gb{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px 14px;align-items:end}
.gb label{display:block;font-family:Arial,Helvetica,sans-serif;font-size:13px;font-weight:bold}
.gb input:not([type=checkbox]),.gb textarea{display:block;width:100%;margin-top:3px;padding:5px 6px;border:2px inset #c9d3da;
  background:#fff;font-family:"Courier New",monospace;font-size:15px;color:var(--ink)}
.gb .wide,.gb .check{grid-column:1/-1}
.gb .check{font-weight:normal;font-size:14px}
.gb button{justify-self:start;font-size:15px;padding:5px 18px}
.small-ink{font-size:13.5px;color:var(--muted)}

/* the advertisements: few, labelled, and the only thing here set in the brand's own type */
.ad{font-family:"DM Sans",Arial,sans-serif;margin:16px auto;max-width:728px;text-align:center}
.ad-label{display:block;font-family:"Space Mono","Courier New",monospace;font-size:10.5px;letter-spacing:.12em;color:#94a3b8;margin-bottom:3px}
.paper .ad-label{color:var(--muted)}
.ad a{display:flex;align-items:center;gap:14px;text-decoration:none;color:#0d1419;background:var(--aqua);
  border:4px outset var(--aqua-light);padding:9px 14px;text-align:left}
.ad a:hover{border-style:inset}
.ad-logo{flex:0 0 auto;width:72px;height:72px;background:#fff;border:2px inset #c9d3da;padding:2px}
/* One phone cut out of the three-phone Ither picture (1000 x 500) and stood upright. Measured in that picture: each
   phone is 231 x 467 and leans 32 degrees to the left; the conversation's center is at (506, 249), the report's
   at (788, 249). The box is the phone's own outline; the picture is scaled, shifted and turned behind it. */
.ad-phone{flex:0 1 auto;display:block;position:relative;overflow:hidden;aspect-ratio:231/467;min-width:0;max-width:100%}
.ad-phone img{position:absolute;display:block;max-width:none;height:auto}
.ad-phone.talk img{width:432.3%;left:-168.7%;top:-3.2%;transform-origin:50.59% 49.72%;transform:rotate(31.62deg)}
.ad-phone.report img{width:432%;left:-290.4%;top:-3.1%;transform-origin:78.81% 49.72%;transform:rotate(32.2deg)}
.ad-tower .ad-phone{width:100%;max-width:176px}
.ad-pair{flex:0 1 auto;display:flex;align-items:center;justify-content:center;gap:6px;min-width:0;max-width:100%}
.ad-pair .ad-phone{width:118px}
.ad-arrow{flex:0 0 auto;font-weight:700;font-size:22px}
/* where the two phones and the words do not fit side by side, the words go underneath */
.ad-box a{flex-wrap:wrap;justify-content:center}
.ad-box .ad-copy{flex:1 1 200px}
.ad-copy{flex:1;min-width:0}
.ad-head{display:block;font-weight:700;font-size:20px;letter-spacing:.02em;line-height:1.2}
.ad-body{display:block;font-size:14.5px;line-height:1.35;margin:3px 0 6px}
.ad-btn{display:inline-block;background:#0d1419;color:#fff;font-weight:700;font-size:13px;padding:3px 12px;border:2px outset #94a3b8}
.ad-banner .ad-head{animation:blink 1.6s steps(2,start) 4}
@keyframes blink{50%{opacity:.25}}
@media (prefers-reduced-motion:reduce){.ad-banner .ad-head{animation:none}}
.ad-tower{margin:14px 0 4px}
.ad-tower a{flex-direction:column;text-align:center;padding:12px 8px;gap:8px}
.ad-tower .ad-head{font-size:17px}
.ad-box{max-width:560px;margin:2.2em auto}
.ad-wanted a{background:#fffbe6;border:4px dashed var(--aqua-ink);color:var(--ink);text-align:center}
.ad-wanted .ad-head{font-family:"Courier New",monospace;font-size:19px}
.ad-wanted .ad-btn{background:var(--aqua-ink)}
.paper .ad-wanted{margin:2.4em auto}

.foot{max-width:1180px;margin:20px auto 40px;padding:14px 18px;text-align:center;font-family:Arial,Helvetica,sans-serif;font-size:14px;
  border:4px ridge var(--aqua);background:var(--navy)}
.postnav{border:2px groove #94a3b8;padding:6px;margin-bottom:10px}
.sponsor{margin:8px 0 0;font-family:"Space Mono","Courier New",monospace;font-size:12.5px;letter-spacing:.06em;color:#cbd5e1}
.sponsor-foot{margin:4px 0 12px}
.sponsor-foot a{display:inline-flex;align-items:center;gap:10px;font-family:"DM Sans",Arial,sans-serif;font-weight:700;font-size:16px;
  color:#fff !important;text-decoration:none}
.sponsor-foot img{background:#fff;border:2px inset #c9d3da;padding:2px}
.small{font-size:12.5px;color:#cbd5e1}
.badges{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin:12px 0}
.badge{font-size:10.5px;line-height:1.25;padding:4px 8px;border:2px outset #94a3b8;background:#c9d3da;color:#0d1419;min-width:88px}

@media (max-width:900px){
  body{font-size:17px}
  .frame{grid-template-columns:minmax(0,1fr)}
  .side{position:static;max-height:none}
  .side ol{columns:2}
  .masthead{flex-direction:column}
  .ad-tower{display:none}
  .gb{grid-template-columns:minmax(0,1fr)}
}
@media (max-width:520px){.ad a{flex-direction:column;text-align:center}}
@media print{
  body{background:#fff;color:#000}.ticker,.ad,.side,.foot,.mascot,.jump,.xlink,.guestbook{display:none}
  .frame{display:block}.paper{border:0;padding:0}
}
'''
