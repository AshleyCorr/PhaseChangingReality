"""Build the web pages of the Side Nerd Blog at sidenerdapps.com/mainnerd/, one post per paper. Usage:

    python scripts/build_main_nerd_pages.py --tectonic PATH/TO/tectonic
    python scripts/build_main_nerd_pages.py --skip-figures            the pages again, the figures as they are
    python scripts/build_main_nerd_pages.py --skip-figures --only 1   post 01 alone, as if no other were written

Writes docs/public/mainnerd/:

    index.html                    the blog's front page: the list of posts, newest first
    comingsoon/index.html         where "next post" lands while the next post is not up yet
    assets/main-nerd.css          the stylesheet, shared by every post
    <slug>PL/index.html           a post's plain-language edition, chapter by chapter
    <slug>/index.html             its technical manuscript, section by section, where the post has that page
    <slug>Knobs/index.html        its interactive companion, where it has one
    assets/<slug>/                its figures (SVG), its PDFs, its mascot GIF and its share image

The posts are listed in scripts/main_nerd_posts.py. Post 01, "The curled torus burps", has the slug
01thecurledtorusburps, both editions as pages, and a companion made from docs/public/curling_ladder_tube.html.
A post is built once its plain-language source is in its paper's folder; until then it is skipped with a note, and
the post before it keeps the placeholder as its "next post". A PDF that has not been compiled yet is left off the
pages, with a note.

The pages are made from the LaTeX sources of the papers (docs/papers/<paper>/), so the text and the numbers on the
pages are the papers' own; nothing is retyped. Where a post has both editions as pages, each chapter of the plain
page links to the matching section of the technical page and back, so that the two can be read side by side. The
pages are plain static HTML, meant to be copied by the site's owner into sites/sidenerdapps/mainnerd/ of the Side
Nerd marketing repository. Their links are root-absolute (/mainnerd/...), so preview them with a web server rooted
at docs/public:   python -m http.server --directory docs/public   then open http://localhost:8000/mainnerd/

Needs Tectonic (to render the figures) and PyMuPDF (to turn them into SVG). The figures of a plain edition are
drawn in its LaTeX source; they are compiled once more with the `preview` package, which puts each figure on its
own tightly cropped page.
"""
import argparse
import html
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from main_nerd_latex import (Converter, environments, match_brace, read_args, roman, slug,   # noqa: E402
                               strip_comments)
from main_nerd_posts import POST_01, POSTS   # noqa: E402
from main_nerd_site import (CSS, STYLESHEET, page_coming_soon, page_index, page_knobs, page_plain,   # noqa: E402
                            page_technical)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/public/mainnerd"            # the blog; one folder per post, named by its slug
SHARED_DIR = OUT / "assets"                    # the stylesheet, and one folder per post
FIGURE_SIZES = OUT.parent / "mainnerd_figures.json"      # kept outside the folder that gets published

# post 01 under the names it had when it was the only post
COMPANION = ROOT / POST_01.companion_source                 # the interactive companion's source
CHAPTER_TO_SECTION, SECTION_TO_CHAPTERS = POST_01.chapter_to_section, POST_01.section_to_chapters


# ------------------------------------------------------------------ which posts, and what each has so far
def source_of(post):
    """The plain-language edition's LaTeX. A post is built once this file is there."""
    return ROOT / post.paper_dir / post.plain_source


def assets_dir(post):
    return SHARED_DIR / post.slug                   # the post's figures, PDFs and pictures


def shown(path):
    """A path as the notes print it: from the repository root, for a file inside the repository."""
    return path.relative_to(ROOT).as_posix() if ROOT in path.parents else str(path)


def pdfs_of(post):
    """(the field of the post that names it beside the pages, the compiled PDF in the paper's folder), per edition."""
    paper = ROOT / post.paper_dir
    return (("pdf_plain_name", (paper / post.plain_source).with_suffix(".pdf")),
            ("pdf_tech_name", (paper / post.technical_source).with_suffix(".pdf")))


def as_it_stands(post):
    """The post with what is not there yet taken out of it, so that no page links to a file that is missing: a PDF
    that has not been compiled, a share image that has not been made. Each gets a NOTE."""
    missing, pictures = {}, assets_dir(post)
    for field, pdf in pdfs_of(post):
        if getattr(post, field) and not pdf.exists():
            print("NOTE: post %02d has no %s yet; its pages leave that PDF out" % (post.number, shown(pdf)))
            missing[field] = ""
    if post.share_image and not (pictures / post.share_image).exists():
        first = POSTS[0].number                             # whose share image stands in (see share_image)
        print("NOTE: %s is missing from %s; the pages refer to %s"
              % (post.share_image, pictures, "it" if post.number == first else "post %02d's" % first))
        missing["share_image"] = ""
    if post.mascot and not (pictures / post.mascot["file"]).exists():
        print("NOTE: %s is missing from %s; the pages refer to it" % (post.mascot["file"], pictures))
    return replace(post, **missing)


def posts_to_build(posts=POSTS):
    """The posts whose plain-language source is there, oldest first, each as it stands today."""
    ready = []
    for post in posts:
        if source_of(post).exists():
            ready.append(as_it_stands(post))
        else:
            print("NOTE: post %02d is not built: %s is not there yet" % (post.number, shown(source_of(post))))
    if not ready:
        raise SystemExit("no post has its plain-language source, so there is nothing to build")
    return ready


def stop(post, message):
    """End the build, naming the post and its source, where the text lacks something a page is built around."""
    raise SystemExit("post %02d (%s): %s" % (post.number, shown(source_of(post)), message))


# ------------------------------------------------------------------ figures
def render_figures(post, tectonic):
    """SVG files for every figure of the post's editions, in its asset folder; returns {name: (width, height)}
    in CSS pixels."""
    try:
        import pymupdf
    except ImportError:                                    # pragma: no cover
        raise SystemExit("PyMuPDF is needed to write the figures: pip install pymupdf")
    sizes, paper, assets = {}, ROOT / post.paper_dir, assets_dir(post)
    assets.mkdir(parents=True, exist_ok=True)

    def save(page, name):
        (assets / name).write_text(page.get_svg_image(text_as_path=True), encoding="utf-8", newline="\n")
        sizes[name] = (round(page.rect.width * 4 / 3), round(page.rect.height * 4 / 3))

    with tempfile.TemporaryDirectory() as tmp:
        figures = compile_figures(post, tectonic, Path(tmp))
        if figures:
            with pymupdf.open(figures) as doc:
                for i, page in enumerate(doc, 1):
                    save(page, "plain-fig-%02d.svg" % i)
    for stem in post.technical_figures:
        with pymupdf.open(paper / (stem + ".pdf")) as doc:
            save(doc[0], "tech-%s.svg" % stem.replace("_", "-"))
    return sizes


def compile_figures(post, tectonic, tmp):
    """The plain edition compiled in `tmp` with each figure alone on a page; the PDF, or None if it has no figure."""
    tex = source_of(post).read_text(encoding="utf-8")
    if not environments(strip_comments(tex), "figure"):
        return None
    for f in (ROOT / post.paper_dir).glob("fig*.pdf"):
        shutil.copy(f, tmp / f.name)
    hook = (r"\usepackage[active,tightpage]{preview}" "\n" r"\setlength\PreviewBorder{3pt}" "\n"
            r"\renewenvironment{figure}[1][]{\begin{preview}\begin{minipage}{\textwidth}\centering" "\n"
            r"  \renewcommand{\caption}[2][]{}\renewcommand{\label}[1]{}}{\end{minipage}\end{preview}}" "\n")
    assert tex.count(r"\begin{document}") == 1
    (tmp / "figs.tex").write_text(tex.replace(r"\begin{document}", hook + r"\begin{document}"),
                                  encoding="utf-8", newline="\n")
    run = subprocess.run([tectonic, "figs.tex"], cwd=tmp, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if run.returncode:
        raise SystemExit(run.stdout.decode("utf-8", errors="replace")[-2000:])
    return tmp / "figs.pdf"


def size_key(post, name):
    """A figure's name in FIGURE_SIZES. Post 01's are bare, as they were when it was the only post; a later post's
    carry its slug."""
    return name if post.number == 1 else "%s/%s" % (post.slug, name)


def recorded_sizes(post, recorded=None):
    """The post's figure sizes as an earlier run wrote them down, for --skip-figures."""
    if recorded is None:
        recorded = json.loads(FIGURE_SIZES.read_text()) if FIGURE_SIZES.exists() else {}
    lead = size_key(post, "")
    return {key[len(lead):]: tuple(size) for key, size in recorded.items()
            if key.startswith(lead) and "/" not in key[len(lead):]}


def write_sizes(sizes):
    """Record the sizes just rendered (`sizes`, by slug). A post that was not rendered this time keeps the sizes
    it had, since its SVG files are still the ones they describe."""
    recorded = json.loads(FIGURE_SIZES.read_text()) if FIGURE_SIZES.exists() else {}
    merged = {}
    for post in POSTS:
        mine = sizes[post.slug] if post.slug in sizes else recorded_sizes(post, recorded)
        merged.update((size_key(post, name), size) for name, size in mine.items())
    FIGURE_SIZES.write_text(json.dumps(merged, indent=1) + "\n", encoding="utf-8", newline="\n")


# ------------------------------------------------------------------ shared pieces
def read_latex(path):
    """A LaTeX source without its comments. A backslash that ends a line is a control space, as backslash-space is,
    and is written that way for the converter."""
    return re.sub(r"(?<!\\)\\\n", r"\\ \n", strip_comments(path.read_text(encoding="utf-8")))


class Blocks:
    """Finished pieces of HTML held out of the text while the rest is converted."""

    def __init__(self):
        self.items = []

    def hold(self, text, own_paragraph=True):
        self.items.append(text)
        token = "\x00%d\x00" % (len(self.items) - 1)
        return "\n\n%s\n\n" % token if own_paragraph else token

    def restore(self, text):
        return re.sub("\x00(\\d+)\x00", lambda m: self.restore(self.items[int(m.group(1))]), text)


def plain_text(fragment):
    text = re.sub(r"<[^>]+>", "", fragment)
    return html.unescape(re.sub(r"\s+", " ", text)).strip()


def first_sentence(fragment, limit=220):
    text = plain_text(re.sub(r'<span class="tag[^>]*>[^<]*</span>', "", fragment))
    m = re.match(r"(.{20,%d}?[.?!])(\s|$)" % limit, text)
    return (m.group(1) if m else text[:limit]).strip()


def bibliography(conv, inner, heading_id="sources"):
    items = []
    for chunk in inner.split(r"\bibitem{")[1:]:
        key, body = chunk.split("}", 1)
        items.append((key, conv.inline(body.strip())))
    out = ['<ol class="sources" id="%s">' % heading_id]
    for n, (key, body) in enumerate(items, 1):
        out.append('<li id="src-%d" value="%d">%s</li>' % (n, n, body))
    out.append("</ol>")
    return "\n".join(out)


def tabular_body(env_text):
    """The rows of a tabular or tabularx, its column specification removed."""
    m = re.search(r"\\begin\{(tabularx?)\}", env_text)
    i = m.end()
    _, i = read_args(env_text, i, 2 if m.group(1) == "tabularx" else 1)
    j = env_text.index("\\end{%s}" % m.group(1))
    return env_text[i:j]


def paragraphs(conv, blocks, body, id_prefix=""):
    """The text between blank lines as <p> elements; a held block alone in a paragraph stands by itself."""
    out = []
    for chunk in re.split(r"\n\s*\n", body):
        chunk = chunk.strip()
        if not chunk:
            continue
        if re.fullmatch("\x00\\d+\x00", chunk):
            out.append(chunk)
            continue
        text = conv.inline(chunk).strip()
        if not text:
            continue
        if text.startswith("<strong>The experiment:</strong>"):       # the line that says where a chapter stands
            out.append('<p class="trail">%s</p>' % text)
            continue
        ident = ""
        m = re.match(r"<em>([^<]{3,80}?)\.?</em>", text)             # a run-in head gets an anchor
        if m and id_prefix:
            ident = ' id="%s%s"' % (id_prefix, slug(m.group(1)))
            text = '<em class="runin">' + text[4:]
        out.append("<p%s>%s</p>" % (ident, text))
    return "\n".join(out)


# ------------------------------------------------------------------ the plain-language edition
def text_start(post, body):
    """Where the plain edition's text starts: at the post's opening words or, where it names none, at the second
    paragraph. The title block before that is the page's own header."""
    if post.opening_words:
        return body.find(post.opening_words)
    m = re.search(r"\S.*?\n[ \t]*\n\s*", body, flags=re.S)
    return m.end() if m else -1


def convert_plain(post, sizes, technical=None):
    """`technical` is the converted technical edition, for a post that has it as a page; mentions of its sections
    then link across."""
    tex = read_latex(source_of(post))
    body = tex.split(r"\begin{document}", 1)[1].split(r"\end{document}")[0]
    start = text_start(post, body)
    if start < 0:
        stop(post, "the text does not have the words it is said to open with, %r" % post.opening_words)
    body = body[start:]                                                # the title block is the page's own header
    body = re.sub(r"\\renewcommand\{\\(arraystretch|refname)\}\{[^}]*\}", "", body)

    keys = re.findall(r"\\bibitem\{([^}]+)\}", body)
    cites = {k: n for n, k in enumerate(keys, 1)}
    labels, figures = {}, environments(body, "figure")
    for n, (_a, _b, inner) in enumerate(figures, 1):
        m = re.search(r"\\label\{([^}]+)\}", inner)
        if m:
            labels[m.group(1)] = (str(n), "#fig-%d" % n)

    def cite_href(key, num):                      # the technical paper: its page or its PDF, if there is either
        return post.technical_href if key == "tech" and post.technical_href else "#src-%d" % num

    conv = Converter(cites, labels, cite_href)
    blocks = Blocks()

    # figures, last first so that positions stay valid
    for n in range(len(figures), 0, -1):
        a, b, inner = figures[n - 1]
        i = inner.index(r"\caption")
        (cap,), _ = read_args(inner, i + len(r"\caption"), 1)
        cap_html = conv.inline(cap)
        name = "plain-fig-%02d.svg" % n
        try:
            w, h = sizes[name]
        except KeyError:
            stop(post, "figure %d has not been rendered yet; run once without --skip-figures" % n)
        fig = ('<figure class="fig" id="fig-%d"><img src="%s%s" width="%d" height="%d" loading="lazy" '
               'alt="%s"><figcaption><b>Figure %d.</b> %s</figcaption></figure>'
               % (n, post.assets, name, w, h, html.escape("Figure %d: %s" % (n, first_sentence(cap_html))), n,
                  cap_html))
        body = body[:a] + blocks.hold(fig) + body[b:]

    def display(m):
        conv.used_mathjax = True
        return blocks.hold('<span class="eq">\\[%s\\]</span>' % html.escape(m.group(1).strip(), quote=False),
                           own_paragraph=False)

    body = re.sub(r"\\\[(.*?)\\\]", display, body, flags=re.S)

    for a, b, inner in reversed(environments(body, "center")):
        if r"\begin{tabular" in inner:
            piece = conv.table(tabular_body(inner))
        else:
            piece = '<p class="pull">%s</p>' % conv.inline(inner.strip())
        body = body[:a] + blocks.hold(piece) + body[b:]
    for a, b, inner in reversed(environments(body, "tabularx")):
        whole = body[a:b]
        body = body[:a] + blocks.hold(conv.table(tabular_body(whole), ident="translations", cls="glossary")) + body[b:]
    for a, b, inner in reversed(environments(body, "itemize")):
        inner = re.sub(r"\\setlength\{[^}]*\}\{[^}]*\}", "", inner)
        items = [conv.inline(x.strip()) for x in re.split(r"\\item(?![a-zA-Z])", inner)[1:]]
        body = body[:a] + blocks.hold("<ul>%s</ul>" % "".join("<li>%s</li>" % x for x in items)) + body[b:]
    for a, b, inner in reversed(environments(body, "thebibliography")):
        inner = inner[inner.index(r"\bibitem"):]
        body = body[:a] + blocks.hold('<h2 id="sources-head">Sources</h2>\n' + bibliography(conv, inner)) + body[b:]

    body = re.sub(r"\\par\\Needspace\{[^}]*\}\\vspace\{[^}]*\}\s*\{\\color\{ink\}\\LARGE\\bfseries ([^}]*)\}\\par\s*"
                  r"\\vspace\{[^}]*\}",
                  lambda m: blocks.hold('<h2 id="continuing">%s</h2>' % conv.inline(m.group(1))), body)

    # chapter openers, sub-headings and boxes
    chapters, chapter, seen = [], [0], set()

    def block_macro(name, n_args, make):
        nonlocal body
        pos = 0
        while True:
            i = body.find("\\" + name + "{", pos)
            if i < 0:
                return
            args, j = read_args(body, i + len(name) + 1, n_args)
            piece = blocks.hold(make(*args))
            body = body[:i] + piece + body[j:]
            pos = i + len(piece)

    def make_chapter(kicker, title):
        num = int(kicker.split()[0])
        name = conv.inline(kicker.split(r"\enspace", 1)[1].strip())
        chapters.append(dict(num=num, kicker=name, title=conv.inline(title)))
        return "\x01CHAPTER %d\x01" % num

    block_macro("chapterpage", 2, make_chapter)
    # sub-headings need to know their chapter, so walk the text in order
    out, pos = [], 0
    for m in re.finditer(r"\x00(\d+)\x00|\\sub\{", body):
        if m.group(1) is not None:
            item = blocks.items[int(m.group(1))]
            if item.startswith("\x01CHAPTER"):
                chapter[0] = int(item.strip("\x01").split()[1])
            continue
        if m.start() < pos:
            continue
        j = match_brace(body, m.end() - 1)
        title = conv.inline(body[m.end():j])
        ident = "ch%d-%s" % (chapter[0], slug(title)) if chapter[0] else slug(title)
        while ident in seen:
            ident += "-2"
        seen.add(ident)
        out.append(body[pos:m.start()])
        out.append(blocks.hold('<h3 id="%s">%s</h3>' % (ident, title)))
        pos = j + 1
    body = "".join(out) + body[pos:]
    block_macro("takeaway", 1, lambda x: '<aside class="takeaway"><b>The point to keep:</b> %s</aside>'
                % conv.inline(x))
    block_macro("whyhere", 1, lambda x: '<p class="why"><b>Why we went here.</b> %s</p>' % conv.inline(x))

    text = blocks.restore(paragraphs(conv, blocks, body))
    # split into the opening, the chapters and the back matter
    parts = re.split("\x01CHAPTER (\\d+)\x01", text)
    opening = parts[0]
    for k in range(1, len(parts), 2):
        num = int(parts[k])
        next(c for c in chapters if c["num"] == num)["html"] = parts[k + 1]
    if not chapters:
        stop(post, "the text has no chapter opener (\\chapterpage)")
    last = chapters[-1]
    if last["html"].count('<h2 id="continuing">') != 1:
        stop(post, "the last chapter must end with one closing heading set as paper 1 sets \"Continuing into the "
                   "technical paper\"; it has %d" % last["html"].count('<h2 id="continuing">'))
    cut = last["html"].index('<h2 id="continuing">')
    last["html"], back = last["html"][:cut], last["html"][cut:]
    notes = "".join('<li id="fn-%d">%s <a href="#fnref-%d" aria-label="back to the text">↩</a></li>' % (k, t, k)
                    for k, t in enumerate(conv.footnotes, 1))
    numbers = {c["num"] for c in chapters}
    sections = {s["roman"]: s["id"] for s in technical["sections"] if s["roman"]} if technical else {}

    def link_up(piece):
        return link_up_plain(piece, post, numbers, sections)

    return dict(opening=link_up(opening), chapters=[dict(c, html=link_up(c["html"])) for c in chapters],
                back=link_up(back), notes=notes, mathjax=conv.used_mathjax, n_figures=len(figures))


def link_up_plain(text, post, chapters, sections):
    """Mentions of the chapters there are (`chapters`, their numbers) become links within the page. Mentions of the
    technical paper's numbered sections link across, where the post has that page (`sections`: numeral to anchor)."""
    text = re.sub(r"(Chapters?)\u00a0(\d+)((?:, \d+)*)( and\u00a0(\d+))?",
                  lambda m: _chapter_links(m, chapters), text)
    text = re.sub(r"Section\u00a0([IVX]+)\b",
                  lambda m: '<a href="%s#%s" target="mn-technical">Section\u00a0%s</a>'
                  % (post.tech, sections[m.group(1)], m.group(1)) if m.group(1) in sections else m.group(0), text)
    return text


def _chapter_links(m, chapters):
    def one(n):
        return '<a href="#ch%s">%s</a>' % (n, n) if int(n) in chapters else n
    out = "%s\u00a0%s" % (m.group(1), one(m.group(2)))
    for extra in re.findall(r"\d+", m.group(3) or ""):
        out += ", " + one(extra)
    if m.group(5):
        out += " and\u00a0" + one(m.group(5))
    return out


# ------------------------------------------------------------------ the technical manuscript
def convert_technical(post, sizes):
    tex = read_latex(ROOT / post.paper_dir / post.technical_source)
    body = tex.split(r"\begin{document}", 1)[1].split(r"\end{document}")[0]
    meta = {}
    for name in ("title", "author", "email", "affiliation", "date"):
        i = body.index("\\%s{" % name)
        (meta[name],), j = read_args(body, i + len(name) + 1, 1)
        body = body[:i] + body[j:]
    body = body.replace(r"\maketitle", "")
    (a, b, abstract), = environments(body, "abstract")
    body = body[:a] + body[b:]

    keys = re.findall(r"\\bibitem\{([^}]+)\}", body)
    cites = {k: n for n, k in enumerate(keys, 1)}

    # numbers: sections, figures, tables, equations
    labels, sections = {}, []
    pos, count = 0, 0
    for m in re.finditer(r"\\section(\*?)\{", body):
        j = match_brace(body, m.end() - 1)
        title = body[m.end():j]
        after = re.match(r"\s*\\label\{([^}]+)\}", body[j + 1:])
        label = after.group(1) if after else None
        numbered = not m.group(1)
        if numbered:
            count += 1
        ident = ("sec-" + label.split(":")[1]) if label else "sec-" + slug(title.split()[0])
        if title.startswith("Methods"):
            ident = "sec-methods"
        if title == "Introduction":
            ident = "sec-intro"
        sections.append(dict(start=m.start(), end=j + 1 + (after.end() if after else 0), title=title, id=ident,
                             roman=roman(count) if numbered else ""))
        if label:
            labels[label] = (roman(count), "#" + ident)
    n_fig = n_tab = n_eq = 0
    for m in re.finditer(r"\\begin\{(figure\*?|table\*?|equation)\}", body):
        end = body.index("\\end{%s}" % m.group(1), m.end())
        lab = re.search(r"\\label\{([^}]+)\}", body[m.end():end])
        kind = m.group(1).rstrip("*")
        if kind == "figure":
            n_fig += 1
            if lab:
                labels[lab.group(1)] = (str(n_fig), "#fig-" + lab.group(1).split(":")[1])
        elif kind == "table":
            n_tab += 1
            if lab:
                labels[lab.group(1)] = (roman(n_tab), "#tab-" + lab.group(1).split(":")[1])
        else:
            n_eq += 1
            if lab:
                labels[lab.group(1)] = (str(n_eq), "#eq-" + lab.group(1).split(":")[1])

    conv = Converter(cites, labels)
    blocks = Blocks()
    abstract_html = conv.inline(abstract.strip())

    def replace_envs(name, make):
        nonlocal body
        for a2, b2, inner in reversed(environments(body, name)):
            body = body[:a2] + make(inner) + body[b2:]

    fig_no = [n_fig + 1]

    def make_figure(inner):
        fig_no[0] -= 1
        stem = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", inner).group(1).replace(".pdf", "")
        name = "tech-%s.svg" % stem.replace("_", "-")
        (cap,), _ = read_args(inner, inner.index(r"\caption") + len(r"\caption"), 1)
        lab = re.search(r"\\label\{fig:([^}]+)\}", inner).group(1)
        cap_html = conv.inline(cap)
        w, h = sizes[name]
        return blocks.hold('<figure class="fig wide" id="fig-%s"><img src="%s%s" width="%d" height="%d" '
                           'loading="lazy" alt="%s"><figcaption><b>FIG. %d.</b> %s</figcaption></figure>'
                           % (lab, post.assets, name, w, h,
                              html.escape("Figure %d: %s" % (fig_no[0], first_sentence(cap_html))),
                              fig_no[0], cap_html))

    replace_envs("figure*", make_figure)
    tab_no = [n_tab + 1]

    def make_table(inner):
        tab_no[0] -= 1
        (cap,), _ = read_args(inner, inner.index(r"\caption") + len(r"\caption"), 1)
        lab = re.search(r"\\label\{tab:([^}]+)\}", inner).group(1)
        caption = "<b>TABLE %s.</b> %s" % (roman(tab_no[0]), conv.inline(cap))
        return blocks.hold(conv.table(tabular_body(inner), caption=caption, ident="tab-" + lab))

    # tables come in two environments; number them in document order
    spans = sorted([(a2, b2, inner) for a2, b2, inner in environments(body, "table*")]
                   + [(a2, b2, inner) for a2, b2, inner in environments(body, "table")])
    for a2, b2, inner in reversed(spans):
        body = body[:a2] + make_table(inner) + body[b2:]

    eq_no = [n_eq + 1]

    def make_equation(inner):
        eq_no[0] -= 1
        conv.used_mathjax = True
        lab = re.search(r"\\label\{eq:([^}]+)\}", inner)
        inner = re.sub(r"\\label\{[^}]+\}", "", inner).strip()
        return blocks.hold('<span class="eq"%s>\\[%s \\tag{%d}\\]</span>'
                           % ((' id="eq-%s"' % lab.group(1)) if lab else "", html.escape(inner, quote=False),
                              eq_no[0]), own_paragraph=False)

    replace_envs("equation", make_equation)

    def display(m):
        conv.used_mathjax = True
        return blocks.hold('<span class="eq">\\[%s\\]</span>' % html.escape(m.group(1).strip(), quote=False),
                           own_paragraph=False)

    body = re.sub(r"\\\[(.*?)\\\]", display, body, flags=re.S)
    replace_envs("acknowledgments", lambda inner: blocks.hold(
        '<h2 id="acknowledgments">Acknowledgments</h2>\n<p>%s</p>' % conv.inline(inner.strip())))
    replace_envs("thebibliography", lambda inner: blocks.hold(
        '<h2 id="references-head">References</h2>\n'
        + bibliography(conv, inner[inner.index(r"\bibitem"):], "references")))

    # cut into sections (positions were taken before the replacements, so find the headings again)
    pieces, heads = [], list(re.finditer(r"\\section(\*?)\{", body))
    for k, m in enumerate(heads):
        j = match_brace(body, m.end() - 1)
        after = re.match(r"\s*\\label\{([^}]+)\}", body[j + 1:])
        start = j + 1 + (after.end() if after else 0)
        end = heads[k + 1].start() if k + 1 < len(heads) else len(body)
        sec = sections[k]
        text = blocks.restore(paragraphs(conv, blocks, body[start:end], id_prefix=sec["id"].replace("sec-", "") + "-"))
        pieces.append(dict(id=sec["id"], roman=sec["roman"], title=conv.inline(sec["title"]),
                           html=link_up_technical(text, post)))
    return dict(meta={k: conv.inline(v) for k, v in meta.items()}, abstract=link_up_technical(abstract_html, post),
                sections=pieces, mathjax=conv.used_mathjax)


def link_up_technical(text, post):
    """'PL, Ch. 5' and 'PL, Chs. 2 and 3' become links to those chapters of the plain page."""
    def chapters(m):
        out = "PL, %s\u00a0" % m.group(1)
        out += '<a href="%s#ch%s" target="mn-plain">%s</a>' % (post.base, m.group(2), m.group(2))
        if m.group(3):
            out += ' and\u00a0<a href="%s#ch%s" target="mn-plain">%s</a>' % (post.base, m.group(3), m.group(3))
        return out
    return re.sub(r"PL, (Chs?\.)\u00a0(\d+)(?: and\u00a0(\d+))?", chapters, text)


# ------------------------------------------------------------------ the blog
def convert(post, sizes):
    """(plain, technical): the post's editions as the page functions take them. `technical` is None for a post
    that has no technical page."""
    technical = convert_technical(post, sizes) if post.has_technical else None    # first: the plain page links
    return convert_plain(post, sizes, technical), technical                       # to its section anchors


def blog_pages(posts, converted):
    """Every page of the blog as {its path under OUT: its text}. `posts` are the posts that are up, oldest first,
    and `converted` holds each one's editions by slug (see convert)."""
    pages = {}
    for post in posts:
        plain, technical = converted[post.slug]
        pages[post.slug + "PL/index.html"] = page_plain(post, plain, technical, posts)
        if technical:
            pages[post.slug + "/index.html"] = page_technical(post, technical, plain, posts)
        if post.has_companion:
            companion = (ROOT / post.companion_source).read_text(encoding="utf-8")
            pages[post.slug + "Knobs/index.html"] = page_knobs(post, companion)
    pages["index.html"] = page_index(posts)
    pages["comingsoon/index.html"] = page_coming_soon(posts)
    pages["assets/" + STYLESHEET] = CSS.lstrip("\n")
    return pages


def copy_pdfs(post):
    """The PDFs travel with the pages, so their links do not depend on which branch a repository shows."""
    assets_dir(post).mkdir(parents=True, exist_ok=True)
    for field, pdf in pdfs_of(post):
        if getattr(post, field):
            shutil.copyfile(pdf, assets_dir(post) / getattr(post, field))


# ------------------------------------------------------------------ main
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tectonic", default="tectonic")
    parser.add_argument("--skip-figures", action="store_true", help="reuse the SVG files already in assets/")
    parser.add_argument("--only", type=int, nargs="+", metavar="N",
                        help="build only the posts with these numbers, as if the others were not written yet")
    args = parser.parse_args()
    posts = posts_to_build([post for post in POSTS if not args.only or post.number in args.only])
    if args.skip_figures:
        sizes = {post.slug: recorded_sizes(post) for post in posts}
    else:
        sizes = {post.slug: render_figures(post, args.tectonic) for post in posts}
        write_sizes(sizes)
    converted = {post.slug: convert(post, sizes[post.slug]) for post in posts}
    for name, text in blog_pages(posts, converted).items():
        path = OUT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    for post in posts:
        copy_pdfs(post)
        plain, technical = converted[post.slug]
        print("post %02d, plain page: %d chapters, %d figures, MathJax %s"
              % (post.number, len(plain["chapters"]), plain["n_figures"],
                 "needed" if plain["mathjax"] else "not needed"))
        if technical:
            print("post %02d, technical page: %d sections, MathJax %s"
                  % (post.number, len(technical["sections"]), "needed" if technical["mathjax"] else "not needed"))
    print("written to", OUT)


if __name__ == "__main__":
    main()
