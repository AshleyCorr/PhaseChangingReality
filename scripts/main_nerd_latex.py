"""A small LaTeX-to-HTML converter for the two editions of paper 1 (the plain-language edition and the technical
manuscript). It handles the commands those two files use and nothing else; an unknown command stops the build, so
that nothing is dropped silently. Used by scripts/build_main_nerd_pages.py.

Not a general converter. Mathematics that is only letters, numbers and common symbols becomes plain HTML (so it
can be read, searched and quoted without a script); anything else is left as TeX for MathJax.
"""
import html
import re

GREEK = {"lambda": "λ", "varepsilon": "ε", "tau": "τ", "Delta": "Δ", "sigma": "σ", "pi": "π", "kappa": "κ",
         "Omega": "Ω", "delta": "δ", "chi": "χ"}
SYMBOL = {"times": "×", "ge": "≥", "le": "≤", "geq": "≥", "leq": "≤", "neq": "≠", "to": "→", "approx": "≈",
          "simeq": "≃", "pm": "±", "ne": "≠",
          "in": "∈", "sim": "∼", "propto": "∝", "equiv": "≡", "ddagger": "‡", "ldots": "…", "cdot": "·",
          "mid": "|", "infty": "∞"}
WORD = {"ln": "ln", "min": "min", "max": "max", "exp": "exp"}
TAGS = {"Exact": ("exact", "EXACT"), "Meas": ("meas", "MEASURED"), "MeasX": ("meas", "MEASURED, EXPLORATORY"),
        "Pub": ("pub", "PUBLISHED"), "Ours": ("ours", "OUR READING, UNVERIFIED"),
        "Claim": ("claim", "THE AUTHOR'S CLAIM"), "Sketch": ("pub", "SKETCH, NOT DATA")}
IGNORE = {"small", "footnotesize", "scriptsize", "Large", "LARGE", "large", "bfseries", "raggedright", "centering",
          "noindent", "selectfont", "medskip", "bigskip", "smallskip", "maketitle", "hline", "toprule", "midrule",
          "bottomrule", "linewidth", "textwidth", "baselineskip", "strut", "sffamily", "relax", "par"}
IGNORE_ARGS = {"color": 1, "vspace": 1, "Needspace": 1, "thispagestyle": 1, "fontsize": 2, "setlength": 2,
               "label": 1, "pagestyle": 1}


class Unknown(Exception):
    pass


def strip_comments(t):
    t = re.sub(r"(?m)^[ \t]*%.*\n", "", t)                 # a comment line goes with its newline
    return re.sub(r"(?<!\\)%.*", "", t)


def match_brace(s, i):
    """Index of the brace closing the one at s[i]."""
    assert s[i] == "{", s[i:i + 30]
    depth = 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError("unbalanced braces")


def read_args(s, i, n):
    """n brace arguments starting at or after s[i] (white space skipped). Returns (list, next index)."""
    out = []
    for _ in range(n):
        while i < len(s) and s[i] in " \t\n":
            i += 1
        j = match_brace(s, i)
        out.append(s[i + 1:j])
        i = j + 1
    return out, i


def slug(text):
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return text[:60].strip("-") or "x"


# ------------------------------------------------------------------ mathematics
def simple_math(tex):
    """Plain HTML for mathematics made of letters, numbers and common symbols; None if it needs a typesetter."""
    out, i, n = [], 0, len(tex)

    def group(k):
        """The group or single token starting at tex[k]; returns (html or None, next index)."""
        if k < n and tex[k] == "{":
            j = match_brace(tex, k)
            return simple_math(tex[k + 1:j]), j + 1
        if k < n and tex[k] == "\\":
            m = re.match(r"\\([a-zA-Z]+)", tex[k:])
            if not m:
                return None, k
            return simple_math(m.group(0)), k + m.end()
        return (simple_math(tex[k]) if k < n else None), k + 1

    while i < n:
        c = tex[i]
        if c == "\\":
            m = re.match(r"\\([a-zA-Z]+)", tex[i:])
            if not m:
                sym = tex[i + 1] if i + 1 < n else ""
                if sym in "{}%":
                    out.append(sym)
                elif sym in ",;:! ":
                    out.append("\u2009" if sym in ",;:" else ("" if sym == "!" else " "))
                else:
                    return None
                i += 2
                continue
            name = m.group(1)
            i += m.end()
            if name in GREEK:
                out.append("<i>%s</i>" % GREEK[name] if name.islower() else GREEK[name])
            elif name in SYMBOL:
                s = SYMBOL[name]
                out.append(" %s " % s if name in ("times", "ge", "le", "geq", "leq", "neq", "to", "approx", "simeq",
                                                   "ne", "in", "sim", "propto", "equiv") else s)
            elif name in WORD:
                out.append(WORD[name])
            elif name in ("text", "mathrm", "rm"):
                if name == "rm":
                    return None
                (arg,), i = read_args(tex, i, 1)
                out.append(html.escape(arg))
            elif name in ("quad", "qquad", "enspace"):
                out.append(" ")
            elif name in ("boldsymbol", "mathbf"):
                (arg,), i = read_args(tex, i, 1)
                inner = simple_math(arg)
                if inner is None:
                    return None
                out.append("<b>%s</b>" % inner)
            elif name in ("tfrac", "frac"):               # only the small fractions that have a character
                pair = re.match(r"\s*(?:\{(\d)\}|(\d))\s*(?:\{(\d)\}|(\d))", tex[i:])
                if not pair:
                    return None
                top, bottom = pair.group(1) or pair.group(2), pair.group(3) or pair.group(4)
                glyph = {("1", "2"): "½", ("1", "4"): "¼", ("3", "4"): "¾", ("1", "3"): "⅓", ("2", "3"): "⅔"}
                if (top, bottom) not in glyph:
                    return None
                out.append(glyph[(top, bottom)])
                i += pair.end()
            elif name in ("left", "right", "bigl", "bigr", "Bigl", "Bigr"):
                pass
            else:
                return None
            if i < n and tex[i] == " ":
                i += 1
        elif c in "_^":
            sub, i = group(i + 1)
            if sub is None:
                return None
            out.append("<%s>%s</%s>" % (("sub" if c == "_" else "sup"), sub.strip(), ("sub" if c == "_" else "sup")))
        elif c == "{":
            j = match_brace(tex, i)
            inner = simple_math(tex[i + 1:j])
            if inner is None:
                return None
            out.append(inner)
            i = j + 1
        elif c.isalpha():
            out.append("<i>%s</i>" % c)
            i += 1
        elif c.isdigit() or c in ".,()[]|/!':; ":
            out.append(c if c != "'" else "′")
            i += 1
        elif c in "=<>+":
            out.append(" %s " % html.escape(c))
            i += 1
        elif c == "-":
            prev = "".join(out).rstrip()
            binary = bool(prev) and (prev[-1].isalnum() or prev[-1] in ")]>")       # after a term: a minus sign
            out.append(" − " if binary else "−")
            i += 1
        elif c == "~":
            out.append("\u00a0")
            i += 1
        elif c in "\n\t":
            out.append(" ")
            i += 1
        elif c == "&":
            return None
        else:
            return None
    text = "".join(out)
    text = re.sub(r"<sup>\s*−\s*", "<sup>−", text)
    text = re.sub(r"\(\s*−\s*", "(−", text)
    text = re.sub(r"(=|&lt;|&gt;|[≥≤≈≃→×∈∼∝≡≠])\s+−\s+", r"\1 −", text)
    return re.sub(r" {2,}", " ", text).strip()


def math_inline(tex):
    plain = simple_math(tex.strip())
    if plain is not None:
        return '<span class="m">%s</span>' % plain
    return r"\(" + html.escape(tex.strip(), quote=False) + r"\)"


# ------------------------------------------------------------------ inline text
class Converter:
    """Holds what the inline conversion needs to know about one document: its numbers and its links."""

    def __init__(self, cites, labels, cite_href=None):
        self.cites = cites              # key -> number
        self.labels = labels            # label -> (text shown, href)
        self.cite_href = cite_href or (lambda key, num: "#src-%d" % num)
        self.footnotes = []
        self.used_mathjax = False

    def math(self, tex):
        out = math_inline(tex)
        if out.startswith(r"\("):
            self.used_mathjax = True
        return out

    def inline(self, s):
        out, i, n = [], 0, len(s)
        while i < n:
            c = s[i]
            if c == "$":
                j = s.index("$", i + 1)
                out.append(self.math(s[i + 1:j]))
                i = j + 1
            elif c == "\\":
                m = re.match(r"\\([a-zA-Z]+)\*?", s[i:])
                if not m:
                    sym = s[i + 1]
                    i += 2
                    if sym in "%&_#$":
                        out.append(html.escape(sym))
                    elif sym == "\\":
                        out.append("<br>")
                    elif sym in ", ;":
                        out.append("\u2009" if sym == "," else " ")
                    elif sym == "(":                      # already-converted mathematics passes through
                        j = s.index(r"\)", i)
                        out.append(r"\(" + s[i:j] + r"\)")
                        i = j + 2
                    else:
                        raise Unknown("\\" + sym + " near: " + s[max(0, i - 40):i + 40])
                    continue
                name = m.group(1)
                i += m.end()
                if name in ("emph", "textit"):
                    (a,), i = read_args(s, i, 1)
                    out.append("<em>%s</em>" % self.inline(a))
                elif name == "textbf":
                    (a,), i = read_args(s, i, 1)
                    out.append("<strong>%s</strong>" % self.inline(a))
                elif name in ("texttt", "path"):
                    (a,), i = read_args(s, i, 1)
                    out.append("<code>%s</code>" % html.escape(a.replace("\\_", "_")))
                elif name == "url":
                    (a,), i = read_args(s, i, 1)
                    out.append('<a href="%s">%s</a>' % (html.escape(a), html.escape(a)))
                elif name == "href":
                    (u, a), i = read_args(s, i, 2)
                    out.append('<a href="%s">%s</a>' % (html.escape(u), self.inline(a)))
                elif name == "cite":
                    (a,), i = read_args(s, i, 1)
                    nums = sorted(self.cites[k.strip()] for k in a.split(","))
                    back = {self.cites[k.strip()]: k.strip() for k in a.split(",")}
                    out.append("[" + ", ".join('<a class="cite" href="%s">%d</a>' % (self.cite_href(back[v], v), v)
                                               for v in nums) + "]")
                elif name == "ref":
                    (a,), i = read_args(s, i, 1)
                    text, href = self.labels[a]
                    out.append('<a href="%s">%s</a>' % (href, text))
                elif name == "footnote":
                    (a,), i = read_args(s, i, 1)
                    self.footnotes.append(self.inline(a))
                    k = len(self.footnotes)
                    out.append('<sup class="fn"><a id="fnref-%d" href="#fn-%d">%d</a></sup>' % (k, k, k))
                elif name in TAGS:
                    cls, text = TAGS[name]
                    out.append('<span class="tag tag-%s">%s</span>' % (cls, text))
                    if i < n and s[i] == "\\" and s[i:i + 2] == "\\ ":
                        i += 2
                        out.append(" ")
                elif name == "texorpdfstring":
                    (a, _b), i = read_args(s, i, 2)
                    out.append(self.inline(a))
                elif name in ("enspace", "quad", "qquad"):
                    out.append(" ")
                elif name == "textbullet":
                    out.append("•")
                elif name == "boldsymbol":
                    (a,), i = read_args(s, i, 1)
                    out.append(self.inline(a))
                elif name in IGNORE:
                    pass
                elif name in IGNORE_ARGS:
                    _, i = read_args(s, i, IGNORE_ARGS[name])
                else:
                    raise Unknown("\\" + name + " near: " + s[max(0, i - 60):i + 60])
                if name in IGNORE and i < n and s[i] == " ":
                    i += 1
            elif c == "{":
                j = match_brace(s, i)
                out.append(self.inline(s[i + 1:j]))
                i = j + 1
            elif c == "}":
                raise Unknown("stray } near: " + s[max(0, i - 60):i + 40])
            elif c == "~":
                out.append("\u00a0")
                i += 1
            elif c == "`":
                if s[i:i + 2] == "``":
                    out.append("“")
                    i += 2
                else:
                    out.append("‘")
                    i += 1
            elif c == "'":
                if s[i:i + 2] == "''":
                    out.append("”")
                    i += 2
                else:
                    out.append("’")
                    i += 1
            elif c == "-":
                if s[i:i + 3] == "---":
                    out.append("—")
                    i += 3
                elif s[i:i + 2] == "--":
                    out.append("–")
                    i += 2
                else:
                    out.append("-")
                    i += 1
            elif c == "\x00":                             # a block already converted
                j = s.index("\x00", i + 1)
                out.append(s[i:j + 1])
                i = j + 1
            else:
                out.append(html.escape(c, quote=False))
                i += 1
        text = "".join(out)
        return re.sub(r"[ \t\n]+", " ", text)

    # -------------------------------------------------------------- tables
    def table(self, body, caption="", ident="", cls=""):
        """HTML for the body of a tabular (the text between its \\begin and \\end, column spec removed)."""
        body = re.sub(r"\\(toprule|bottomrule)", "", body)
        parts = re.split(r"\\(?:midrule|hline)", body)
        head_rows = self._rows(parts[0]) if len(parts) > 1 else []
        rows = self._rows(" ".join(parts[1:]) if len(parts) > 1 else parts[0])
        out = ['<div class="tablewrap"><table%s%s>' % ((' id="%s"' % ident) if ident else "",
                                                       (' class="%s"' % cls) if cls else "")]
        if caption:
            out.append("<caption>%s</caption>" % caption)
        if head_rows:
            out.append("<thead>")
            for r in head_rows:
                out.append("<tr>" + "".join("<th>%s</th>" % self.inline(c.strip()) for c in r) + "</tr>")
            out.append("</thead>")
        out.append("<tbody>")
        for r in rows:
            out.append("<tr>" + "".join("<td>%s</td>" % self.inline(c.strip()) for c in r) + "</tr>")
        out.append("</tbody></table></div>")
        return "\n".join(out)

    @staticmethod
    def _rows(text):
        rows = []
        for line in re.split(r"\\\\", text):
            if not line.strip():
                continue
            cells, depth, cur, k = [], 0, [], 0
            while k < len(line):                         # split on & outside mathematics and braces
                ch = line[k]
                if ch == "\\" and k + 1 < len(line):
                    cur.append(line[k:k + 2])
                    k += 2
                    continue
                if ch == "$":
                    j = line.index("$", k + 1)
                    cur.append(line[k:j + 1])
                    k = j + 1
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                if ch == "&" and depth == 0:
                    cells.append("".join(cur))
                    cur = []
                else:
                    cur.append(ch)
                k += 1
            cells.append("".join(cur))
            rows.append(cells)
        return rows


def environments(s, name):
    """Every top-level \\begin{name}...\\end{name} in s as (start, end, inner), in order."""
    out, pos = [], 0
    b, e = "\\begin{%s}" % name, "\\end{%s}" % name
    while True:
        i = s.find(b, pos)
        if i < 0:
            return out
        j = s.index(e, i)
        out.append((i, j + len(e), s[i + len(b):j]))
        pos = j + len(e)


def roman(n):
    out = ""
    for value, sym in ((10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while n >= value:
            out += sym
            n -= value
    return out
