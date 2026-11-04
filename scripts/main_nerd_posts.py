"""The posts of the Side Nerd Blog, sidenerdapps.com/mainnerd/, as data. A post is one paper, and what differs
from one post to the next is here: its number and folder name, where its LaTeX is, which editions it has, and the
frame text written for it (summary, question headings, quick answers, key terms). What every post shares (the page
shell, the advertisements, the stylesheet) is in scripts/main_nerd_site.py; the conversion of the papers' own text
is in scripts/build_main_nerd_pages.py.

The questions and answers restate the plain-language edition and carry its numbers; change them there first. The
headings that echo common searches ("What was there before the Big Bang?") are bridges to the paper, and each
answer says what the work does not show.

To add a post, write its Post below and put it in POSTS. A text block left empty is left off the pages, and a post
is built only once its plain-language source is in the repository.
"""
from dataclasses import dataclass, field

BLOG = "/mainnerd/"                             # the blog: one folder per post
SHARED = BLOG + "assets/"                          # the stylesheet, and one folder per post
REPO = "https://github.com/EmilySmithCreate/RecreationalPhysics"


@dataclass(frozen=True)
class Post:
    """One post. The addresses of its pages and files follow from `slug`; they are the properties below."""
    number: int
    slug: str                           # the post's folder name; the plain-language edition adds "PL"
    title: str
    published: str                      # ISO date the post first went up
    paper_dir: str                      # the paper's folder, from the repository root
    subtitle: str = ""
    updated: str = ""                   # ISO date of the latest revision, if any
    plain_source: str = "paper_plain_language.tex"     # in paper_dir; each PDF is the same name with .pdf
    technical_source: str = "paper.tex"
    pdf_plain_name: str = ""            # the PDFs' names beside the pages; empty where there is none to offer
    pdf_tech_name: str = ""
    # The words the plain edition's text starts with; the title block above them is the page's own header.
    # Left empty, the text starts at the second paragraph of the document.
    opening_words: str = ""
    cite_key: str = ""                  # the BibTeX key under "How to cite this page"
    has_technical: bool = False         # the technical edition as a page of its own, at /mainnerd/<slug>/
    technical_figures: tuple = ()       # that page's figures: PDF files in paper_dir, named without ".pdf"
    # which section of the technical paper goes with which chapter of the plain edition, and back
    chapter_to_section: dict = field(default_factory=dict)
    section_to_chapters: dict = field(default_factory=dict)
    has_companion: bool = False         # an interactive companion, at /mainnerd/<slug>Knobs/
    companion_source: str = ""          # its source file, from the repository root
    artifact: str = ""                  # where the papers' PDFs link to it; the pages link to the blog's copy
    mascot: dict = field(default_factory=dict)     # file (in the post's assets), width, height, alt, caption
    share_image: str = "og-image.png"   # in the post's assets; a post that has none borrows the first post's
    mid_ad_after: int = 0               # the chapter the mid-article advertisement follows; 0: a third of the way in
    ticker: tuple = ()                  # the post's own lines in the ticker, after its title
    # the frame text; each is left off the page while it is empty
    page_title: str = ""                # the plain page's <title>
    description: str = ""               # what a search result shows for the plain page
    technical_description: str = ""
    companion_description: str = ""
    summary: str = ""
    chapter_questions: dict = field(default_factory=dict)      # chapter number -> (question, bridge)
    big_questions: list = field(default_factory=list)          # (question, answer)
    faq: list = field(default_factory=list)                    # (question, answer)
    key_terms: list = field(default_factory=list)

    @property
    def base(self):                     # the plain-language edition
        return BLOG + self.slug + "PL/"

    @property
    def tech(self):                     # the technical edition
        return BLOG + self.slug + "/"

    @property
    def knobs(self):                    # the interactive companion
        return BLOG + self.slug + "Knobs/"

    @property
    def assets(self):                   # the post's figures, PDFs and pictures
        return SHARED + self.slug + "/"

    @property
    def pdf_plain(self):
        return self.assets + self.pdf_plain_name if self.pdf_plain_name else ""

    @property
    def pdf_tech(self):
        return self.assets + self.pdf_tech_name if self.pdf_tech_name else ""

    @property
    def technical_href(self):
        """Where "the technical edition" leads: its page, its PDF where it has no page, or nowhere."""
        return self.tech if self.has_technical else self.pdf_tech

    @property
    def paper_url(self):                # the paper's folder in the repository
        return REPO + "/tree/main/" + self.paper_dir

    @property
    def headline(self):
        return self.title + (": " + self.subtitle if self.subtitle else "")


# ------------------------------------------------------------------ post 01: what the answer engines read
SUMMARY_01 = (
    "<p><b>In one paragraph.</b> <i>The curled torus burps</i> is a computational physics paper about a "
    "hypothetical sharp change in which flat space emerges from something more curled up. It uses a toy model of "
    "space as a network of dots and links. The toy is the paper's own: a close cousin of the published model "
    "called combinatorial quantum gravity, with one knob turned above the published value, and not that model "
    "itself. In the paper's toy, a tube (a torus curled to four steps around) sits above the flat arrangement in "
    "energy and is stable for now. The paper counts what it costs to start the change (12 units at the main setting, for "
    "every size of tube tested), predicts how long the tube waits by counting its exits, watches the flat arrangement "
    "spread, seals the system to see where the released energy (the burp) goes, and counts what is left behind "
    "(about one four-dot scrap). It also reads what the tube turns into: the tight curl is gone and the energy "
    "is released, but the result is still a narrow tube, six steps around, and a second large direction does "
    "not appear. A toy can show that a mechanism is possible or impossible within a class of models; that is "
    "the whole claim.</p>")

# One searched question per chapter, as a heading, with a bridge to what the chapter does. Each bridge is true
# of the plain-language edition's chapter and claims no more than it does.
CHAPTER_QUESTIONS_01 = {
    1: ("What was there before the Big Bang?",
        "Nobody knows. This chapter sets up one way to study the question without a time machine: treat space "
        "as one arrangement of something deeper, and ask what kind of change could have produced it."),
    2: ("Is space made of something smaller?",
        "Some physicists think so. This chapter describes one published model, combinatorial quantum gravity, "
        "in which space is a network of dots and links and nothing else, and then the toy this paper works in: "
        "a close cousin of that model with one knob turned up, not that model itself."),
    3: ("What existed before space?",
        "The author's hypothesis is that it was not nothing and not chaos, but a specific curled-up arrangement. "
        "This chapter builds a stand-in for that arrangement inside the toy model."),
    4: ("What could have set off the Big Bang?",
        "Unknown. In the toy model a change of this kind needs a push, and the push can be counted exactly: "
        "twelve units at the paper's main setting of the knob (λ = 1.25), the same for every size of tube "
        "tested."),
    5: ("What is a metastable state, or false vacuum?",
        "A state that is stable for now: it lasts until chance hands it enough energy to leave. This chapter "
        "predicts how long the toy's version waits, and checks the prediction."),
    6: ("Can space change phase, the way water freezes?",
        "In the toy model it can: a patch of the new arrangement appears and grows while the old one is still "
        "there beside it. This chapter watches that happen, says what the pictures do and do not prove, and "
        "reads what the tube turns into: a tube six steps around, not a wide sheet."),
    7: ("Where did the energy of the Big Bang come from?",
        "In the author's hypothesis the change itself released it, the way freezing water gives off heat. In "
        "the toy model that release, the burp, can be counted exactly and followed when nothing is allowed to "
        "leave, and the change leaves a small scrap of the old arrangement behind."),
    8: ("How do you test an idea about the universe with a simulation?",
        "By turning one knob at a time with the rules written down first, and publishing every setting, "
        "including the ones that failed. This chapter is that map."),
    9: ("Can a computer simulate the beginning of the universe?",
         "Not the real one. This chapter lists exactly how far these simulations go, where each stops, and what "
         "lies past each edge."),
    10: ("Is this a theory of where the universe came from?",
         "It is a hypothesis, with one mechanism tested in a toy model and no physicist's review yet. This "
         "chapter says what was learned and what would have to come next."),
}

BIG_QUESTIONS_01 = [
    ("What was there before the Big Bang?",
     "Nobody knows, and this work does not settle it. It tests one hypothesis, the author's: that space is one "
     "settled arrangement of something deeper, and that before the change there was a different, specific "
     "arrangement, curled up and stable for now. A toy model shows that a curled arrangement can give way "
     "sharply in a network of dots and links, with a countable push and a countable release. What it turns into "
     "there is a slightly wider tube, not a wide space. That is part of a possibility shown in a toy, not "
     "evidence about the real universe."),
    ("Where did the Big Bang come from?",
     "One hypothesis, tested here only in a toy model: the hot early universe is the energy released when an "
     "earlier arrangement gave way and space opened out, the way water gives off heat when it freezes. In the toy "
     "the release can be counted exactly and grows with the size of the system, while the push that starts it "
     "stays small and fixed at every size tested."),
    ("What was there before time?",
     "This paper cannot say. The toy model has no time in it: its clock counts attempted moves, not seconds. In "
     "the author's wider hypothesis, which this paper does not test, what came before space had order but not "
     "time as we experience it."),
    ("Is space made of something?",
     "Possibly. In the published model this work builds on, space is a network of dots and links with no "
     "positions at all, and flat space is one arrangement of that network. Whether real space is like that is an "
     "open research question."),
    ("Is this an accepted theory?",
     "No. It is a hobbyist's hypothesis, tested on a toy model with the predictions written down first, written "
     "with AI assistance and not yet reviewed by a physicist. The paper says exactly what was and was not shown."),
]

FAQ_01 = [
    ("What is “The curled torus burps” about?",
     "It is a plain-language edition of a computational physics paper by Emily Smith. It asks what a sharp change "
     "from one specific arrangement into flat space would look like, and tests each part of that picture in a toy "
     "model made of dots and links: what it costs to start, how long the wait is, what takes shape, where the "
     "released energy goes, and what remains."),
    ("What is combinatorial quantum gravity, and is this it?",
     "Combinatorial quantum gravity is a published model, built by Carlo Trugenberger, Christy Kelly and Fabio "
     "Biancalana, in which space is a network with no positions at all and the energy is a count of squares that "
     "acts as a curvature. This paper's toy is a close cousin of that model, not that model itself: it turns one knob, "
     "the price of a surplus square (λ), above the published value of 1."),
    ("What is a curled torus?",
     "A torus is a grid that wraps around in both directions, the surface of a doughnut, and every shape in the "
     "paper is one: flat (the sheet, a fat doughnut), curled (the tube, a thin ring) or knot. Curl one direction "
     "of the flat one until it is only four steps around and you have a tube. In the toy the tube has one long "
     "direction where the flat sheet has two, and it sits 4(λ − 1) units of energy per dot above the flat sheet: "
     "one unit per dot at λ = 1.25."),
    ("Is flat space the same as endless space?",
     "No. Flat is about each place's surroundings; whether space goes on for ever is a separate question. In the "
     "toy every shape closes on itself, like a doughnut, and a count made from inside matches an endless grid's "
     "until half the way around, so a doughnut much wider than its inhabitants looks endless to them. Real space "
     "is measured to be flat on the largest scales; whether it is endless has not been measured."),
    ("What is the burp?",
     "The burp is the energy given off when the tube's tight curl comes undone. For a clean change it is "
     "exactly 4(λ − 1) per dot, so it grows with the size of the tube, while the push needed to start it does "
     "not."),
    ("Does the tube open into a wide space?",
     "Not in this toy as it stands. Read from the saved wiring, the tube ends as a tube six steps around where "
     "it was four: flat at every dot, and still narrow. The toy's rule charges for loops of four and for nothing "
     "longer, so nothing pays for widening further."),
    ("How much energy does it take to start the change?",
     "Twelve units at λ = 1.25: the cost of the cheapest partner swap out of a perfect tube, counted exactly, the "
     "same at every size from 48 to 288 dots. In 320 sealed runs, no tube left with a budget below 12 and every "
     "tube left with 12 or more."),
    ("How long does the tube wait before it starts?",
     "Count the exits a sweep offers, multiply by the chance each is taken, and you predict the mean wait with "
     "nothing fitted. Across twelve conditions the measured mean wait divided by the predicted one averaged 0.98. "
     "The waiting is memoryless, like rolling a die until a six comes up."),
    ("What is left behind after the change?",
     "About one four-dot scrap of the old arrangement per sheet, holding 14 units, whatever the size of the tube, "
     "in the sealed runs with the most room. What the scrap does afterwards is the subject of the next paper."),
    ("Does this show how the universe began?",
     "No. A toy can show that a mechanism is possible or impossible within a class of models; that is the whole "
     "claim. The clock in the simulation is a sampling rule, not physical time, and no physicist has reviewed the "
     "work."),
    ("Was AI used to write this?",
     "Yes. The author used Claude (Anthropic) and ChatGPT/Codex (OpenAI) to assist with code, analysis and "
     "writing, and is responsible for the result. The predictions for the main tests were written down before the "
     "runs."),
]

KEY_TERMS_01 = ["what came before the Big Bang", "emergent spacetime", "combinatorial quantum gravity",
                "graph model of space", "curled torus", "metastable state", "false vacuum decay", "activation energy",
                "first exit time", "memoryless waiting time", "nucleation and front", "sealed system (microcanonical)",
                "Creutz reservoir", "relic defect", "pre-registration", "Markov chain Monte Carlo",
                "decompactification"]

POST_01 = Post(
    number=1, slug="01thecurledtorusburps", title="The curled torus burps",
    subtitle="A small push, a large release, and the question of what survives", published="2026-10-10", updated="2026-10-14",
    paper_dir="docs/papers/curled_torus",
    pdf_plain_name="the-curled-torus-burps-plain-language.pdf",      # copied beside the pages by the build
    pdf_tech_name="the-curled-torus-burps-technical.pdf",
    opening_words="Everything you have ever seen", cite_key="smith2026curledtorus",
    has_technical=True, technical_figures=("fig_tori", "fig1", "fig_lambda"),
    chapter_to_section={1: "sec-intro", 2: "sec-model", 3: "sec-model", 4: "sec-cost", 5: "sec-exit",
                        6: "sec-conversion", 7: "sec-sealed", 8: "sec-lambda",
                        9: "sec-discussion", 10: "sec-discussion"},
    section_to_chapters={"sec-intro": [1], "sec-model": [2, 3], "sec-cost": [4], "sec-exit": [5],
                         "sec-conversion": [6], "sec-sealed": [7], "sec-lambda": [8],
                         "sec-discussion": [9, 10], "sec-methods": [9]},
    # The companion, "The Tube's Curling Ladder", was first published as a Claude artifact, and the papers' PDFs
    # link to it there. The blog hosts its own copy, built from the same source file, and the web pages link to that.
    has_companion=True, companion_source="docs/public/curling_ladder_tube.html",
    artifact="https://claude.ai/artifact/F4sBf955J5ZT7rCq8uebW8",
    mascot=dict(file="sleepy-cow.gif", width=208, height=254,
                alt="A line drawing of a shaggy cow curled up asleep; in the next frame it stretches both front "
                    "legs and lets out a small burp.",
                caption="Fig. 0. A curled arrangement, stable for now, gives way and burps. Round cow not "
                        "available, artist's impression. Not a torus."),
    mid_ad_after=4, ticker=("Twelve units start an escape", "What opens a space?"),
    page_title="The curled torus burps: what came before space? A hypothesis tested in a toy model, in plain "
               "language",
    description="What came before space? A plain-language physics paper tests one hypothesis in a toy model: a "
                "curled-up arrangement gives way for a push of 12 units, releases a burp of energy and leaves "
                "a scrap behind. What it turns into is still a narrow tube.",
    technical_description="Technical edition: activated escape and conversion of a curled torus in four-regular "
                          "bipartite graph energies next to combinatorial quantum gravity. Exact barrier of 12 "
                          "units, first-exit times from counted moves, sealed-reservoir runs and relics.",
    companion_description="Turn the toy model's knobs yourself: an interactive companion to \"The curled torus "
                          "burps\". Exact numbers at every setting; measured results only where the simulations "
                          "ran.",
    summary=SUMMARY_01, chapter_questions=CHAPTER_QUESTIONS_01, big_questions=BIG_QUESTIONS_01, faq=FAQ_01,
    key_terms=KEY_TERMS_01)


# ------------------------------------------------------------------ post 02: the plain-language edition only
SUBTITLE_02 = "How many scraps a sharp change leaves, and which of them last"

SUMMARY_02 = (
    "<p><b>In one paragraph.</b> <i>What the change leaves behind</i> is the second paper in a series about a "
    "hypothetical sharp change in which space emerges from something more curled up. It works in a toy model of "
    "space as a network of dots and links. The toy is the paper's own: a close cousin of the published model "
    "called combinatorial quantum gravity, with one knob turned above the published value, and not that model "
    "itself. In the toy a curled tube changes all at once into a wider tube and gives off energy (the burp), and "
    "the change leaves small scraps. This paper counts them, reads what kinds there are, and asks how they come to "
    "be. One start leaves about one "
    "scrap, whatever the size of the tube. More starts leave more, but fewer than one each. The common scrap, a "
    "curled loop of four dots, heals away in warm surroundings and is frozen in when its surroundings cool. "
    "Cooled slowly enough, nearly all of them heal, yet most of the leftover energy stays, held by two rarer "
    "kinds of scrap that no cooling in these runs removed. Three predictions written down beforehand failed, and "
    "they are reported. A toy can show that a mechanism is possible or impossible within a class of models; "
    "that is the whole claim.</p>")

# One searched question per chapter, as a heading, with a bridge to what the chapter does. Each bridge is true
# of the plain-language edition's chapter and claims no more than it does.
CHAPTER_QUESTIONS_02 = {
    1: ("What happens in a sharp phase change?",
        "One arrangement gives way to another all at once and gives off energy, the way water gives off heat when "
        "it freezes. This chapter repeats, briefly, the sharp change the first paper studied in a toy model: a "
        "curled tube of dots that turns into a wider tube."),
    2: ("Is anything left over from before the Big Bang?",
        "Nobody knows. In the toy model a sharp change usually stops one small piece short of finishing. This "
        "chapter says what that scrap is, and that a change started in one place leaves about one, whatever the "
        "size of the tube."),
    3: ("How do defects form when a material changes phase?",
        "A change that starts in many places at once does not always finish cleanly. In the toy model more "
        "starting places leave more scraps, though fewer than one each."),
    4: ("Do defects heal over time?",
        "In warm surroundings the toy's common scrap does. This chapter holds one in a bath at three warmths and "
        "times how long it lasts."),
    5: ("What does it mean for something to freeze in as the universe cools?",
        "A leftover freezes in when its surroundings get too cold to heal it before it has healed. This chapter "
        "runs that race in the toy, one scrap at a time."),
    6: ("Does cooling slowly remove defects?",
        "Mostly, in the toy: cooled slowly enough, nearly all of the common scraps heal. But most of the leftover "
        "energy stays, and this chapter shows what the slow cooling did not touch."),
    7: ("What kinds of defect can a sharp change leave?",
        "In the toy, three, read link by link: a curled loop, a double loop and a twist. Each is stuck on its own "
        "and holds a fixed amount, and none is a seam of the kind first tested."),
    8: ("Why do some defects never heal?",
        "One possible reason, seen in old runs and not yet tested: most scraps are free, but some are joints "
        "between two stretches of different width, and a joint may not be removable while the widths differ."),
    9: ("Could something left over from the birth of space still be around?",
        "This paper cannot say. It shows which leftovers last in one toy model, lists the three predictions that "
        "failed, and says what has to be computed next."),
}

BIG_QUESTIONS_02 = [
    ("Is anything left over from before the Big Bang?",
     "Nobody knows, and this work does not settle it. It tests one part of a hypothesis, the author's: that space "
     "emerged from something else by a sharp change, and that such a change leaves a remainder. She wants to test "
     "whether a remainder can last. In a toy model the common leftover heals away unless its surroundings cool, "
     "and two rarer kinds lasted through every cooling tried. That is a result about a toy, not evidence about "
     "the real universe."),
    ("What is a relic, in physics?",
     "Something made in an earlier era that survives into a later one. This paper studies relics of a toy model "
     "only: small arrangements of dots and links left behind when a curled arrangement changes sharply into a "
     "flatter one."),
    ("Do defects from a phase change last forever?",
     "Not necessarily. In the toy the common defect heals in any warm surroundings and survives only if its "
     "surroundings cool first. Two rarer kinds were still there after the slowest cooling these runs could "
     "afford, which is not the same as lasting forever."),
    ("Is this an accepted theory?",
     "No. It is a hobbyist's hypothesis, tested on a toy model with the predictions written down first, written "
     "with AI assistance and not yet reviewed by a physicist. The paper says exactly what was and was not shown."),
]

FAQ_02 = [
    ("What is “What the change leaves behind” about?",
     "It is a plain-language edition of a computational physics paper by Emily Smith, the second in a series. The "
     "first paper found that a curled arrangement in a toy model of dots and links changes all at once and gives "
     "off energy. This one asks what that change leaves behind: how many leftover pieces, what they are, and "
     "whether they last."),
    ("What is the burp?",
     "The energy given off when the toy's curled tube changes into a wider one: one unit for every dot at the "
     "paper's setting. The name comes from the first paper in the series."),
    ("What is a scrap?",
     "A small piece left over when the change stops short of finishing. Read from the wiring there are three "
     "kinds: a curled loop of four dots holding 14 units, a double loop of eight dots holding 8, and a twist of "
     "eight dots of two kinds holding 20 to 30."),
    ("How many scraps does a change leave?",
     "A change started in one place leaves about one, whatever the size of the tube, from 64 to 288 dots. Started "
     "by hand in one, two, four or eight places it left 1.10, 2.15, 2.50 and 3.40 on average: more starts, more "
     "scraps, but fewer than one each. A tube long enough to start itself in many places left about one curled "
     "loop for every 400 to 480 dots."),
    ("Do the scraps last?",
     "The common one, the curled loop, heals away in every warm bath tried, in 500 to 80,500 sweeps of the "
     "simulation. If the bath cools first it is frozen in. The double loop and the twist were unchanged by every "
     "cooling run, the slowest lasting 300,000 size-fair sweeps; whether a slower one would remove them is not "
     "known."),
    ("What happened when many scraps were cooled slowly?",
     "The curled loops fell off a cliff: slowing the cooling from 10,000 to 100,000 fair sweeps lost two in "
     "three, and from 30,000 to 300,000 lost twelve in thirteen. The energy held in leftovers fell much less, "
     "from 4.3 percent of the burp to 1.3 percent, because the other two kinds stayed."),
    ("Which predictions failed?",
     "Three. That scraps would grow with the size of the tube. That the curled loops would fall gently as the "
     "cooling slowed. And that the leftovers which survive "
     "cooling would be seams between regions, predicted by the author. All three are reported in the paper."),
    ("What is combinatorial quantum gravity, and is this it?",
     "Combinatorial quantum gravity is a published model, built by Carlo Trugenberger, Christy Kelly and Fabio "
     "Biancalana, in which space is a network with no positions at all and the energy is a count of squares that "
     "acts as a curvature. This paper's toy is a close cousin of that model, not that model itself: it turns one "
     "knob, the price of a surplus square (λ), above the published value of 1."),
    ("Does this show what is left over from the birth of the universe?",
     "No. A toy can show that a mechanism is possible or impossible within a class of models; that is the whole "
     "claim. The clock in the simulation is a sampling rule, not physical time, and no physicist has reviewed the "
     "work."),
    ("Was AI used to write this?",
     "Yes. The author used Claude (Anthropic) and ChatGPT/Codex (OpenAI) to assist with code, analysis and "
     "writing, and is responsible for the result. The predictions for the main tests were written down before the "
     "runs."),
]

KEY_TERMS_02 = ["what is left over from before the Big Bang", "relic defect", "defects after a phase transition",
                "nucleation and growth", "annealing", "freeze-out", "dynamical arrest", "emergent spacetime",
                "combinatorial quantum gravity", "graph model of space", "metastable state",
                "isolated system (microcanonical)", "pre-registration", "Markov chain Monte Carlo"]

PAGE_TITLE_02 = ("What the change leaves behind: is anything left over from before space? A question tested in a toy "
                 "model, in plain language")
DESCRIPTION_02 = ("Is anything left over when space emerges? A plain-language physics paper counts the leftover "
                  "pieces of a sharp change in a toy model of dots and links, and follows which of them last.")
TICKER_02 = ("Three kinds of scrap", "The common one heals. What holds the others?")

POST_02 = Post(
    number=2, slug="02whatthechangeleavesbehind", title="What the change leaves behind", subtitle=SUBTITLE_02,
    published="2026-10-13", updated="2026-10-14", paper_dir="docs/papers/relic",
    pdf_plain_name="what-the-change-leaves-behind-plain-language.pdf",
    pdf_tech_name="what-the-change-leaves-behind-technical.pdf",
    opening_words="A sharp change rarely finishes the job.", cite_key="smith2026changeleavesbehind",
    ticker=TICKER_02, page_title=PAGE_TITLE_02, description=DESCRIPTION_02,
    summary=SUMMARY_02, chapter_questions=CHAPTER_QUESTIONS_02, big_questions=BIG_QUESTIONS_02, faq=FAQ_02,
    key_terms=KEY_TERMS_02,
    mascot=dict(file="meadow-calf.gif", width=272, height=204,
                alt="A line drawing of a shaggy calf stretching and burping in a meadow of flowers; in the next "
                    "frame it stands up, looking sheepish, beside the flattened patch of grass where it lay.",
                caption="Fig. 0. A burp, and the patch it leaves behind. Round cow still not available, artist's "
                        "impression. Not a torus."))

POSTS = [POST_01, POST_02]              # oldest first
