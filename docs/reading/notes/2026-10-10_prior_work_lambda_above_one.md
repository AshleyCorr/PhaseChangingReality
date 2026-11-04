# Prior-work search: has anyone run this model above its curvature value (λ > 1), or studied one ordered arrangement turning into another in it?

CLAUDE.md rule 15; TASKS, "The reviews of 6 October", item 9 (a). An earlier note,
`2026-10-09_prior_work_curled_torus.md`, transcribes the search of 23 September; this one is a fresh search made
on 10 October without sight of it, and the two reach the same verdict on the coefficient. One note for two sentences, with a separate verdict
for each.

**The sentence it backs** (where it will appear, and its exact wording): two sentences, (a) and (b).

- **(a)** The plain-language edition of paper 1 (to be written): "We have not found published work that runs this
  model with the surplus price raised above its curvature value (λ > 1)."
- **(b)** The technical paper 1, `docs/papers/curled_torus/paper.tex`, introduction (lines 56 to 58): "A different
  kind of change has not, as far as we have found, been studied in this model: one \emph{ordered} arrangement turning
  into another."

**Date of the search:** 10 October 2026, 05:04 to 05:35 ET by the clock (`date` run at the start, before writing and
at the end).

**Who searched:** an assistant agent (Claude), started fresh for this task, with read access to the repository and
web access. **The owner has not read the passages quoted here.** Quotations were taken from HTML-to-text conversions
of arXiv and ar5iv pages and from `pdftotext`; each must be checked against the PDF before it is quoted in a paper.
Nothing here was taken from private correspondence; the one point that rests on it is cited to VISION Update 18,
where it is already recorded.

**Kinds of statement** (rule 10). "Published": what a paper says, with how much of it was read. "Ours, exact":
arithmetic on the model's definitions. "Ours, unverified": a judgment about what a paper does or does not cover.

## Where was searched

arXiv has no full-text search of its own; its search covers titles, abstracts, authors and comments. The full-text
searches below are INSPIRE's (`fulltext:`) and Google Scholar's. Counts are the totals each source reported; "looked
at" means the titles were read, and abstracts or text where the next section says so.

| Source | Terms used | Results looked at | Nearest hit |
|---|---|---|---|
| arXiv (full-text search) | *arXiv's own search reaches titles and abstracts only; full text was reached through INSPIRE and Google Scholar, next rows.* `all:"combinatorial quantum gravity"` (8); `abs:Ollivier AND abs:gravity` (11); `abs:Ollivier AND abs:"phase transition"` (6); `"cycle condensation"` or `"condensation of cycles"` or `"condensation of short cycles"` (3); `"random regular graphs"` with cycles or squares and "chemical potential", "first order" or "phase transition" (6); `au:Trugenberger` (93; the 60 most recent, 2007 to 2026); `au:Biancalana AND abs:graph` (3); `au:Kelly_C AND abs:Ollivier` (1); `"hard-core" AND graphs AND squares` (14); `"emergent geometry" AND "random graphs"` (0); `Ollivier AND "Monte Carlo"` (4); `"Ollivier-Ricci" AND emergent` (12); `"random regular graph" AND "4-cycles"` (1); `decompactification` with graph, graphs, network or lattice (11); `emergent AND graph AND nucleation` (1); `metastable AND "regular graphs"` (16); `"quantum graphity"` (18) | All of them, by title; abstracts of about 30 | Nothing at λ > 1. No paper on the model newer than [T25] (v2, April 2026), as the note of 5 October found |
| Google Scholar | `"combinatorial quantum gravity"` (79 reported; all eight pages, 77 titles); the same since 2024 (20; both pages); `"Ollivier" "hard-core" "regular graphs" squares` (12; first 10); `"Ollivier curvature" "random regular graphs" "phase transition"` (18; first 10); `"cycle condensation" graphs` (321; first 10, almost all unrelated); `"Trugenberger" "local term" Ollivier` (2) | As stated | Three 2026 items not on arXiv (below), none of them a simulation of this model as far as their titles, snippets or abstracts show |
| INSPIRE-HEP | Full text: `fulltext:"combinatorial quantum gravity"` (52); the same with `metastable` (5); `Ollivier` + `"hard-core"` + `squares` (27); `"Ollivier curvature"` + `"random regular graphs"` (12); `"cycle condensation"` + `graphs` (1); `Ollivier` + `metastable` + `torus` + `"random graphs"` (2: [T24], [T25]); `"random regular graphs"` + `decompactification` (0); `"emergent geometry"` + `"random graphs"` + `nucleation` (2); `hypercubes` + `"random regular graphs"` + `"first-order"` (7); `"quantum graphity"` + `nucleation` (7); `Ollivier` + `"local term"` + `"global term"` (1: [T24]) | All, by title | [T24] and [T25] are the only texts with Ollivier, metastable, torus and random graphs together |
| Papers citing the closest source (forward citations) | [KTB19]: Google Scholar 35, Semantic Scholar 32, INSPIRE 31, OpenAlex 27. [T17] (Trugenberger, JHEP 09 (2017) 045, arXiv:1610.05934; not in `REFERENCES.bib`): 61, 50, 47, 42. [GV21]: 15, 12, 13, 15. [T25]: 4, 2, 1, 1. [T24]: Google Scholar 3, INSPIRE 2. [KTB20]: INSPIRE 22, OpenAlex 21. Kelly and Trugenberger 2018 (arXiv:1811.12905): INSPIRE 6. [T22]: INSPIRE 8. [T23]: INSPIRE 4. Trugenberger's 2023 review (arXiv:2311.17526): INSPIRE 4. [Akara21]: INSPIRE 15. [Kelly22]: Google Scholar 1, INSPIRE 0 | Every list, by title | The only citers of [T25] are a paper on symmetry breaking on graphs (Evnin, arXiv:2512.09480), one on triangles in rewired random regular graphs (Akara-pipattana and Nechaev, arXiv:2604.23152), a philosophy preprint, and the corrigendum |
| The closest source's own reference list | [T25], all 101 entries (arXiv HTML); [KTB19], all 59 (ar5iv) | All, by title | [T25] refs. 76 to 78: [Akara21], [GV21], [GKLM23] |
| Texts searched for terms | [T25] (arXiv HTML): local term, global term, saturat-, penal-, coefficient, weight, hard-core, allotrope, metastab-, torus, cylinder, compactif-. [KTB19] (ar5iv): Secs. 3.3 and 4 read; mean field, vacua, barrier, metastab-. [T24] (arXiv HTML v1): barrier, lifetime, simulat-, nucleat-, tunnel, dataset. [Kelly22] (the local PDF, 18,018 lines of extracted text): mean field, local correction, interpolat-, deform-, one-parameter, tunable, relative strength, nucleat-, decompactif-, metastab-, torus, cylinder, three squares, barrier. Kelly and Trugenberger 2018 and [KTB20] (ar5iv): metastab-, hysteresis, mean field, barrier, torus. [T17] (ar5iv, a short rendering that may be incomplete): the same. [Akara21] (PDF): Sec. 2 and the discussion. Akara-pipattana and Nechaev 2026 (arXiv HTML): quantum gravity, Ollivier, metastab-, square | As stated | [KTB19] Sec. 3.3.2 and Sec. 4; Kelly and Trugenberger 2018, Sec. IV; [Kelly22] Sec. 4.5.1.1 (all below) |
| Semantic Scholar keyword search | Five queries on the model's vocabulary | Three ran (20, 0 and 7 results); two were refused (HTTP 429) on two tries | Nothing new |
| General web search | Eight queries: the model's name with Ollivier and random graphs; the name with 2025 or 2026; Ollivier with metastable, torus, decompactification; emergent geometry with first order, nucleation, metastable; regular graphs with 4-cycles, hard-core, local term; allotropes with simulation; and the two 2026 titles below | About ten results each | [CP12]; Quach and others 2012 (below) |
| `REFERENCES.bib` and earlier notes in this folder | The comment "LITERATURE CHECK, 2026-09-23" in `REFERENCES.bib` (an earlier search for work varying the local term's coefficient: none found); `2026-10-05_trugenberger_programme.md`; `decompactification_2026-09-25.md`; `docs/papers/curled_torus/v2_changes.md`, 6 October candidates, item 4; the entries [KTB19], [T25], [GV21], [Kelly22], [T24], [GKLM23], [DQM25], [Akara21], [Roost24], [GM04], [CJR09], [BSV10], [GHR10], [BPS10] | Read | The 23 September check reached the same answer on the coefficient from fewer sources |

**Searches that could not be run, said plainly.** (1) Three 2026 items that Google Scholar lists as mentioning the
model could not be opened: an Authorea preprint (HTTP 403; its abstract was read through OpenAlex instead), an SSRN
preprint (HTTP 403; title and one snippet only) and a ResearchGate PDF (connection refused; title and one snippet
only). (2) Two of five Semantic Scholar keyword searches were refused. (3) For three Google Scholar queries only the
first page of ten was read. (4) [T24]'s citing works were taken from Google Scholar and INSPIRE only (the OpenAlex
lookup returned a different paper and was discarded). (5) Unpublished work, talks and code cannot be checked this
way, the model's author's included.

## What came closest

### For sentence (a): which settings of the surplus price are in print

**Published, as far as this search found: λ = 0, λ = 1, and, in the text of two papers, an outright ban on surplus
squares (the limit λ → ∞). Nothing at a finite λ above 1, and nothing between 0 and 1.**

- **λ = 1, the curvature value.** [KTB19], [T25], [T22], [KTB20] (three links per point), [Kelly22]. This is the
  model.
- **λ = 0, squares rewarded with no saturation.** [KTB19] Sec. 3.3.1 and its Figs. 5 and 6 (the "mean field" action on
  bipartite hard-core graphs; baby universes; read today from ar5iv). [GV21] (read in full by the owner, 21 September;
  not re-read). [Kelly22] Sec. 4.2.5 compares the mean-field and exact actions and says of [GV21] that it "studied
  combinatorial quantum gravity in the mean-field approximation" (term search of the PDF). Kochergin, Khaymovich,
  Valba and Gorsky, arXiv:2305.14416 (abstract only): random regular graphs with a reward per short cycle of each
  length, a clustering transition; again no saturation term. [T25] also cites [GKLM23] for the first-order transition
  with the global term only; its abstract (read today) is about a matrix model for spanning forests on planar graphs
  and does not mention that transition, and its full text was not read.
- **A ban on surplus squares (λ → ∞).** Two published sentences, both read today:
  - [KTB19] Sec. 4 (ar5iv text): the configuration space is restricted to graphs in which no edge carries more than
    2D − 2 squares, "We study the annealed dynamics of the mean field action ... in the configuration space" so
    restricted, where "both the exact action ... and the mean field action ... agree".
  - Kelly and Trugenberger, "Combinatorial Quantum Gravity: Emergence of Geometric Space from Random Graphs",
    arXiv:1811.12905 (J. Phys.: Conf. Ser., 2019; journal details from a search result, to verify; it is
    [KT19] in `REFERENCES.bib`, a correction made on review the same day: the search had reported it as
    missing), Sec. IV (ar5iv text; the rest of the paper was term-searched only): "we can adopt the simpler
    mean field action expressed in terms of the total number of squares provided we explicitly exclude configurations
    with more than (2d−2) squares per edge."
  - VISION Update 18 records the model's author's position that his code imposes only the hard-core rule. What the
    published text says and what the code ran are two different facts, and this search can only speak to the first.
    **New for the record:** Update 18 calls the cap "our reading of one sentence"; the same statement is in a second
    paper by the group. Whether that changes anything is the owner's call; nothing was edited.
  - *Ours, exact:* under such a ban the curled torus of paper 1 is not an allowed state at all. Its N links along
    the curled direction each carry three squares (two plaquettes and the ring of four), which is why S = 1.25 N,
    X = N and H = 4(λ − 1)N. So the ban is not a place where paper 1's change could have been seen.
- **0 < λ < 1 and finite λ > 1.** Not found. [Kelly22] varies nothing between or beyond the two actions (term search:
  no "interpolat-", "one-parameter", "tunable", "relative strength" in that sense). [T25]'s Eq. (22) has no
  coefficient on the local term, and the letter λ appears in the review only as a Compton wavelength.
- **Other energies that keep geometry from collapsing, in other models (not this one).** [Akara21] (Sec. 2 and the
  discussion read today from the PDF): exponential random graphs with no hard constraint, a two-star term plus a
  reward for each pair of points joined by exactly two paths of length 2; the authors say what they took from CQG is
  the hard-core condition, not the curvature energy. It has no local term to weight. Akara-pipattana and Nechaev,
  arXiv:2604.23152 (term search of the HTML): triangles in rewired random regular graphs under a fugacity that
  happens to be called λ; a long-lived clustered state and hysteresis between random and clustered; cites [T17] and
  [T25] only for the use of Ollivier curvature. [Lamas25] (abstract only): a different Hamiltonian with a
  connectivity term. Tee and Trugenberger, arXiv:2102.12329 (abstract only): when Forman and Ollivier curvature
  agree; no simulation of a family between them that we could see from the abstract.
- **Seen by title, snippet or abstract only, and not this model as far as those show.** Agourakis and Gerenutti, "A
  Finite-Size Crossover in Ollivier–Ricci Curvature of Random Regular Graphs" (Authorea, 2026; abstract via OpenAlex):
  the sign of the mean curvature of random regular graphs against density, no dynamics. Euzen, "Quantum Relational
  Substrate ..." (SSRN, 2026; snippet only): its own framework, "related to spacetime-allotrope ideas" in CQG.
  Vassallo, "Emergent Spacetime and Protomatter from Ollivier–Ricci Flow with Discrete Cartan Torsion" (ResearchGate,
  2026; snippet only): a Ricci-flow model on simplicial complexes. Oggad, arXiv:2606.20943 (abstract): concentration
  of measure and spectral dimension.

### For sentence (b): one ordered arrangement turning into another

**Not found: any simulation, measured rate, counted barrier or front for such a change in this model. Found: two
places where the question is addressed in words, and one where it is said to be open.** The sentence's word
"studied" is therefore too broad as printed.

- **[KTB19] Sec. 3.3.2 (read today, ar5iv text): an argument that one such change cannot happen at λ = 1.** The paper
  sorts states by squares per point: below 1, exactly 1 (the flat lattice among them), above 1. It asks whether a
  flat vacuum, once reached, can go on to "some classical solution" of the over-full kind, and argues that "one
  requires an infinite number of Glauber transitions (edge switches)" to get from one to the other as N → ∞, "akin
  to the existence of an infinite energy barrier". Its Figs. 6 and 7 are quenches from random starts under the two
  actions, not a prepared ordered state followed as it changes. *Ours, exact:* the curled torus has 1.25 squares per
  point and a non-zero share of over-full links, so it belongs to the paper's over-full class. *Ours, unverified:* this
  is the nearest thing in print to paper 1's question. It is a heuristic argument (the paper's word), at λ = 1 where
  the two arrangements cost the same, about the opposite direction (flat to over-full), with no rate, barrier count
  or simulation of the change. It does not do what paper 1 does, but it does discuss a change between ordered
  arrangements, and paper 1's introduction should cite it.
- **[T24] and [T25] (dark-matter sections; [T24] term-searched today, read in full by the project on 19 September):
  a proposal, no calculation.** Domains of one ordered tiling inside another, "metastable states, which, depending on
  their free energy difference to the minimum and the barrier in between, may be extremely long-lived" ([T25]).
  [T25] also describes, in words, the free-energy minimum moving from one number of squares per point to another as
  the coupling falls. [T24] states that "no datasets were generated or analyzed". No barrier, lifetime or conversion
  is computed in either. Paper 1 already cites [T24] in the sentence after (b).
- **[Kelly22] Sec. 4.5.1.1 (read today from the PDF text): the relevant class of states is said to be unexplored.**
  After classifying the flat states with two squares on every link (torus or Klein bottle), it turns to four-link
  graphs whose links carry two or three squares, which is the class the curled torus is in (*ours, exact*), and says:
  "The author is not aware of any relevant existing results in the literature and has not managed to establish any
  of utility." Sec. 4.5.2 adds that "a more complete study of the classical phase of 4-regular combinatorial quantum
  gravity is perhaps desirable". This supports sentence (b) as of April 2022.
- **[GV21]** (not re-read): hysteresis between the random phase and the phase of hypercubes; random to ordered, not
  ordered to ordered. **[KTB20]** (term search): the ordered states with three links per point are prism graphs and
  Möbius ladders (strips of squares closed into a cylinder or with a twist); which of the two appears is discussed,
  and no change from one to the other is followed.
- **The decompactification relatives of `v2_changes.md` item 4, all about other models** (abstracts re-read today
  from arXiv; the full texts were searched by an assistant agent on 25 September, not today). [GM04]: a vacuum with
  curled extra dimensions is unstable to their opening, by tunneling or thermal fluctuation over a barrier.
  [CJR09]: de Sitter space nucleates regions with a different number of large dimensions. [BSV10]: tunneling
  between vacua of different effective dimension in a six-dimensional Einstein–Maxwell theory. [GHR10]: one curled
  direction opens from a lower-dimensional parent, with an observable signature. [BPS10] (abstract only, then and
  now): the same idea with its inflationary spectrum. Each of these **does** study one ordered arrangement turning
  into another, several with a barrier or a nucleation rate, in continuum gravity or field theory; none is a graph
  model and none is this model. So sentence (b) is only true with its words "in this model", and the outlook is the
  place to cite them.
- **Graph models other than this one** (abstracts only unless said). [CP12]: a first-order change from disorder to
  a nearly two-dimensional manifold. Quach, Su, Martin and Greentree, "Domain structures in quantum graphity",
  arXiv:1203.5367: metastable domain structures and defects at domain boundaries in the ordered phase of quantum
  graphity, made by annealing. Wilkinson and Greentree, arXiv:1506.07588: the "ripening" of a quantum-graphity graph
  into disjoint pieces unless a term is added. Spector and Schwartz, arXiv:1808.05632: a lattice ground state whose
  defects "behave like particles of quantized mass that attract one another". These are about ordered phases with
  domains or defects in other energies; none follows one ordered arrangement converting into another with a
  different number of large directions. None is in `REFERENCES.bib` except [CP12].

## Verdict

**(a): NOT FOUND, for any finite λ above 1** (and for any λ between 0 and 1). The sentence may be written, linking
this note, **on condition that it does not hide the ban**: a prohibition is the limit of raising the price, and two
papers describe one in their text. Wording the search supports:

> We have not found published work that runs this model with the surplus price set to any finite value above its
> curvature value (λ > 1). In print, as far as we have found, are the price at zero (λ = 0), the curvature value
> itself (λ = 1), and, in the text of two papers, a version in which surplus squares are forbidden outright.

A shorter form, if space is tight: "We have not found published work that sets the surplus price anywhere other than
zero, its curvature value, or (in the text of two papers) an outright ban." The sentence exactly as drafted, with
"raised above its curvature value (λ > 1)" and no mention of the ban, is not recommended. Whether a public page also
says that the model's author states his code has no such ban is the owner's decision: that statement reached the
project by private correspondence (VISION Update 18; CLAUDE.md, "Public-facing work").

**(b): NOT FOUND, for a simulation, a measured rate or a counted barrier of such a change in this model; but the
word "studied" must go.** [KTB19] Sec. 3.3.2 argues about one such change (and concludes it cannot happen at λ = 1
in a large system), and [T24] and [T25] propose long-lived ordered domains without computing anything. Wording the
search supports, for `paper.tex`:

> A different kind of change has not, as far as we have found, been simulated or measured in this model: one
> \emph{ordered} arrangement turning into another. It has been discussed: Ref. [KTB19] argues (its Sec. 3.3.2) that
> at λ = 1 a flat vacuum cannot reach an over-full one in a large system, and Refs. [T24, T25] propose long-lived
> domains of one ordered arrangement inside another; neither simulates such a change.

and for a plain-language page: "We have not found a published simulation of one ordered arrangement turning into
another in this model." The sentence must keep "in this model": in continuum gravity such changes are well studied
([GM04], [CJR09], [BSV10], [GHR10]).

What would change the verdict (a search not yet done, a paper not yet read):

- Reading in full what was only searched for terms or read as an abstract: [Kelly22] (347 pages; a parameter on the
  local terms could be written in a way the term search missed), Kelly and Trugenberger 2018, [KTB20], [GKLM23],
  Kochergin and others 2023, Tee and Trugenberger 2021, and the Russian-language thesis of D. Kochergin (2024) that
  cites [GV21].
- Opening the three 2026 items that could not be fetched (Agourakis and Gerenutti; Euzen; Vassallo).
- The model's author or his co-authors saying that such runs exist unpublished. This is a question for him, to be
  asked when the owner chooses (her decision of 5 October puts requests to him after his talk).
- A check of the quotations above against the PDFs, by the owner or a physicist, before either sentence is posted.
- A new paper on the model. As of today none newer than [T25] was found on arXiv, INSPIRE or Google Scholar.
