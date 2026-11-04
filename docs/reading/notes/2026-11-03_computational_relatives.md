# Reading: three "the universe is a computation" relatives of the Klein-bottle computer

Read 2026-11-03 by a fresh assistant agent, prompted by the owner's idea of today (her hypothesis, not a result):
the inner two-dimensional shell of a black hole, a Klein bottle of one-sided dark-matter loops around a
zero-dimensional knot, is a computer whose symbols are the allowed (curl, twist, spin) sets under her rules, and "the
simulation plays out all the options in probabilities." Read against the claim's steps 1 to 10
(`docs/papers/programme_draft.md`) and VISION Updates 17, 20, 21. Targets, never evidence (rule 13); every physics
judgement is ours, unverified. All three texts came from PDFs via PyMuPDF; check quotes against the pages.

| Work | Read status | Source | Steps | One-line fit |
|---|---|---|---|---|
| Gorard 2020 [Gorard20] | In part: Sec. 2.2, 2.3 (eqs. 25 to 30), 3.4 (Bell), plus the 2026-09-23 reading | Publisher's PDF, local copy in `docs/reading/` | 10, 8 | Closest relative of "all options at once"; phases put in by hand, Bell met by non-locality across branches, not by counting |
| Lloyd 2005/2018, arXiv:quant-ph/0501135 v9 | In part: abstract, intro, Sec. 1.1, 1.2, 3.1 to 3.4 openings, 3.7, 4, 5 | arXiv v9 PDF | 10, 8, 3 | The computer whose events *are* spacetime, quantum at the gates; her versions are his histories, made classical by decoherence |
| 't Hooft 2014/2016, arXiv:1405.1548 v3 [tHooft14] | In part: contents; Sec. 1.1, 1.2, 2.1, 3.5, 3.6, 3.6.2, 4.1, 4.3, 6 opening, 10.3, 10.3.1 | arXiv v3 PDF, 259 pp. | 1, 10 | A classical-symbol computer under quantum mechanics, as hers is; the price is one real state and superdeterminism |

## 1. Gorard

**Core claim** (the 2026-09-23 bib note needed no correction). A hypergraph is rewritten by a rule; where the rule
could fire in several places the model follows every order at once (a "multiway system"). Each branch is a basis
state, a cut across branches a superposition (eq. 25). Measurement is "completion": extra rules collapse equivalent
branches to one (eq. 28). Probabilities are squared "path weights" (eq. 29), complex because the multiway causal
graph is argued to need a metric with three signs (p. 567): the phase is argued, not derived from counting.

**Bell.** Two non-local routes: Wolfram's long-range edges joining a particle pair outside 3+1 space; and Gorard's
own, causal links between events on *different branches*, "an explicitly nonlocal theory", violating CHSH "in much
the same way as ... the de Broglie-Bohm" theory (pp. 587 to 588).

**Fit.** Step 10 word for word: her versions are his branches, "plays out all the options" is the multiway system.
Step 8's "simple rules make the whole pattern" is the Wolfram project's premise. The nearest published relative of
the Klein-bottle computer. It would require a rule that can fire in several places, a norm carrying a phase (Update
21's missing ingredient, Hardy's axiom), and correlations across branches, outside space. It does not give
probabilities from counting versions; Update 20 (counting is an instruction set, S <= 2) stands.

## 2. Lloyd

**Core claim.** "What is quantized here is information." At each gate the qubits either scatter (an event) or do
not; each pattern of events is a "computational history" with its own causal structure and action, and the
computation is a superposition of 2^n of them (eq. 1.1). Each history embeds as a discrete spacetime obeying the
Einstein-Regge equations (Sec. 2). Classical spacetime emerges by Gell-Mann and Hartle's decoherent histories (Sec.
3.7); black-hole evaporation is unitary or nearly so (Sec. 3.2); "as soon as a newly minted bit interacts with
another bit, a new volume of spacetime is created". Which computation is ours is open (Sec. 4); one candidate weighs
programs by length, so simpler computations weigh more.

**Bell.** Not discussed. He need not: the symbols are qubits and the gates carry amplitudes, so quantum mechanics is
an input.

**Fit.** Step 10: the versions are his histories, and the rule she adopted (amplitudes plus a record in the
surroundings) is his Gell-Mann-Hartle step. Step 8: program-length weighting is the published form of "simple rules
make the pattern". Step 3: new volume where a new bit first interacts is a relative of her daughter opening in the
zero-dimensional space (a target only). It requires quantum symbols carrying amplitudes; classical symbols with
probabilities put her with 't Hooft.

## 3. 't Hooft

**Core claim** (abstract only before, 2026-09-22). Quantum mechanics is "an instrument to statistically describe a
world where the physical laws, at their most basic roots, are not quantum mechanical at all" (Sec. 1.2). Underneath
is a deterministic automaton whose "ontological states" form a special basis; the universe is in exactly one "with
probability 1", and the Schrodinger equation only permutes them (the cogwheel, ch. 2). Superpositions are
"templates", "not ontological" (Sec. 2.1); the Born rule is inherited from the initial distribution of ontic states
(Sec. 4.3); observables diagonal in that basis are "beables".

**Bell.** Superdeterminism (Sec. 3.6, 3.6.2, 10.3): the detector settings have roots going "all the way to the Big
Bang"; the source's ontic state carries "non-local information about the future", never harvestable; an "ontology
conservation law" removes counterfactual settings. He concedes an "apparent non-locality" may remain (10.3.1) and
that no ontological basis for the Standard Model has been found (Sec. 4.1). Black holes enter only through the
holographic bound.

**Fit.** Step 1: a finite table of (curl, twist) symbols is what he calls beables, and a computer over classical
symbol sets is his cellular automaton, the nearest of the three in kind. Step 10 is the tension: "versions all stand
at once" is Everett's reading, which he rejects. Her phrase "plays out all the options in probabilities" sits on the
fork: all options real is Gorard and Lloyd and needs amplitudes; one real and the rest bookkeeping is 't Hooft, paid
for with superdeterminism. Update 20 found the toy's counting is a classical instruction set; 't Hooft is the
published position that accepts that and keeps determinism.

## 4. Cheapest next steps (ours, unverified)

1. **Exact, no run, serves all three.** On the complete state spaces at N = 16 and 18 (`small_graphs.py`), build the
   multiway graph of the switch move from one state, count the paths reaching each class after k steps, and compare
   with Update 17's renaming counts. If they agree, the versions are Gorard's path weights without phase; if not,
   "all the options" and "versions" are two different counts and she must say which she means.
2. **Exact, from her symbol table.** Does the rule send each allowed set to one next set ('t Hooft's cogwheel) or
   to several (branching)? That decides which relative she has before anything is built.
3. **A decision owed under S1.** Amplitudes on the options (Lloyd) or probabilities ('t Hooft); step 10 says
   amplitudes.

## Proposed REFERENCES.bib entries (not edited into the file)

```bibtex
@article{Lloyd05,
  author  = {Lloyd, Seth},
  title   = {A theory of quantum gravity based on quantum computation},
  year    = {2005},
  eprint  = {quant-ph/0501135},
  archivePrefix = {arXiv},
  note    = {arXiv v9, 2018-04-10, 58 pp.; no journal reference on arXiv (to verify). Read in part 2026-11-03 (sections in docs/reading/notes/2026-11-03_computational_relatives.md). A quantum computation as a superposition of histories, each a discrete spacetime obeying the Einstein--Regge equations; classical spacetime by decoherence. Bell not discussed: the symbols are qubits. Popular book: Programming the Universe, Knopf 2006 (not read).}
}
```

For `tHooft14` (in the file, abstract only), a replacement note: *Read in part 2026-11-03 (sections as in the note's
table) from the arXiv v3 PDF; Springer book 2016 (pagination to verify). One real ontological state at a time,
superpositions are templates; Born rule inherited (Sec. 4.3); Bell met by superdeterminism (Sec. 3.6.2, 10.3). Our
fork remains contextuality, not superdeterminism.*
