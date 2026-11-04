# The series: what each paper claims, what is established, what is missing

Written 23 September 2026, night, at the owner's request ("work through each of the other pieces with
similar detail; where needed design new experiments and predictions"). One entry per paper, in the
order they depend on each other. Every claim is at the level of the model; what it would mean for the
universe is the owner's framework, stated in paper 7 with its status beside each piece.

**Status words, used the same way throughout.** *Established*: pre-registered, run, verdict on the record.
*Exact*: arithmetic or exhaustive enumeration, no sampling. *Failed*: pre-registered and the prediction did
not hold. *Designed*: pre-registered or drafted, not yet run. *Idea*: the owner's, recorded, no test yet.

---

## Paper 1. Metastable curled tori and their front-driven decay (drafted)

**Claim.** In the graph energy of combinatorial quantum gravity with its local term's coefficient raised
above 1 (not CQG itself, which is λ = 1 only; the author's condition of 24 September), a torus with one
direction curled shows activated escape and rapid conversion in counting observables. The initial gap is exact;
the release subtracts the final energy. A dominant connected component does not prove a unique front,
and the global seed scan measures escape. About one relic remains in the tested cold sealed boxes.

**Status.** Established at λ = 1.25 (T7, T9, T10, T11); exact results for all λ; the λ map (T8, O38) run and
written into section V: sharp wherever the torus is stuck (1.05 to 1.35), edge between 1.35 and 1.40, verdict
inconclusive by the letter for a stated reason (the memoryless-wait criterion was too tight).

**Current status, 9 October.** Unpublished reader manuscript, with supplement and reproducibility manifest.
Submission or endorsement logistics do not imply scientific review. See [measurement scope](measurement_scope.md).

**Draft.** `docs/papers/curled_torus/paper.tex`, `paper.pdf`; review page
https://claude.ai/artifact/3TcXFYr51m955zTkgitXwR. *Added 10 October:* the plain-language edition,
`paper_plain_language.pdf`, whose chapters the manuscript's sections now follow, and its interactive companion,
"The Tube's Curling Ladder", https://claude.ai/artifact/F4sBf955J5ZT7rCq8uebW8 (source
`docs/public/curling_ladder_tube.html`). The edition closes by previewing short papers 2 and 5 of the list
below ("What the change leaves behind"; "How curled directions open").

---

## Paper 2. The leftover: size, number, and what makes it

**Claim to test.** The change leaves remnants of a fixed size, and their number is set by how the change
started (per seed), not by the size of the space. That is the model-level version of "dark matter is the
scrap of the change, abundant through many seeds" (VISION Update 16).

**Established.** One leftover per converted tube, however large (T10, ONE RING, HOWEVER LARGE, N up to 288);
its position is uniform, neither at the seed nor where fronts meet (T11, NEITHER); read from its wiring, it
is one column of the tube left curled, costing 24λ − 16, occasionally a two-point twist (20 units) or a
combination of small defects (8 to 45 units) (O13, O15, O28).

**Exact, new tonight (T8 pre-registration).** Near λ = 1 the column keeps a large share of the lump:
(24λ − 16)/(4(λ − 1)N), 22 % at λ = 1.25 and 72 % at λ = 1.05 for N = 64, falling with N.

**Established the same night (T17, O33): BETWEEN.** One to eight planted seeds in one tube leave 1.10, 2.15,
2.50, 3.40 leftovers on average; slope 0.29 ± 0.04 per extra seed. More seeds, more scrap, but fewer than one
each; both the owner's ONE PER SEED and our ONE PER TUBE failed. Next: where do the extra leftovers go when
fronts meet (merge, anneal)?

**Still to design.**
- *Does a leftover move?* Needed by papers 2 and 5. A column in a flat sheet, fixed coupling, followed for
  a long run; record its position every block. Cheap. Draft below (T19).
- *Does the leftover's share depend on λ as the exact formula says?* Read from T8's saved final states.

**What would have counted against the owner's picture, and did not happen.** ONE PER TUBE in T17. What did
happen is weaker than her hypothesis: the scrap grows with the number of seeds, but less than one per seed.

---

## Paper 3. The fertile window: which λ makes a lump, a stuck X, and a clean birth

**Claim to test.** There is a window of λ above 1 in which X is stuck for now, the change releases a lump,
and the new space can take the lump without melting; the owner's λ ≈ 1.25 sits inside it. With the rules
fixed between generations (the owner's decision of 23 September), the reason our λ is where it is would be
counting across loops, not evolution: a universe drawn at random belongs to the most prolific loop.

**Established.** At λ = 1.25: all of paper 1, and T9 (BONFIRE WITH A THRESHOLD: the new space survives only
if its surroundings can hold the lump; the needed room scales with N). At λ = 1.5: X is not stuck (T7).

**Running.** T8, the λ map of stuckness, sharpness, release and scrap.

**Measured the same night (T18, O34): PROPORTIONAL.** The room needed is 12, 36, 72 stores at λ = 1.10, 1.25,
1.40 (N = 96): it grows faster than the lump. A bigger λ gives a bigger lump and a harder birth.

**Honest limit.** The model has no black holes, so it cannot count offspring. It can map the ingredients
(stuckness, push, lump, room, scrap); fertility itself is the owner's framework.

---

## Paper 4. Closed regions: can concentrated energy re-curl space? (the black-hole piece)

**Claim to test.** In a closed region, energy concentrated into space can fold it back towards X (a black hole
in the owner's picture is a re-curled region that can "burp" a new space when it gets the activation energy).

**Established, and against.** O20: a sealed sheet given a budget of energy melts rather than folds, at every
budget and every λ tried; with named points. The exact move census (corrected 23 September): from perfect
space no single move adds a square, so a fold must run through a seam.

**Why it is the hardest.** The owner's theory has points interchangeable, and under that weighting a complete
fold (a whole closed piece, e.g. a 4-cube with 192 symmetries) gains enormously while a partial fold gains
nothing. That is the one route by which folding could win, and it is exactly the one named-points runs
cannot see. Doing it properly needs the symmetry count of every proposed state, which costs 0.1 to 24 s per
count at N = 160 with the present tool (Q15).

**Designed, and the blocking tool now exists.**
1. ~~*Exact, small.*~~ *Withdrawn the same night:* at N = 16 and 18, the only enumerable sizes, there is no flat
   sheet at all (the 4 x 4 torus is the 4-cube; the first flat torus is 6 x 6 = 36), so fold against melt cannot
   be posed there.
2. *A fast symmetry count: done (Q20).* igraph counts in about a millisecond where the old counter took up to
   45 s, agreeing exactly; a chain with interchangeable points built on it reproduces the exact interchangeable
   averages at N = 18. Per-move counting is affordable up to about N = 64.
3. *Next:* pre-register the closed-region test with the owner's prediction (her statement: open space spreads
   and heals; a closed region folds): O20's sealed sheet at N = 36 and 64, λ = 1.25, the same budgets, run with
   named and with interchangeable points side by side; the reading is the folded share of the damage.

**Measured the same night (T21, O36): MELTS EITHER WAY at N = 64**, the size where complete folds exist; a
fragile lean towards folding at N = 36 under the bath. The owner's closed-region fold is not seen for energy
spread through a box. The untested route is a local push with cold space around it.

---

## Paper 5. Gravity as the wish to refold: do leftovers pull on each other?

**Claim to test.** Leftovers attract at a distance through counting (entropy), and the pull depends only on
their energy, not their wiring (the equivalence principle). The owner allows kinds of dark matter to differ
slightly; ordinary matter must not (measured to one part in 10¹⁵).

**Established, exact.** No energetic force at a distance: defects sharing no square are exactly additive in
energy, with an attraction of 16 only where they touch (O22). So anything gravity-like must come from
counting.

**Designed tonight.**
- T19, *mobility*: does a leftover column move through the sheet at fixed coupling, and how far? Without
  motion there is no pull to measure by watching; with it, two columns can be released and their separation
  followed.
- *The interchangeable-points pull, exactly.* For two identical defects in a torus, the interchangeable weight
  of each placement is its symmetry count. Computable exactly for every separation. Our expectation: the
  count is large only at special placements (exactly opposite, aligned), so the "force" it makes is a set of
  preferred symmetric placements, not a smooth attraction. If so, that is a clean negative for the simplest
  version of the mechanism, and it should be said.
- **Run the same night (O32), and it came out that way:** at 18 separations in a 12 x 12 torus, two identical
  simple defects have 1 or 2 symmetries at every ordinary distance and 4 only when exactly opposite. No pull;
  a factor-2 preference for maximum separation. What remains open is the entropic force when the surroundings
  are free to rearrange, which the interchangeable chain (Q20) can now measure.

**What would count against.** Immobile leftovers, or a pull that depends on wiring for defects of equal energy.

---

## Paper 6. Quantum behaviour from counting versions

**Claim (VISION Update 17).** Superposition is the set of undetectably different versions of the world;
measurement is an interaction that makes them distinguishable.

**Status, plainly: the weakest leg.**
- Rung 0 (T15): INCONCLUSIVE by the letter; versions exist in the parts, and in a world with three different
  things there is exactly one, as in a melt.
- Rung 3a: CLASSICAL for every arrangement, by argument. Counting versions of one fixed arrangement is a
  hidden-variable theory and cannot violate Bell. The owner's prediction failed.
- Rung 0b: the loop of four is not a resonator in the sheet. Prediction failed.

**What the reading of Smolin adds (23 September).** His real-ensemble quantum mechanics ([Smolin11]) gets
Schrödinger's equation from real copies that *interact* (a non-local copy rule) and carry a *phase*. Those
are exactly the two ingredients the counting lacks. His "classical because it has no copies" matches rung 0.
And there is a prediction that separates the owner from him: a system unique in the universe but isolated
and symmetric should behave quantum-mechanically for her and not for him.

**Next, needing no decision.** Rung 1: do the chain's time-fractions equal the counts in the interchangeable
ensemble at N = 16 and 18? Validation, exact. Draft pre-registration below (T20).

**Needing the owner's decision first.** Whether to add an interaction among versions (Smolin's route, a new
rule, VISION decision under S1); whether to redefine the object as the cap; whether Part V of the public page
is restated.

---

## Paper 7. The framework, for the public page

**What it is.** The owner's claims 1 to 6 and the cosmological cycle, each with a status box: which paper
tests it, the verdict, and what would count against it. Told in her words, with every claim saying whose it is.

**What it must not be.** A claim that the framework is established. Today one piece is established (paper 1),
several are designed or running, and the quantum leg has failed as stated. VISION S5 (a physicist has read it)
is not met.

**Can it be published within a week?** Paper 1 as a preprint: yes, if Carlo or another physicist endorses.
Papers 2 and 3: possibly, if T17, T18 and T8 come out clean and the owner reviews them. Papers 4 to 6: not
honestly within a week; they can be published as a *programme*, with their tests named and their current
status shown, which is a stronger thing to publish than a framework stated as settled.

---

## Draft pre-registrations written tonight

T17 (leftover per seed) is committed and running. T18 (room needed against λ) is committed before its runs
(see PREREGISTRATION.md). T19 (leftover mobility) and T20 (quantum rung 1) are drafted in this file only, for
the owner's predictions in the morning; nothing is run under them.

### T19 (pre-registered and run): ANNEALS at every coupling (O35)

Take the final states of T10 or T17 that hold exactly one curled column in an otherwise flat torus. Continue
each at λ = 1.25 and g = 1.5 (and a warmer g = 2.0) for 100,000 sweeps, recording every 500 sweeps the
columns of the vertices at d = 1. Observable: the column position's mean squared displacement against time,
and whether the leftover survives. Predictions for the owner: (a) it moves (diffuses) at g = 1.5; (b) it
moves only at the warmer coupling; (c) it does not move before it anneals away or at all. Ours: (c) at
g = 1.5, since T10's leftovers stayed in place over 30,000 sweeps in a cold box; unknown at 2.0.

### Quantum rung 1 (pre-registered as T15 rung 1): AGREES (O37)

As `docs/design/quantum_loop_design.md` rung 1: at N = 16 and 18, the interchangeable-ensemble probability of
each class (exact, from the counts) against the chain's long-run time-fraction with the renaming correction
applied per recorded configuration. Prediction (validation, not physics): agreement within sampling error.

---

## Night of 24 September 2026: what was launched, and who to ask, per paper

Written at the owner's request to move the programme's pieces to measured status, with the predictions inferred
from her stated positions (marked so in `PREREGISTRATION.md` until she confirms them). American spelling from
here on, at her request.

### Launched tonight (all pre-registered first)

| Test | Paper | Piece | What it settles | Where it runs |
|---|---|---|---|---|
| T24 | 1 (and the programme's piece 2) | 2 | The λ map with the energy check read from each decay's saved wiring, so a thermal excitation cannot fail a cell | AWS Batch, 28 jobs, via the new push-triggered queue |
| T25 | 2 | 6 | The scrap race: does the leftover freeze in when the box cools faster than it heals | laptop, 6 jobs |
| T26 | 4 | 8 | The local spark: energy in one vertex's store, or disorder in one patch, in a cold sealed sheet; named and interchangeable points | laptop, 8 jobs |
| T27 | 4 | 8 | T21's melt with its energy draining at four leak rates: does it fold before it flattens | laptop, 5 jobs |

### Who to ask, with what data (public names, public questions; the private correspondence stays in `docs/outreach/`)

The owner's rules apply (memory, 23 September): one narrow question the recipient can answer, a bias toward
UVA so a coffee is possible, the AI disclosure sentence, and a check of the recipient's 2024 to 2026 papers
before the letter is drafted. Nothing below has been sent; each is a candidate with the question and the data
that would go with it.

| Paper | Candidate | The question, in one line | The data to attach |
|---|---|---|---|
| 1, curled torus | Marija Vucelja (UVA Physics; physics of sampling, Metropolis dynamics on graphs) | For a single-move barrier under a local switch chain, is a transmission coefficient near 0.6 that does not change with the barrier height what she would expect, and is there a standard way to estimate it from the move set? | T22's table (O39): first exit on time in all six cells, κ = 0.56 to 0.68 from λ = 1.05 to 1.25 |
| 2, the relic | Carlo Trugenberger (the model's author; he named the lifetime of allotropes as his question) | How long does a stuck region last at λ = 1, and does a cooling schedule freeze it in? He asked the first half; T19 and T25 answer it for the relic at λ > 1 | T19's annealing times (O35) and T25's freeze-out time, once read; the allotrope's adjacency list is already requested (his reply pending) |
| 3, the fertile window | Diana Vaman (UVA Physics; composite and emergent gravity) | The λ deformation is the published curvature plus a penalty on a third square per edge: is there a continuum counterpart to adding such a local term to a discrete Einstein–Hilbert action, and what would it do at large scale? | The exact edge-by-edge form H = 4 Σ_e [(2 − S_e) + λ (S_e − 2)₊], the dimension ladder (O41), and T18's room-against-λ (O34) |
| 4, the black-hole piece | Kent Yagi (UVA Physics; with Brustein and Medved treated the black-hole interior as a different state of matter) | If a collapsed region were an ordered re-curled arrangement rather than a random one, which observable in ringdown or tidal response would notice the difference? | T26 and T27's verdict (fold, melt, or heal), once read; O20 and T21 as the spread-energy baseline |
| 5, curve-first gravity | Diana Vaman, or Adam Solomon (UVA; modified gravity, Love numbers) | Held until the three-dimensional model passes its reproduction gate; no 2D number is carried into a claim about gravity (the owner's decision) | O22 (no energetic force at a distance, exact) and O32 (no counting pull at fixed wiring) as the negative baseline |
| 6, quantum from counting | Samson Abramsky (UCL) or Rui Soares Barbosa (INL) for the argument; Trey Boone (UVA Philosophy) for its form | Is the argument of rung 3a sound as stated: that a renaming of a fixed arrangement fixes or swaps each loop's pairs at once and so is a noncontextual instruction set? | PREREGISTRATION T15 rung 3a and O29; the exact renaming counts of Q15 and O28 |
| companion, six links (piece 11) | Carlo Trugenberger | [T22] Fig. 3's protocol, its Boltzmann weight and whether its graphs are bipartite (held for his next reply; one question at a time) | O43: our curve under both readings of the axis, and the shapes |
| piece 12, the allotrope | Eryk Kopczyński (drew [T25] Fig. 9; RogueViz) | The adjacency list of the drawn allotrope | The exact statement that if no edge carries three squares the local term vanishes and its stability does not depend on λ |

### Status of each paper after tonight's launches

- **Paper 1:** drafted; T24 will replace T23 in section V if it reads clean, and the memoryless claim then rests on
  a run whose gates were all fixed before it ran.
- **Paper 2:** T10, T11, T17, T19 on the record; T25 running. Drafted tonight (`docs/papers/relic/paper.tex`) with
  T25's section left as the pre-registered question.
- **Paper 3:** T8, T18, T22, T23 on the record; O41 exact. Drafted tonight (`docs/papers/fertile_window/paper.tex`).
- **Paper 4:** O20, T21 on the record; T26 and T27 running. Not drafted: its first result is tonight's.
- **Paper 5:** waits on the six-link gate. **Paper 6:** waits on the owner's decision about a rule for canceling
  versions. **Paper 7:** the programme page, kept current.

---

## The gravity path (paper 5), sketched 24 September 2026, night, at the owner's asking

The owner's test, in her terms: **does curved space move matter?** Everything below is design; nothing is
pre-registered or run, and the dynamic rungs wait on the six-link gate (Gate C′). The exact rungs need no gate.

1. **Exact, no run: the energetic interaction of two relics in a flat 3D torus.** The 2D result (O22) is that H is
   a sum of per-edge terms, so two defects sharing no square are exactly additive, with a term of 16 only where
   they touch. The same arithmetic on a 6 × 6 × 6 torus with two curled columns at every separation. Expected:
   the same (no force at a distance). A pull here would be a surprise and would change everything below.
2. **Exact, no run: the counting pull at fixed wiring.** O32's measurement in 3D: the symmetry count of the torus
   with two identical relics at every separation, with the fast counter (Q20). Expected, as in 2D: a count that
   is large only at special placements, so no smooth attraction. Both rungs are a day's work and settle whether
   anything gravity-like can come from the arrangement alone.
3. **T28, the dynamic pull (the real test; pre-register with her prediction first).** Two relics in a 3D torus
   (6 × 6 × 6 or 8 × 8 × 8) at λ in the 3D window (1 < λ < 1.2, O41) or at 1.25 for one curled direction, held at a
   coupling where the sheet rearranges but does not melt (the first job is to find that window, as the programme
   says), with interchangeable points (her theory's weighting; per-move counting is affordable to about N = 216
   with igraph at a millisecond a count). Observable: the distribution of the separation over a long run, which
   gives the potential of mean force F(r) = −g ln P(r) up to a constant. Predictions on offer: (a) F falls with
   r decreasing (a pull), and by the same amount for two relics of equal energy but different wiring (the
   equivalence principle); (b) F is flat (no pull); (c) F rises (a push). What would earn the name gravity, from
   the programme: attractive, growing with the energy present, the same for every kind of relic, and weakening
   with distance roughly as 1/r² in 3D. A named-points control runs beside it, so that any pull can be
   attributed to counting or to the energy.
4. **The curling half, her way round: does the space around a relic curve using the relic's energy?** Read on
   the same runs: the local-dimension census and the square deficit as a function of distance from a relic,
   against the flat torus. If the deficit reaches beyond the relic's own vertices and scales with its energy,
   that is curvature sourced by matter, in the model's terms.

What blocks step 3 today: the six-link kernel's reproduction gate (rule 2), and a coupling window for a 3D sheet
that rearranges without melting, which nobody has measured. What does not block steps 1 and 2: nothing.

### 25 September, early morning: verdicts and the 3D launch

- **Paper 2:** T25 read, FREEZES IN, t* = 30,000 sweeps (O47). The paper's section 7 can now be written from the record.
- **Paper 4:** T27 read, STAYS MELTED (O48); T26 at 21 of 22 cells, all MELTED [corrected: 27 cells, 26 melted, one healed; O64], the owner's confirmed prediction heading
  for a fail. The next version of the black-hole question is three-dimensional (the fold she means is of three
  intertwined directions), after T30 shows what a curled region looks like at six links.
- **Piece 11 / the companion paper:** the owner's decision to proceed (VISION Update 24); T30 and T32 on Batch with her
  predictions; the exploratory window scan for T28 on the laptop. The gravity test's exact rungs need no gate.

### 25 September, morning: the four-direction test, and T26's verdict

- **T26 read: MELTS** (22 of 22 cells; corrected, O64: 27 cells, 26 melted, one healed). The owner's confirmed prediction failed. Piece 8's next form is six-link.
- **Piece 13 (VISION Update 25):** time as a fourth curled direction. T33 on Batch with her prediction (a tied pattern:
  three together from the three-curled start; four together or a singleton then three from the gas) and ours (one at a
  time). Exact before the runs (O50): the walls rise from rung to rung (doubling exactly at λ = 1.25 only) and the flat state's wall is 128, so a hot bath
  can drive a cascade in four dimensions without melting the flat state. A companion paper for the four-direction result
  would follow the three-direction one; neither is drafted.

### 25 September, morning: the overnight Batch results, read

- **Paper 1 / piece 2:** T24 INCONCLUSIVE by the letter for a third distinct reason; gate 3′ passed in all 28 cells; the
  rare long wait (24 to 76 τ) at small N is the thing to study next (O52). The edge break-up held a third time.
- **The six-link companion:** T32 FIXED WALL (16 at every size, sharp); T30 FIRST ONLY (the directions open one at a
  time; the gas of 6-cubes descends two rungs) (O54). *Corrected 25 September, midday:* the first direction opened
  fully in 28 of 84 torus replicas, partly in most others; the verdict is carried by the gas cell (O54, correction). *Corrected again (O64):* the gas is reported separately as
  registered, so the tori's verdict is NEVER OPENS. The reproduction gate: Gate C′ run A FAILS, run B ONE POWER; a
  factor of two matches the ordered side and the hot tail points at non-bipartite graphs (O53). The companion paper can
  now be drafted from T30, T32, O41, O49 and O50, with the open gate stated.
- **Who to ask, updated:** the model's author, one question: whether his 3D action sums over edges once (a factor of 2
  against ours) and whether his 3D graphs carry triangles and pentagons; the data to attach is O53's table.
  *Revised 25 September, evening:* held for when the six-link paper is ready for him to read (it would be the first he hears
  of our 3D work); the triangles-and-pentagons half is the weaker reading (O53 addendum).

---

## The (D, λ) map: what parameters lead to a burp, exact and measured so far (25 September 2026)

The owner's program (VISION Update 26): which number of directions goes with which λ, how interconnected the directions
are, and how the snap relates to the degree-of-freedom relationships. What is on the record, all at g = 1.5 for the
measured entries and exact for the walls. Every curled direction costs 4(λ − 1) per point, at every D (O41, O50).

| D (links) | Curled state | Stuck for now (exact wall > 0) | Wall at λ = 1.25 | Fixed with size? | Opening pattern | Source |
|---|---|---|---|---|---|---|
| 2 (4) | one curled (the tube) | 1 < λ < 1.6 (minimum of 32−16λ and 64−40λ); metastable at g = 1.5 for 1.05 ≤ λ ≤ 1.35 | 12 | yes, 48 to 192 points | dominant component, clock-dependent waits, release εN−H_final; about one relic in cold boxes | paper 1; T7 to T24 |
| 2 (4) | both curled (a gas of 4-cubes) | never above λ = 1 | downhill | | falls apart at once | O40 |
| 3 (6) | one curled (4 × L × L′) | 1 < λ < 2 (96 − 48λ) | 36 | | never opened: T41 NEVER OPENS at both λ (O81); T48 STAYS, 32 of 32 (O91); T54 untied (computed 11 Oct): the energy of a completed patch peaks near 7.5 columns at about 225 at λ = 1.25, an energy of completed patches and not a barrier (O118 and its correction) | O41, O81, O91, O118 |
| 3 (6) | two curled (4 × 4 × L, L ≥ 18) | 1 < λ < 1.5 (96 − 64λ) | 16 | yes, 192 to 512 points (T32) | first direction fully open in 28 of 84, partly in most others (corrected); the second's wall (36) unpaid; near-flat only at a hot bath (T30) | O49, O54 |
| 3 (6) | all three (a gas of 6-cubes) | 1 < λ < 1.2 (96 − 80λ) | downhill | | at λ = 1.10: joins and opens two of three directions in 21 of 24, no rest between, never one space (T30-gas) | O41, O54 |
| 4 (8) | one curled | 1 < λ < 2.5 (160 − 64λ) | 80 | | not run | O50 |
| 4 (8) | two curled | 1 < λ < 1.67 (160 − 96λ) | 40 | | not run | O50 |
| 4 (8) | three curled (4 × 4 × 4 × 36) | 1 < λ < 1.43 (160 − 112λ) | 20 | | T33: a push equal to the cheapest move buys one move and stalls (NO CASCADE); T40: pushes 20 to 160 never start (27 Sep readings, O79 to O84); T48 eight links STAYS at 1.25, ADVANCES ONLY at 1.40 (O91) | O50, O79 to O84, O91 |
| 4 (8) | all four (a gas of 8-cubes) | 1 < λ < 1.25 (160 − 128λ) | 0 (the edge of the window) | | T33 at λ = 1.10: NO CASCADE; T49: the 8-cube never opens fully (O92) | O50, O92 |
| flat space's own wall | 2D: 32; 3D: 64; 4D: 128 (the same at every λ) | | | | a cold sheet accepts no move in any D | O22, O51, O50 |

What the map says so far (*ours*): the walls rise from rung to rung at every D (they double exactly only with eight links at
λ = 1.25; O55, correction), so nothing in the energy ties the
directions together, and the release of one rung in a cold bath does not pay the next; more directions widen the
window in λ for the partly curled states and raise the flat state's own wall, so a four-direction change can be driven
hot without melting where a two-direction one cannot. The "interconnection" the owner asks about is therefore not in this
energy; if it exists in reality it is an ingredient this family lacks, which is a thing the map can say plainly.

*Added 19 October 2026 (consistency audit): the table above is the untied ladder, which is the rule again since 16
October.* From 9 to 16 October the hypothesis's three-direction runs carried the owner's tie "all at the last" (VISION
Update 47); she withdrew it on 16 October (Update 58), and every tied verdict stands as scored (T54 tied, T56 and T57
tied under this tie; T44 to T46 under the earlier ties). *Exact, six links (O106):* under that tie every rung sits 3a per point above flat, a = 4(λ − 1), the
whole release coming at a point's last opening, and the cheapest single moves out, for λ = 1.05 to 1.25, are: the
slab (one curled, 4 × L × L) 39.2 falling to 4.0; the rod (two curled, 4 × 4 × 18) 30.4 to 24.0; the gas of 6-cubes
14.4 to 8.0; flat space 64 at every λ. Against the untied walls at λ = 1.25 (36, 16, downhill) the tie lowered the
slab's wall and raised the others'. Under any table of that form a damaged point costs no tie, so the tie also
rewarded breaking (O114, O115; one of the reasons in Update 58).

### 25 September, morning: paper progress

- **Paper 1:** a paragraph on the two 120-decay repeats (T23, T24) added to Sec. V, with the three different criteria
  and the rare long wait stated.
- **Paper 2:** Sec. 7 written from T25 (FREEZES IN, the survival table); abstract and discussion updated. Complete as a
  draft.
- **Paper 3:** T24 written in; a new section, "The same map across the number of directions", with the exact-wall table
  and the six-link measurements; the 3D window sentence corrected to O49.
- **Paper 4, drafted:** `docs/papers/black_hole/paper.tex`, "Does concentrated energy re-curl space? Three sealed tests
  ... and a melt every time" (O20/T21, T26, T27).
- **The six-link companion, drafted:** `docs/papers/six_links/paper.tex` (O41, O49, O50, T32, T30, the gas; the
  reproduction gate as its own section; the eight-link section left for T33).
- None has a PDF (Tectonic is not installed system-wide); all are American-spelled; none has been read by the owner.

### 25 September, afternoon: S5 moves; the owner's direction

- **Paper 1 submitted to arXiv with the model's author's endorsement** (the owner's report). His reading is awaited, not
  prompted.
- **The owner's direction** (VISION Update 28): other models with loop and exchange rules (`docs/design/loop_exchange_brief.md`);
  gravity as the target, at least directionally, at one λ (`docs/design/gravity_brief.md`); the (D, λ) map is groundwork for
  the new model.
- **Exact today (O56):** the one-switch curled column in six-link flat space is not a relic (120 units, not a dip); at fixed
  wiring there is no pull at a distance from energy or counting, as in two dimensions. The stable six-link relic is the
  first thing the gravity test needs, and it is not yet shown to exist.

---

## The two-week plan (written 25 September 2026, 13:12 ET, at the owner's direction)

The owner's direction of 25 September: run every claim at once, iterate each several times, use the rented machines
generously, read widely in parallel, write the papers here; do it by the book, because that is what persuades; the goal is
a body of work she can hand to physicists within about two weeks. What that means for each paper, as of today. "In flight"
means pre-registered and on the Batch queue.

| Paper (pieces) | Where it stands | In flight | Next iteration | What finishes it |
|---|---|---|---|---|
| 1, the burp (1, 2) | Submitted to arXiv; the window inconclusive by the letter three times, for stated reasons | T38 (the rare long wait: what the stragglers are) | A fourth λ map written with the owner's bar (a majority sharp at some size, stragglers reported), if she wants one | Her read of T38; a v2 with the stragglers explained |
| 2, the relic (5, 6) | Drafted; T10, T11, T17, T19, T25 on the record | T37 (many natural seeds; nucleation-and-growth with nothing fitted) | If long tubes end defected: does a defected space heal or last (a planted defected state followed in time) | T37 read and written in; the question of the defected end state answered |
| 3, the fertile window (4, 11, 13) | Drafted; the (D, λ) map table | T39 (the cascade window), T40 (the push in four directions), T41 (room for the burp in three) | The window's edges in λ with the owner's parameter questions (how many directions, what λ); a clean domain's size per seed against λ (from T37) | T39 to T41 read and written in; the map complete for two and three directions |
| 4, black holes (8) | Drafted; three melts in two dimensions; T34 melts at its first size | T34 (512 points), T42 (interchangeable points at λ = 1.02, the counting half) | If T42 folds: the packed protocol with interchangeable points, larger sizes | T34 and T42 read; the paper states which half of her mechanism this family can host |
| 5, gravity (7) | Drafted today: no pull at a distance in this family, and why (a gapped flat space); what a pull needs | Nothing: the next step needs her decision (gravity brief, section 6) | With option A adopted: two relics drifting together in two directions, then the 1/r shape in three, then clumping and re-curling | Her decision, then those three runs |
| six links (11, 13) | Drafted; the eight-link section written from T33 | T39, T40, T41 | T40's pattern at and above the true push | T39 to T41 read; the reproduction gate still open (a question for the model's author, held for this paper) |
| allotropes (12) | T36 read: none lasting on a torus | A construction being designed (a planted allotrope) | The planted allotrope's lifetime at λ = 1, the model author's own question | Its reading, sent with the six-link paper |
| quantum (9) | Weakest leg; the phase-per-swap form does not exist as stated | Nothing | Needs her decisions: the sign convention; whether to adopt a rule over histories | Her decisions |

**Compute.** The Batch environment runs 16 one-CPU jobs at once (about $0.78 an hour when full); the account allows 30. On
25 September about 140 jobs were queued (T37 to T42), roughly 150 CPU-hours, about $7. To use the budget the owner named
(at least $20 a day), the ceiling must rise: `terraform/batch.tf` `max_vcpus` from 16 to 30 and the deploy workflow run
(`deploy_manual`, "deploy"); that is hers to apply. Above 30 needs a quota request to AWS.
*Done 25 September, about 14:50 ET, with the owner's permission:* the live ceiling was raised to 30 through the AWS CLI, matching
the committed terraform (so a later deploy changes nothing).

**Decisions waiting on her, in order of what they unblock:** (1) the gravity rule (option A of the gravity brief:
a massless field on the points with relics as its sources); (2) the exchange sign's convention (points or pairs);
(3) confirming or replacing the predictions inferred for her in T36 to T42; (4) whether to write a fourth λ map with her bar.

## 26 September, early morning: the owner's ledger, and the tie run (VISION Update 33)

- **The (D, λ) map gains a column it did not have: how the releases split by order of opening.** Exact (O71): with every
  direction costing 4(λ − 1) per point, the three releases are equal at every λ and D, and walls, being returned, cannot
  change that; only strain in the partly open states can (the tie). Walls are fixed with size and releases grow with the
  region, so any ledger in which walls take a fixed share (the owner's 44 of 66) describes a patch of fixed size, about
  one 6-cube at λ = 1.25. The follow tie's walls at six links, λ = 1.25 (X, two-curled, one-curled): κ = 0: −4, 16, 36;
  κ = 1: 20, −6, 20; κ = 2: 44, −30, 4; κ = 2.5: 56, −42, −4. The middle rung loses its wall first at every λ from 1.05 to
  1.40, so "large, smaller, none" never occurs.
  *(Annotated 19 October 2026, consistency audit: this column is superseded. The story in which the order of opening
  sets the three releases was set aside on 9 October (VISION Update 50); the follow tie it uses was replaced by "all at
  the last" the same day (Update 47), and that tie was withdrawn on 16 October (Update 58). The numbers stand as exact
  results about the follow tie.)*
- **The counting drive against size** (O72): per point 0.33, 0.16, 0.06 in ln A for four, six and eight links, plus the
  exchange of identical cells, which grows with their number; with six links at g = 1.5 the gas beats flat space below
  λ = 1.031 at 10³ cells and 1.126 at 10²⁴. O55's "does not move with N" corrected.
- **Launched: T44** (paper 3 and the six-link paper; pieces 11 and 14): the gas of eight 6-cubes under the follow tie at
  λ = 1.25, κ = 1, 2, 2.5, pushed with its exact wall, baths N/2 and 2N, eight replicas, six Batch jobs. Does one push
  open all three directions into one flat space? Her inferred prediction: yes, at κ = 2 and 2.5. Ours: not all, at any κ.
- **Still unread in the bucket** as of this morning (all jobs finished, per `cloud/status/latest.md`): the rest of T37
  and T38, and T39 to T43. Reading them needs the download step of the handoff.

## 26 September, 05:08 ET: the programme regrouped, and the papers aligned to the groups (VISION Update 34)

The owner regrouped the programme into four groups and took the allotropes out of it. New piece numbers (old in
brackets); the papers each group feeds:

| Group | Pieces | Papers |
|---|---|---|
| **The burp: how X opens** | 1 opens sharply (1); 2 not a fluke (2); 3 the rule (3); 4 room for the burp (4); 5 how the curled directions open, three and four (11 and 13); 6 the burp's three shares, with the charge per kind (14) | Paper 1 (pieces 1, 2; submitted); paper 3, the fertile window (3, 4, 6); the six-link paper, retitled as the paper on how curled directions open (5), with T44 its tied chapter |
| **The scrap** | 7 the scrap: size, number, freeze-in (5 and 6) | Paper 2, the relic; its question "what is the scrap" is open again now that dark matter is a direction's release |
| **What it must give back** | 8 curve-first gravity (7); 9 quantum from counting (9); 10 time and the speed limit (new) | Paper 5, gravity (8); paper 6, quantum (9); a new outline, time and the speed limit (10): the front's top speed against c, then the owner's "third pays for time", with nothing run |
| **The cycle** | 11 black holes as re-curled regions (8); 12 the strange loop (10) | Paper 4, black holes (11); paper 7, the framework, with the loop's sizing (12) |

**Parked:** the allotropes paper (old piece 12, `docs/parked/allotropes.md`), kept for the correspondence with the model's
author; T43 (finished on the cloud, unread) answers his question and goes with it. **Owed:** the reading on dark charges
(`docs/reading/notes/2026-09-26_dark_charge.md` holds search summaries only; this session's network blocked arXiv).

## 5 October 2026: many short papers, each anchored on a published model (the owner's decision, VISION Update 42)

The owner: "I'd rather have 15 seven-page papers, brought together as the full theory by a 16th, than one 90-page
paper." Each short paper supports one part of the theory, can be read and checked alone, and where possible takes
up the model or the open question of one researcher's most recent work: replicate what they published, then extend it
with this project's code and compute. That gives each paper a natural reader. The list below replaces the seven-paper
grouping above as the working plan; piece numbers are the twelve of VISION Update 34. "Anchor" is the published work
the paper builds on; the readings are in `docs/reading/notes/` (5 October for most).

| # | Short paper | Piece | Anchor (whose model or question) | Replicate first | Then extend | State |
|---|---|---|---|---|---|---|
| 1 | The curled torus burps | 1, 2 | Kelly, Trugenberger, Biancalana [KTB19]; [T25] | [KTB19] Fig. 8a (done) | a sharp change between two ordered states at λ > 1 | Written; revised 10 Oct after the AI review of 9 Oct, and again 14 Oct from the owner's own read (plain edition ten chapters on the one question; technical paper in plainer words); both on the owner's company blog (sidenerdapps.com/mainnerd); on hold at arXiv pending a reader or a journal |
| 2 | What the change leaves behind | 7 | Nucleation and growth (KJMA); freeze-out | the KJMA exponent (done, T37) | the scrap's kinds, number, place, lifetime; how much freezes in (T51); free scraps and joints | Rewritten 13 Oct around the three kinds, renamed 14 Oct (was "What the burp leaves behind"); both editions drafted; the plain one revised from the owner's notes of 14 Oct (free scraps and joints, the column's wiring, O126); the technical draft has no figures yet and owes three |
| 3 | Links that reach far, left where starting points meet | 7 | Markopoulou and Smolin [MS07]; Quach and co-authors [QSMG12]; Hossenfelder [H13] | a frozen boundary between domains, in this model | their count, reach and scaling with size; whether cooling heals them; whether the opening splits X in two | First look only (O95, O100); to be pre-registered |
| 4 | The order of the transition in three directions | 5 (Gate C) | Trugenberger [T22] | his D = 3 figure (needs its size and whether its graphs were two-sided) | cooling against heating at 500 points with replica exchange | The bounded joint project to offer the model's author |
| 5 | How curled directions open, three and four | 5 | [T22]; Mulder's theorem [Mul79]; the topology conjecture of [AGGN22]; the decompactification relatives [GM04], [CJR09], [BSV10], [GHR10] | the exact ladder (done) | one at a time; a gas never joins; a connected X; the true barrier | Draft brought up to date 11 Oct (`six_links/paper.tex`, with a PDF): thirteen registered tests, T56 stage 1 and the shape reading (O115, O116) included; not yet read by the owner; T54's first stage computed, not read |
| 6 | The rule, the room, and which settings give a birth | 3, 4 | The local term of [T25] | none needed | the map over the curling cost and the number of directions; room; melting moving with size | Draft brought up to date 11 Oct (`fertile_window/paper.tex`, with a PDF): T58 to T60 stage 1, the product (a flat torus six around, O116), the room, the scrap, one table across directions; T60's held-out size (O117); not yet read by the owner |
| 7 | The make-up at birth and the three shares | 6 | Planck 2018; asymmetric dark matter [KLZ09], [PV13]; Verlinde [Ver16] | the published numbers (done, O89) | which shapes of shares match birth; run them | Outline. The owner chose the shape "all at the last" on 9 Oct and gave her best guess for the three shares on 10 Oct (VISION Updates 47, 52); needs a physicist's check; 20 Oct: in the stiff toy no coupling holds the thin, steady density of missing squares that her dark energy (Update 70) would be, on the flat sheet (O141) or the burp's product (O142); her position that the network's end temperature and empty de Sitter's are one (Update 76) gives one equation whose solution needs a defect priced at about 146 times the end temperature; soft space (Updates 76, 77) is the ingredient designed for a steady density |
| 8 | Dimensions that open in order: a crossover or a sharp change? | 5, 6 | Stojkovic [St14] | his crossover in a layered lattice | set beside an opening with a fixed push and an exact release | Idea |
| 9 | What a cut hides | 8 | Jacobson [Jac95]; Verlinde [Ver11] | none needed | the hidden count: does it follow the boundary or the volume (T52) | Read 19 Oct (O137): a relic's window hides 5 shapes, flat 2 at r = 1 (the control fails) and 1 at r = 2; the scaling NOT READ (r = 2 too slow). Next: the flat control's second shape, then a faster count at r = 2 |
| 10 | An entropy action across wirings | 8 | Bianconi [Bia24] | her vacuum action on a regular lattice | the same action on flat, curled and leftover wirings: her "variation with respect to the topology" | First computation done 10 Oct (O111): flat, curled, re-glued and knots at 64 points; the action rewards squares without limit. Across 445 saved wirings it acts like this model's energy with a coefficient below 1 (0.001 to 0.55). Drafted 10 Oct in two versions, technical and plain language (`docs/papers/entropy_action/`). The question for her is drafted (local); the metric solved per wiring and a sheet holding one knot are not done |
| 11 | A pull between leftovers | 8 | Verlinde [Ver11]; the G-field of [Bia25] | none needed | the pull in a warm bath in three directions; the design of a quantity on the points | The technical draft of 25 Sep (`docs/papers/gravity/paper.tex`, the exact negatives and what a pull needs) revised 10 Oct with where the two routes stand, and a plain-language draft beside it. Its measurements are not run: the hidden count is read and does not decide the route (T52, O137), the warm sheet's reach and the pull are not started, the design of the quantity on the points is not written; 20 Oct: the design of the quantity on the points is written (`docs/design/soft_space_brief.md`, with the reading `docs/reading/notes/2026-10-20_soft_space_prior_work.md`), decided in two stages (Update 77), and stage A's kernel built and checked (Q26); the first registered pull run waits for her prediction |
| 12 | Concentrated energy does not fold space | 11 | Black holes as random bubbles [T25]; Smolin [Smolin06]; as benchmarks only, the black-hole-cosmology lineage (Pathria, Popławski, Fife; unread, O103 (j)) | none needed | seven registered tests and what they need (T21, T26, T27, T34, T42, T47 part B, T57) | Draft brought up to date 11 Oct (`black_hole/paper.tex`, with a PDF): seven registered tests, T57 included (MELTS, tied and untied; O114); not yet read by the owner; 16 Oct: a section on the author's mechanism of that day (Update 59) added; 19 Oct: a paragraph on her Update 70 positions (her decision that the loop's coupling is fixed by its total energy, with our exact reading in the toy's terms; dark energy as the thermal defect density, adopted as a hypothesis to test, with rule 13) added; the PDF not rebuilt since 14 Oct |
| 13 | Counting versions cannot beat Bell | 9 | Smolin [Smolin11]; Hardy [Hardy01] | none needed | the rungs of T15; the exchange sign | Results in hand; outline; 21 Oct: piece 13 (interference and records, the published rule checked in an exact toy) folded into piece 9 by the author, so this paper carries both |
| 14 | A front's pace | 10 | Ellis [Ell14] | none needed | steady on a fair clock; faster where energy sits (T47, T50) | Results in hand; outline; 20 Oct: T75 read (O140), NO SIGNAL in the stiff toy, a disturbance spreading by jumps through the far-reaching switch; the same protocol in soft space, where waves travel at one speed, is the next registered run |
| 15 | A feed, not a spark: energy-coupled nucleation of the slab | 5, 11 | Kramers' escape with a slowly rising temperature; paper 1's two-channel law as the known case | the tube's first exits under a feed against 1 − exp(−∫Γ[g(t)]dt) (a sandbox look first) | the untied slab fed at fixed rates, shared store against one point's store, capacity scanned; first escape, a persisting front and completion recorded separately; the held-out size | Stage 1 read 17 Oct (O132): no replica of 96 opens or reaches a front; BREAKS in 22 cells of 24, SITS in 2 (out of time, damage rising); her inferred OPENS fails, ours fails by the letter at the larger size; the held-out size read 17 Oct (O133): every cell's word as written, no front at any size; T68 stage 1 read 17 Oct (O134): a ramped or pulsed rate, or a feed at 16 or all points, at λ 1.25 to 1.60, no front in 1,728 replicas; the held-out size read 18 Oct (O136): 106 of 108 cells as written, the two misses marginal, no front in 864 replicas, T68 complete at three sizes; T69 (the lowest path) is the exact test; VISION Update 62's knobs and T70 the next; the author's handoff, VISION Update 59, H1 to H4; 21 Oct: T71 read (O144), the slab cycles under the layer form, each daughter exactly its curling cost while the chain keeps pace |
| — | Allotropes dissolve | parked | Trugenberger [T24] | none needed | the first numerical data on them (T36, T43) | Results in hand; kept for the correspondence with him. *Parked and unnumbered (19 Oct, consistency audit): it was paper 15 from 5 to 16 October, when number 15 went to the feed paper above; every citation of paper 15 since then means the feed paper.* |
| 16 | When an observer finds itself: light cones, a universe's age, and two pictures of dark energy (was "The loop and its measure") | 12, 14 | [BHKP07] (the causal diamond); [LBS16] (the 2016 observer-window paper); the measure problem through [Freivogel11], [BFY08]; the stellar and hard-steps inputs of T72. Hofstadter (the strange loop) and Smolin [Smolin06] are not cited in the draft yet, nor is the branching count of Update 59 | the 2016 paper's 0.1 % and peak (T72's gate: the share 0.17 %, inside the registered factor of two; the classes read on the owner's decision to take the gate as passed (its peak missed the registered window by the letter; O138)) | the measure as decided (Updates 59, 67, 68, 70, 71; Update 70 gives class D its hard-steps delay shape) and its first two calculations (T72, T73) | Drafted 19 Oct (`loop_measure/paper.tex`, no PDF built on this machine); 20 Oct: retitled, and a section added comparing the author's constant dark energy (the diamond closes) with [T24]'s falling one (if the acceleration ends there is no event horizon, the diamond never closes and the measure would make our age early), with T77 run under declared stand-ins (21 Oct; her constant-dark-energy prediction holds on the control; O143); waits for her read; rule 13 caveat throughout; 21 Oct: T77 run and read (O143) |
| 17 | A phase-changing reality | all | the sixteen above | | the assessment that brings them together | After the others |
| 18 | Particles as knots in a network | 6, 7, 9 | Kosterlitz, Thouless, Halperin, Nelson and Young (defect pairs in two dimensions); Kleinert and Zaanen (gravity from defects); Kibble and Zurek; asymmetric dark matter [KLZ09], [PV13]; Sakharov's conditions | none chosen | the three knots as kinds with fixed masses; curl and uncurl as matter and antimatter; pairs under a surface-keeping rule; strings; the excess (unequal prices, or inherited from the stores); mass as bound field energy and the conversion to physical units | Outline 20 Oct (`particles/outline.md`), on the author's request; nothing run for it; numbered 18, after the synthesis's number, so as not to renumber |
| 19 | The toy read through Carroll's reader | 9, 1 | Cao, Carroll and Michalakis [CCM16]; Cao and Carroll [CC17] | none needed (an estimator applied to our wirings; its control cases are exact) | the reader's count against the toy's local dimension on windows; a window with stored matter against empty space (T91, T94) | Drafted, two editions, 2 to 3 November (`docs/papers/carroll_reader/`); number proposed 3 Nov by Fable, the owner's to confirm. Verdicts MIXED (T91) and READS DIFFERENTLY (T94, her prediction). Three questions for Carroll at its end. Next: a state built from the twist-curl rules (design brief under S1 first) |

Notes. (a) Papers 5, 12, 13, 14 and 15 are mostly negative or bounding results already measured; each is short and
can be written now. *(15 in this note, written 5 October, is the allotropes paper, now parked and unnumbered.)* (b) Papers 3, 8 and 10 each begin with one narrow question to their anchor's author, which doubles
as the route to a reader (the questions are at the end of the reading notes). (c) Paper 4 is the only one proposed as
joint work. (d) Nothing here changes the rule that a paper at λ ≠ 1 is not called combinatorial quantum gravity. (e) Added
6 October (O103): the narrow statement the three AI reviews propose for papers 1 and 2 together, which the author may
adopt: a modified emergent-geometry toy supports a stable-for-now compactified arrangement, local nucleation, a
propagating geometric change and a calculable release in two directions, with the freeze-out of defects under test;
the cosmological interpretation comes later. Paper 1's revision candidates are in `curled_torus/v2_changes.md`.

## 12 October 2026: where the price on loops of six belongs (the owner's question)

Asked whether T62 and T64 deserve a paper of their own, and which papers they belong in. *Ours:* not a paper of
their own yet. They are a change to the rule, so their home is **paper 6** (the rule, the room, and which settings
give a birth), as a section: what the rule charges for decides what the product is. Loops of four only: a tube six
around. Loops of six as well: mostly wide, flawed nearly everywhere. They also change what two other papers have to
ask. **Paper 2** (what the change leaves behind): the flaws become the main obstacle, and paper 1's own leftovers turn
out to sit at the joint between two widths (O116, addendum of 12 October). **Paper 5** (how curled directions open,
three and four): an opened direction should be measured for how far around it is, and a rule may have to pay for each
further step. A short paper of its own would be earned by two things not yet in hand: the 384-point runs holding, and
a way to get the wide product clean (slower cooling is the first thing to try). Paper 1 states the result in its
closing section, in a few sentences.

## 13 October 2026: a second set of relatives and one vocabulary (the owner's request)

The owner is considering G.-W. Chern's condensed-matter theory group at the University of Virginia as readers,
first of paper 2, and asked for our wording to match theirs where the concept is the same. What was read, the
map of terms, the words that look alike and are not, and the comparison with Trugenberger's programme are in one
place: `docs/reading/notes/2026-10-13_chern_group_vocabulary.md`. *Ours, in one sentence:* the model is
Trugenberger's, deformed; the questions asked of it (how one ordered state turns into another, isolated or in a
bath, and what defects remain) are the kind this group asks of its own models; neither is evidence for the
hypothesis (rule 13). Nothing has been sent to anyone.

| Short paper | Their work that bears on it | What it gives | Done on 13 October |
|---|---|---|---|
| 1, the curled torus | [JC26]; [LLC26a] | "isolated" and "microcanonical" for sealed; "minimax barrier" for the barrier the paper says the 12 units is not; "domains separated by defects" for T61's endpoints | Four insertions and two references in the technical manuscript; rebuilt. The plain-language edition is unchanged |
| 2, what the change leaves behind | [JC26]; [JYC25]; [BBJC26]; [FZC26] | The defect vocabulary; the caution that an exponent fitted over a short range is an apparent one | The technical draft rewritten with T51, T55, O116's wording and O123, and a draft plain-language edition written. The owner has read neither |
| 3, links that reach far | [JC26] Sec. III | Walls form where independently nucleated domains meet | Note only |
| 5, how curled directions open | [LLC26a] | The method for the barrier T54 did not price | A pointer at the head of the draft |
| 6, the rule, the room | [JC26]; [BaC26] | A dynamical phase diagram; entropy maximized under what the dynamics retains | A pointer at the head of the draft |
| 12, concentrated energy | [BBM26]; [BFC26]; [YC24] | Two kinds of melting; recovery after an energy pulse | A pointer at the head of the draft |
| 14, a front's pace | [FZC26]; [BBJC26] | A propagating interface between two phases; a growth law written as a crossover | Note only |
| 15 (as numbered on 13 October; now the parked, unnumbered allotropes row), allotropes dissolve | [JC26] Sec. III.2 | A criterion for arrest: the energy available against the local barrier | Note only |

A candidate reader for paper 2, beside the model's author in the table of 24 September: G.-W. Chern (the note,
section 7, has the question; the reading note of 25 September named him first). The owner's rules apply: one
narrow question, the AI disclosure sentence, a check of his recent papers, which the note records.

For paper 1 the candidate of 24 September, M. Vucelja, was looked at again the same day (the note, section 8).
Her joint paper with Chern is not the connection; her own work on relaxation and mean first passage times
under Metropolis dynamics is, and it touches paper 1 in three places (the first exit, the reservoir of stores,
and paper 2's cooling). The work that would serve a reader from that side is in
`curled_torus/handoff_2026-10-13_measurement_side.md`.

## 14 October 2026: paper 1 tightened, and where its cut material belongs (the owner's notes)

Reading paper 1 on 14 October the owner asked that it keep to its one question and that each side topic be teased
once, in the paper it belongs to. Her rules for all the papers are gathered in `docs/papers/feedback_2026-10-14.md`.
Where paper 1's cut material now belongs:

| Cut from paper 1 | Its home |
|---|---|
| The scrap's kinds, where it sits, free scraps and joints | Paper 2 (paper 1 keeps the cold-box count, one paragraph, and a tease) |
| Trapped heat does not widen the tube (T61) | Paper 6, beside the price on loops of six (T62, T64): what decides how wide the product is |
| The price on loops of six | Paper 6 (12 October section above); paper 1 no longer states it |
| Named against interchangeable dots | Papers 12 and 5, which run both; a paper of its own if the counting drive is tested near λ = 1.02 |
| Six and eight links per dot, more directions | Paper 5 (paper 1 keeps one tease, in its last chapter) |
| The random-tangle route (the published question) | Paper 4; paper 1 keeps its reproduction and one sentence on the jump |

The fair clock is now named in paper 1 (plain edition, Chapter 5; the technical paper, Sec. II), and papers 2, 6 and
14 use it. The same day the other drafts (papers 5, 6, 10, 11, 12 and 2) were given the plain-wording pass and the
consistency checks of the rules file; the technical drafts of papers 2, 5, 6, 10, 11 and 12 have no figures, and what
each would need is listed below (drawn by script from saved results, as paper 1's are; sketches labelled so).

| Paper | Figures owed |
|---|---|
| 2, what the change leaves behind | relics against size and against seeds (T10, T17, T37); the cooling result (columns, energy, pieces of each kind against cooling time, T51); the three kinds' wiring |
| 5, how curled directions open | the ladder and its walls for six and eight links, untied and tied; outcomes per test; counts at each d against time; the saved wiring of an opened gas; distance profiles of the OPENS replicas; T54's barrier against patch size; T56 tied against untied; the gate curve |
| 6, the rule, the room | the price list against λ; first exits against the count (T22, T59); the four maps' criteria; the largest component against size with the held-out points; the edge; widths of the end states; births against room (T18); walls against λ; the trade-off |
| 10, an entropy action | the action against squares for the 445 wirings; λ_eff against c₀; the action per cell for the three rungs; a sketch of the tube and its re-glued twin |
| 11, a pull between leftovers | Table I as two log-log panels; a sketch of two defects at separation r; symmetry count against placement |
| 12, concentrated energy | outcome shares per cell for the seven tests; T21's folded share against budget; the scar against time (T42, T34); T57's damage against energy; the six-link ladder under the tie; the counting crossover |
