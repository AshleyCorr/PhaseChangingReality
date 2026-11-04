# The next set of experiments and looks, edited after two red-team reviews (26 October 2026)

**Written 2026-10-26 by an assistant session without the owner. Every item is proposed; nothing is decided or
registered; the owner's predictions are owed where an item says so; labels per rule 10** (exact: arithmetic or complete
enumeration; measured: a registered run with its verdict; look: a sandbox look under rule 16, which can kill a claim and
never crown one; ours: an interpretation, unverified; hers: the owner's). A new energy term, rule or move set is a dated
VISION decision of hers under S1; no item below adopts one. No item below was run, launched or committed by this
session. The programme page is not republished by an agent session (handoff, section 1).

**Rule-obedience line, stated once and referred to by every item (exact; O159).** The hard rule, as the owner confirmed
it on 25 October, holds the square count at exactly 3N for six links (N for four). The flat tori obey it: 6 × 6 × 8 at
six links (S = 864), 16 × 10 and L × L at four links (S = N). The following do **not**: the rod 4 × 4 × 18 (S = 3N + 144,
curl 576, anticurl 0), the slab 4 × 12 × 12 (S = 3N + 144), the 6-cube and 8-cube, the three saved damaged push-look end
states (curl/anticurl 202/206, 185/205, 493/61), the curled shell of item 8, and the 62-point square-free piece of item 6
(S = 0). Each item says which start it uses.

**Decisions that gate items** (hers; not repeated here, see `docs/design/owner_questions_2026-10-24.md`): **Q1** (the
hard rule as is, compound moves, the soft form μ(S − 3N)², or X from damaged states), **Q2** (where a rule-obeying X's
anticurl sits), **Q3** (the pair tie f = (0, 0, a, 0)), **Q4** (pricing broken points). Also open: the **clock** (fair or
chain, hers since Update 46) and **soft space's brief** (Update 40, unwritten).

---

## 1. Summary (ten lines)

1. First, and only when the 54 jobs have landed: read T74 at the held-out size L = 16 under Amendment 6, naming the
   commit the jobs ran; all three λ = 1.10, c = 2R cells are the live ones, not one (measured, O145; Amendment 6).
2. The T71 per-birth reader (already written, untracked) touches no registered runner and can be committed now; the
   `layer_feed` guard fix and the `.meta.json` commit-hash helper wait until item 1 is committed.
3. Three sandbox looks need no decision and no prediction and run now: three switches at radius 1 out of flat 6 × 6 × 8
   (script exists), the cube-face look with its expectation corrected (exact: C4 □ K2 is the 3-cube, so the rod's
   wrap-around squares are cube faces under a graph-only definition), and the four-link size census with held-out L = 12.
4. The fed-slab ledger checks from the T71 and T74 logs are exact and cheap, but the graph-free "bucket" null must name
   the rules it keeps and read the 0.4 to 2.2 % residual, not "the third digit"; and clean-switch attempts per fed unit
   are fixed by construction (1/r), so the size count that means anything is per birth.
5. A rule-obeying curled X is an existence-by-construction census, priced before it is called X: under the rule every
   balanced X sits 4λ·curl above flat space with no 4(λ − 1) release left (exact, O159 (i)), which the owner should see
   before Q2 is answered.
6. The pair tie on those candidates, the soft-form scan and the push on a rule-obeying X are all blocked on Q1 to Q3
   and, for the push, on a kernel that does not exist (compound moves or the soft form, with rule-6 tests).
7. Abandon clauses that decided the owner's questions (option (b) from four-link arithmetic, option (c) from one look,
   "the twin X" from a non-exhaustive search, an inferred Z2 as a kill) are removed; looks report their words and stop.
8. The dipole look in the toy's light field is kept as the cheapest look that bears on H4, with words that can score
   its own expected outcome (AT CHANCE / EXCESS); it is a non-test of her mechanism and says so.
9. The record's debts come with the program: the red-team pass, the prior-work notes (paper edits as drafts for her
   approval), the KCM bibliography and Crossref debt of O159, and T9's drift, owed since 22 September.
10. Owner's predictions are owed only for registered runs (items 4's follow-ups, 10, 11); the looks run without them.

---

## 2. The program, in order

### Item 1. Gate: read T74 at the held-out size L = 16 (measured; handoff Task A)

- **Kills:** T74's one registered size claim (the period between births does not change with L; hers, confirmed
  25 October) and the three stage-1 repeat claims (fed, timing, unfed).
- **Runs:** nothing new. Fetch and accept the 54 files through `accept_inbox.py` (none is in `results/` or `cloud/inbox`
  as of this writing, so nothing can be read yet); `t74_heldout_score.py` and `configs/t74_heldout_prediction.json`
  untouched after the data is seen; `analyse_t74.py` for the per-cell words; the per-birth reader of item 2 extended to
  T74 so short births are not hidden by the running average (O159). **The reading names the commit the 54 jobs ran**
  (O159's condition), taken from the queue manifest and the workflow log, since the runs' `.meta.json` carry no hash.
- **Words and rules:** Amendment 6 as written: HOLDS / MISSES, every miss named, "within 10 %" beside it where true;
  REPEATS STAGE 1 / CHANGES for each of fed, timing, unfed. Coverage quoted from the rule in force: nominal 99.7 % per
  range, 0.060 for at least one miss among the 18 (the "about 96 %" figure belongs to the rejected rule that left out
  the fit's error, Amendment 6).
- **Live cells (measured, O145 and Amendment 6):** all three λ = 1.10, c = 2R cells (meals 16, range 6.405 to 6.512;
  meals 256, range 6.311 to 6.749; steady, 6.415 to 6.445), where clean switches fired in stage 1 and six cells read
  MIXED. Report them first and separately, with clean switches per replica and each birth's energy against a N_slab.
- **Ours (unverified):** HOLDS in the 15 cells away from λ = 1.10, c = 2R, because the period is a N_slab / min(c, R) by
  construction (exact; Amendment 6 and O160 already say so, and say that HOLDS there crowns nothing about her picture).
  **Rule 11 if MISSES:** (1) not a falsification of the gate mechanism; (2) the period is clock-free by the same
  arithmetic (feed, cap and gate all ∝ N_slab, one layer proposal a sweep), so the first candidate is the fluctuation of
  the two layer stores beside the cut (O145's own cause), then a too-narrow range, then implementation only if a miss
  appears at c = R/2 or λ ≥ 1.25; (3) the exact log counts of item 4 tell these apart; (4) the fed-slab kernel is set
  aside pending an independent reimplementation if a c = R/2 cell misses by more than 10 % or the ledger identity
  (item 4 (a)) fails anywhere.
- **Owner's prediction:** on record (Amendment 6, "yes but the surplus matter"). **Cost:** laptop, under ten minutes
  once accepted. **Waits on:** the 54 jobs finishing.

### Item 2. The T71 per-birth reader now; the guard fix and the hash helper after item 1 (code; O158 (b), (d); O159 item 4; handoff Task D)

- **Kills:** nothing physical; closes O158 (b), (d) and the two audit items F30, F31.
- **Now (touches no registered runner):** commit `scripts/read_t71_births.py` and `tests/test_read_t71_births.py`
  (written, untracked); check Codex's 5,148 and 4,620 births in the two clean-switch cells (λ = 1.10, r = 1/16, L = 8 and
  12), including the first logged row, with births above one curling cost up to 1.25; record as an addendum to O158.
- **After item 1 is committed (the record's condition: no registered runner changes while its held-out stage is
  unread):** replace the three `pool + 1e-12 < bill` tests in `layer_feed` with strict `pool < bill`; a test at a pool
  short by 1e-13 (refused); no test of exact equality on floats (the runs never reach it; if wanted, construct it with
  integer-valued stores); the docstring's "energy is conserved exactly" reworded as an accounting identity of the chosen
  ledger; `tests/test_layer_feed.py:93` given a second seed or marked seed-dependent; the `.meta.json` commit hash and
  dirty-tree flag through one shared helper with a test (TASKS, 6 October item 5), existing results untouched. Then one
  stage-1 T74 replica rerun from its config **as a post-fix regression check** against the committed stage-1 file (O160
  already did the pre-launch rerun; this one has a different purpose).
- **Words:** REPRODUCES / DIFFERS (if DIFFERS: the sweep where the trajectories part and whether any scored number
  moves).
- **Ours (unverified, not exact):** REPRODUCES. No logged store went below zero (exact, O158 (b)), but the logs are
  sampled and the strict guard can refuse a move the old guard accepted when the pool was short by under 1e-12, which
  changes the trajectory from that move on; so "no verdict changes" is ours until the rerun reads.
- **Cost:** laptop, about 30 minutes plus the suite. **Waits on:** nothing for the reader; item 1 committed for the rest.

### Item 3. Three looks that need no decision: compound moves at radius 1, cube faces read correctly, and the four-link size census (look plus exact; handoff Tasks B and C; Q1 option (b)'s exact count)

**3 (a). Three switches at radius 1 out of flat 6 × 6 × 8** (start obeys the rule).
- **Kills:** "the hard rule with a local sequence of up to three switches at radius 1 can start a clean opening from
  flat space" (if ZERO), or Update 128's "frozen" as corrected by O159 (if FOUND).
- **Computes:** run `sandbox/hard_rule_triples_look.py` as it stands (words ZERO / nonzero declared; radius 1 only: the
  second switch within one link of the first's four points, the third within one link of the eight; its own estimate
  17 s for stages 1 and 2 and about 800 s for stage 3 in eight slices; prune by the exact bound that a switch gains at
  most 2(deg − 1) squares). Report whether stage 3 finished or the four-link 16 × 10 fallback was used, and the measured
  maximum gains against the bound. **Added exact reading:** switches are reversible, so ZERO at k local switches out of
  flat space also means no clean opening can *end* in flat space by k local switches. **Scope:** the kill is "local
  sequences of k switches at radius 1"; non-local sequences are untouched. The [Swap17] carry-over is about a fixed
  triangle count, not squares (read 26 October; to verify against the PDF).
- **If ZERO AT THREE:** four switches on the pruned pairs **on the four-link flat torus only** (fan-out far smaller;
  bound 6 a switch). **A four-link word only**: it abandons nothing about six-link option (b), where the four-switch
  count is left undone and said so. For any found move: curl and anticurl made, dH at λ = 1.25, and whether the end
  state has a point with more open directions than the torus has (clean against damage).
- **Words:** ZERO AT THREE (radius 1) / FOUND k AT THREE; four links: ZERO AT FOUR / FOUND k AT FOUR; per found move
  OPENS / DAMAGES.
- **Ours:** ZERO AT THREE at radius 1; at four on the four-link torus, FOUND and every found move DAMAGES.
- **Cost:** laptop, 15 to 90 minutes. **Waits on:** nothing. One LOG line by the session that runs it.

**3 (b). Cube faces, local neutrality, and the Euler count** (flat torus obeys the rule; the rod and the saved push states
do not, read anyway and marked).
- **Kills:** the Gauss–Bonnet note's claim (ours, unreviewed) *as a graph statement*. **Exact, found by the review and
  checked by hand:** in the rod 4 × 4 × 18 every wrap-around 4-cycle round a length-4 direction, with its translate one
  step in another direction, is C4 □ K2, which is the 3-cube Q3 (induced, no extra links). So under
  `sandbox/cube_face_look.py`'s own definition ("a face of at least one 3-cube subgraph") every wrap-around square is a
  cube face, the rod reads ALL FACES with S = 3.5N, and "every 4-cycle a face ⇒ S = 3N" is false as a graph statement.
  WRAP ONLY can be read only through an embedding (coordinates), which presupposes what the toy is meant to derive;
  whether an embedding-defined face is acceptable is hers (question 7 below).
- **Reworded check (ours, unverified; arithmetic):** S = 3N follows for a closed cubical 3-manifold with six links a
  point because each vertex link is a triangulated 2-sphere on 6 vertices (12 edges, so 12 squares a point, S = 3N),
  whatever the squares per link. So the check becomes: does every point's link triangulate a sphere with 12 edges?
  The same count allows a rule-obeying X with a 5-square link beside 3-square links at one point (the non-octahedral
  6-vertex sphere), curl and anticurl paired at a point: a **fifth Q2 placement**, ours, added to the owner's list.
- **Computes:** the existing look as written (ALL FACES / WRAP ONLY / MIXED; NEUTRAL / LOBED with "small" fixed as
  within the shuffled spread) plus the vertex-link Euler count per point. **The growth look is dropped:** a larger flat
  torus has S = 3N by arithmetic, and growth as the only allowed move is a decided rule (Updates 72, 75), not a result;
  nothing from it goes on the record beyond a LOG line.
- **Ours:** flat ALL FACES, NEUTRAL, every link a 12-edge sphere; rod ALL FACES (corrected sign), links not spheres
  where the curl sits.
- **Cost:** minutes. **Waits on:** nothing.

**3 (c). The four-link size census with a held-out size (exact; rule 14 for any "at every size" sentence).**
- **Computes:** count-keeping singles, radius-2 pairs and radius-1 triples out of the flat L × L torus at L = 6, 8, 10;
  **held-out L = 12** with the number written first (ours: 0 at every k ≤ 3 for L ≥ 6). The N = 18 pair line is dropped:
  O159 (f) already records the rule-keeping classes at N = 18 joined by single switches.
- **Words:** FROZEN AT ALL SIZES / THAWS AT SMALL L (first L named). This is the held-out size behind hypothesis H-c;
  the six-link flat torus count has one size and is said to.
- **Cost:** laptop, under an hour. **Waits on:** nothing.

### Item 4. Is the fed-slab burp anything but the ledger? Exact checks from the logs (exact; no run)

- **Kills:** "the slab decides when to burp", "the graph's wiring enters T74's fed-phase verdicts".
- **Computes:** (a) the ledger identity per fed-phase birth in every T71 and T74 file: energy in since the last birth −
  clean-switch spending − daughter's energy = 0 to 1e-9; (b) **the bucket null, restated:** a graph-free model that
  keeps, by name from `layer_outside.py`, the reservoir, relaxation, `edge_flow`, `take_evenly`, `rho`, the cap and the
  gate **on the two layer stores beside the cut** (not one number: O145's rule-11 answer (2) found the unfed equal words
  decided by those two stores' fluctuation, so a one-number bucket reads DIFFERS there for a known reason), same flow
  seeds, scored against every stage-1 fed-phase period and timing word; the quantity read is the **residual** per cell
  against a N_slab / min(c, R) (stage 1: 0.4 to 2.2 %, 15 of 18 fits below the exact value, larger with meals of 256 and
  c = 2R; measured, Amendment 6 table), not "the third digit"; (c) per replica, the share of births whose preceding
  interval held zero graph moves, with widenings counted as splits (O158 (c)); (d) at λ = 1.10, c = 2R, whether the
  shortfall count is seed-linked or memoryless; (e) **the clock count, corrected (exact, from `layer_outside.py` and the
  T74 config):** a sweep makes N_slab clean two-switch attempts and the feed is r N_slab a sweep, so attempts per fed
  unit are 1/r at every L, fixed by construction; what grows ∝ N_slab is attempts **per birth** (births every 16a or 32a
  sweeps whatever L). Count clean switches per birth against N over L = 8, 12, 16, set against the stage-1 pair of 7 and
  7 fed-phase clean moves at L = 8 and 12 (measured, Amendment 6; small numbers), where ∝ N_slab predicts 2.25× more.
- **Words:** IDENTITY HOLDS / FAILS; RESIDUAL PATTERN (meals, c) / NO PATTERN; WIRING-FREE SHARE above or below 90 %;
  SEED-LINKED / MEMORYLESS; PER-BIRTH SWITCHES ∝ N / FIXED.
- **Ours:** identity holds everywhere; the residual grows with meals of 256 and c = 2R; wiring-free share above 90 % at
  λ ≥ 1.25; memoryless; per-birth switches ∝ N on the chain clock, which the (7, 7) pair already leans against. If (a),
  (b), (c) read so, the next programme page (republished by the owner, not by an agent) should say T71 and T74 show the
  ledger and cap working as registered at three sizes and show no graph mechanism for the burp: a kill of a reading,
  not of her picture, written before any new fed-slab run is registered.
- **Registered follow-ups, after her word only (her prediction owed for each; nothing launches before it is on the
  sheet):** the gate null (one T74 cell at gate 0.5 and 2 a N_slab; ours PROPORTIONAL to 3 %); and the one
  non-tautological size claim, **clean switches per birth** against N at λ = 1.10, c = 2R, fit on L = 8, 12, 16,
  **held-out L = 24** (2,304 points), with a **fixed sweep budget** stated (the period is fixed in sweeps, cost per sweep
  ∝ N: about 2.5 hours a job on Batch from 2.25 × 67 minutes, 1 to 3 jobs); ours GROWS with exponent near 1 on the chain
  clock; **abandon condition tied to the claim:** abandon the λ = 1.10 track if the fitted exponent's range at L = 24
  excludes 1 in either direction. The clock control (both clocks at L = 8) is moved out of this item: a fair-clock mode
  of the fed-slab runner is a change to a registered runner, new code with rule-6 tests, and belongs under the clock
  decision (item 9).
- **Cost:** laptop, an hour to write, minutes to run. **Waits on:** item 1 for the L = 16 files; nothing else for the
  stage-1 part.

### Item 5. Damaged starts under count-keeping moves: a backbone or diffusion? (look; Q1 option (c) made exact, reported, not decided)

- **Kills:** "the hard rule from damaged starts reaches clean space" as a look's word; it does **not** remove option (c),
  which is hers (handoff section 5).
- **Computes:** from the three damaged push-look end states (42, 57 and 14 count-keeping switches each; these starts
  obey neither the rule nor her X, and **share one origin**: saved end states of one protocol, the sealed push on the
  forbidden rod, so "all three" is one construction, not a sample). **Walk defined by moves, not sweeps:** apply one of
  the known count-keeping moves, recompute the count-keeping set only within radius 2 of it (new sandbox code, no
  `src/` change), with a full recount every k = 50 moves as a check (the full enumeration is about 275,000 `try_switch`
  calls, O159 (b), about 0.14 s at the measured rate, so sweeps of N moves would be days). Budget 2,000 moves a start.
  Record the set of points ever touched, the displacement of the damaged region's center, and whether any point of the
  clean rod interior is ever moved.
- **Words:** FROZEN BACKBONE / DIFFUSING / COOPERATIVE (touched set grows only when two defects meet).
- **Ours:** FROZEN BACKBONE at all three, clean interior never touched (the KCM reading, Ritort and Sollich 2003,
  read 26 October; to verify against the PDF).
- **Cost:** laptop, about two hours including the local recount code. **Waits on:** nothing.

### Item 6. Does a rule-obeying curled X exist in the toy, by construction, and what does it cost? (exact census, scoped; kills nothing about Q2)

- **Kills:** "a rule-obeying one-curled slab or rod can be built by whole-row removal at these sizes" (per placement,
  by these constructions only); "the rod between two slabs already has two hands" (by the mirror renaming).
- **Priced before anything is called X (exact; O159 (i); red team 2's missing (1)):** a torus with k sides of length 4
  carries curl k·N and anticurl 0, so a rule-obeying one-curled slab must hold anticurl N (N/4 missing squares) and a rod
  2N (one sixth of its faces open); under the rule H = 4λ·curl for every balanced X, so a balanced X sits 4λ·curl above
  flat space (the rod's 576 curl: 2304λ) with **no 4(λ − 1) release left**: its opening is a descent with nothing to
  burp. This is stated to the owner with Q2, since it changes what "burp" can mean under the rule.
- **The 62-point square-free piece** (the incidence graph of the projective plane of order 5: 6-regular, bipartite,
  hard-core valid, S = 0, anticurl 744; its being the (6,6)-cage is general knowledge, to verify): the scratch result
  "not a dip at λ = 1.25, cheapest switch gains 8 squares, dH = −128" is **not cited as exact until the look is committed**
  with its words and its rule-obedience line (S = 0 does not obey), the pattern O159 (b) caught. A rod 4 × 4 × L (curl
  32L) beside a (v₆) piece (anticurl 24v) balances when 4L = 3v (v = 32, 40, 48; whether (32₆) and (40₆) exist is
  general knowledge, to verify, carried into the docstring); (31₆) exists and never balances such a rod. **A disconnected
  rod-plus-piece contradicts the decision that X is one connected piece (Updates 39, 44)** unless grafted; no such
  candidate carries forward before her answer to question 1 below.
- **Computes, per placement (i) collar, (ii) spread thin, (iii) hollow core, (iv) in the two slabs, (v) the fifth of
  item 3 (b) (curl and anticurl at one point), (vi) a grafted square-free piece:** build the smallest candidate by
  whole-row removal or rewiring on slab 4 × 6 × 6 and 4 × 8 × 8, rod 4 × 4 × 8 and 4 × 4 × 12, the twin X at the
  smallest joinable size; each: 6-regular, bipartite, valid, **S = 3N checked before anything else is read**, curl and
  anticurl split by the Euler count of item 3 (b), count-keeping single switches and radius-2 pairs out of it and whether
  every one costs; the mirror renaming (Q15's method) of each twin candidate; per-region sums Σ(S_e − 4) over slab A,
  rod, slab B (which report only what the construction put there).
- **Words:** NONE BUILT / EXISTS, FROZEN / EXISTS, STUCK / EXISTS, MOVES (where); SWAPPABLE / DISTINGUISHED; region sums
  OPPOSITE / SAME SIGN / ALL NEUTRAL. **No abandon clause:** NONE BUILT means these builders found none at these sizes.
- **Held-out number (rule 14):** the count of count-keeping single switches out of each candidate that EXISTS, fitted at
  the two smaller sizes, predicted with a range at the third before it is built.
- **Ours:** NONE BUILT for (i) and (iii) at these sizes; EXISTS, FROZEN for (iv); SWAPPABLE for every symmetric build, so
  "opposite hand" is untestable without a signed twist label (hers, Update 97).
- **Cost:** laptop, an hour (builders are the only new code). **Waits on:** nothing for the census; Q2 and question 1
  below for which candidate carries forward.

### Item 7. The pair tie and the (D, λ) map in the rule's own units, on item 6's candidates (exact; blocked on Q1 and Q3)

- **Kills:** "under the pair tie the second opening is easier than the first" (Update 124's cascade, which rests on a
  forbidden X and 4(λ − 1) units); "the counting drive can make X near λ = 1 under the rule".
- **Prior work cited, not repeated:** `sandbox/pair_tie_patch_look.py` (LOG, 25 October) has the patch energies of the
  first opening (rod rows) and the second (slab helix), tied and untied, λ = 1.15 to 1.40, on the forbidden rod and
  slab; T54 is registered and read (ΔH(k) = a k − b k² exactly at L = 8, 12, 16; FIXED WALL at tied 1.25 and 1.40, RISES
  AGAIN elsewhere; held-out numbers held). Those numbers do not carry over to a balanced X, where anticurl terms enter.
- **Computes, on every candidate that EXISTS:** the patch **defined under the rule before computing** (curl and anticurl
  removed together, the count kept at every step; the words "barrier" for a fitted peak and "critical patch" for a seed
  are withdrawn, ASSUMPTIONS near line 3504, and are not used); with H = 4·anticurl + 4(λ − 1)·curl + tie, f = (0, 0, a,
  0), the cheapest allowed move out split into clean opening against break; whether any placement makes the two slabs'
  openings separable (the only way "half each" is a toy quantity); the O55/O72 counting drive set against 4λ a unit; the
  light field's free energy for a balanced X against the flat torus of the same N (**ours, unverified** that the
  leading ln I₀(β) per square cancels under the rule; report the sign at β = 0.5, 1, 2).
- **Words:** CASCADE / NO CASCADE / BREAK FIRST; SEPARABLE / ONE RELEASE; light LOWERS / RAISES / UNDER ONE UNIT.
- **Ours:** BREAK FIRST; ONE RELEASE (and under the rule that one release is a descent of 4λ·curl, item 6); the renaming
  count cannot pay 4 a unit at λ ≥ 1; UNDER ONE UNIT.
- **Cost:** minutes to an hour. **Waits on:** Q1 (which move set defines "out"), Q3 (scored as hers only after it), items
  3 (a) and 6. Feeds paper 3 with a rule column.

### Item 8. Her north and south pole: a dipole round a trapped knot in the toy's light field? (look; the one toy handle on H4, as a non-test)

- **Kills:** nothing about her mechanism (the toy has no spin, handedness, weak force or black-hole spin); it says
  whether the toy's own field puts a dipole round a knot more often than round flat space.
- **Computes:** `sandbox/u1_gate_topology.py` (untracked output in `sandbox/out`, 24 October) measures a winding number on
  the gate, not flux through closed surfaces; the **surface-flux count is new sandbox code**, costed separately (about an
  hour to write, minutes to run): net flux through every closed square surface round a 6-cube or 3-cube knot in a curled
  shell and in flat space, β = 2 and 8, winding move on. Start states: the shell and the rod do not obey the hard rule;
  said in the docstring. DeGrand and Toussaint 1980 is unread (to read before any citation); that compact U(1) at β = 2
  has poles at a finite density everywhere and at β = 8 almost none is general knowledge, to verify, carried into the
  docstring.
- **Words (corrected so the expected outcome can be scored):** the chance rate of poles and of dipoles (zero net flux,
  nonzero first moment) declared first from the equal flat volume; then AT CHANCE / EXCESS at the knot, per β.
- **Ours:** AT CHANCE at both β. No abandon clause: the toy's field was never a candidate sorter.
- **Cost:** laptop, about an hour and a half in all. **Waits on:** nothing.

### Item 9. Method, before the gated runs: rule 11's four answers for the hard-rule family, and the clock arithmetic for the owner (ours; no run)

- **Does:** (a) drafts, **before items 3, 5 and 6 are read**, the four answers of rule 11 for the case in which they all
  read frozen or none built: does that falsify the hard rule as stated (ours: it says the rule forbids a clean opening
  from flat space by local moves and that no X we can build obeys it at these sizes); implementation, parameters,
  finite size or the mechanism (the held-out L = 12 census of item 3 (c) is the finite-size check); what cheaper test
  tells them apart (item 3 (a)'s four-link word; the soft form at one μ); what would make us abandon the branch (hers to
  set; the draft offers "no clean count-keeping path of any length found and the soft form meets only in damaged
  states"). (b) Gives her the exact clock numbers so the clock decision is made on arithmetic: attempts per fed unit
  1/r on the chain clock at every L; per birth ∝ N_slab; on the fair clock (one fair sweep = N/96 chain sweeps, O90)
  attempts per fair sweep grow faster with N, not slower. (c) Costs a fair-clock mode of the fed-slab runner (new code,
  rule-6 tests) if she wants both clocks scored at one control size.
- **Cost:** an hour of writing. **Waits on:** nothing. Nothing here is run or recorded beyond this file and the handoff.

### Item 10. The soft form μ(S − 3N)² from both sides (measured; Q1 option (a); needs her dated decision; held behind it in the order)

*[27 October, 16:42 UTC: her dated decision is given, "proceed w soften" (VISION Update 133). Still owed: μ's
scaling, her prediction, her clock word and the starting X (questions A1 to A4).]*

- **Kills:** "the soft rule lets clean space open without damage"; "hard and soft forms agree" at the toy's sizes.
- **Runs, cell count written out:** μ ∈ {0.25, 1, 4, 16} × **one scaling fixed in the registration** (μ held fixed, or
  μ₀/N; the reading of Q1 says the scan must fix one in advance, and running both and choosing afterward is the rule-13
  risk; if she wants both, they are registered as separate cells with the same words and no choice afterward) × starts
  {flat, item-6 X, damaged push state} × sizes {4 × 8 × 8, 6 × 6 × 8, held-out 8 × 8 × 8} × g ∈ 4 values × 4 replicas =
  576 jobs, or 144 at one g; **one clock stated in the design** (ours: chain sweeps, 10⁵ a replica; on the fair clock
  10⁵ fair sweeps at N = 512 is 5.3 × 10⁵ chain sweeps a replica, hours a job); the seconds-a-sweep figure at each size
  measured on the laptop before the manifest is sized (Batch runs 96 at once).
- **Words:** MEET / DO NOT MEET per μ (flat and damaged starts agree in ⟨X⟩, ⟨S⟩ and local dimension); FROZEN /
  ACTIVATED (Arrhenius in μ/g, the predicted barrier at least 16μ + 64 from the single switch's loss of 4 squares and the
  cheapest flat move's 64, both exact, O60 and Update 128); DAMAGED (any point with more than three open directions in
  the met state). **Size caveat:** T9's drift (16/g against ln N, owed since 22 September; item 12) is the caveat on
  every warm scan at fixed coupling and is run or stated before this is read.
- **Ours:** ACTIVATED with the Arrhenius slope; DO NOT MEET at μ ≥ 4 within the budget; DAMAGED wherever they meet.
  Abandon the soft form as a route to a clean opening if DAMAGED at every μ where they meet.
- **Owner's prediction:** owed before launch. **Waits on:** her dated decision on Q1 (a) (a new energy term under S1),
  her clock word, items 3 (a) and 6, the T9 caveat.

### Item 11. The push on a rule-obeying X, with "paired in place" read (look, then measured; blocked on Q1 to Q3 and on a kernel)

- **Kills:** Update 127's reading (ours) that breakage is her anticurl pair. **Not** Z2: Z2 ("a balanced X opens with
  curl and anticurl vanishing in pairs") is the assistant's inference, and nothing is scored against her until she
  states or restates it (question 3 below).
- **Kernel line (new code, uncosted until Q1):** `run_sealed_curled_d.py` has no compound-move or soft-form mode. Either
  needs rule-6 tests (networkx brute force, incremental exactness, constraint preservation, same seed): about a day's
  work for the soft form, more for compound moves, whose proposal set item 3 (a) first has to find non-empty.
- **Runs:** `pair_tie_push_look.py`'s protocol (sealed, one push into one point, E bracketing 4λ·curl of one row,
  4 replicas, 10,000 sweeps, clock stated) on item 6's X, λ = 1.10, 1.25, 1.40; **three sizes minimum**, fit on two, the
  largest held out with a number, or the run makes no size claim. Read curl and anticurl per sweep, the census d = 0 to
  6, pieces, and the pairing read: for every curl link the distance to the nearest anticurl link against shuffled
  positions, and Q(r) at r = 1, 2, 3 with "small" declared as within two shuffled standard deviations at r = 3.
- **Words:** OPENS IN PAIRS / OPENS WITH LEFTOVER / BREAKS (any d > 3) / STAYS; PAIRED / UNPAIRED.
- **Ours:** STAYS under compound moves; BREAKS under the soft form at small μ and STAYS at large μ; UNPAIRED wherever
  anything moved. Abandon "a push opens a rule-obeying X cleanly in this family" if BREAKS or STAYS at every setting at
  the held-out size.
- **Owner's prediction:** owed (for the run, and separately her word on Z2). **Cost:** the kernel, then laptop hours for
  the look, 40 to 120 Batch jobs for the registration. **Waits on:** Q1, Q2, Q3; items 3 (a), 6, 7 (and 10 if the soft
  form is chosen).

### Item 12. The record's own debts (method; rules 12, 15, 1)

- **Red team (rule 12):** a fresh Codex session, read-only, started from the commit that records item 1, told by its
  brief not to read O157 to O161 or this file (**blindness is by instruction, verified by the brief's reading list, not
  a pin**), given the repository and five questions, never the conclusions: is T74's size claim an arithmetic identity
  and what did the held-out stage test; could `t74_heldout_score.py` have read anything from the L = 16 files that
  shaped the prediction file (the hash helper of item 2 makes this checkable from the next run on; until then,
  timestamps against `sandbox/heldout_2026-10-23/`); are the ranges' coverage claims consistent with the observed miss
  rate; does the per-birth reader count the first logged row; is the guard fix complete across all release modes.
  Recorded as ~~O162~~ O163 (27 October: O162 was taken by the ledger reading) with what it found and what it got wrong.
- **Prior-work notes (rule 15), drafts for her approval where they touch a paper (her feedback rules apply to every
  draft; TASKS 6 October item 9 (a) marks the paper-1 rewording as hers):** link
  `docs/reading/notes/2026-10-10_prior_work_lambda_above_one.md` from `fertile_window/paper.tex:73` and
  `programme_draft.md:509`; the proposed paper-1 rewording to "simulated or measured" with [KTB19] Sec. 3.3.2 cited, as a
  draft; `did-space-snap-open_v1.html:1001` ("a scrap that stays put") is a **different claim** and gets its own search
  note or the proposed reword "none of our runs holds one"; `relic/paper_plain_language.tex:651` from "Nobody has
  traced" to "we have not found" with a note, as a draft. Two new searches on the template: a fed and gated slab as a
  black hole eating in meals; count-preserving growth as the only allowed move. And the general-knowledge claims this
  program leans on carry "to verify" into each look's docstring ((32₆) and (40₆) exist; the 62-point graph is the
  (6,6)-cage; pole densities in compact U(1); C4 □ K2 = Q3 is checked by hand in item 3 (b) and needs no note).
- **Bibliography and Crossref debt of O159 (rule 1), KCM first because it bears on Q1:** Ritort and Sollich 2003,
  Fredrickson and Andersen, Kawasaki, Costeniuc et al. 2006, Bowick and Giomi 2009, Chatterjee and Diaconis, DeGrand and
  Toussaint 1980, Jaffe; entries with read status in `REFERENCES.bib` before any of them is cited in a paper.
- **T9's drift** (16/g against ln N with errors, several replicas; TASKS T9), the size caveat on every warm run at fixed
  coupling, before item 10 is read.
- **Cost:** one audit session; two agent readings of an hour each; T9 a few laptop hours. **Waits on:** item 1
  committed (red team); nothing (notes, bibliography, T9).

---

## 3. Hypotheses (ours, unverified; each falsifiable and tied to an item)

- **H-a (items 1, 4).** T74's size claim HOLDS away from λ = 1.10, c = 2R because the period is the kernel's arithmetic
  (a N_slab / min(c, R)); the fed-phase periods and daughter energies in T71 and T74 are bookkeeping of the registered
  gate and cap; a per-layer graph-free model reproduces every stage-1 fed-phase period to within the 0.4 to 2.2 %
  residual and the residual grows with meals of 256 and c = 2R. Falsified by IDENTITY FAILS, or by a residual outside
  that range with NO PATTERN.
- **H-b (item 4 (e); restated after the review).** On the chain clock clean-switch attempts per fed unit are fixed by
  construction (1/r), so no size effect can come from them; clean switches **per birth** grow ∝ N_slab. Falsified by
  PER-BIRTH FIXED; the stage-1 pair (7, 7) at L = 8 and 12 already leans against it, and L = 16 decides. The fair-clock
  direction is not claimed until the proposals-per-fed-unit arithmetic of item 9 (b) is written.
- **H-c (items 3 (a), 3 (c), 5).** Under the hard rule flat space is frozen to every local sequence of up to three
  switches **within radius 1 of the first** on the 6 × 6 × 8 torus (one six-link size) and on the four-link torus at
  L = 6 to 12 (held-out L = 12); the first count-keeping move, if any, is at four switches and damages. Falsified by
  FOUND k AT THREE at radius 1, by THAWS AT SMALL L, or by a found four-switch move that OPENS.
- **H-d (item 6).** By whole-row removal or rewiring a rule-obeying curled X can be built only when its anticurl is
  carried by a structured region (rows removed across the slabs, or a separate square-free piece), not hidden in a
  collar or a core of a 4-wide shape; every such X is frozen in its interior. Falsified by a collar or core candidate
  that EXISTS at these sizes, or by MOVES in a rod interior. (Existence beyond these constructions is not claimed.)
- **H-e (items 6, 7).** The toy's twins are mirror-identical by construction (SWAPPABLE); under the rule curling costs
  4λ a unit, so the pair tie gives no cascade, the cheapest exit from any balanced X is a break, and the counting drive
  cannot pay 4 a unit at λ ≥ 1. Falsified by DISTINGUISHED without an asymmetric choice of ours, by CASCADE with a clean
  opening cheapest, or by a counting gain above 4 a unit.
- **H-f (item 5).** From damaged starts the count-keeping moves form a frozen backbone: the touched set stops growing
  within about 200 moves and the clean interior is never reached. Falsified by DIFFUSING or by a clean interior point
  moved.
- **H-g (items 10, 11).** Wherever the soft form lets the flat and damaged starts meet, the met state is damaged, and a
  push on a rule-obeying X either stays or breaks; nothing opens in pairs. Falsified by MEET without DAMAGED, or by
  OPENS IN PAIRS with PAIRED at the held-out size.
- **H-h (item 8).** The toy's light field shows no excess of poles or dipoles round a trapped knot at either β over the
  equal flat volume. Falsified by EXCESS at either β.

## 4. What the toy cannot test (said once)

- **Matter against antimatter at the burp (H4) and the spinning hole's sorting of handed strands:** no matter,
  antimatter, spin, weak force, black-hole spin or time in the toy. Item 8 is the most it offers and is a non-test; a
  signed twist label (Update 97) would test the label's arithmetic, not her mechanism, and is hers to decide. Published
  relatives (Vilenkin 1978; leptogenesis; Boyle, Finn and Turok 2018) are targets, unread.
- **Handedness:** exact, the wiring carries no sign (Update 123); "opposite hand" is untestable until she decides a
  signed label, and item 6 says only whether the twins are even distinguishable.
- **Balanced curvatures (H5):** under the rule curl − anticurl is zero by the rule itself; a curl is a handle, not a
  curvature of either sign, once squares are read as faces, and that reading needs an embedding (item 3 (b)); per-region
  sums report what a construction put there.
- **"Each twin gets half" (hers):** exact, the pair tie fixes the total and the order, not the split; a release per twin
  needs separable openings (item 7), which the toy may not have; and under the rule a balanced X has no 4(λ − 1) release
  at all (item 6), so what "the burp" is under her rule is a question for her, not a measurement.
- **Time, the front's speed, the parent's return of surplus energy, dark energy as w = −1:** no time in the toy, only
  the count of moves on a stated clock; the stiff sheet holds no steady thin density (O146), so the dark-energy test
  waits on soft space's brief and her decision, and T76's L = 32 stays unrun.
- **Anything about the real universe:** every item is a statement about this family of toys.

## 5. Questions for the owner (one line each; Q1 to Q4 are not repeated, see `owner_questions_2026-10-24.md`)

1. **Global or local rule (raised by item 6):** does "squares held at 3N" apply to the whole graph, so a far-away
   square-free piece can balance a curled rod, or region by region (Q(r) ≈ 0 everywhere)? (If global, does a
   disconnected rod-plus-piece X override Updates 39 and 44, X as one connected piece?)
2. **The price of X under the rule (item 6, exact):** a balanced X sits 4λ·curl above flat space with no 4(λ − 1)
   release left; is a descent with nothing to burp still "the burp" in your picture, or does the burp need the rule
   relaxed somewhere?
3. **Z2 (item 11):** is "a balanced X opens with curl and anticurl vanishing in pairs" your prediction, restated in your
   words, or not yours? (Nothing is scored against you until you say.)
4. **The clock (items 4, 9, 10):** chain sweeps, fair sweeps, or both scored at one control size, given the arithmetic
   of item 9 (b) (attempts per fed unit fixed by construction on the chain clock; per birth ∝ N_slab)?
5. **The fifth placement (item 3 (b)):** may a rule-obeying X carry curl and anticurl at the same point (a 5-square link
   beside 3-square links, the non-octahedral vertex link), or must they be apart as in Q2's four placements?
6. **Soft space's brief (Update 40):** still unwritten; may an agent draft it for your decision, or does it wait on you?
7. **Faces (item 3 (b)):** for the Gauss–Bonnet reading, is a "face" defined by the wiring alone (then the rod's
   wrap-around squares are cube faces and the reading is dead as a graph statement) or by the embedding (coordinates,
   which the toy is meant to derive)?
8. **Paper edits (item 12):** approve or amend the proposed rewordings (paper 1 to "simulated or measured" citing
   [KTB19] Sec. 3.3.2; the scrap sentence to "none of our runs holds one"; "Nobody has traced" to "we have not found")?
9. **Predictions owed for registered runs only:** item 4's follow-ups (gate null; per-birth switches at L = 24), item
   10 (the soft form, with your dated Q1 (a) decision), item 11 (the push). The looks (items 3, 5, 6, 8) run without one.

## 6. Red-team log (each refutation and what was done with it)

**Red team 1.**
- Item 1, "only live miss is meals 256" (serious): fixed; all three λ = 1.10, c = 2R cells named live (O145's six MIXED
  cells); Amendment 6's own coverage numbers quoted; the commit hash of the 54 jobs added to the reading (from the
  manifest and workflow log; the `.meta.json` helper goes in with item 2 after item 1).
- Item 2, the rerun already done (minor): restated as a post-fix regression check; exact-equality test dropped;
  short-by-1e-13 test kept.
- Item 3 (b), one-number bucket refuted by O145 (serious): the null is now per-layer (the two stores beside the cut) with
  its kept rules named from `layer_outside.py`, restricted to fed-phase periods and timing words, reading the residual.
- Item 3 follow-up, L = 24 abandon rule decoupled, cost unstated, fair-clock runner uncosted (serious): abandon tied to
  the exponent's range at L = 24; fixed sweep budget and 2.5-hour estimate stated; clock control moved to item 9 (c) as
  costed new code.
- Item 4, four-link word abandoning six-link option (b); kill scoped wrongly (serious): the four-link count abandons
  nothing about six links; reversibility reading added as exact; kill scoped to radius-1 sequences; [Swap17] marked
  triangles, to verify.
- Item 5, rod WRAP ONLY has the wrong sign because C4 □ K2 = Q3; growth look vacuous and promotes a look (serious):
  expectation corrected to ALL FACES; the note's claim marked false as a graph statement; embedding-defined face put to
  her (question 7); growth look dropped.
- Item 6, abandon condition unfalsifiable, (v) makes existence almost free, disconnected X against Updates 39 and 44,
  "ours" assumed the global reading (serious): split into existence by construction (no abandon clause) and the
  global-or-local question (question 1); connectedness decision cited; 4L = 3v balance and (31₆) noted.
- Item 7, 25 October patch look uncited, withdrawn words, light-field line stated as exact (minor): the patch look and
  T54's reading cited; patch defined under the rule before computing; withdrawn words removed; light-field line ours.
- Item 8 (a), cost off by orders of magnitude; N = 18 pairs repeat O159 (f) (serious): walk redefined by moves with local
  recount within radius 2 and full recount every 50 moves; budget 2,000 moves a start; N = 18 line dropped.
- Item 10, Z2 as a kill (owner-decision creep), kernel missing (serious): Z2 removed from kills and put to her (question
  3); kernel line with rule-6 tests added; Q1 the hard gate.
- Item 11, words cannot score the expected outcome; the script measures winding, not flux (serious): words replaced by a
  declared chance rate then AT CHANCE / EXCESS; surface-flux code costed as new; abandon clause dropped.
- Item 12, paper edits are hers; the scrap sentence is a different claim; "pinned" overstated (minor): edits marked as
  drafts for her approval; the scrap sentence given its own note; blindness stated as by instruction.
- H-b unsupported, sign unargued, T74 has no registered clock (serious): H-b restated on the chain clock per birth; the
  fair-clock direction withheld until item 9 (b)'s arithmetic is written.
- Item 9, both scalings run with a free choice afterward; clock presumed (minor): one scaling fixed in the registration
  or separate cells with the same words; one clock stated and scored.
- Missing (1) to (9): hash added (items 1, 2); bibliography debt added (item 12); 25 October patch look and O159 (f)
  cited (items 7, 3 (c)); connectedness decision cited (item 6); kernel and fair-clock mode costed (items 11, 9 (c));
  C4 □ K2 = Q3 used (item 3 (b)); T9 added (items 10, 12); "to verify" carried into docstrings for the general-knowledge
  claims (item 12); predictions owed only for registered runs, nothing launches before they are on the sheet, and the
  programme page is not republished by an agent (header, item 4).

**Red team 2.**
- Item 1 and H-a, two meal cells at risk, 96 % misattributed, "the clock" as a cause of the period, no L = 16 files yet
  (minor): all three cells named; coverage corrected; the clock dropped from rule 11 (2) for the period and kept for the
  clean-switch rate; "no files in the worktree" stated.
- Item 3 (e) and H-b, attempts per fed unit fixed by construction (serious): accepted as exact; the count is now per
  birth; the (7, 7) stage-1 pair set against ∝ N; the L = 24 abandon condition is about per-birth counts.
- Item 3 (b), "third digit" decided before the run (serious): the bucket's kept rules are named; the residual per cell
  (0.4 to 2.2 %) and its pattern are the quantity read.
- Item 2, "no verdict changes" not exact (minor): labelled ours until the rerun reads.
- Item 4, radius-2 prediction has no script; six-link option abandoned on a four-link count; "every size" from one size
  (serious): words matched to the script (radius 1); four-link word separate; the four-link L census with held-out L = 12
  is the held-out size for H-c, whose scope is now radius 1.
- Item 6, the 62-point piece quoted as exact from an uncommitted scratch (minor): not cited as exact until the look is
  committed with its rule-obedience line (S = 0 does not obey).
- Item 6 and H-d, abandon clause unfalsifiable, no held-out number, preempts Q2 (serious): abandon clause removed;
  held-out number named (count-keeping single switches out of each candidate, fitted at two sizes, predicted at the
  third); the shape of X left to her.
- Item 7, T54 already registered and read; cannot run next week (minor): T54 cited; only the pair-tie and rule columns
  computed; the barrier quantity defined under the rule first; moved behind Q1 and Q3.
- Item 8 (a), abandoning option (c) decides Q1 for her; three starts share one origin (serious): the look reports its
  word and stops; the shared origin stated.
- Item 9, cost not derived, clock contradictory, placed before its decision (serious): the cell count written out (576 or
  144), one clock stated, seconds a sweep to be measured first; held behind her Q1 (a) decision.
- Item 10, two sizes cannot hold one out; blocked on Q1 to Q3; Z2 owed twice (minor): three sizes minimum or no size
  claim; listed as blocked; Z2 a separate question.
- Item 11, wrong path; abandon clause abandons a non-candidate (minor): path corrected to `sandbox/`; abandon clause
  dropped; kept as a non-test bearing on H4.
- Item 5, the claim tests the wrong hypothesis; the Euler count allows a fifth placement (minor): check reworded as the
  Euler count on the vertex link; the fifth placement added to Q2's list as ours (question 5).
- Program-wide, predictions owed on looks (minor): owed only for registered runs; looks run now.
- Missing (1) to (7): the balanced X priced before it is called X and put to her (item 6, question 2); a held-out size
  for H-c (item 3 (c)); the clock arithmetic supplied to her as exact (item 9 (b), question 4); the hash helper placed
  (item 2); the per-birth reader moved ahead of item 1 (item 2); rule 11's four answers for the hard-rule family drafted
  before the looks (item 9 (a)); the fifth Q2 placement (item 3 (b), question 5).
