# Every registered test, one line each (as of 4 November 2026)

Verdict words are as scored on the record. "Owner" is her prediction and whether it held; "(inf.)" means the
assistant inferred it from her positions and marked it so; "—" means none recorded. "Where" names the archive's
`PREREGISTRATION.md` section and `ASSUMPTIONS.md` O entry, which hold the full reading (rule 17: one fact, one
home). Tests registered in this repository from its creation on go in this repository's own
`PREREGISTRATION.md` and are added here.

| Test | What it asked | Verdict as scored | Owner's prediction | What it means now | Where |
|---|---|---|---|---|---|
| T1–T5 | Tooling: rectangular tori; the full energy with λ; connectivity readers; ergodicity at N ≤ 18; parallel tempering | DONE | — | exact averages at N = 16, 18 validate every sampler; tempering trusted to g ≈ 3 at N = 160 | TASKS T1–T5, Q8, Q9, Q11 |
| Gate A | Reproduce the published λ = 0 behaviour (jump, hysteresis, shattering into knots)? | PASSED (owner's decision) | — | the model reading and sampler reproduce the published λ = 0 case | TASKS Gate A |
| Gate B | Reproduce [T25] Fig. 3 at λ = 1? | OPEN; cannot pass with either variant (matches [KTB19] Fig. 8a to rms 0.005 instead) | — | the two published figures disagree; a question for the authors | TASKS Gate B, O27 |
| Gate B‴ | Is Fig. 3 the printed (half-coefficient) Hamiltonian, our λ = ½? | NOT REPRODUCED at every λ | REPRODUCED or CLOSER: failed | the coefficient is not the explanation | PREREG Gate B‴, O149 |
| Gate C | Reproduce the published six-link curve [T22] Fig. 3? | FAILED | — | six-link results carry the open-gate caveat (U24) | TASKS, O43 |
| Gate C′ | Is the 3D axis a 1/g² reading; is the 2D figure compressed too? | A FAILS; B ONE POWER | — | every axis reading exhausted; question for the model's author | PREREG Gate C′, O53 |
| T6 | Is the change out of the random phase first order? | λ = 0: FIRST ORDER; λ = 1, 1.25, 1.5: INCONCLUSIVE, final | — | out of the random phase at λ ≥ 1 no latent heat above 1.3 per point at N = 100 | PREREG T6, O11, O12, O17 |
| T7 | Is the tube → flat change sharp (memoryless wait, two orders side by side, one front)? | TWO-STATE CHANGE at N = 64, 96, 192 | — (ours held) | the order → order route is sharp; see corrections | PREREG T7, O13 |
| T8 | Does the sharp change hold as λ comes down toward 1? | INCONCLUSIVE by the letter; sharp at every stuck λ; edge between 1.35 and 1.40 | — | — | PREREG T8 |
| T9 | Sealed, does the tube's release feed the change or stop it? | BONFIRE WITH A THRESHOLD | — (ours held) | "melted" class was damage, not melt (O123) | PREREG T9, O14 |
| T10 | Does the leftover grow with the tube's length? | ONE RING, HOWEVER LARGE | — (ours grows: failed) | the leftover is one fixed defect, not a density | PREREG T10, O15 |
| T11 | Does the leftover sit where two fronts meet? | NEITHER | — (ours seam: failed) | a four-point column along the tube | PREREG T11, O16 |
| T12 | Does a two-number coarse law govern the run? | NOT ESTABLISHED | — | front cost fixed with size | PREREG T12 |
| T13 | At λ = 1 is the author's jump equilibrium or the lattice branch surviving on heating? | INCONCLUSIVE; lattice branch seen at every size | — | his protocol's jump is a cold-ascent artefact as far as sampled | PREREG T13 |
| T15 rung 0 | Do the leftovers carry versions (renamings) and distinct spectra? | INCONCLUSIVE by the letter | versions exist and label type: fails by the letter | whole-graph versions are the sheet's symmetries | PREREG T15 r0, O28 |
| T15 rung 3a | Can renaming-versions reproduce the GHZ contradiction? | CLASSICAL, by argument, for every arrangement | strongly contextual: failed | counting versions cannot give quantum correlations | PREREG T15 r3a, O29 |
| T15 rung 0b | Is the closed loop of four a resonator inside the sheet? | INCONCLUSIVE; 6 of 7 DISSOLVED | survives localised: failed | the loop alone is not the object a wave rule acts on | PREREG T15 r0b, O30 |
| T15 rung 1 | Does the interchangeable chain spend time in proportion to the count? | AGREES | — | tool validated | PREREG T15 r1 |
| T16 | Does the published correlation length grow with N? | NONE (his protocol); NOT READABLE (tempering) | — | divergent tendency not reproduced | PREREG T16, O31 |
| T17 | Does each planted seed leave its own leftover? | BETWEEN (slope 0.29 per seed) | ONE PER SEED: holds at k = 2, fails beyond | — | PREREG T17, O65 |
| T18 | Does the room the new space needs grow with the lump? | PROPORTIONAL | — (ours held) | — | PREREG T18 |
| T19 | Does a leftover move, stay or anneal at fixed coupling? | ANNEALS at every coupling | — | a leftover is not permanent at fixed temperature | PREREG T19, O35 |
| T21 | Does a sealed sheet with interchangeable points fold rather than melt? | MELTS EITHER WAY | a closed region folds: failed | — | PREREG T21 |
| T22 | Near λ = 1, do exits fall back into the tube? | FALL-BACKS by the letter | — | first exit on time; recrossings general (κ ≈ 0.6) | PREREG T22 |
| T23, T24 | The λ map at 120 decays per cell: sharp across the window; break-up at the edge? | window INCONCLUSIVE; edge BREAK-UP BEGINS AT THE EDGE | SHARP and BREAK-UP: edge held, window not by the letter | — | PREREG T23, T24, O44, O52 |
| T25 | Does a planted scrap freeze in as the box cools? | FREEZES IN | FREEZES IN (inf., confirmed): held | — | PREREG T25, O47 |
| T26 | Does energy packed into one place re-curl cold space? | MELTS (26 of 27 cells) | RE-CURLS: failed | — | PREREG T26, O48, O64 |
| T27 | Does a melt fold before it flattens when its energy drains? | STAYS MELTED | FOLDS (inf.): failed | — | PREREG T27, O48 |
| T30 | Six links: does one push open both curled directions of a rod? | tori NEVER OPENS; gas FIRST ONLY | ALL AT ONCE: failed | — | PREREG T30, O54, O64 |
| T32 | Six links: is the activation fixed with size? | FIXED WALL, sharp at 16 | FIXED WALL: held | — | PREREG T32, O54 |
| T33 | Eight links: in what pattern do four curled directions open? | NO CASCADE | tied pattern: not supported | the push, not the tie, was tested | PREREG T33, O62 |
| T34 | Six links: does concentrated energy fold flat space? | MELTS (no point folded) | RE-CURLS: failed | "melt" = flickering scars (O82) | PREREG T34, O69 |
| T36 | Do low-square regions persist at λ = 1 (allotropes)? | TRANSIENT | ALLOTROPES (inf.): failed | — | PREREG T36, O59 |
| T37 | In long tubes with many natural seeds, does the scrap grow with the space? | MANY SEEDS, MANY SCRAPS | (inf.): held | the long warm tubes melted; "mosaic" withdrawn | PREREG T37, O77, O88 |
| T38 | Are the rare long waits a second population? | TWO POPULATIONS | (inf.): held | most "waits" were the detector's (O109); see T58, T59 | PREREG T38, O84, O109 |
| T39 | Is there a bath size at which the first release pays the second wall? | NO WINDOW | WINDOW with CASCADE (inf.): failed | — | PREREG T39, O81 |
| T40 | How big a push starts four curled directions? | NEVER STARTS (pushes 20 to 160) | tied pattern (inf.): not reached | — | PREREG T40, O81 |
| T41 | Three directions: does the new space need room for the burp? | NEVER OPENS | ROOM NEEDED (inf.): failed | — | PREREG T41, O81 |
| T42 | Does concentrated energy fold six-link space with interchangeable points near λ = 1.02? | interchangeable HEALS; named MELTS by the letter | RE-CURLS (inf.): failed | named did not melt either: one flickering scar (O82) | PREREG T42, O82 |
| T43 | How long does a planted allotrope last at λ = 1? | DISSOLVES at every coupling | ALLOTROPE LASTS (inf.): failed | — | PREREG T43, O83 |
| T44 | Under the follow tie, does one push open all three directions of a cube gas? | NOT ALL at every κ | ONE PUSH OPENS ALL (inf.): failed | — | PREREG T44, O79 |
| T45 | Which budget-fitted order of openings opens with one push? | NOT ALL in all four settings | B (inf.) and C (hers): failed | the 5.36 fit's target later withdrawn (O89) | PREREG T45, O79 |
| T46 | Her four-direction order with a tie: one push opens all? | NOT ALL | ONE PUSH OPENS ALL: failed | — | PREREG T46, O79 |
| T47 | A: is the front's speed fixed? B: does flat four-direction space re-curl when time keeps energy? | A MIXED by the letter; B HEALS | ACCELERATING: failed; FRONT: failed | the fair clock (O90) came from A | PREREG T47, O76, O80 |
| T48 | Does a connected curled torus open into one space when released energy stays local? | control ONE SPACE (1.25) / ADVANCES ONLY (1.40); two-curled ADVANCES ONLY / DAMAGED; one-curled STAYS | ADVANCES ONLY (hers for ≥ 2 curled): held in 2 of 4 settings | — | PREREG T48, O91 |
| T49 | Does a single hypercube open all its directions, and in what order? | 6-cube PART OPEN; 8-cube STUCK | ours adopted as hers: holds where stuck | — | PREREG T49, O92 |
| T50 | Does the tube's front slow where energy sits? | NO EFFECT (1, 3); SPEEDS (6, 10) | SPEEDS: held at 6 and 10 | the front runs opposite to clocks near mass | PREREG T50, O93 |
| T51 | How does the frozen scrap fall as cooling slows (fair clock)? | CLIFF; P1 fails at one point; P2 holds | GENTLE: failed | columns heal; pieces of 2 and 8 do not | PREREG T51, O104 |
| T52 | What does a cut round a relic hide (the hidden count)? | C1 FAILS at r = 1; SOMETHING HIDDEN; scaling NOT READ | flat hides 1: fails at r = 1 | the entropic route neither opened nor closed | PREREG T52, O137 |
| T53 | Does the rod open by itself in a warm bath, from one place or several? | ADVANCES ONLY at every (L, λ); SEVERAL | OPENS IN A WINDOW: failed; SEVERAL: held | a steady bath opens the first direction only | PREREG T53, O105 |
| T54 | The smallest opened patch of a slab that grows, priced exactly (held out L = 16) | held-out numbers HOLD; words RISES AGAIN (unscored) / FIXED WALL | CRITICAL PATCH: not scored / fails by the letter | the numbers are completed-patch energies, not barriers (O118) | PREREG T54, O118 |
| T55 | What are the pieces of 2 and 8 that no cooling removes? | KNOT at both sizes | SEAM: failed | the scrap is three knots of fixed energy | PREREG T55, O107 |
| T56 | Under "all at the last", does the slab open in a warm bath? | STAYS / ADVANCES ONLY; control NO DIFFERENCE; SEVERAL | OPENS into ONE SPACE: failed | the tied slab breaks; stage 2 not run | PREREG T56, O115 |
| T57 | Does too much concentrated energy curl flat six-link space under the tie? | MELTS in all four groups | RE-CURLS: failed | — | PREREG T57, O114 |
| T58 | What were the three long waits at λ = 1.30 doing (replay)? | MIXED (1 LOW THRESHOLD, 2 TRUE WAIT, 0 SECOND CURL); P1–P3 hold | SECOND CURL: failed | first exit is one memoryless population | PREREG T58, O110 |
| T59 | Were the λ = 1.05 long waits chance, and is the 7 % the looks? | TRUE WAIT; OFFERED AS COUNTED; ON THE COUNT; HIDDEN EXITS | chance; hidden exits: held | paper 1's count stands | PREREG T59, O112 |
| T60 | The λ map a fourth time, wait timed without the detector, held-out size | SHARP ACROSS THE WINDOW; BREAK-UP BEGINS AT THE EDGE; AS PREDICTED at N = 288 | SHARP and BREAK-UP: held | red-team pass owed before paper 1 quotes it | PREREG T60, O113, O117 |
| T61 | If the burp's heat stays, does the tube come out wider? | PATCHWORK at every λ and size | WIDENS: failed | no clean torus wider than six in 720 end states | PREREG T61, O121 |
| T62, T64 | With a price on loops of six, does the tube widen? | WIDENS at μ ≥ 0.05–0.20; STAYS SIX AROUND at ≤ 0.10 (T62) | WIDENS at every price: held at 3 of 4 | the wide product is flawed nearly everywhere | PREREG T62, T64, O127, O128 |
| T63 | With that price, what does the curled tube open into? | NOT READ (control bar mis-set) | — | rerun as T64 | PREREG T63 |
| T65 | Does the tied slab open from one spark? | no cell OPENS; tied PARTIAL / DAMAGED; control FALLS BACK; held-out AS PREDICTED | opens in room: failed | one spark buys a sprinkle of pairs | PREREG T65, O130 |
| T66 | Does flat 3D space heal damage faster than 2D? | FASTER IN 3D at 216; MIXED at 512; held-out not run | FASTER at every size: failed at 512 | the observable measures a fluctuation (flaw disclosed) | PREREG T66, O129 |
| T67 | Does the untied slab open from a fixed-rate feed? | BREAKS / SITS; held-out AS PREDICTED | OPENS from one spot (inf.): failed | — | PREREG T67, O132, O133 |
| T68 | Does a varying or many-point feed open the slab? | MIXED; no OPENS or FRONT; held-out 106 of 108 cells | OPENS at some setting (inf.): failed | — | PREREG T68, O134, O136 |
| T69 | Does the lowest path out of the slab pay for itself? | MIXED / AS PREDICTED by cell | PAYS: not supported within reach | the first clean opening costs 64 | PREREG T69, O135 |
| T71 | Does the slab cycle under the layer form (gate g1, daughter ledger)? | CYCLES everywhere; held-out AS PREDICTED (152 of 152) | held, one cell excepted | the control was underfed per point; no graph mechanism for the burp (a graph-free bucket reproduces the births) | PREREG T71, O144, O160 |
| T72 | When is an observer like us most likely, by its universe's age? | TYPICAL under A, B, C, D | "agree" with ours: fails on C | the counting unit decides, not the stars | PREREG T72, O138 |
| T73 | Does the causal-diamond weight move the observer peak? | MOVES | peaks move: held; no word changes: failed | — | PREREG T73, O139 |
| T74 | Fed by meals through a curl edge, does the slab burp, wait, dissolve? | BURPS 102 of 108; cold DISSOLVES, warm FED; held-out HOLDS | ours adopted as hers: fails in part | the period is arithmetic by construction | PREREG T74, O145, O161 |
| T75 | Does a disturbance spread in the toy's space at one speed? | NO SIGNAL at every (N, g) | SPREADS: failed | far-reaching switches make jumps, not signals | PREREG T75, O140 |
| T76 | Does the density of missing squares stay constant as space grows? | no steady thin density; option 2 MIXED | CONSTANT: failed | the stiff sheet cannot hold a thin density; moves to soft space | PREREG T76, O141, O142 |
| T77 | Does a fading dark energy make our age early under the diamond weight? | fading rows lower every share | constant: held | — | PREREG T77, O143 |
| T78, T79 | Does a curled ring lose more entropy than a flat patch; is it a penalty on curl or on density? | CURL PENALISED, FLAT; PENALTY ON CURL | CURL PENALISED, GROWING: fails on growth; PENALTY ON CURL: held | λ > 1 is not caused by the network's curvature | PREREG T78, T79, O147, O148 |
| T80, T81 | Random-walk dimension; Eq. (45) in form | DRAFTS, not run | owed | — | PREREG T80, T81 |
| T82 | The twin gate | WITHDRAWN before any run | — | — | PREREG T82 |
| T83 | Is curl on a warm 2D sheet felt at a distance? | LONG by the letter; a defect's own footprint in substance | SHORT: fails by the letter (gate was the assistant's error) | reach needs the links of three dimensions | PREREG T83, O165 |
| T84 | Do two relics pull together via relaxed link lengths, and is light delayed? (exact landscape, held out L = 24) | SHAPE ONLY | CLOSES and DELAYED: held; our band failed | — | PREREG T84, O167 |
| T85 | Does one α fix both light's delay and the pull (γ = 1)? | DRIFTS ACROSS b and KNOTS; held-out numbers held | ONE α: fails by the letter | the drift across b was the test's own definition | PREREG T85, O168 |
| T86 | Which object carries a wave under the adopted rule? | LEAKS for every kind | CARRIES (inf., confirmed): failed | nothing the toy makes confines a wave | PREREG T86, O169 |
| T87 | Does a curled loop of four have a sign of its own? | SIGNED at every radius | SIGNED (inf.): held | the sign is the bipartition's | PREREG T87, O170 |
| T88 | Does a rod fed at one end open as one front from that end? | NOTHING below the wall; ONE FRONT at 30 a store; SEVERAL at 60; held-out holds | ONE FRONT (inf.): holds at 30, fails at 60 | the front is paid by the fed stores, not by its release | PREREG T88, O171 |
| T89 | Does the crossover coupling drift with ln N at λ = 1 (held out N = 1156)? | UNSCORED by the registered gate; LOGARITHMIC DRIFT (amendment and stage 3) | — | TASKS T9 closed: the coupling falls as 1/(a + b ln N) | PREREG T89, O172 |
| T90 | Does the quantum toy on the class spaces behave? | BEHAVES, PREDICTIONS MISSED | — | quantum switching favours wirings with most moves | PREREG T90, O173 |
| T91 | Carroll's mutual-information reader on the toy's states | MIXED; held-out size claim HOLDS; sheet word MISSED | — | the flat lattice reads 3 to 4 (taxicab) | PREREG T91, O174 |
| T92 | Does a curled X pay an entropy penalty against flat space at λ = 1? | CURL FAVORED at both sizes | warm X prefers to open: failed | — | PREREG T92, O176 |
| T93 | Does the opening start at a zero-dimensional cell inside the rod? | NOTHING OPENS (design error: rod melted) | not tested | an even per-vertex feed cannot ask the question | PREREG T93, O177 |
| T94 | Does a window with stored matter read differently through Carroll's reader? | READS DIFFERENTLY in every case | READS DIFFERENTLY: held | — | PREREG T94, O178 |
| T95 | Do move paths count the same thing as renamings? | NEITHER (they run opposite ways) | — | "the options" are copies, a third count | PREREG T95, O179 |
| T97 | The curl-pair daughter: does a surplus grow with λ? | GROWS THEN FLAT; surplus share = 1/R | A SURPLUS GROWS WITH λ: holds up to the full curl | the billion-to-one picks a size, not a λ, under this form | PREREG T97, O180 |
| T98 | Carroll's reader with shares on black diagonals only | READS DIFFERENTLY; exactly 2 at R = 3 | READS DIFFERENTLY: held | — | PREREG T98, O181 |
| T99 | Does λ move the leftover's share of the release? | λ-FREE at the edge | λ-SET (inf.): failed | the share counts knots; no λ nears a billionth | PREREG T99, O183 |
| T100 | The γ test in its consistent form on fresh knots | DRIFTS ACROSS b and KNOTS; NO DELAY at one point; held-out held | ONE α (inf.): failed | — | PREREG T100, O182 |
| T101 | Does the toy's own clock (moves per point) run slower near a knot? | FASTER NEAR KNOTS | SLOWER (inf.): failed | the move count runs opposite to clocks near mass | PREREG T101, O184 |
| T101 Am. 1 | The same at λ = 1 with one planted knot, g = 1, 1.5, 2 | ran overnight 3 to 4 Nov; all nine replicas ended ≥ 98.9 % flat; **not yet read by its registered words** | SLOWER (inf.): pending | — | PREREG T101 Am. 1; O186 reserved |
| T102 | Is the toy's Klein relative a store or a knot? | KNOT (32(λ − 1) at every L; wall ≈ 21) | STORE (inf.): failed | in this toy a Klein store is a knot of fixed size | PREREG T102, O185 |
| T103 | The two kinds as her moves on a flat torus: does the number of Klein stores rise with λ? | MIXED (2 stores in 240 cells; effectively none at every B) | RISES: failed | the ledger is spent making pairs before any store can be paid, so under this form stores cannot form; the same-sign curls left with no room, the raw material for stores, rise with λ (0.01 to 0.18 per square); net twist exactly zero; books balance to the digit | this repository's PREREG T103 |
| T104 | T103 with the store's price from gravity (a join costs the ledger nothing): do stores rise with λ now? | MIXED (2 stores in 240 cells; effectively none) | RISES: failed | the cause is the sweep's order, ours: cancelling link after link before joining empties the sheet and leaves no square without room; with room judged all at once (sandbox look) stores form and rise with λ; the clock form is hers (T105) | this repository's PREREG T104 |
| T105 | T104 under her clock (room judged once on the made state, all pairs at the same instant): do stores rise with λ? | RISES (none at λ ≤ 1.10; 0.10, 0.18 to 0.23, 0.42 to 0.48 per square at 1.25, 1.50, 2.00; held-out size 24 within ±30 %) | RISES: held | under her rules and clock, stores form once the release packs the sheet and their number rises with λ; both signs equal; net twist zero; stores sit together more as λ rises | this repository's PREREG T105 |
| T107 | The seed on the boundary of the loose remnant, and the kind on each side: a foot on each side; the outside foot grows, the inside stops; the feet joined (read); mirror images under her kind rule; the size (read) | MIXED (1 HOLDS 0.75 to 1.00, 0.85 held out; 2 outside larger at +500 HOLDS at 32, 48, 64 and FAILS at 128 (0.35), the inside bounded HOLDS; 3 joined 0.75 to 1.00; 4 mirror images HOLDS, 0.68 to 0.81, the sign matching the outward kind in 54 of 57; 5 the size 0.021 to 0.009, falling) | her five (hers by adoption): 1, 2b, 4 held at every length; 2a failed at the held-out length | the seed straddles the loose remnant's edge and her kind rule sets the sign at every length; "larger at a fixed time" was a criterion that scales with the band (said before the run); the replacement is growth, T109 on her words; the size's trend is T108 | this repository's PREREG T107 |

Not registered or never written: T14 (became T17), T20, T28 (the gravity test, designed only), T29, T31, T35,
T70, T96 (not needed once "the options" were decided).

## Readings corrected after scoring (the verdict stands; the picture does not)

- **T7 and S2′.** S2′ was written after T7 had met four of its five parts; T7 is never presented as having cleared it.
- **"Opens into a sheet" (T7, T8, T23, T24, T60).** 9,208 of 9,215 exactly flat end states are the tube with its seam shifted by two, six around (O116).
- **T9 "melted".** A sealed tube cannot melt at these λ; every "melted" run kept 94 % or more of its squares (O123).
- **T10, T11 "ring".** A four-point column lying along the tube (O16).
- **T30.** First read FIRST ONLY (the analyser had pooled the gas); as registered the tori are NEVER OPENS.
- **T34, T42, T57 "melt".** One to a few flickering scars, nothing curled (O82).
- **T37.** The long warm tubes melted; the "mosaic" and the "factor of ten" readings are withdrawn (O88, O89).
- **T38.** TWO POPULATIONS supersedes UNCLEAR; O109 then read most long waits as the detector's; T58 and T59 found one memoryless population.
- **T45, T46.** The 5.36 target of their fits is withdrawn (O89).
- **T54.** The numbers are completed-patch energies, not barriers (O118).
- **T56.** The one rod replica that read OPENS is a rod glued differently; stage 2 is not run, so nothing may say the slab stays shut "at every size".
- **T61.** The "flat and eight around" end state is two pieces (a torus and a Klein bottle).
- **T65.** The tie was withdrawn after launch; the untied control is the cell that speaks to the hypothesis.
- **T66.** The observable measures a fluctuation, not healing speed; disclosed, not repaired.
- **T71.** The control was fed 1/N_slab of the arms' feed per point; a properly fed control is a new cell.
- **T83.** The scoring gate admitted near-perfect sheets; LONG is a defect's own footprint.
- **T84.** The chain form was withdrawn; the exact landscape is what was scored.
- **T85.** The drift across b for an isotropic knot is the test's own definition; T100 is the consistent form and still drifts by ±30 %.
- **T89.** UNSCORED by the registered gate (the lattice branch on the heating leg); LOGARITHMIC DRIFT under a post-hoc amendment and a prospective stage 3.
- **T93.** The rod melted because the design fed every store above the wall.
- **T102.** Its registration text was appended one minute after its run began; the predictions were written before and unchanged.
