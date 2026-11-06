# The toy, term by term, and the code map (4 November 2026)

Every test uses one energy, built from terms added one at a time. A run switches on the terms it needs; the
energy is their sum. Each term says where it came from: **published** (taken from a paper, equation named),
**ours** (added by this programme, with its dated decision in the archive's `VISION.md`), or **looks only**.
Written for the four-link toy; with six links the first term reads 16(3N − S).

```
E = 16(N − S) + 4X                              terms 1, 2: the published model (total curvature)
  + 4(λ − 1) X                                  term 3: the curling cost (ours)
  + Σ_points f(d_p)                             term 4: the pair tie, looks only (ours)
  + μ_N N                                       term 6: the layer form's price per point (ours; zero so far)
  + (κ/2) Σ_links (φ_p − φ_q)² + q Σ_points e_p φ_p    term 7: soft space, built, nothing run (ours)
  + ½ δᵀ K δ + α Σ_links |c_e| δ_e              term 8: lengths on the links; K exact; the scale mode fixed by her rule (ours)
  + β Σ_squares (1 − cos θ_□)                    term 10: lattice light, built (published: Wilson's lattice gauge theory)

E + stores + daughters = energy at the start + energy fed        term 5: the books (stores: Creutz's demon, published; the rest ours)
S = N  (S = 3N with six links)                                   the zero-total rule: curl = uncurl (ours, decided; softened)
16(3N − S) + 4λX = 4·uncurl + 4(λ − 1)·curl                      exact, six links: why uncurl costs more than curl
```

Symbols: N points (under the layer form the count can change); S small squares; X surplus squares (each link on
a third square counts one at four links, a fifth at six); λ the curling knob (1.25 in most runs; at λ = 1 the
toy is the published model, with the factor-2 correction its author reported on 23 October 2026, so the printed
local term is our λ = ½); μ_N the layer form's price per point; φ soft space's number on a point; e_p the curling
energy at a point; δ_e a link's length above the lattice's; K the stiffness of the published curvature in the
lengths (exact: 4 times the lattice Laplacian on each axis family); α the one constant coupling a link's curl to
its length (negative, so light is delayed near matter); θ_□ the light field's phase around a square and β its
stiffness; d_p the number of open directions at a point. Curl on a link is a square above flat, uncurl a square
below.

| Term | What it charges or does | Source | Used in |
|---|---|---|---|
| 1 | Square reward, 16(N − S). Flat space closes its share of small squares; each missing square costs 16. | Published: Trugenberger's review [T25] Eqs. (8), (21), (22); [KTB19]. Checked against the text. | Every run |
| 2 | Local term, + 4X. A link on a third square pays back what the extra square earned. With term 1, the published total curvature. | Published: [T25] Eq. (22) with the author's factor-2 correction ([Q28] in the archive) | Every run |
| 3 | Curling cost, + 4(λ − 1)X. Each link curled tight costs extra; zero wherever space is smooth; the burp per point is this cost. | Ours, 24 Sep 2026 (Update 23) | Every run at λ ≠ 1 |
| 4 | Pair tie. A point with exactly one of the pair's two curled directions open pays one curling cost, so curling two costs the same as curling one; the whole release comes when the second opens. | Ours, 25 Oct (Update 124), her decision, form to confirm | Looks only; piece 5 |
| 5 | The books: stores, feed, gate, daughters. Stores hold spare energy so the total is conserved to the last digit; a feed adds energy at a set rate; the gate lets the slab split only when the stores beside the cut hold its whole curling cost; the release leaves into a daughter's ledger. | Stores: Creutz's demon (1983). Feed, gate, ledger: ours (Updates 66, 69) | Sealed and fed runs |
| 6 | Layer form, + μ_N N. A whole layer compacts, six around to four, or opens again, with no point broken; μ_N allowed and zero so far. | Ours, 18 Oct (Updates 63 to 65) | T71, T74 |
| 7 | Soft space, first stage. One number on each point that slows its clock and stretches its links; waves of one speed; dips where knots sit. | Ours, 20 Oct (Update 77); its class is a stratified scalar theory (Bragança and Lemos 2018) | Built, nothing run |
| 8 | Lengths on the links. A length on every axis link; the published curvature's second-order change is 4 times the lattice Laplacian per axis family, gapless, uniform scaling free; a link's curl couples to its length with α; the total length fixed near matter (her scale rule). | Ours, 28 Oct (Updates 136, 137, 143); the carrier of gravity by her decision of 4 Nov (Update 157) | T84, T100; pieces 8, 11 |
| 9 | Amplitudes and records (not energy): versions carry amplitudes, and a record in the surroundings keeps them apart. | Published quantum mechanics (decoherence); adopted 15 Oct (Update 57) | An exact toy only |
| 10 | Lattice light. A phase on every link, squares as its loops; light is its travelling wave, never faster than one link per tick. | Published: Wilson's lattice gauge theory. Built 24 Oct (Update 96) | Looks only |
| rule | Zero total. Curl and uncurl held equal: squares equal points. The published hot phase is forbidden. Exact: clean six-link space cannot move under the hard rule by any one, two, three or restricted four switches; softened 27 Oct (Update 133). | Ours, 25 Oct (Updates 105, 108, 133) | Exact looks only |
| rule | Growth in empty space. A double row is added to every empty ring with a fixed chance per sweep, never removed: a one-way clock. | Ours, 25 Oct (Updates 108, 118, 119); brief before anything is built | Nothing yet |
| state | Curl and anti-curl on the squares: one c in [−1, 1] per square, sign the kind (+ black diagonal, matter; − white, antimatter), pairs with equal absolute value, only \|c\| = 1 a stable store. No dynamics yet. | Ours from her rules, 3 Nov (Update 156); `docs/design/curl_pairs_build_2026-11-03.md` | T97 |
| moves | The two kinds as moves on the curl state: full pairs across shared links, cancellation into light where there is room, a Klein store (two same-sign halves joined, a flag) where there is none; seeded sweeps, room judged once on the made state (her clock, 4 Nov), the store's price from gravity (a one-way counter), freeze when nothing changes. | Her rules, 3 and 4 Nov; stand-ins ours; `graphity.curl_pair_moves`, `docs/design/curl_pair_moves_2026-11-04.md` | T103, T104, T105 |
| twist energies | A reader, not a term: an energy in MeV for a dark particle, a Klein store, the electron and the proton, in three parts, the count part (hers, the electron as the unit), a twist part (the elastic-rod form, a stand-in of ours, constant free) and a store's binding (free). Exact with nothing fitted: under the rod form a store is 4 dark particles and the dark particle is the heaviest per length; her billion would need a binding of about 170 TeV per store; on the sheet stores are one in five of the pairs, not one in a billion (T105). | Her rules; stand-ins ours; `graphity.twist_energy`, `docs/design/twist_energies_2026-11-04.md` | none (nothing fitted or registered; her answers owed) |
| state | Twist on the diagonals: particles as loops of diagonals with twist, curl and spin; net twist conserved under pair creation. | Ours from her rules, 30 Oct; `docs/design/twisted_diagonals_build_2026-10-30.md` | Exact checks |
| rules | How the toy moves. Points of two kinds, each with four (six, eight) links; no two points share more than two neighbours (the hard-core rule); a move swaps two links; a move raising the energy by ΔE is taken with chance e^(−ΔE/g) at coupling g. In some runs arrangements that differ only by names count once (interchangeable points). | Published: [T25] Fig. 1b, Eq. (28); Metropolis. Interchangeable points: Betre and Lewis 2025, Eq. (72) (Update 12) | Every run |

Naming: at λ = 1 and with terms 1, 2 only, the toy is Trugenberger's model. Anywhere else it is our family
around the published model, never "combinatorial quantum gravity" (the model's author's condition).

## What is exact about the toy and stays true whatever the sampling

- **Zero is the floor.** The hard-core rule forbids a triangle beside a square, so no valid arrangement sits
  below flat space: space is stable, not metastable.
- **No long-range force at fixed wiring.** H is a sum of per-edge terms, so defects sharing no square are
  exactly additive; an attraction of 16 only where they touch. Flat space is gapped (the cheapest move out costs
  32, 64 or 128 at four, six, eight links), and the warm sheet's correlation length is about two links. Nothing
  living in the wiring alone carries a pull farther than a step or two; that is why the carrier is a length on
  the links.
- **The curled torus** (the tube, 4 × L, one direction curled) sits 4(λ − 1) per point above flat space at λ > 1;
  the cheapest move out costs 12 at λ = 1.25 and the push that starts it is 12 at every size from 48 to 288.
  Below λ = 1 the ladder inverts (folding pays). A fully curled X with every direction closed can be one
  connected piece only up to 4^D points; larger, it is a gas of hypercubes.
- **What the tube opens into** is a tube re-glued six around, not a wide sheet (9,208 of 9,215 exactly flat end
  states); "flat", "opens" and "one space" are counts at a point or an edge, not extents.
- **The wiring pairs curl and anti-curl**: every square of a bipartite wiring has one black-to-black and one
  white-to-white diagonal, so a surplus of one kind lives only in the state on the squares.
- **A loop is one-sided exactly when its total twist is an odd number of halves**; no loop of thirds is
  one-sided; the cheapest one-sided loop costs 3/4.
- **Renamings and copies are different counts.** A melted graph has exactly 1 renaming, a perfect 160-point
  sheet 320, a shattered gas about 10^29. The count over paths of the move is a third count, anti-correlated with
  the renamings (T95). "The options" are copies, L = (n!)²/A (her decision).
- **The Klein relative** of the tube (4 × L with a reflection seam) is valid, bipartite and paired, has 16
  renamings at every length, is a strict dip behind a wall of about 21, and sits one double column's energy,
  32(λ − 1), above flat at every length (T102).
- **A sealed tube cannot melt** at the curling costs run: its own energy can break at most about one square in
  sixteen at λ = 1.25. Melting needs energy from outside. Write "damaged" or "flawed", never "melts".
- **The fair clock**: one fair sweep is N/96 chain sweeps, so every local pair of links is offered equally often
  at every size.

## Code map (`src/graphity/`)

| Module | What it is |
|---|---|
| `cqg.py` | The four-link kernel: the published model plus the curling knob; `run_chain(..., lam, cap, glauber)` returns squares, surplus squares, acceptance. A unit test pins the capped path bit for bit. |
| `cqg_d.py` | The same at any number of links (six for three directions, eight for four); at four links draw for draw with `cqg.py`. |
| `squares.py`, `connectivity.py`, `dimension.py` | The readers: squares on an edge; pieces, largest piece, baby universes, hypercubes; the local square-count signature d(v) and connected pieces of a vertex set. |
| `graphs.py`, `small_graphs.py`, `symmetry.py`, `hidden.py` | Tori and starting states; every valid state at N ≤ 18 (exact averages, the sampler's validation); the renaming count; the hidden count of a region. |
| `sealed.py`, `sealed_d.py`, `sealed_tie_d.py`, `spark.py` | Sealed and leaky runs with a bath of stores (energy conserved to the last bit); the direction tie (archive history); the local spark. |
| `tempering.py`, `wang_landau.py`, `wl_ising.py` | Parallel tempering (validated against exact averages at N = 18); flat-histogram sampling and its Ising check. |
| `interchangeable.py`, `interchangeable_d.py` | Chains in which the points are interchangeable (weight divided by renamings). |
| `contraction.py`, `clean_move.py`, `layer_feed.py`, `layer_outside.py`, `row_growth.py` | The layer form: whole layers compact or open, the clean two-switch move, the fed slab with its gate and daughter ledger (T71, T74), growth by whole rows. |
| `soft_d.py`, `soft_field.py`, `edge_wave.py`, `lengths.py`, `regge.py` | The zero-total rule in its soft form; soft space (term 7); dynamical edge waves; lengths on the links with the exact stiffness (term 8, the gravity carrier); Regge-style curvature in lengths. |
| `u1.py`, `diagonals.py` | The lattice photon (term 10); body diagonals derived from the cube edges, and light on them. |
| `twist.py`, `curl_pairs.py` | Particles as loops of diagonals with twist, curl and spin (net twist conserved); curl and anti-curl as a state on the squares. |
| `coupled.py`, `correlation.py`, `exits.py`, `objects.py`, `sixloop.py` | Two chains on one random stream; the published correlation length; every exit from the tube and every fall-back; two nine-square objects counted exactly; a price on loops of six (ours). |
| `energy.py`, `mc.py`, `full_curvature.py`, `general_chain.py` | Parked: Konopka's graphity model and the full Ollivier energy with triangles and pentagons. |
| `analysis.py`, `results.py` | Error bars and autocorrelation; writing results without ever overwriting. |

`scripts/` holds the runners and analysers of every registered test (named by test number where one exists);
`configs/` every config that was run; `tests/` the rule-6 checks. Sizes: the two-direction tube is 4 × L;
disorder → order runs use N = 4p² with p prime (the model author's advice); the slab is 4 × L × L.

## What the toy cannot do (so that nobody asks it to)

It has no time (only the count of moves, which runs faster near knots, T101). Its flat space takes no move at
all below a coupling of about 6 and melts above it, so no disturbance travels at a steady pace in the stiff toy.
No six-link knot is a dip, so no knot moves in three directions. Its energy cannot tell curl from anti-curl; the
state on the squares does. It counts no observers. Nothing in it is about the real universe.
