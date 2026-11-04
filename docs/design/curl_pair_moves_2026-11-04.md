# The two kinds as moves: the form to fix before anything runs (4 November 2026, draft for her)

*Written on 4 November 2026 after she asked whether item 1 of `STATE.md` (the dynamics of the two kinds as moves) can
proceed. The dynamics is decided in words (`curl_pairs_build_2026-11-03.md`, section 3; Update 156 addendum). Between
those words and a program lie the seven choices below. Each has a plainest default, **ours**, so that she can answer the
set at once (rule 20: her answers are read back as one set). Nothing here has run. Under S1 nothing runs until she has
answered or accepted each line. Module to build: `graphity.curl_pairs` (moves added to the state that exists); tests:
rule 6 (brute force, exactness of increments, constraint preservation, same seed).*

## Current definitions (from `HYPOTHESIS.md`, Part 2, unchanged here)

- A square holds one real c in [−1, 1]: + is curl on the black diagonal (matter), − anti-curl on the white (antimatter).
- A pair is made as (+c, −c) on the two squares sharing a link. Only |c| = 1 stores energy stably.
- A cancelling pair leaves as two photons when there is room. With no room, two same-twist one-sided half-twist curls
  join into one two-sided full twist in a √2 Klein store, which cannot move, collects photons of matching spin, and pays
  the barrier plus the λ cost.
- λ sets the number of pairs; the freeze sets the surplus. Twist in two dimensions comes in halves only.

## The seven choices (defaults ours; hers to accept or change)

| # | Choice | Plainest default (ours) | What it decides |
|---|---|---|---|
| 1 | **The twist state in halves.** | Twist is not a second number: every square with c ≠ 0 carries τ = ½·sign(c), so no curl, no twist and the sign is the colour. A **store** is the one exception: a flag on a square meaning τ = ±1 (two halves joined) with c = ±1. State per square: c, and store ∈ {0, +1, −1}. | Whether a curl can exist with no twist in two dimensions (she said spin needs twist present; a photon is half a curl). If yes, τ is a second number and the state doubles. |
| 2 | **The size of a made pair.** | Pairs are made full: (+1, −1), costing 2 units of the release. No partial pairs. | Whether "curl is continuous" means partial pairs are made (then: what sizes, and what makes them grow to 1). |
| 3 | **A photon.** | A ledger entry: a cancellation returns 2 units to the ledger, counted as two photons. No object moves on the links. | Whether a photon must be an object that travels (`graphity.twist.Photon`) and can be re-absorbed, or can collect in a store (the store "collects photons of matching spin"). The default cannot do the collecting. |
| 4 | **Room.** | A +1 and a −1 on squares sharing a link cancel when at least one square beside either of them is empty (c = 0). Otherwise there is no room. | Her "if there is space": one empty neighbour, or more. |
| 5 | **The store's price.** | Two same-sign full curls on squares sharing a link, with no room, join into one store on one square (the other emptied) at a barrier B paid from the ledger, plus the λ cost 4(λ − 1) for the square it curls. B is a knob with **no value**: the toy's own Klein relative has a wall of 21 (T102) but she has said her store is something else (owed 3). | B, or a rule that gives it. If B is left open the first run reports every B in a fixed list and claims nothing from it. |
| 6 | **The clock and the freeze.** | Sweeps in a fixed random order (seeded). Each sweep: make pairs while the ledger holds 2 or more; cancel every pair with room; join every same-sign pair without room if the ledger can pay B. The run freezes when a sweep changes nothing. No temperature; no drain to a bath. | Whether the region cools (the sealed stores drain, as in T74) or the freeze is only "no move left". The surplus at the freeze is the result. |
| 7 | **Where it runs, and what is measured.** | A flat R × R torus region with the ledger started at the release 4(λ − 1)·N, R = 8, 12, 16 with R = 24 held out, λ at paper 1's window, seeds fixed. Measured: the net surplus Σc, the stored Σ|c|, the number of stores, each against λ and R. | The geometry (a flat region, or the opened tube six around), the sizes, and her prediction in her words for the held-out size. |

## Her prediction (owed, in her words, before the first run)

What the net surplus does as λ rises and as R grows, and how many stores form. T97 measured, under her first pair rules
on a sheet, a surplus that grows with λ up to one full curl and then stays, its share of the gross the boundary row's,
1/R. The store-to-dark-particle ratio she gave on 4 November (the notebook) cannot be scored on this build, which has
λ units only; it waits for a build with the twist rules' energies.

## What this build can and cannot show

It can show whether her words make a surplus at all, what sets it (λ, the freeze, R), and whether stores form with no
room. It cannot show the billionth, the dark particle's mass, or anything about the sky.

## Her answers (4 November 2026, verbatim in `sandbox/NOTEBOOK.md` the same day) and what was built

1 accepted (a full curl carries a half twist of its sign; a store is a flag). 2 accepted. 3 accepted for the first run
("photons are diagonal waves"; collecting not built). 4: her rule is that a pair cancels when three-dimensional space
can accept the photons, and a failed pair's two halves each join a matching twist, so a failed pair makes two stores of
opposite sign; the sheet's stand-in for "can accept" is an empty square beside either square (ours). 5: no B given
("21 what?": 21 units of E, where a missing square costs 16 and the push is 12); scanned 0, 4, 12, 21. 6 accepted.
7: her prediction, "as lambda rises, we have more net dark matter in the new spacetime. lambda is related to the number
of klein bottles in the network." Built as `graphity.curl_pair_moves`, tests `tests/test_curl_pair_moves.py`, run as
T103: no store forms under this form (the reading is under T103 in `PREREGISTRATION.md`). Her question of the same
day, "can each klein be a store and the set together is a knot?", is a definition (free) with a reader built for it
(the largest connected set of stores), zero in every cell of T103 since no stores formed.

## After T103, T104 and T105 (4 November 2026)

Her two further decisions the same day: the store's price comes from gravity, one-directional, never lost as it is
spent (built as a one-way counter, a count not an energy, ours as the stand-in); and room is judged once on the made
state for every pair at the same instant (her clock; "Yes to 1"). The sequential sweep of choice 6, ours, was what left
no square without room (T103, T104 readings). Under her clock the stores rise with λ (T105, her prediction held).
