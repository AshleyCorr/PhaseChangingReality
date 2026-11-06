# Pre-registration

Tests registered in this repository from 4 November 2026 on. T1 to T102 and the gates are in the archive's `PREREGISTRATION.md`; `RESULTS.md` here has one line each. Numbering continues from T103. Each section: the question, the design with sizes and seeds, the held-out size where a size claim is made (rule 14), the owner's prediction in her words (or marked inferred), ours, the verdict words, and, after the run, the Reading.

## T103. The two kinds as moves: does the number of Klein stores rise with λ? (registered 4 November 2026, before the run)

**Question.** Under her rules for the two kinds as moves (`docs/design/curl_pair_moves_2026-11-04.md`; her answers of
4 November in `sandbox/NOTEBOOK.md`), on a flat torus whose ledger holds the release 4(λ − 1) per point: how many
Klein stores stand at the freeze, and how does that number move with λ?

**Design.** `configs/t103_curl_pair_moves.json` through `scripts/run_curl_pair_moves.py`
(`graphity.curl_pair_moves`). Sides 8, 12, 16, with **24 held out** (rule 14); λ = 1.05, 1.10, 1.25, 1.50, 2.00 (ours;
paper 1's 1.25 among them); barrier B = 0, 4, 12, 21 in the toy's energy units, scanned because she gave no value
(21 is the toy's own Klein relative's wall, T102, which she has said is a different object); three replicas, seeds
in the config. Moves per sweep: make full pairs across shared links while the ledger pays 2; cancel opposite neighbours
where an empty square is beside either (light leaves, never returns); join same-sign neighbours with no room into one
store at B + 4(λ − 1) from the ledger. Freeze when a sweep changes nothing. Stand-ins ours, listed in the module's
docstring.

**Her prediction (hers, verbatim, 4 November):** "as lambda rises, we have more net dark matter in the new spacetime.
lambda is related to the number of klein bottles in the network." Scored on the store count: **RISES** if at every side
and every B the mean store count never falls from one λ to the next and is higher at 2.00 than at 1.05; **FLAT** if
it is the same at every λ within the replicas' spread; **NONE** if no store forms at any setting; **FALLS** if it is
lower at 2.00 than at 1.05 everywhere; **MIXED** otherwise. Reported at every B; no verdict depends on choosing a B.

**Ours, said so they can fail.** The net twist stays near zero at every setting (a torus has no boundary row, and her
failed pair makes one store of each sign). The store count per square at the held-out side 24 lies between half and
twice its value at side 16 at the same (λ, B). The largest connected set of stores is a look for her question of
4 November (each Klein a store, the set a knot) and carries no verdict.

**What it cannot show.** The billionth, the dark particle's mass, the store-to-dark-particle energy ratio she
predicted on 4 November (this build has λ units only), or anything about the sky.

**Reading** (4 November 2026, `results/t103_curl_pair_moves.csv`, 240 rows, every run frozen within 7 sweeps).
Verdict by the registered words: **MIXED**, and in substance none: two stores in 240 cells (side 8, λ = 1.25, B = 0,
one replica; side 24, λ = 2.00, B = 4, one replica), zero everywhere else, at every B. **Her prediction failed.** The
cause is exact, not statistical: pairs are made while the ledger can pay, so the ledger is spent on pairs before any
join is considered; a join needs B + 4(λ − 1) from the ledger, and at the freeze the ledger holds at most 0.025 per
point. What her rules do make rises with λ exactly as she said: the same-sign curls left standing with no room, the
raw material of stores, go from 0.009 per square at λ = 1.05 to 0.178 at λ = 2.00 on side 24 (0.010 to 0.187 on side
16). Ours held: the net twist is exactly zero in every cell; the books balance to the last digit in every cell. The
held-out size claim is not scorable (zero against zero). The store cluster is zero everywhere. In the form rule 11
asks for: *stores rise with λ* → failed because the store's price is paid from a ledger that pair-making empties
first → replacement **hers to give**; the candidate we name (ours) is that the store's price is held in the two curls
that join, her word of 4 November ("the binding energy held"), so a join costs the ledger nothing → the test that
tells: the same run with the join paid from the curls, registered as T104 once she decides, scored on the same words.

## T104. The store's price from gravity: do the Klein stores rise with λ now? (registered 4 November 2026, before the run)

**Question.** T103's question with the store's price in her form given after T103 (hers, verbatim, 4 November, the
notebook): "Gravity, which is one directional and doesn't get lost as it's spent but increase as it's spent to make
more dense particles"; "there is no room for photons or for mobious strips to move, so Klein bottles form".

**Design.** `configs/t104_curl_pair_moves_gravity.json` through the same runner and module, `price_from = "gravity"`:
a join costs the ledger nothing and a one-way counter grows by B + 4(λ − 1) per store (a count, not an energy, ours
as the stand-in; the toy has no gravity in this state). Every other move, the sizes (8, 12, 16; **24 held out**), the
λ list, the B list (now entering only the counter) and the replica count are T103's; fresh seeds. Rule 11: T103's
previous claim → failed because the ledger was emptied by pair-making → her replacement (the price from gravity) →
this test.

**Her prediction (hers, the same words as T103):** "as lambda rises, we have more net dark matter in the new spacetime.
lambda is related to the number of klein bottles in the network." Scored on the store count with T103's words:
**RISES** / **FLAT** / **NONE** / **FALLS** / **MIXED**, the same definitions; B cannot change the count under this
form, so one verdict.

**Ours, said so they can fail.** Stores form now, and their count per square rises with λ as the no-room same-sign
curls did in T103 (0.01 to 0.18 per square), at roughly half that density since two curls make one store. Stores of
the two signs come out equal within the replicas' spread, net twist near zero. The store count per square at the
held-out side 24 lies between half and twice its value at side 16 at the same λ. The largest connected set of stores is
a look for her knot question and carries no verdict.

**Reading** (4 November 2026, `results/t104_curl_pair_moves_gravity.csv`, 240 rows, every run frozen within 7 sweeps).
Verdict by the registered words: **MIXED**, and in substance none again: two stores in 240 cells (side 24, λ = 1.50 and
2.00, one replica each), zero everywhere else. **Her prediction failed under this form too.** The books balance to the
digit; the net twist is zero everywhere; the no-room same-sign curls rise with λ as in T103. The cause is exact and is
ours, not hers: a diagnostic at side 24, λ = 2 shows the made sheet holding 193 same-sign links with no room beside
either square, and the cancel phase, run link after link before the join phase, emptying the sheet from 528 full
squares to 90 and leaving zero no-room squares before any join is tried; one cancellation opens room for its
neighbours and the pass chains through. Rule 11: *stores rise with λ* → failed because the sequential sweep cancels
before it joins (a convention of ours, choice 6 of the design document) → replacement: room judged once on the made
state, all pairs at the same instant (hers to accept) → the test that tells: a sandbox look (`sandbox/LOG.md`, 4
November) shows stores forming and rising with λ under that clock; T105 is to be registered on her word.

## T105. Her clock: do the Klein stores rise with λ? (registered 4 November 2026, before the run)

**Question.** T104's question under her clock, decided 4 November after T104's reading ("Yes to 1"): room is judged
once on the made state, for every pair at the same instant; the pairs with room cancel into light and the same-sign
neighbours without room join into stores, each square used once, cancellations first. The store's price from gravity
(T104's form). Rule 11: T104's claim → failed because the sequential sweep cancelled link after link before joining,
ours → her clock → this test.

**Design.** `configs/t105_curl_pair_moves_her_clock.json` through the same runner and module (`clock =
"simultaneous"`). Sides 8, 12, 16, **24 held out**; the λ list, the B list (entering only the gravity counter), the
replica count as T104; fresh seeds. The sandbox look of 4 November (`sandbox/LOG.md`) ran this clock at B = 0 on sides
8, 16, 24 with other seeds; T105 is the registered run and the look does not score it.

**Her prediction (hers, the same words):** "as lambda rises, we have more net dark matter in the new spacetime. lambda
is related to the number of klein bottles in the network." Scored on the store count with T103's words: **RISES** /
**FLAT** / **NONE** / **FALLS** / **MIXED**; one verdict, since B cannot change the count.

**Ours, said so they can fail.** Stores form; none at λ = 1.05 and 1.10 and a count rising from 1.25 on, since the
ledger at the low λ cannot pack the sheet (the sandbox look); stores of the two signs equal within the replicas'
spread; net twist zero. **Held-out size 24**, written before the run: stores per square at 24 within ±30 % of the
mean at 16 at the same λ, from the look's 0.108 → 0.107 at 1.25, 0.260 → 0.232 at 1.50, 0.488 → 0.417 at 2.00 (ours;
the look's seeds differ). The largest connected set of stores is reported for her knot question, no verdict.

**Reading** (4 November 2026, `results/t105_curl_pair_moves_her_clock.csv`, 240 rows, every run frozen within 13
sweeps). Verdict: **RISES. Her prediction held.** At every side the mean store count never falls from one λ to the
next and is higher at 2.00 than at 1.05: none at 1.05 and 1.10, then per square about 0.10 at 1.25, 0.18 to 0.23 at
1.50, 0.42 to 0.48 at 2.00 (sides 8, 12, 16, 24). Stores of the two signs equal within the spread (side 24, λ = 2:
124.9 against 123.8); net twist exactly zero in every cell; the books balance to the digit. B does not change the
count (the cells at different B differ only by their seeds). **Held out, side 24:** stores per square at 24 against
16 are 0.88, 0.96 and 1.03 of the value at 1.25, 1.50 and 2.00, inside the ±30 % written before the run: the size
claim held. The largest connected set of stores (her knot question, a look): 2.5 to 4 stores at 1.25, 5 to 9 at
1.50, 21 to 38 at 2.00, so stores sit together more as λ rises; whether that is her knot is hers to read. What this
shows: under her rules and her clock, on a sheet, Klein stores form only once the release packs the sheet, and their
number rises with λ. What it does not show: the billionth, the dark particle's mass, the store-to-dark-particle
ratio, or anything about the sky (λ units only).

## T107. The seed on the boundary of the loose remnant, and the kind on each side (registered 6 November 2026, before the run)

**Question.** Her picture of 5 and 6 November: the burp's cycle leaves a loose remnant at the centre of the black
hole, lower in density than the layers round it; the next seed starts at the remnant's edge with a foot on each
side; one daughter is one foot; and the first pair made across the boundary sets the outside foot's kind, every pair
after following it. Decided 6 November (late): the curled slab is a doughnut of squares, the toy's 4 x L tube is that
doughnut unrolled, and the loose band is its inner rim. Does the seed straddle the boundary, does the outside foot
grow while the inside foot stops, are the feet joined, and does her kind rule make the two feet mirror images? The
sandbox looks of 5 and 6 November (`sandbox/LOG.md`: loose band, boundary feet, feet kind) saw all four at L = 64 and
cannot score anything (rule 16); T107 runs on fresh seeds at three lengths with the fourth held out.

**Design.** `configs/t107_boundary_feet.json` through `scripts/run_t107_boundary_feet.py`
(`graphity.loose_band`, `graphity.curl_pair_moves` with `kind_rule`). Lengths **32, 48, 64**, twenty replicas each;
**128 held out** (`configs/t107_boundary_feet_heldout.json`, run only after the numbers below are filled in from the
three). The cell is T37's (lambda 1.25, g 1.5, no cap). The band: the columns whose circular distance from the centre
column is strictly less than L/16 (3, 5, 7 and 15 columns at 32, 48, 64 and 128; the look of 5 November called the
band at 64 "8 columns" and printed "of 32 band points" for what are 28, a labelling slip, now exact), loosened by
k = 4 valid switches whose four points all lie in the band. The chain runs in blocks of ten sweeps; the escape is
the first block at which at least eight points not made to read sheet (d = 2) by the loosening do so; the feet are
read at 0, 50, 200 and 500 sweeps after the escape, the pieces counted over all sheet points. At 500 sweeps after
the escape her curl state is put on the wiring's squares and her clock run (price from gravity, B = 12, room judged
once on the made state) under her kind rule and, as the control, under the coin of T103 to T105. A square's side is
read from the circular mean of its corners' columns against the band's half width.

**Her predictions (hers by adoption, 6 November, late; the wording of the night before pasted back with "this all
seems right").** Scored at every length, each a share of the twenty replicas, and at the held-out length against the
number written below before it runs (rule 14):

1. **A foot on each side.** At 200 sweeps after the escape the seed has new sheet points on both sides of the band's
   edge in more than half of the replicas: **HOLDS** if above 0.5 at every length, **FAILS** if at or below 0.5 at any.
2. **The outside foot grows and the inside foot stops.** At 500 sweeps after the escape the outside foot is larger
   than the inside foot in more than half of the replicas, and the inside foot never exceeds the band's point count:
   **HOLDS** / **FAILS** by the same rule, the two parts scored separately.
3. **The feet are joined** (read, not scored): the share of replicas whose two feet share one connected piece of
   sheet at 500 sweeps after the escape, reported at every length.
4. **The feet are mirror images under her kind rule.** Among the replicas that made a straddling pair, the outside
   foot's net twist carries the outward kind's sign and the inside foot's the reverse in more than half: **HOLDS** /
   **FAILS**; the coin control is reported beside it and expected at chance.
5. **The size falls with length** (its own test, T108 when registered; not scored here): her formula
   (red - blue)/(red + blue + all stored energy), about one in a billion, is read at each length as
   net_twist_out / (stored_on_squares + held_in_stores) and reported, no verdict.

**Ours, said so they can fail.** From the looks at 64: a foot on each side in 0.85 to 1.00 at 200 sweeps; the
outside larger than the inside in 0.8 or more at 500 sweeps, the inside at most the band's count; joined in about
0.9 at 500 sweeps; mirror images in 0.6 to 0.75 under the rule, 0.1 to 0.25 under the coin; the size ratio about
one in fifty, falling slowly with length if at all. **Held-out length 128**, to be written here from the three
fitting lengths before it runs: the shares for 1, 2 and 4 with a range of plus or minus 0.15 round the mean of the
three, and the escape sweep's median within a factor of two of the trend.

**The held-out numbers, written 6 November 2026 from the three fitting lengths, before L = 128 ran** (by the
procedure above: the mean of the three, plus or minus 0.15, capped at 1; the escape median within a factor of two of
the trend). Fitting lengths read (`scripts/analyse_t107.py`, `results/t107_boundary_feet_*.csv`): a foot on each
side at +200: 0.75, 1.00, 1.00; the outside foot larger at +500: 0.95, 0.80, 0.60; the inside foot bounded by the
band at every length; joined at +500: 0.85, 1.00, 0.90; mirror images under her rule: 0.79, 0.76, 0.68 (of 19, 17
and 19 replicas with a straddling pair); median escape sweep 40, 20, 20; the size ratio 0.021, 0.013, 0.010. So, for
**L = 128**: a foot on each side **0.92 (0.77 to 1.00)**; the outside foot larger **0.78 (0.63 to 0.93)**; mirror
images **0.74 (0.59 to 0.89)**; the escape's median sweep **20 (10 to 40)**. *Ours, said before the run so that it
can fail:* the outside-larger share falls with length across the three (0.95, 0.80, 0.60), and the reason is in the
criterion, not the picture: the inside foot's size is bounded by the band, which grows with L (12, 20, 28 points),
while the outside foot grows at a rate per sweep, so at a fixed 500 sweeps the comparison tilts toward the inside as L
grows (a criterion that scales with N; CLAUDE.md's warning). A straight line through the three puts the share at 128
near 0.1, below the registered range; if it fails there, the failure is scored as written and the replacement
criterion (the outside foot's growth rate, or the comparison at a time that scales with L) is registered as a new
test, never read into this one. Two more notes before the run: (a) the registered control for prediction 4, "the coin
at chance", was ill-posed as written (ours): under the books the two feet's twists always sum to zero, so "opposite
signs" holds whenever either is nonzero under any rule; the proper control, stated now, is the share of replicas whose
outside foot's sign matches the outward kind among those with nonzero twist, against one half by chance; at the
fitting lengths that share is 15 of 15, 13 of 14 and 13 of 15 (41 of 44), the coin having no outward kind at all.
(b) The band is 15 columns at 128 (60 points).

**Reading** (6 November 2026, night; `results/t107_boundary_feet_feet.csv`, `_kind.csv`, and `_heldout_*.csv`,
read by `scripts/analyse_t107.py`; 80 replicas, every one escaped, the books balanced to the digit in every cell).
Verdict: **MIXED: three of the four scored predictions HOLD at every length, the held-out length included; one
failed at the held-out length as the registration said it might.**

1. *A foot on each side* at +200: 0.75, 1.00, 1.00 at 32, 48, 64; **0.85 at 128**, inside the 0.77 to 1.00 written
   before the run. **HOLDS.**
2. *The outside foot larger at +500:* 0.95, 0.80, 0.60; **0.35 at 128**, below the 0.63 to 0.93 written before the
   run. **FAILS at the held-out length.** *The inside foot bounded by the band:* at every length (its +500 median
   4, 10, 13, 22 against bands of 12, 20, 28, 60). **HOLDS.** Rule 11, the failure's form: the claim "the outside
   foot is larger than the inside at a fixed 500 sweeps" → failed because the inside foot's size is set by the
   band, which grows with L, while the outside foot grows per sweep (at 128 its median is 3 at +200 and 9 at +500,
   the inside 20 and 22), so the comparison at a fixed time tilts toward the inside as L grows, a criterion that
   scales with N, said before the run → replacement: her words "the outside foot grows and the inside foot stops"
   scored on growth, not size: the outside foot's gain from +200 to +500 against the inside's (read here, not
   scored: the outside gained more than the inside in 0.80, 0.90, 0.85, 0.70 of replicas; the inside's median gain
   0, 0, 0, +4; the outside's +42, +10, +8, +3) → new test, T109 when she gives her words, with a time that
   scales with L or the gain scored directly.
3. *Joined* at +500 (read): 0.85, 1.00, 0.90, 0.75.
4. *Mirror images under her rule* among replicas with a straddling pair: 0.79, 0.76, 0.68; **0.81 at 128** (13 of
   16), inside the 0.59 to 0.89 written before the run. **HOLDS.** The proper control, stated before the held-out
   run: the outside foot's sign matched the outward kind in 15 of 15, 13 of 14, 13 of 15 and 13 of 13 replicas
   with a nonzero twist, 54 of 57 against one half by chance; the registered "coin at chance" was ill-posed (above)
   and is not scored.
5. *The size ratio* (read): 0.021, 0.013, 0.010, 0.0088, falling with length; the escape's median sweep 40, 20, 20,
   20 (128 inside the 10 to 40 written before the run).

What this shows: on the toy's tube, the doughnut unrolled, the next escape starts at the loose remnant and its seed
straddles the remnant's edge at every length; under her kind rule the two feet come out as mirror images with the
sign set by the first pair across the boundary, at every length; the outside foot grows while the inside stays
within the remnant. What it does not show: that the outside foot is *larger* than the inside at any fixed time (it
is not, once the remnant is big), the size of the surplus (one in fifty to one in a hundred here, falling with
length, against her one in a billion), or anything about the sky.
