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
