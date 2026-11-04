# Curl and anti-curl on the squares: the form fixed before anything runs (3 November 2026)

*Written on the owner's instruction of this evening ("curl and anti curl do: the loop of the white to white vs the black
to black diagonal on the same square. You need to build that into the toy. proceed w S1"). Under S1 the form is fixed
here, in writing, before any run; every line says whose it is. Module: `src/graphity/curl_pairs.py`; tests:
`tests/test_curl_pairs.py`. Nothing here derives anything; her rule is put in and read back exactly. The model with any
dynamics on this state is our family around the published model, never combinatorial quantum gravity.*

## 1. Her rule, and the exact fact under it

**Hers.** The two kinds exist in the four-link toy: on each square, the black-to-black diagonal carries curl (matter) and
the white-to-white diagonal carries anti-curl (antimatter). At the burp all the barrier energy returns as equal pairs of
the two that cancel; λ sets the surplus that cannot cancel, which then curls the space.

**Exact.** Every square of a bipartite wiring has two black corners and two white, so exactly one diagonal of each kind
(`wiring_is_paired`, checked on the flat torus and the 4-cube). The wiring alone therefore always holds the two kinds
paired, one of each per square. The energy counts squares and cannot tell the diagonals apart (the colour-term brief,
section 2, stands). So the record's sentence "the toy holds nothing that comes in opposite kinds" (Updates 42, 153) was
wrong as a statement about the wiring and right only about the energy; it is corrected in Update 156. A surplus of one
kind can live only in a state on the squares, which is what is built.

## 2. The state (the first form; ours, from her rules)

| quantity | values | the rule it carries | whose |
|---|---|---|---|
| c on each square | a real number in [−1, 1] | curl is continuous (3 November) | hers |
| sign of c | + is curl on the black diagonal, matter; − is anti-curl on the white, antimatter | the two kinds are the two diagonals | hers |
| a pair | made as (+c, −c) on two squares | pairs have equal absolute value | hers |
| stability | only \|c\| = 1 is a stable store | only a full curl stores energy stably | hers |
| one value per square | a square holds curl or anti-curl or nothing, not both | the two diagonals of one square are the cancelled pair | **ours**, from "pairs that cancel"; hers to change |

Readers, exact: net curl Σc (her net matter), stored Σ|c| (energy held, in full curls), the number of stable squares, a
census by kind, and the same-colour pairs carrying each kind in the form the Carroll reader of T94 takes.

## 3. What is not built; her decisions of 19:18 UTC on the three choices

**Decided by her the same evening (VISION Update 156, addendum).** (1) A pair is made across a shared side: in the square
toy, on the two squares sharing a link, matter on one and antimatter on the other, so a row's edge may hold only the
matter curl and separated sides carry separated surpluses. (2) A cancelling pair leaves as two photons; with nowhere to
go, the energy joins two same-twist one-sided half-twist curls into one two-sided full twist in a root-2 Klein store,
which cannot move, collects photons of matching spin, and pays the barrier plus the λ cost. (3) λ sets the number of
pairs; the freeze sets the surplus. Two barriers: the first paid in curl, the second going at once and keeping its
antimatter surplus. The dynamics is now specified in words and is to be written as moves, with rule-6 tests, before any
run; (2)'s second half needs item 4 below. The choices as they stood before her answer are kept under them for the record.


1. **Where a pair is made.** On two squares sharing a link (neighbours), or anywhere in the opened region.
2. **When a pair cancels.** A +c and a −c of equal size on neighbouring squares cancel; into a photon that leaves (her
   "if there is space", so only when an empty square is beside them), or into a frozen pair that stays (her dark
   matter, the Möbius strip) when there is no room.
3. **λ's part.** The release of an opening is 4(λ − 1) per point; the plainest reading pays it into pairs, each pair
   costing 2|c|, so λ sets how many pairs an opening makes. The surplus then comes from pairs that fail to meet before
   the region freezes, so it is set by the cooling and the seeds as much as by λ. Whether λ enters a second way, in the
   odds that a pair cancels, is hers.
4. **The twist.** Her 3 November rules give twist in thirds (matter) and halves (dark) on the carriers; this module holds
   curl only. Twist on the squares' diagonals is the two-dimensional form of `graphity.twist` and waits on item 1.

## 4. First exact uses, no decision needed

- **The Carroll reader with matter only.** T94 put shares on both diagonals of every square (a cancelled pair in this
  state's words). The same windows with shares on black diagonals only (`stored_pairs(MATTER)`) read a window of matter
  against one of pairs; to be registered with her prediction before it is computed.
- **The census of saved end states.** The wiring of every saved end state is paired by the exact fact above, so no saved
  state carries a surplus in the wiring; the colour census of T88 (1 November) found none and could not have.

## 5. Her question of the same evening, answered here for the record

"What would λ be to leave the billion-to-one?" cannot be computed until items 1 to 3 are decided and a run is made; the
state gives the toy a surplus to count, which it did not have. Our expectation, said now so it can fail: the surplus is
set mostly by the freeze (item 2) and not by λ alone, as the knots' number was (O107).
