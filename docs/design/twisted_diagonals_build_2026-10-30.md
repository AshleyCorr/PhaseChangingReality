# The twisted-diagonal state: the form fixed before it is coded (30 October 2026)

*Written on the owner's decision of this afternoon ("item 3 of the diagonals brief and is not built -> proceed w build").
Under S1 the form is fixed here, in writing, before any run; every line says whose it is. The rules it encodes are the
standing sheet `twist_curl_rules_current.md` (hers), read in full before this was written; where the sheet leaves a
choice, the choice is marked **ours** and is hers to change. Module: `src/graphity/twist.py`; tests:
`tests/test_twist.py`. This is her Option B (the diagonals brief, decision 5): particles are loops of twisted diagonals
built as variables, and the wiring's knots are the scrap. Nothing here derives anything; a rule is put in, and the
results are what follows from the rules together.*

## 1. The lattice (exact, from the sheet)

An L × L × L torus of points coloured by the parity of x + y + z (black even, white odd; a convention, ours). A **box
diagonal** is a step whose displacement is a signed permutation of (1, 1, 2): 24 of them, length √6, same colour to
same colour. A **long diagonal** is (1, 1, 2m) in any signed permutation with m ≥ 2 (her 1 × 1 × 4, √18). A **face
diagonal** is a signed permutation of (1, 1, 0): 12, length √2, same colour. Body diagonals (1, 1, 1) change colour and
carry light, not twist; they are not part of this state.

## 2. The state

A **loop** is a closed sequence of points on the torus whose consecutive steps are box diagonals, long diagonals or face
diagonals (one kind per loop; a loop of box diagonals is matter, of face diagonals dark, by the sheet), all of one colour
(automatic for these steps). Each loop carries:

| quantity | values | rule it encodes | whose |
|---|---|---|---|
| twist τ_i on each diagonal i | multiples of ⅓ on box diagonals; multiples of ½ on face diagonals; **0 on a long diagonal** | twist in thirds (matter) and halves (dark); the long diagonal holds curl with no twist | hers |
| sign of τ_i | either sign on any diagonal | a positive particle goes over 1 (⅔, ⅔) and twists back (−⅓); the neutrino holds +θ and −θ; so the colour sets which sign is "matter" and does not forbid the other | hers (the sheet); the reading that colour does not constrain the sign was ours and is **hers from 2026-10-30** ("agree") |
| curl per loop | an integer ≥ 1 | one curl of energy per loop | hers |
| spin s per loop | −½, 0, +½; **non-zero only if some τ_i ≠ 0** | half spin on a full curl; spin needs twist present, not net twist (30 October) | hers |

A **particle** is a named set of loops. Its readers (exact, from the sheet's section 2):

- **charge** = Σ τ_i over all its diagonals; a particle on its own must have whole charge (a loop alone may carry a third);
- **spin** = |Σ s_ℓ| over its loops (spins can cancel while twists add: the pion);
- **mass as a count** = the number of looped diagonals, a long diagonal counting one (Update 148);
- **the parts, each its own quantity** (30 October, §5 of the relations document): L = loops, n = diagonals, τ⁺ = Σ|τ_i|,
  T = Σ τ_i, s = Σ s_ℓ, C = Σ curl (held half curls included), **orb** = the spin a loop has absorbed from photons, carried
  as orbital motion (her decision, 2026-10-30, 20:06 UTC: "we need to represent this in the list of energy types"). No rule
  combines them into MeV here; MeV is the measured total only.
- **net twist** of a set of particles = Σ T, the quantity pair creation conserves (the sheet, "Pairs and annihilation";
  the sheet's open item 6, defined here for the first time).

## 3. What is checked, and what is not

Checked by the module (and raised as an error): a step that is not an allowed diagonal; a loop that does not close on
the torus; a twist not a multiple of its kind's increment; twist on a long diagonal; spin without twist present; mixed
step kinds in one loop. **Not checked, open:** the stability rule (sheet, open item 1); whether a loop of three can hold
its twist (the electron's size n, item 3); whether twisted diagonals may cross or share points (allowed for now; ours, and hers from 2026-10-30, "agree"); the photon (not a loop of box diagonals; it stays unbuilt).

## 4. The energy

*Revised 2026-10-30 on her decisions of that afternoon.* The twist term adopted as the working form (hers: "it's okay to work with
this, just carry the note. we'll see if we can derive later"), with the writhe term she asked for ("writhe seems like a
great idea and already has available math"):

    E_twist(loop) = ε [ Σ_i τ_i² + Σ_i (τ_{i+1} − τ_i)² ]  +  ε_w Wr²

elastic plus mismatch on the diagonals (E4 of `sandbox/twist_energy_options.py`), and the coil's own cost through the
writhe Wr of the loop as a closed polygon (the Gauss double integral; the Călugăreanu–White–Fuller theorem Lk = Tw + Wr
and the elastic-rod and DNA models are the published math, general knowledge, to verify). Two constants, ε and ε_w;
neither pinned yet. **Rule-13 note, carried:** E4 was chosen because its rankings of her test particles were the ones
she wanted; it is not derived. **Her requirement on the term (hers, this day):** a charge of 2 or more on one loop must
cost more than two separate particles, since like twists repel and a free particle carries a whole twist; under E4 alone
charge 2 on six diagonals costs exactly two electrons, so a repulsion piece is still owed (the simplest, ours, in the
look: a per-loop cost in the net twist squared). **Her correction on the periodic form:** periodic in a loop's net
twist (whole charge free, fractions costing) is her "a full twist is a requirement" as an energy, the confinement of
thirds, and is not what was decisive against; the per-diagonal form was. Proposed for her, not adopted: that term
beside E4. **The question the writhe term raises (ours):** the conserved quantity of a closed ribbon is Lk = Tw + Wr,
so twist can turn into coil; if charge is the net twist it is not conserved under coiling; the published reading would
make charge the linking number, twist held plus coil. Hers to decide. *Checked 2026-10-30 on her question (exact; `sandbox/linking_number_check.py`): charge as Lk is not consistent with the state as built, since a closed ribbon's Lk is a whole number, so a loop alone could not carry a third, and the writhe of a closed pentagon of box diagonals is a generic real, so the twist held could not stay in thirds; it would give her the whole-twist requirement for free on planar loops. Charge as net twist, with writhe a shape cost outside charge, is consistent; the theorem's conservation law is then not used, and that note is carried. Hers.* **DECISION (hers, 2026-10-30, 17:47 UTC): charge is the net twist; writhe is a cost of shape outside charge; the theorem's conservation law is not used (note carried).** Her mechanism, recorded and not adopted: twist is made in packages of a third at the creation of surplus matter from uncurling directions, and spreads to a continuous value over a bound loop (or a coiled loop with gluons); a sign switch needs a shallow, near-neutral stretch, perhaps what the writhe cost is about. *Ours, exact:* the mismatch piece prices that stretch (a change D over k diagonals costs ε D²/k), so the term and her picture together argue for continuous twist on the diagonals inside a bound particle with the loop's total in thirds; that is a change to the increment check of section 2 and is hers to decide. Her picture of 2026-10-30, 17:53 UTC: a kink in a coil takes twist into a loop while keeping it spread. *Ours, exact:* with charge = net twist, twist cannot pass into coil (that is the rod model's move, under Lk); what lowers the cost is length, ε T²/n on a flat loop of n diagonals, and the coil is how a long loop sits in a small region, unpriced here, with the writhe cost setting how tight it may be. *Corrected the same hour, exact (`sandbox/electron_size_under_thirds.py`):* ε T²/n is the continuous case. With twist in thirds on each diagonal, as built, the elastic piece of charge 1 is at least ⅓ at every length and the mismatch piece is 0 on the triangle and 2/9 on every longer loop, so **the triangle is the unique cheapest loop at charge 1: the electron's size n = 3 (section 3, open item) is selected by the working term as it stands.** Continuous twist would remove that selection and owe a size rule. Her agreement of this hour: charge is the net twist and a coil cannot absorb twist. *Predicted then enumerated, 18:29 UTC (`sandbox/neutrino_pion_cheapest.py`):* the neutrino's cheapest loop is any of four or more diagonals (2/3; the triangle 8/9); the pion's loops cost 1/3 and 4/9 at every length; only the electron selects a length. **Her hypothesis, not adopted:** the neutrino as a dark loop of face diagonals with +½ and −½, no charge, little mass, gravity only; allowed as built (cost 3/2 at four or more; 48 face triangles exist, one diagonal per plane), but 'no curl' and 'cancelling mass' would be rule changes, and a Möbius strip proper has a net half twist, which a lone loop may not carry. *Her reply, 18:42 UTC, recorded (hypotheses):* curl signed, the neutrino +1 and −1 curl netting 0; face diagonals form together three per cube; the neutrino's mass the mass of its spin alone; the photon half a curl, spin 1, at c. *Exact:* every face triangle is the three faces of one unit cube meeting at a corner (8 per cube). Each of the four is a change to the sheet's table (signed curl; half curl; spin 1 on one loop; a rule for mass from the parts) and is hers to decide. Previously here: "None is added." (the sheet's
section 4 stood until this afternoon). The sheet's section 4 stands: the state carries curl as a count per loop (one curl of energy each) and no
twist term (the κ form was ours and is not adopted). So the first thing the state can do is score the sheet's own
particles exactly, and conserve net twist; what it cannot yet do is move.

## 5. The first exact check (run with the tests, nothing registered)

The sheet's four test particles, built by hand and read by the module: electron (one loop of three box diagonals, twists
−⅓, −⅓, −⅓, spin ½): charge −1, spin ½, mass 3. Positive pion (two loops: ⅔ + ⅓ on one, their spins opposed ... as the
sheet reads it, "twists ⅔ + ⅓, spins opposed"): charge +1, spin 0. Neutron (three loops, −⅓, −⅓, +⅔ net, spins netting
½): charge 0, spin ½. Neutrino (one loop, +⅓ and −⅓, spin ½): charge 0, spin ½, twist present ⅔. A pair (loop with T and
its mirror with −T): net twist 0. The photon is excluded (not a loop of box diagonals).

## 7. The photon, first form (her decision of 2026-10-30, 19:19 UTC: "proceed, add")

Her words of that evening, put in order: the photon is a curve on the two-colour body diagonal; it carries half a curl
and spin 1 and no twist; its spin carries it one link per sweep; the angle its curl sits at changes as it moves and
repeats every four links, an oscillation along a line of cubes joined corner to corner; when it meets matter it hands on
its spin and its curl. Built as `Photon` in `graphity.twist`: an open path of `length` body diagonals (1, 1, 1) in one
signed form, `position` links advanced, curl ±½, spin ±1, charge 0, `angle_degrees` = 90° × (position mod 4), colour
alternating link by link. The meeting with matter is not built.

| quantity | values | whose |
|---|---|---|
| diagonal | body, (1, 1, 1) signed; two-colour; length √3 (she wrote "root 2" and corrected it to √3 at 19:34 UTC) | hers |
| curl | +½ or −½ | hers |
| spin | +1 or −1 | hers; the sign reading ours, from her ledger |
| twist, charge | 0 | hers |
| motion | one link per sweep; the curl's angle a quarter turn per link, period four | hers |

**Her annihilation ledger** (hers): A (+1 twist, 1 loop, ½ spin) and B (−1 twist, −1 loop, ½ spin) make C (+½ loop,
spin 1, no twist) and D (−½ loop, the opposite direction, spin 1, no twist). Built as `annihilate`, with `ledger` summing
net twist, curl and signed spin over particles and photons. **Two consequences, built and confirmed by her (19:33 UTC):** curl is now a
signed non-zero integer (her B), and `mirror` reverses curl as well as twist and spin; and the ledger balances (0 = 0 in
twist, curl and spin) only with B's spin −½ and the photons' spins opposite, which is how it is built. **The count of diagonals** (her decision, 19:54 UTC): a photon's two-colour diagonals are not matter's and
count no mass, so the count reads 6 before and 0 after; the 4 is the photon's period, not a size. **Her hypothesis for the
meeting, not built:** the loop that takes the photon's spin and curl is an electron, which goes to a higher energy level; the
excited state's values (a half curl and a spin 1 held on a loop) are not in the sheet's table and are to be written first. **Written, 20:06 UTC (her "yes write it"):** a loop holds an excitation, the list of photons it has absorbed, each
as (half curl, spin 1), until it re-emits one (`Loop.held`, `absorb`, `emit`); the loop's own spin stays as it was and the
absorbed spin is `orb`; `total_curl` adds the held half curls; the ledger sums T, C, s, orb, J = s + orb and n, and
absorb then emit leaves every sum unchanged. Not decided: which loop may absorb (published: a bound electron, not a free
one) and the energy of a level. Section 2's "curl ≥ 1" and
section 3's "the photon stays unbuilt" are superseded by this section.

## 8. The twist constant by kind; the second anchor (her decisions of 2026-10-30, 22:00 UTC)

**DECISION (hers, "agreed, proceed"): the twist constant may differ by kind of diagonal**: ε on box and long diagonals,
ε_face on face diagonals (`twist_energy(..., eps_face=None)`; one constant for both when it is not given). *Exact,
predicted then enumerated (`sandbox/neutrino_by_kind.py`):* the two-loop neutrino under signed curl is cheapest with the
twist cancelling across its loops (+⅓ on one, −⅓ on the other; or ±½ on face diagonals), box pair 2/3, face pair 3/2 at
every length from three, no size selected; her dark (face) pair wins iff ε_face/ε_box < 4/9, the same threshold whether
the twist cancels across the loops or within each. Signed curl costs nothing in the term. ε_face is pinned by nothing
yet.

**The second anchor** (her "proceed"; `sandbox/anchor_muon_tau.py`; exact; published ratios to verify). The electron costs
ε/3. The flat pion as built costs 7ε/9, ratio 7/3 against 273 measured, with no constant free: **the flat pion cannot
anchor** (killed). The muon as her coil of minimum size pins ε_w/ε = (206.768/3 − 5/9)/Wr_μ²: 118.4 (tight one-turn
coil, Wr 0.760) or 79.9 (wide, 0.925). The tau as the next coil of the same family lands nowhere within 1 per cent (the
ladder goes as Wr², steps of 25 to 100 per cent; nearest −21 and +5.4 per cent). One of 48 cross-family combinations
lands at −0.45 per cent (wide one-turn muon, tight four-turn tau), predicted to occur by chance at that spacing and, under
rule 13, not a result: neighbours −46 and +60 per cent, no rule selecting the coils. What would make it one: a stability
rule that picks the coil (her candidates of 29 October), then the tau held out (rule 14).

## 9. The photon ladder: energy as curl per link (her decision of 2026-10-31, 16:25 UTC)

**Her picture (2026-10-31):** the photon's curl angle may turn less than a quarter turn per link, so it takes more than four
links to return to its start; the repeat is the wavelength and 4 links is the shortest. **DECISION (hers): the photon's
energy is its curl per link, set by its wavelength**, "the one that matches measurement" (per-packet energy goes with
frequency, the photoelectric effect; general knowledge, to verify), not curl times length; the length is the pulse.

*Exact:* returning exactly to the start needs a whole repeat n ≥ 4, so the curl per link is 2/n: ½, ⅖, ⅓, 2/7, ¼, ...
a discrete ladder with no extra rule. Built: `Photon(period=n)`, `energy = |curl| = 2/n`, `angle_degrees = 360/n` per
link; `Loop.held` and `emit` carry the period; `annihilate(period=4)`. What it changes (ours, exact): the excitation
ladder is the photon ladder (the level is the held photon); a held 2/n splits into two ladder photons only where
1/n = 1/k + 1/j has whole solutions (n = 4: 6 + 12 or 8 + 8; n = 5: 10 + 10 only); annihilation's two photons have a
wavelength nothing in the toy pins (measured: one definite one, 511 keV each); the electron's energy under the twist
term (ε/3) and a photon's (2/n) are different currencies until a rule ties them (rules page, open item 2). No color can
be named for 4 links: the toy has no length unit; if the link were the Planck length (not decided) 4 links is about
6 × 10⁻³⁵ m and green light repeats every ~10²⁸ links.

**Gravity waves on the edges** (`sandbox/edge_wave_look.py`; predictions written first, all held). The carrier is the
edges' stiffness K = 4(6I − A) per axis family (gravity_link_brief, step 1). Given an inertia (the look's assumption;
the toy has no time): one speed for every long wave, ω = 2|k| (1.95 at 8 links, 1.80 at 4, 1.27 at 2), as the world's
gravitational waves share c with light; the shortest edge wave is 2 links, not 4, because a stretch has no turning
angle; the "+" polarization (x-links lengthen where y-links shorten) is carried by the axis edges exactly, the families
never coupling; the "×" polarization changes the cubes' angles and would need the face diagonals (ours, geometric, not
computed); spin 2 is not in the toy. So the edges carry no ladder like 2/n.

## 6. What this build does not claim

That any of these loops is stable, that the counts are masses in MeV, or that the lattice is space. It is the first
object in the toy that is her particle by her rules, and its only immediate use is that the rules can now be checked
against each other exactly, and that a wave or a chain can later act on it.
