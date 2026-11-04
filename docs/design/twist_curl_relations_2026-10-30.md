# Twist and curl: the relations as equations and inequalities, the observed table in her counts, and the 2D phase

*Written 2026-10-30 at the owner's request of 29/30 October ("read the papers that you listed and take math that they
brought that doesn't contradict our theory ... build out the table and the fully simplified equations and inequalities
representing the relationships between twist and curl observed in 3d and what it means for the 2d phase"). Every line
says whose it is: **hers** (her rule, `docs/design/twist_curl_rules_current.md`), **published** (a source, with how much
was read: `docs/reading/notes/2026-10-30_braid_lineage_reads.md` and `2026-10-30_helon_model_full_read.md`),
**exact** (arithmetic or geometry, checked), **measured** (a PDG value, from memory, to verify), **ours, unverified**.
Nothing here is a result about reality. The sandbox lane applies: the table and the integer test kill but never crown
(`sandbox/observed_particle_table.py`).*

## 1. The quantities

| Symbol | Meaning | Values | Whose |
|---|---|---|---|
| D | number of directions of the lattice | 2 (the paper 1 toy, four links a point) or 3 (six links) | exact |
| S_e | squares on link e | 0 ≤ S_e ≤ 2D − 1 (the hard-core rule caps it; flat is 2D − 2) | published [T25], exact |
| c_e | curl on link e, c_e = S_e − (2D − 2) | −(2D − 2) ≤ c_e ≤ +1; in 3D −4 to +1, in 2D −2 to +1 | hers (name), exact (range) |
| C(ℓ) | curl of a closed loop ℓ, the sum of c_e over the links its diagonals span | whole numbers; "what is real is the total round a closed loop" | hers |
| τ_d | twist on diagonal d | matter: 0 or ±⅓ on box diagonals (1,1,2); dark: 0 or ±½ on face diagonals (1,1,0); sign by colour | hers |
| T(ℓ) | net twist of a loop, Σ τ_d round it | a whole number of thirds (matter) | hers |
| Q | electric charge | Q = T(ℓ) for a free loop (in units of e) | hers |
| s | spin | ½ or 1 (and 0 for opposed pairs) | hers |
| n | mass count: looped diagonals, a long diagonal counting one | whole numbers | hers (Update 148) |
| m | mass | m = u n, u = m_e / n_e | hers; u exact given n_e |
| Wr | the coil (writhe of the loop as a curve) | real; 0 for every planar loop | exact |

## 2. The relations, 3D, as they stand

**Her rules written as constraints** (hers unless marked):

    (R1)  twist only where curl:            τ_d ≠ 0  ⇒  c_e ≠ 0 on the links of d's box
    (R2)  thirds on box diagonals:           τ_d ∈ {0, +⅓, −⅓} for matter;  sign(τ_d) = colour of d's end points
    (R3)  halves on face diagonals:          τ_f ∈ {0, +½, −½} for dark twist
    (R4)  charge is net twist:               Q = Σ_{d∈ℓ} τ_d  ∈ ℤ for a free particle        (confinement: Q ∉ ℤ only stacked)
    (R5)  spin × curl = ½:                   s · C = ½  ⇒  (C, s) ∈ {(½, 1) photon, (1, ½) matter};  s = 0 for two loops spinning against each other
    (R6)  mass is a count:                   m = u · n,   n = number of looped diagonals (a long diagonal counts one)
    (R7)  the unit:                           n_e = 3 as it stands (open: n_e > 3 if a 3-loop cannot hold its twist)
    (R8)  equal spacing (stability):         like twists sit at equal spacing round the loop  ⇒  a charge-±1 loop of three thirds is (S L^k)^3, n = 3(k + 1)
    (R9)  no uncharged spin:                 T(ℓ) = 0 and s ≠ 0 does not occur for a matter loop  (the neutrino must live elsewhere: see §5)
    (R10) pairs:                              curl is made with anticurl, +θ with −θ;  Σ_all τ_d = T_total is fixed after the burp

**Exact consequences** (ours, from R1 to R8; each checked in the sandbox or by arithmetic):

    (E1)  n ≥ 3 for any loop in 3D (three box-diagonal steps close; `proton_loop_fold.py`), and every closed loop of box diagonals is one colour (parity)
    (E2)  a charged lepton has n ≡ 0 (mod 3) under R8;  charged objects at n = 3, 6, 9, ...;  the gap below the muon is not made by R8
    (E3)  a loop of three box diagonals is planar, Wr = 0 (all 16 triangles);  4-loops Wr = 0;  |Wr| up to 1.000 at 5, 1.564 at 6
    (E4)  the four test particles pass R4 and R5: e (−1, ½), n (0, ½), γ (0, 1), π± (±1, 0)
    (E5)  the integer test: if R6 holds, n_X = n_e · (m_X / m_e) is a whole number for every particle X

**The published relations taken, that do not contradict R1 to R10** (published; sources in the reading note):

    (P1)  Lk = Tw + Wr                                   Călugăreanu, White, Fuller: linking of a ribbon's edges = its twist + its writhe
          — the one published relation between a twist and a coil; it needs a ribbon (a framing), which a loop of diagonals does not yet carry
    (P2)  Q = ⅓ (N_+ − N_−)                              rishon counting (Harari, Shupe 1979): charge = net count of charged constituents in thirds
          = ⅓ (a + b + c)                                 BHKS 2008, Eq. 3: charge = sum of the three strands' twists over three
    (P3)  three per particle                             rishons, helons, braids: "three helons seems to be the minimum ... stable, non-trivial structure" (BT05 §II)
    (P4)  Ω = β (N_+ + N_− − N_0)/3,  Q = (β + Ω)/2      BT05 Eqs. 3–5: hypercharge and the Gell-Mann–Nishijima relation from counting alone
    (P5)  C: flip every twist's sign;  P: reverse the order round the triplet;  T: both;  CPT = 1     BHKS 2008 Eqs. 5–7
    (P6)  interactions add twist strand by strand:  [a,b,c] + [d,e,f] = [a+d, b+e, c+f]              BHKS 2008 Eqs. 8–10
    (P7)  N_B = Σ β conserved: braids and anti-braids are made and destroyed together                 BT05 §III, Eq. 6 (lepton number)
    (P9)  a crossing is a twist:  σ₁ → [½, ½, −½],  σ₁⁻¹ → [−½, −½, ½],  σ₂ → [−½, ½, ½]     BHKW 2012 Eq. 2.1: a crossing of two strands under a cap equals half twists on the strands,
          so every braid reduces to twists only ("twisting (electric charge) and braiding are fundamentally the same process", BT05 §IV)
    (P10) twist is quantized by the node's symmetry:  the smallest distinguishable twist is π/3 on a four-valent node  BHKW 2012 §3.1: thirds from a three-fold symmetry, not put in by hand
    (P11) a conserved sum under the moves:  Θ = T_l + T_r + Σ T_i − 2 Σ x_i  (twists minus twice the signed crossings)    BHKW 2012 Eq. 3.4, additive under interaction and decay (Thms 3.1, 3.3)
    (P12) stability:  "any nontrivial braid with internal twists is stable under single evolution moves"                 BHKW 2012 §3.4; a trivial braid is unstable ("dual to a 3-ball")
    (P13) mass as a count of crossings:  conjectured, not computed; makes the interacting (boson-like) braids massless   BHKW 2012 §3.7, citing BT05 and BTMS07
    (P8)  Koide: (m_e + m_μ + m_τ) / (√m_e + √m_μ + √m_τ)² = 0.666661 against 2/3                     measured relation, no accepted derivation (Koide 1983, 1990); unit-free, so it holds for counts n as for masses

**How each maps onto her rules** (ours, unverified): P2 is R4 with "twisted strand" for "twisted diagonal"; P3 is
E1 with a reason (two strands untwist, three cannot) that is about braids and transfers to loops as an analogy only;
P5 is R10's pairs made exact (her antiparticle is the colour-flipped loop; "reverse the order" is the loop run the other
way); P6 is what R10 needs as a rule in the toy, net twist conserved diagonal by diagonal under the moves; P7 is her
"net matter stays the same"; P1 is the relation she called "the self-referential loop of the twist curl relationship",
in its published form, and it says a coil can trade for twist only when the loop is a ribbon with two edges. P9 is the published form of her 'the twists repel somewhat' turned the other way: a coil (crossing) and a twist are
exchangeable, half a twist per strand per crossing, so a loop can hold its twist as shape or as twist, which is what her
'a tight loop needs a gluon to hold its twist' needs as a rule. P10 is the one place in the lineage where thirds are
derived rather than named, and it comes from a three-fold symmetry at a point, which the cubic lattice has along its body
diagonals. P11 is the exact shape a conserved net twist in the toy would take if the coil counts against it. P12 is R1
and R9 in their words: no twist, no stability. P13 is the published cousin of R6, with crossings for diagonals. P4 and P8
have no counterpart in her rules yet; P4 is the kind of count-to-rule result she asked for on 29 October, and P8 is the
sharpest known constraint on any mass rule for the charged leptons.

**What is not taken, with its kind (rule 23):** BT05's "no H+ with H− in one triplet" (theory structure, inherited
from Harari and Shupe; her proton holds both); BHKS's unbounded generations from a neutrino series [a, 2 − 2a, a − 2],
a even (theory structure; nothing measured supports a fourth generation); BHKS's half-integer twists on non-orientable
ribbons (allowed there, unused; her halves are the dark twist, a different thing); the rishon hypercolour force
(theory structure, added to fix problems the braid picture claims to avoid).

## 3. The observed table in her counts (step 1 of the path; `sandbox/observed_particle_table.py`)

Masses are PDG values from memory, **to verify** before any number leaves the sandbox. n is for n_e = 3; k = (n − 3)/3
is the gluons per run under R8 if the object were three thirds equally spaced.

| particle | mass MeV | Q (thirds) | spin | n (n_e = 3) | n mod 3 | content |
|---|---|---|---|---|---|---|
| electron | 0.510999 | −3 | ½ | 3.00 | 0 | lepton |
| muon | 105.658 | −3 | ½ | 620.30 | — | lepton |
| tau | 1776.93 (PDG 2025; 1776.86 before 30 Oct) | −3 | ½ | 10,431.7 | — | lepton |
| neutrino | < 8 × 10⁻⁷ | 0 | ½ | < 0.005 | — | lepton |
| photon | 0 | 0 | 1 | 0 | 0 | boson |
| π⁺ | 139.570 | +3 | 0 | 819.40 | — | u anti-d |
| π⁰ | 134.977 | 0 | 0 | 792.43 | — | u anti-u, d anti-d |
| K⁺ | 493.677 | +3 | 0 | 2,898.3 | — | u anti-s |
| ρ(770) | 775.26 | +3 | 1 | 4,551.4 | — | u anti-d, spins parallel |
| proton | 938.272 | +3 | ½ | 5,508.46 | — | uud |
| neutron | 939.565 | 0 | ½ | 5,516.05 | — | udd |
| Λ | 1115.68 | 0 | ½ | 6,550.0 | — | uds |
| Δ⁺⁺ | 1232 | +6 | 3/2 | 7,233 | — | uuu |
| Ω⁻ | 1672.45 | −3 | 3/2 | 9,818.7 | — | sss |
| W | 80,369 | +3 | 1 | 471,836 | — | boson |
| Z | 91,188 | 0 | 1 | 535,349 | — | boson |
| Higgs | 125,200 | 0 | 0 | 735,031 | — | boson |

(The script prints 37 rows, the quark masses among them with their scheme dependence noted.)

**The integer test, E5, run** (exact arithmetic on measured ratios; ours):

- m_μ/m_e = 206.768 283 0 ± 4.5 × 10⁻⁶; m_p/m_e = 1836.152 673 4 ± 5.7 × 10⁻⁷; m_τ/m_e = 3477.23 ± 0.23 (to verify).
- **n_e = 3 fails:** the muon would be 620.30 diagonals, the proton 5,508.46, the tau 10,431.7. None is whole; the muon
  misses by 0.30 where the allowed error is 4 × 10⁻⁵. (The record's "621 = 3 + 3 × 206" rounded 206.77 to 206.)
- Over n_e = 1 to 5,000, the muon alone is whole within error at 343 values (the first: 82, 164, 246, ...); all three
  together at 3 values (3,871; 4,061; 4,264), against a chance expectation of 3.8. So no n_e is singled out: the passes
  are what three unrelated ratios produce by accident.
- **Kind: measurement** (rule 23), after the arithmetic. What it kills: *m = u · n with one unit u and n a whole
  count*, for the electron, muon, proton and tau together, at every n_e up to 5,000 beyond chance. What it does not
  kill: mass as a count plus something else (an energy of the coil, of the twist, or of the gluon runs, which the
  standing rules already say is a separate ledger, §4 of the rules sheet); or masses that are not simply additive over
  parts (binding). Under rule 11 the four answers: (i) it falsifies R6 in its one-unit form, not the count idea;
  (ii) the failure is in the mechanism as stated, not implementation, parameters or size; (iii) the test that tells
  the readings apart is whether a second term with no free constant (the coil's energy, say, at the loop's own Wr) makes
  620.30 − 3 × 206 = 0.30 for the muon and 0.46 for the proton out of the same rule; (iv) abandon "mass is only a
  count" if no such term without a fitted constant closes both gaps.
- P8 (Koide) survives the change of units unchanged, so any mass rule for the leptons has to reproduce 2/3 to six
  figures; a count rule does so only if the counts themselves do.

## 4. The 2D phase: the two degrees of freedom, twist and curl, as a discrete set of values

*Corrected 2026-10-30 at the owner's words: "Don't think about 2d as a flat sheet it is not. It is the 2 degrees of
freedom twist and curl and their rules. They are not in space, or where they are doesn't matter ... not a plane: a set of
discrete values ... handedness ... a twist that can be charged creating a new force. Curl can be free in space, travel
like a photon." The first version of this section read "2D" as the paper 1 four-link sheet; that reading is withdrawn.
What was exact about that sheet is kept at the end, marked as facts about that toy and nothing else.*

**The value set** (hers; exact where marked): the phase is the set of pairs (τ, c) a carrier may hold, with the rules
that join them, and not a place.

| degree of freedom | carrier | allowed values | handedness | whose |
|---|---|---|---|---|
| curl c | a link (what is real is the total round a loop, C) | c_e ∈ {−4, …, 0, +1} on a link; C = ½ (free curl, the photon) or whole (a loop) | sign | hers; range exact |
| twist τ | a diagonal, and only where there is curl | matter: {−⅓, 0, +⅓} on box diagonals; dark: {−½, 0, +½} on face diagonals | the sign, "what we call handedness" | hers |

**The allowed combinations, from R1 to R10** (ours, exact from the rules; T = net twist, C = curl of the object, s = spin):

| C | T | s | what it is | rule |
|---|---|---|---|---|
| 0 | 0 | 0 | flat space; no curl, so no twist | R1 |
| 0 | ≠ 0 | — | **forbidden** | R1 |
| ½ | 0 | 1 | the photon: free curl, travelling, with the quarter twist and anti-twist beside it that cancel | R5, §3 of the rules sheet |
| ½ | ≠ 0 | — | not allowed as it stands: a photon carries no net twist | hers (photon rules) |
| 1 | ±⅓, ±⅔, ±1, … (thirds) | ½ | a matter loop: charge T, spin ½ in the direction of its twist; free only when T ∈ ℤ | R2, R4, R5 |
| 1 | 0, with twists ±θ present that cancel | ½ | the neutrino (hers, 30 October): no net twist so no charge, spin because twist is present; **its mass, her idea of the same day:** a loop whose twists cancel weighs nothing ("a ½ spin cancels mass when there is no twist"), which at the loop level survives the neutron (below) and at the object level does not; to test | R9 narrowed; idea, hers |
| 3 loops | −⅓ −⅓ +⅔ = 0 | ½ | **the neutron** (added at her request): three loops each with net twist, held by gluon runs, net charge 0, spin ½, 939.57 MeV; its loops do not cancel one by one, so under the loop-level reading it keeps its mass while the neutrino loses its own; *exact:* it differs from the proton (+⅔ +⅔ −⅓) by one loop's sign, and their mass difference is 2.531 electron masses, 7.59 units at n_e = 3, not whole (`sandbox/neutron_spin_mass_kill.py`) | hers; arithmetic ours |
| 1 | 0, no twist at all | ½ | **not seen**: an uncharged coil takes no spin | R9 |
| 1 | 0 | 0 | an uncharged coil, a long diagonal: curl with no twist, no spin, no charge (her gluon) | hers, Update 147 |
| 1 + 1 | T₁ + T₂ | 0 | two loops spinning against each other, each with its twist (the pion) | hers |
| n loops | Σ T | ½ or 0 | the proton ring and the hadrons: twist and anti-twist held apart by gluon runs | hers |

Two consequences the table makes visible (ours, exact from the rules):

- **Curl is the carrier and twist is the passenger.** Every row with twist has curl; the rows with curl and no twist are
  the photon and the gluon, which is why those two are the force carriers in her picture and matter is not: "a twist that
  can be charged creating a new force" is the C = 1, T ≠ 0 row, and "curl can be free in space, travel like a photon" is
  the C = ½, T = 0 row.
- **A tension inside the rules, flagged, hers to resolve.** R2 says a twist's sign is the colour of its diagonal's ends
  (black +, white −), and E1 says every closed loop of box diagonals is one colour. Together they give any one loop
  twists of one sign only. The proton ring (⅔, ⅔, −⅓ in one loop) contradicts that, and so does "a positive particle goes
  over 1 and twists back". One of the two statements has to give: either the sign is not fixed by the colour (then what
  fixes it is open), or the proton is not one ring. Kind: theory structure of her own rules; no measurement bears.

**Facts about the paper 1 four-link sheet, kept only as facts about that toy** (exact): its curl range is c_e = S_e − 2
∈ {−2, −1, 0, +1}; its (1,1) diagonal joins like colours and three of them cannot close; every closed loop of its
diagonals is planar. None of this is about the two-degree-of-freedom phase.

## 5. Mass as separate parts, each in its own unit (her instruction of 30 October)

Her words: "Spin must cancel mass somehow, so we need to separate out the twist spin charge and loop parts of mass and stop
using MeV for all parts because they may not add and subtract cleanly like a single unit." So from here the record keeps
the parts apart and writes MeV only for the measured total.

| part | symbol | counted how | unit | whose |
|---|---|---|---|---|
| loop (curl) | L | number of closed loops, one curl each | u_L | hers |
| diagonals | n | looped box diagonals, twisted or not (the long diagonal counts one) | u_n | hers (R6 as it stood) |
| twist present | τ⁺ | sum of the magnitudes of the twists on the loops | u_τ | hers |
| charge | T | net twist, Σ of the signed twists | u_T | hers |
| spin | s | ½ per loop with twist present; 0 for a coil with none | u_s | hers (R5, R9 narrowed) |
| measured mass | m | — | MeV | measured |

The combination m = F(L, n, τ⁺, T, s) is **open**. What is fixed about it (hers): spin takes mass away ("spin must cancel
mass somehow"), so F falls with s where the twists cancel; the neutrino (n ≥ 3, τ⁺ = 2θ, T = 0, s = ½) weighs almost
nothing and the neutron (three loops, τ⁺ = 4/3, T = 0, s = ½) weighs 939.57 MeV, so the cancelling has to act loop by loop
and not on the object (§4, the neutron row).

What is already published and bears (general knowledge, to verify before any citation): (a) masses do not add in standard
physics; a proton's quarks carry a few MeV of its 938, and the rest is the energy of the field between them, which is her
"gluon runs" with their own unit; (b) rest mass and motion combine as squares, E² = m² + p², so a part held as motion
would enter F under a square root; (c) Koide's relation ([Koide83], [Koide90]) is exact in square roots of the three
charged-lepton masses, and Brannen's form of it writes √m as one constant plus a cosine at three phases a third of a turn
apart (Brannen, 2006, not read; to verify), the first published place thirds appear in a mass formula. These are good
news in her sense: existing math to apply, not to invent.

Kill tests on the way, each with its expectation written first and none crowning anything: additive one unit, killed
(§3); count plus one curl's energy per loop, killed (`mass_ledger_kill.py`); whole counts in √m with one unit, killed, and
in m², unreadable from the tau's error (`mass_parts_combination_kill.py`). What is left is what she asks for: parts in
their own units, combined by a rule to be stated, and the first such rule to test is the Koide–Brannen one, since it is
published, exact to six figures for the leptons, and built on thirds.

**Candidate equations** (her request of 30 October, later: "how might it be a more complex equation involving positive and
negative twists in available values, spin, loops (with and without twists), ratios of different emergent values ... we need
that equation to equal the measured masses"). Each shape below says how many constants it has, how many masses it meets and
what it predicts (rule 13). The withdrawn step: "2.531 electron masses" for the neutron–proton difference assumed addition
and tests only the additive form.

- **E-A, published, already her shape** (Koide in Brannen's form; `sandbox/koide_brannen_look.py`):
  √m_g = A (1 + √2 cos(δ + 2πg/3)), g = 0, 1, 2 for tau, electron, muon. A constant part A and a signed part at an
  angle, the three leptons a third of a turn apart. Two constants, three masses, one prediction. Fitted on the electron and
  muon alone, the held-out tau comes out 1776.97 MeV against 1776.93 ± 0.09 (0.4 σ; PDG 2025, corrected 2026-10-30 from 1776.86 ± 0.12), and δ = 0.222222 rad, which is 2/9
  to six digits (Brannen 2006, not read; to verify). In her terms: the electron is light because its two parts nearly
  cancel (17.72 − 17.00); exact cancellation is at 3/8 of a turn and the electron sits 0.006 turn from it. **So in this
  shape a massless spin-½ loop is a loop at the cancelling angle**, a place for the neutrino that needs no cancelling of
  "mass" by spin: the angle does it. What it lacks: a rule that reads the angle from a loop's twists (thirds of a turn
  from thirds of twist is the obvious candidate and is not derived), and a rule for several loops.
- **E-B, hers to state, the loop sum:** √m = Σ_loops a_ℓ, with a_ℓ = A (1 + √2 cos φ_ℓ) and φ_ℓ read from the loop's
  twists, loops without twist (gluon runs) entering as their own term. *Exact (`koide_brannen_look.py`, part 4):* with no
  new constant, three signed lepton amplitudes added never give the proton or neutron; the nearest is three loops at the
  muon's angle, 950.9 MeV, 1.3 % high. So the nucleons need either their own angles or a term for the runs. Killed in its
  no-constant form; open with one.
- **E-C, the spin term:** her "spin must cancel mass somehow" as a signed part, √m = u_n n − u_s s + …, loop by loop. Not
  tested: it has at least two constants and the leptons alone (three masses) cannot separate it from E-A; the nucleons
  and the pion would. Hers whether to write it out.

**Form (hers, 30 October): a set of equations and inequalities, possibly recursive.** "It can be a set of equations/
inequalities. They may not immediately combine to one line without being recursive, which is a feature not a bug." So the
target is a system, not a formula. Already in that form: R1 to R10 (selection rules and inequalities, §2); Koide as a
constraint on three masses at once, (Σ√m)² = (3/2) Σm, with the thirds-of-a-turn spacing a second constraint rather than a
closed formula; and R8's equal spacing, (S L^k)³ with n = 3(k + 1), which is recursive in k. A recursive rule is also what a
simulation runs: one rule applied to its own output. Candidates below are read as members of such a system.

What the first precise equation would be for (hers): the starting point for the simulations, which means the parts
(L, n, τ⁺, T, s) must be readable from a saved wiring; today the toy reads curl and the counts, and twist has no reading
rule yet (§6, item 3).

## 6. Open after this document (hers to decide, in order)

1. R6 in its one-unit form fails the integer test (§3). Keep mass as a count plus a second ledger, or replace it.
2. The neutrino's mass: hers of 30 October places it as a loop with cancelling twists (spin from twist present), which answers
   the helon objection; the mass bound (below 10⁻⁶ MeV against 0.51 MeV for any loop at the electron's unit) is still a
   measurement against it. Her answer of the same day, to test: a loop whose twists cancel weighs nothing. At the loop level
   the neutron does not kill it; at the object level it does (`sandbox/neutron_spin_mass_kill.py`). Still open under it: the
   three neutrino masses differ by about 10⁻⁸ MeV, so the three kinds cannot differ by whole counts of any unit near the
   electron's (measurement); and whether a loop with cancelling twists can be pulled apart or always stays whole.
   Also hers: whether 'sign by colour' gives, since she accepts twists of both signs on one loop.
3. Whether R10 is to be built as P6 (net twist conserved diagonal by diagonal under the toy's moves): the first exact
   piece of the diagonals build, no constant.
4. Whether a framing (a ribbon with two edges) is given to her loops so that P1 applies, or the coil stays a shape.
5. P4's counterpart: does any count on her loops give hypercharge and the Gell-Mann–Nishijima relation.
