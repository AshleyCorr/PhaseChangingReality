# What could carry a pull in this family: the candidates, side by side (1 November 2026)

*Written at the author's request ("can we try the entropic route and identify some additional carrier options"). Ours,
unverified by a physicist, except where a line is marked exact or measured and points at its O entry. It gathers what
the record already holds (`gravity_brief.md` sections 2, 6, 7; `soft_space_brief.md`; O22, O31, O55, O60, O61, O87,
O97, O147, O148) and adds one candidate that did not exist when those were written. Nothing here is decided; every
candidate that adds an ingredient is a knob under S1 and needs her dated decision before any run.*

## The one fact every candidate has to get past

Flat space in this family is gapped: the cheapest move out of it costs 32, 64 or 128 with four, six or eight links,
at every λ (exact, O22, O51, O50). A medium with a gap carries an effect only over its correlation length, and in the
warm published model that length is about two links at every coupling and every size from 196 to 676 (measured, O31:
the square-density correlation falls to about zero within two steps). So nothing that lives *in the wiring alone* can
carry a pull farther than a step or two. That is why O22 (no energetic force between defects at fixed wiring), O32 and
O56 (no counting pull at fixed wiring) and O58 are one fact seen four times. A carrier is something added, or something
the wiring already has that is not gapped.

## The entropic route, tried as far as it can be without a decision

- **What it is.** A pull from counting: the free energy of the warm surroundings depends on where two relics sit, so
  −g ln P(r) falls as they approach. Verlinde's form is F = T dS/dr ([Ver11], read;
  `docs/reading/notes/2026-10-05_dark_energy_and_entropic_gravity.md`).
- **What is measured.** The warm background does charge a curled ring about two units of entropy beyond a flat object
  of the same squares, a penalty that does not grow with the background's deficit (T78, T79; O147, O148). So the
  surroundings feel a relic. The 22 October look found the likely mechanism: the hard-core rule forbids most switches
  beside a saturated link (12 to 13 % allowed round a curled ring against 34 to 35 % round its uncurled twin).
- **What the one-step look of today says** (`sandbox/pair_handles_look.py`; output
  `sandbox/out/pair_handles_look_2026-11-01.txt`; a look, kill or keep, not a free energy). Two curled relics (short
  handles, one curled ring each, 12 points) built into a flat 20 × 20 torus at separations 2 to 9; the valid switches
  available to their surroundings, against the sum for each alone plus the cross pairs two far-apart relics always
  share (48 × 48 = 2,304). The energy is exactly additive at every separation (O22 again). The count of moves is
  **lower when the relics are close**: 524 fewer at separation 2, 132 at 3, 42 at 4, 4 at 5, and exactly 0 from 6 on.
  Read as entropy at this order, two curled relics have *less* room together than apart, within about five lattice
  steps, and feel nothing beyond. So at the one-step order the entropic interaction between curled relics is a
  short-range repulsion, not a pull; it dies at about the correlation length O31 measured.
- **The decisive prerequisite was already measured.** O103 (d) said: measure the warm sheet's correlation length first;
  if it is short the route is closed without running the pull. O31 measured it on 23 September for another purpose:
  about two links. With today's look agreeing in range and giving the sign, the result of "trying the entropic route"
  at this writing is: in the warm published model at N ≤ 676 an entropic interaction between relics reaches a few
  links and is repulsive at the one-step order. What would reopen it: a coupling at which the correlation length grows
  with N (the critical point of a continuous transition, which T16 did not find up to 676), or a different background.
- **A registered test that would price it anyway**, if she wants the number: two relics pinned in the warm sheet
  (switches that would destroy either are refused), the mean energy ⟨H⟩(r, β) at separations 2 to 8 and couplings from
  the melt down, and the free-energy difference between separations by integration in β from the melt, where the pinned
  ensembles' entropies differ only by the forbidden volume the look above counts. Cost: T78's chains, about a laptop
  day. Her prediction first. It would give a size for the contact-range interaction and nothing beyond.

## Carriers that add something (each a knob; her decision first)

| | What is added | Exact or measured so far | Shape of pull | What it gets wrong, known in advance |
|---|---|---|---|---|
| **A. A massless scalar on the points**, relics as sources in proportion to their energy (gravity_brief §6, option A) | one number per point, energy Σ over links of the squared difference | O61: integrated out exactly, the fluctuation part is a Casimir attraction ~ r⁻⁴ (2D) and r⁻⁶ (3D), about 10⁻³ g at contact; the source term is the graph's Green's function, 1/r in three directions | Newton's; like sources attract | Nordström's scalar gravity: right Newtonian limit, bends no light, −1/6 of Mercury's shift ([Giulini08]; O103 (f)). The clocks-and-rulers scalar that passes the static tests has a preferred frame with α₁ = −8, excluded ([WN72]; `soft_space_brief.md`) |
| **B. The link-length field** (`graphity.edge_wave`, built 1 November; decision (1) confirmed) | one length per link; stiffness K = 4(6I − A), four times the graph Laplacian on a 6-regular graph | exact: gapless (ω² → 0 as k → 0), speed one link per sweep at m = 4, energy conserved, families never couple (tests). *New today, exact by inspection:* the static response to a source is the same Green's function as A's, so a relic that stretches or shortens the links round it in proportion to its energy is a 1/r carrier in three directions, and the field is already in the code | Newton's at the static level; and it is a metric perturbation, not a number on points | as built it is the all-lengths-alike (conformal) part of a metric, a scalar in disguise, so A's light-bending objection applies until the two polarizations (decision (3), the face diagonals' field) carry a traceless part; and as built the lengths have no back-reaction on the switch energy, so B carries a pull only once a rule says what a stretched link costs the wiring. That rule is the knob |
| **C. A quantity on the points that sets the local pace** (VISION Update 40; `soft_space_brief.md`, "clocks") | one number per point; a point's value sets how often it may change | nothing exact yet; Bianconi's G-field is its nearest published relative (O97) | whatever wanders collects where clocks run slow (Newtonian, through the time part of the metric, where slow matter's gravity lives in Einstein's account) | the toy has no clocks; "how often a point may change" is a rate of the sampler, not a time (O93); the preferred-frame theorems apply as to A |
| **D. Interchangeable points** (Update 12, decided) | no new state; an arrangement's weight is divided by its renamings | exact: a drive to curl only at full curling, beating the curling cost only below λ ≈ 1.02 (O55, O72); no counting pull between two defects at fixed wiring (O32, O56) | none at a distance | a weight on whole arrangements, not a field; it carries nothing between two places |
| **E. Long links** ([PWS09], disordered locality) | a few links between far-apart points, kept by rule | nothing built; the hidden count (O87, T52) is the related measurement | not a pull: an energy density that behaves as dark energy if the count grows with volume | the published proposal needs about 10⁶⁰ such links and assumes the growth; nothing here derives it |
| **F. Test particles on the geometry** (de Bakker and Smit's route) | walkers whose paths follow the wiring | nothing built | Newton's, in dynamical triangulations | needs a geometry that fluctuates without melting, which this flat space does not do (gravity_brief §6, D) |
| **G. Loop phases** (a phase per square) | a U(1) on squares | nothing built; the literature says vector modes | like relics repel | a route to light, not to gravity (gravity_brief §6, B) |

**Where this leaves the choice** (ours). B is the new entry and the cheapest to try, because the field exists and is
tested. The exact first look, needing no decision: the static Green's function of K on the six-link torus with one
relic as a source, set beside A's O61 numbers (they should coincide up to the factor 4). The decision it then needs is
the back-reaction rule: what a stretched link costs the wiring, and how a relic sources the stretch. With that rule B is
the discrete cousin of a metric and, with decision (3), the one candidate on this list that could in principle carry two
polarizations. C stays what Update 40 made it: decided in aim, with no form. The entropic route is priced at contact
range, repulsive at the one-step order, and closed beyond it in the warm published model.

## The Bell argument, as a reminder (O29; exact as an argument, unreviewed)

A "version" in Update 17 is a renaming of the points that leaves every relationship intact. Take two small loops in a
fixed arrangement and attach detectors to them. Any renaming either fixes each loop's two pairs of points or swaps them,
and it does so for every loop at once; so each renaming is a complete list of answers, one for every setting of every
detector, written down before any detector is read. Attaching a detector only removes some renamings from the list (the
ones that would move the detector); it never changes the answer a surviving renaming gives. A theory in which every
outcome of every possible measurement exists in advance is a local hidden-variable theory, and Bell's theorem bounds
its correlations: CHSH's S ≤ 2, and GHZ's parity test comes out classical, for every arrangement. So counting the
versions of a fixed arrangement cannot reproduce the quantum correlations. What survived (Updates 20, 21): the
object-level counting, the loop's vibrations, and the need for an amplitude rule (Hardy's fifth axiom) if the quantum
leg is to go further; what did not: "superposition is the set of undetectably different versions of one arrangement".
