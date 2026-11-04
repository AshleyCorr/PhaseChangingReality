# Twist and curl: the standing rules (29 October 2026)

*The one page to read for the particle rules as they stand. Written at the owner's instruction of 29 October ("continue
building on this and scrub older ideas that may create confusion"). It replaces `twist_curl_rules_2026-10-27.md` as the
working sheet; that file is kept as the history and is not edited further. Every line says whose it is: **hers** (her
rule, her words put in order; verbatim in `sandbox/NOTEBOOK.md`), **exact** (arithmetic or geometry, checked),
**ours** (our reading or reasoning, unverified), **open** (hers to decide). Nothing here is a result: a rule is put in,
and the results are what follows from the rules together. Withdrawn ideas are listed at the end by name so that they
are not mistaken for current ones. The model with any of this is our family around the published model, never
combinatorial quantum gravity.*

## 1. The lattice and its two quantities

- **The lattice** (exact): black and white points, six links a point; a link joins black to white. Face diagonals
  (1, 1, 0), length √2; body diagonals (1, 1, 1), √3, black to white; box diagonals (1, 1, 2), √6, black to black or
  white to white. A closed loop of box diagonals is one colour.
- **Curl** (hers; exact in the toy): on a link, c_e = S_e − 4 (flat space has four squares on every link). One extra
  square on a link is one whole curl, 360°. Links are not real; what is real is the total round a closed loop.
- **Twist** (hers): lives on diagonals and only where there is curl: no curl, no twist, no degree of freedom. Matter
  twist in thirds on the box diagonals (the "diagonals"); dark twist in halves on the face diagonals. The sign is the
  colour: black +, white −. A diagonal runs black to black, an anti-diagonal white to white.
- **Gravity and light** (hers): gravity travels through the cube edges (the links; their lengths relax from curl with
  one constant α, the total length fixed near matter); light travels through the body diagonals, whose length is derived
  from the cube's edges by geometry with no new constant. The pyramid of face diagonals with the body diagonals as its
  axes is for score-keeping and for light to pass through; it holds no box diagonals and so no mass.
- **The sectors (hers, 31 October, a position, not yet a decision):** the box diagonals carry ordinary matter; the
  face diagonals carry dark matter and the neutrinos (**her correction, 21:16 UTC: neutrinos are not the dark matter;
  the neutrino is two-sided, two half-twist loops of opposite hand; dark matter is a one-sided loop of surplus curl, a flaw
  left when space opened, which meets gravity and not light.** *Exact:* no loop of box diagonals is one-sided in thirds, so
  the one-sided loop lives on the face diagonals; it costs 3/4, exactly half the neutrino's 3/2, under the term as built;
  its net half twist is not charge, by her rule of 23:26 UTC below, so the whole-charge rule has nothing to refuse;
  `sandbox/one_sided_loop_look.py`); the waves, gravity and light, run
  on the black-to-white lines, the
  edges and the body diagonals. *Exact:* a step changes colour exactly when its coordinate sum is odd, so edges (1) and
  body diagonals (3) change colour and face (2) and box (4) diagonals do not: one-colour loops carry twist and mass,
  colour-changing lines carry waves, with the sign (black +, white -) alternating every step on them. Owed: how a
  face-diagonal neutrino couples to ordinary matter (it is detected through it in the world). Published relative
  (general knowledge, to verify): sterile-neutrino dark matter.

## 2. Particles

- **A particle** (hers): a closed loop of box diagonals, or several, with one curl of energy per loop or full coil.
  **Mass counts looped diagonals**, twisted or not; the coil (the loop's writhe, called 'fold' before 29 October) is a shape,
  not a mass. **Refined (hers, 30 October):** mass has separate parts, twist, spin, charge and loop, each in its own unit, and
  MeV is the measured total only; the parts need not add like one unit, and spin takes mass away where the twists cancel
  (`twist_curl_relations_2026-10-30.md`, §5). **A long diagonal** (hers, 29 October) is a curl held along one untwisted diagonal longer than 1 × 1 × 2, such
  as a 1 × 1 × 4 (√18): her gluon as a lattice object, mass and curl with no twist, charge or coil. **Decided (hers, 29 October, Update 148):** it counts
  one of mass, not its length (`sandbox/long_diagonal_mass.py` has both ladders).
- **The unit** (hers): the closed loop of three white thirds (−⅓, −⅓, −⅓), net twist −1, is the electron: charge −1,
  spin ½, the smallest charged mass. (Exact: 0.511 MeV ÷ 3 = 0.170 MeV per looped diagonal; proton 5,508 diagonals,
  neutron 5,516, charged pion 819, muon 621.) **Open (hers, 29 October):** the electron may hold more than three
  diagonals, if a three-loop of thirds is too tight to hold its twist without a gluon fold; then the unit is 0.511 ÷ n.
  Exact: three box-diagonal steps do close on the lattice (flat triangles, fold 0), so "too tight" is a rule about
  holding twist, not geometry.
- **Charge** (hers, 29 October): charge is twist carried by spin; spins can cancel while twists add. A loop's charge is
  its net twist, a whole number of thirds summed round the loop. A positive particle goes over 1 (⅔, ⅔) and twists back
  (−⅓) to net a whole number; gluon folds keep the opposed twists apart.
  **Narrowed (hers, 31 October, 23:26 UTC): "a charge needs 2 sides and can only be applied on thirds."** Charge is
  the net twist of two-sided loops of box diagonals only; a face diagonal's half twist is twist, not charge, so a
  one-sided face loop has charge 0 and meets no light. Built: `Particle.charge` sums box loops only. Open (hers):
  whether the face half twist is a dark charge of its own kind (Update 34) or no charge at all; as built, none.
- **Spin** (hers, 29 October, "precise"): full spin on half a curl (the photon, spin 1); half spin on a full curl
  (matter, spin ½). A photon's half curl is spent on a loop's |⅓| segment, the only size a photon can act on; the
  photon's curl stops and the loop spins ½ in the direction of that twist. With spin a loop can hold its twist stably in
  less mass; an uncharged coil takes no spin, so no uncharged electron is ever seen. **Narrowed (hers, 30 October):** spin
  needs twist *present*, not net twist: the neutrino is a loop whose twists are plus and minus of the same increment,
  cancelling to no charge, and it spins. Open: its mass under the count rule (a three-loop weighs 0.51 MeV at the unit;
  the bound is below 10⁻⁶ MeV). **Spin with no twist is a second
  kind of force**, probably the one that holds protons and neutrons together.
- **Exact, the four test particles** (O166 and the free electron): electron (−1, ½): one twist, one spin. Neutron
  (0, ½): twists net 0, spins net ½. Photon (0, 1): spin, no twist. Charged pion (1, 0): twists ⅔ + ⅓, spins opposed
  (hers: a quark and an antiquark spinning against each other, each with its twist). All pass.
- **The proton** (hers): a large closed ring of about 5,500 diagonals, charged and spinning, long in shape; its loop
  holds twist and anti-twist and fights itself. Ours, to verify: the proton's measured form factor is close to round,
  and the ring's size follows from the scattering bound on the electron (about 10⁻¹⁸ m), the bound she adopted.
- **Excitation** (hers): a photon's excitation is of the electron's orbit round the nucleus, the known science; photons
  are caught by the electron field and do not reach the nucleus. Ours, to verify: a free electron cannot absorb a
  photon on its own, which agrees.
- **Pairs and annihilation** (hers): curl is always made in pairs, curl with anticurl, twist +θ with −θ; two photons
  closing an open diagonal make a matter and an antimatter loop; a black and a white twist meeting cancel, their curl
  leaving as photons. Spacetime holds only the net matter the burp left: pairs cancel, so net matter stays the same.
  Ours: this conserves net twist (matter minus anti), not the diagonal count; net twist is not yet defined in the toy.

## 3. The photon

- (hers) Half a curl with one unit of spin, moving along the body diagonals as a smooth spiral round a four-cube loop at
  one link per tick (the speed of light, the speed at which flat space opens), costing nothing to pass. The diagonals
  beside it take about a quarter twist and anti-twist, uncharged, gone once the photon is two links on.
- ~~(ours, not built) A photon's energy in the world is continuous; half a curl is one amount in the toy, so the energy
  would have to sit in the spiral's pitch.~~ **Decided (hers, 31 October):** the curl angle may turn less than a quarter
  turn per link, so the repeat (the wavelength) is any whole n ≥ 4 and the curl per link is 2/n; the photon's energy is
  its curl per link, set by its wavelength, the reading that matches measurement (`graphity.twist.Photon(period=n)`;
  build document §9). One unit of spin in and a half out on the loop still leaves a half to be placed.

## 4. The energy, as it stands

    E = 16(3N − S) + 4X             the published curvature, six links (exact)
      + 4(λ − 1) X                  the curling cost (ours; decided)
      + ½ δᵀ K δ + α Σ_e |c_e| δ_e  lengths on the cube edges, K = 4(6I − A) per family (exact), Σδ = 0 per family (hers)
      body diagonals ℓ = √(ℓx² + ℓy² + ℓz²)   derived, no constant (hers)
      twist read from curl (no state of its own); charge, spin and mass by section 2 (hers)

Not in the energy and not built: a twist term (the κ(1 − cos 2πτ) form was ours and is not adopted); a photon's energy
ε (hers: the photon is its own unit, not tied to λ); the growth clock (a double row added to every empty ring); net
twist as a conserved quantity; the second force of spin without twist.

## 5. Open, in order

1. **The stability rule** (hers to propose): what forbids loops of 4, 5, ... diagonals with twist and spin, so that
   nothing charged sits between the electron and the muon (621 diagonals). "No twist, no stability" covers the neutral
   side. Exact: whole-fold loops are not rare (48 at length 5, 7,824 at length 7).
2. **Mass against energy**: the toy gives a loop one curl of energy whatever its length and a mass equal to its length,
   two numbers where the world has one (hers: that is the underlying pattern the world's one number hides). The first
   check is E = mc² as a prediction: one ratio of the two counts for every measured object.
3. **The electron's size n**, and what pins it (the gluon fold on a tight loop).
4. **The rho meson**: the pion with spins parallel, 775 MeV; the rules do not yet make a spin flip cost diagonals.
5. **Where a photon's energy lives** (the spiral's pitch), and where the other half unit of spin goes.
6. **Net twist defined and conserved** in the toy: the first exact piece of the diagonals build, no constant. *Built 2026-10-30: `graphity.twist`, net twist = the sum of twist over a set of particles, conserved by construction under pair creation (`tests/test_twist.py`); the state is `twisted_diagonals_build_2026-10-30.md`.*

## Withdrawn, by name (do not build on these)

- The pyramid as a candidate particle at four diagonals, 0.68 MeV (ours, miscounted; it has no box diagonals).
- "Fold must be whole" as the stability rule (killed by the five- and seven-loop counts).
- Spin angle = ½ × net twist angle, spin as a function of charge (killed by the audit, O166).
- Charge = net twist + photons held, with the photon's half curl as charge (ours, as read; replaced by charge is twist
  carried by spin).
- Mass = |twist| + |fold| (ours; replaced by mass counts looped diagonals).
- A further photon excites the loop (replaced by excitation of the orbit).
- Twist as a state of its own on the diagonals, τ_d = s_d [round(n C_p)/n + h_d/2] (ours; twist only where curl).
- Thirds on the cube body diagonals (Update 97; they are on the 1 × 1 × 2 box diagonals).
- Each ⅓ twist carries 60° of spin, three together 180° (tied to the killed spin rule).
- The proton as a needle; the moment-based bound on the electron's size (the scattering bound is used).
- Prices version A (curl 4(λ − 1), anticurl 4λ); version B is chosen.
- The fit of the direction tie to 5.36; the three-share story by order of opening (Update 50).
