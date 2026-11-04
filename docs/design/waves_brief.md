# Waves in the toy: a design brief for the owner's decision

**Why this exists.** The owner wrote on 31 October 2026: "We need the toy to include waves." Under S1 a new ingredient's
form is fixed in writing by her before it is built. This page lays out what a wave needs that the toy lacks, three forms,
and one recommendation. Form B is built (section 3a); nothing has been run on the toy's states. Kinds are labeled:
exact, built, ours, hers.

## 1. What the toy has and lacks (exact, built)

- **No time.** The toy's only clock is the count of Monte Carlo moves (sweeps). Her light is a `Photon` whose curl
  angle advances by 360/n degrees per link along a body diagonal (`Photon.advance(sweeps)`), one link per sweep: a
  pattern that moves, not a thing that is pushed.
- **A stiffness but no inertia.** Gravity's carrier on the cube edges is the edge lengths' stiffness K = 4(6I − A) per
  axis family (exact, gravity_link_brief step 1). A wave on it needs an inertia, which the toy has nowhere.
  `sandbox/edge_wave_look.py` (31 October) assumed a unit inertia and found one speed for all long waves, 2 links per
  unit time, the shortest edge wave 2 links, the "+" polarization exact on the axis edges and the "×" one needing the
  face diagonals.
- **One c (hers, 26 September, Update 33 answers (b)).** Light and gravity share one speed. In the toy this is a
  condition, not yet a fact.

## 2. Three forms

**A. Kinematic waves (no new constant).** A wave is a profile that moves one link per sweep at a fixed speed, as the
photon already does; add the same for an edge stretch along an axis. *What it buys:* the photon ladder and a
gravitational "+" pulse side by side, both drawable. *What it cannot do:* bend, slow, scatter or interact, so none of
her questions about light near a black hole (31 October) or a front slowing near matter (26 September) can be asked.
Ours: not worth building alone.

**B. Dynamical edge waves (one constant, pinned by her one c).** Each edge length ℓ obeys m ℓ̈ = −K(ℓ − ℓ₀) per axis
family, stepped by leapfrog; the energy ½ m ℓ̇² + ½ ℓᵀKℓ is conserved to rounding. *Exact:* the long-wave speed is
2/√m links per unit time. If light's speed is one link per sweep (the photon as built), then **one c fixes m = 4**:
no free constant. The "×" polarization needs the face diagonals' lengths as a second field (ours, geometric; decided
by her). *What it buys:* gravitational waves that propagate, reflect and superpose on the lattice; the pull brief's
halo (an edge-length dip) as a static limit of the same equation. *What it owes:* a rule-6 test set (energy conserved,
the plane-wave dispersion, the speed at long wavelength, same-seed reproducibility), and a statement that the clock is
the sweep.

**C. The photon in a lengthened lattice (needs the derived diagonal length).** The photon advances one body diagonal per
sweep; where the edges are lengthened by the halo the body diagonal is longer (derived, her decision of 28 October, not
built), so the photon's path takes longer there: light slows and bends near curled matter. *What it buys:* her ring
(light aimed inside a capture radius falls in, outside escapes) as an exact ray computation, and the first toy
statement about light near a black hole. *What it owes:* the derived diagonal length, and a rule for the photon's step
when the diagonal is not a lattice step (interpolation or a per-link delay).

## 3. Recommendation (ours)

B first, with m = 4 pinned by her one c, then C on top of the halo B gives at rest. A alone answers nothing she has
asked. Her decisions needed, in order: (1) the clock is one sweep = one link of light (B and C both need it). *Her
question of 31 October, 23:27 UTC, "Can we write time as links per sweep?", is read as taking this: time in sweeps,
speeds in links per sweep, c = 1 link per sweep. **CONFIRMED by her, 2026-11-01, 01:45 UTC ("Yes proceed"), with the
sweep as written: the chain's sweep (2N attempts) for a wave on a fixed lattice, the fair sweep of O90 (N/96 chain
sweeps) where the wave runs beside the chain.** (2) m is
pinned by one c, not scanned: *read as taken with her "proceed", which answers the recommendation naming it; INFERRED
until she objects.* (3) whether the face diagonals carry a second length field (the "×" polarization) now or
later: *taken by her message of 2026-11-01 ("The face-diagonal length field for the ×' polarization. So face
diagonals are the dark matter and dark antimatter."; VISION Update 149): the face diagonals carry the second field; INFERRED.
Building it waits on the face diagonals' stiffness at the flat point, exact and owed.* (4) C's step rule for a photon on a non-lattice diagonal: open.

## 3a. Built (2026-11-01)

Form B is built: `src/graphity/edge_wave.py`, tests in `tests/test_edge_wave.py` (rule 6: the stiffness against an
independently built dense matrix, the plane-wave frequency exactly, energy conserved without drift, the travelling
wave at one link per sweep, families independent, same seed bit for bit). Exact: ω² = (4/m) Σ 2(1 − cos kᵢ), long-wave
speed 2/√m, so m = 4 gives c = 1 link per sweep; band top ω² = 12; leapfrog dt = 0.05 sweeps. Not built: the face-diagonal
field (decision (3)) and form C (decision (4)). Nothing has been run on a state of the toy; the first use, a "+" pulse
beside the halo of a relic, waits on a pre-registration with her prediction.

## 4. What no form can show

Anything about the real universe. Spin 2 is not in any form here (the edge lengths carry no turning angle); a wave on
the lattice has a preferred frame (the lattice), which the band of Update 36 already carries as a stated condition.
