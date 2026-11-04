# Reading notes: what a graph model needs to produce a pull between relics

> **Status of these notes:** written by an assistant agent on 2026-09-25 from fetched copies of the papers. Some passages
> were read through a tool that summarizes a page, so every quotation must be checked against the paper itself before it is
> used in a paper or a letter. Inferences marked *ours* are unreviewed.


Written 2026-09-25 by an assistant agent (Claude) for the gravity track (`docs/design/gravity_brief.md`;
ASSUMPTIONS O22, O31, O32, O56). Every paper below was fetched and its text searched; section and
equation numbers are from the arXiv version read. Where only an abstract was seen, the entry says so.
Everything marked *ours, unverified* is the agent's inference and has not been checked by a physicist.
Journal references are given only where one was seen in the text or on the publisher's page; otherwise
the arXiv number is the reference.

---

## (a) The answer in one paragraph

**The lead reviewer's hypothesis is right at its core, and needs three corrections.** Right: an
interaction carried by a medium dies off over the medium's correlation length. Fluctuation-induced
forces have a range "related to that of the correlations of the fluctuations" (Kardar and Golestanian,
Sec. I). Gapped quantum ground states have exponentially decaying connected correlations (Hastings and
Koma, Thm. 2.8). A massive mediator gives the Yukawa potential, e^(−mr)/r (Tong, Eq. 3.73). Gapped fracton
models give only an exponentially short attraction (Pretko, Eq. 24). And Trugenberger himself writes that
both phases of this model "are governed by a finite correlation length", which diverges only at the
critical point ([T25] v2, "Cycle condensation", text around Eq. 23). The flat vacuum here is also frozen,
because every move out of it costs 32, 64 or 128 (O56). That is the extreme case of a gap.

The three corrections:

1. **The curvature correlator is the wrong test for a gap.** In four-dimensional Einstein gravity, which
   does have a massless graviton, the curvature two-point function is a pure contact term at tree level.
   It appears away from contact only at one loop, ∝ G²/x⁸, while the *volume* correlator carries a
   tree-level ∝ G/x² (Laiho and Ratliff, Eqs. 218 and 223). In two dimensions, curvature correlators
   vanish because 2D gravity has no propagating degrees of freedom at all (van der Duin and Loll,
   Secs. 5.3 and 6). So O31 is consistent with a gap but does not detect one. The better test is how the
   metric, meaning ball volumes or distances, responds at distance r from one relic.
2. **Gapless is necessary but not sufficient.** A gapless medium only guarantees some power law. For
   Newton's 1/r the relic has to act as a *Gauss-law charge tied to its energy*: a monopole source,
   coupled without derivatives. Otherwise the gapless medium gives Casimir-type power laws that fall
   much faster, such as the 1/R⁴ between stiff inclusions in a membrane (Goulian, Bruinsma and Pincus;
   Kardar and Golestanian, Eq. 17).
3. **The sign depends on the spin of the mode.** Vector-type modes, the "Coulomb phase" that hard local
   constraints produce, make *like* defects repel (Henley, Eq. 4.1; Tong, Sec. 3.5.2; Giulini, Sec. 7).
   Scalar exchange (Nordström) makes like sources attract. So does the tensor theory with the Einstein
   sign, where Gu and Wen get "force ... proportional to m1m2/r2. Two masses with the same sign attract"
   (Gu and Wen 2009, Sec. IV). And 1/r needs three large directions: the same Gauss law gives ln R in two
   (Henley, Sec. 4.2).

In discrete classical models, the robust way to get a protected gapless mode is a hard local constraint
with an extensive, liquid-like set of ground states (Henley's conditions C1 to C3; the three-dimensional
dimers on a bipartite lattice of Huse, Krauth, Moessner and Sondhi). It does not come from a broken
symmetry, and it does not need a critical point tuned by hand.

Neither no-go theorem forbids the directional goal. The Weinberg–Witten theorem assumes Lorentz
invariance and a fixed spacetime, and neither holds here, since space itself emerges (Carlip, Sec. 3.7,
item 4). Marolf's theorem constrains only nonlinear universal coupling and places "no constraints on the
emergence of strictly linear spin-2 degrees of freedom". The one discrete random-geometry model that has
shown a Newtonian pull is four-dimensional dynamical triangulations. There the matter was a test field
moving on the geometry, and the result needed a tuned phase plus extrapolation to the continuum and to
infinite volume (de Bakker and Smit 1997; Dai et al. 2021).

---

## (b) Papers read

### Topic 1. Forces carried by a medium: gapped against gapless

**[KG99] M. Kardar and R. Golestanian, "The 'friction' of vacuum, and other fluctuation-induced forces,"
arXiv:cond-mat/9711071 (Rev. Mod. Phys. 71, 1233 (1999), per the arXiv listing; not checked against the
journal).** Read: Secs. I to III in full (pp. 1 to 6); the dynamic-Casimir sections skimmed.
- Sec. I: two ingredients, "a fluctuating medium" and objects that modify its fluctuations; the range
  "is related to that of the correlations of the fluctuations. The most interesting cases are when the
  interactions are long-ranged, corresponding to scale free fluctuations."
- Sec. II.B (Fisher and de Gennes): near a wall a binary mixture is perturbed "only over a distance of the
  order of the correlation length ξ. Any interaction mediated by the concentration fluctuations must also
  decay with this characteristic length." Power laws appear only at the critical point, Eq. (2).
- Sec. II.C: long-range forces come from "any correlated medium, by which we mean any system with
  fluctuations that have long-range correlations"; besides critical points, "much more common are cases
  where long-range correlations exist due to Goldstone modes of a broken continuous symmetry."
- Sec. III.B, membrane inclusions: the direct disturbances "tend to be short-ranged, falling off
  exponentially with a characteristic length related to the distance over which the lipid membrane
  'heals'". The fluctuation-mediated part exists "as long as the rigidity of the inclusion differs from
  that of the ambient membrane, and fall[s] off as 1/R⁴"; for stiff inclusions it is attractive, Eq. (17).
  Rods: Eq. (18), 1/R⁴ with angular dependence.
- Sec. I and III.B: fluctuation-induced forces "are non-additive, and cannot be obtained by adding
  two-body potentials."
- **Bearing.** This is the textbook form of the hypothesis. The range equals the correlation length
  unless the medium is critical or has soft modes. *Ours, unverified:* O22's exact additivity at fixed
  wiring is what is expected, because a fluctuation-induced pull exists only when the wiring between the
  relics is free to rearrange. The membrane case shows that a gapless medium plus inclusions coupled to
  *curvature* gives 1/R⁴, not a Newtonian law.

**[GBP93] M. Goulian, R. Bruinsma and P. Pincus, "Long-range forces in heterogeneous fluid membranes,"
Europhys. Lett. 22, 145 (1993). Abstract only** (publisher's page); body known through [KG99] Sec. III.B.
- Abstract: a membrane-mediated interaction between inclusions that "falls off as 1/R⁴, can be attractive
  or repulsive depending on the temperature and the elastic properties of the inclusion and the membrane."
- **Bearing.** The standard example of a gapless medium (a tensionless membrane, energy set by bending)
  giving a long-range but non-Newtonian pull between inclusions that only change the local stiffness.
  *Ours, unverified:* a relic that only stiffens or softens the wiring around it would be the graph
  analog, and should be expected to give a fast power law at best.

**[KK06] O. Kenneth and I. Klich, "Opposites attract: a theorem about the Casimir force,"
arXiv:quant-ph/0601011 (journal reference not checked).** Read in full (4 pp.).
- Abstract: "the Casimir force between two bodies related by reflection is always attractive, independent
  of the exact form of the bodies or dielectric properties." Eq. (5): the interaction is
  ∫dκ log det(1 − T_A G₀ T_B G₀), controlled by the propagator G₀ between the bodies. It holds "for a
  scalar field in any dimension"; extension 1 covers finite temperature.
- **Bearing.** The *sign* of a fluctuation-induced force between two identical, mirror-image relics is
  attraction, whatever their shape. Its *range* is set by G₀: *ours, unverified*, a mass m in the
  propagator makes it fall off as e^(−2mr). So the theorem supports "identical relics attract if anything
  carries the force", and the gap still kills the range.

**[HK06] M. B. Hastings and T. Koma, "Spectral gap and exponential decay of correlations,"
arXiv:math-ph/0507008.** Read: abstract, Sec. 1, Sec. 2 (definitions and Theorems 2.6 and 2.8).
- Sec. 1: the "folk theorem ... that a nonvanishing spectral gap above the ground state implies
  exponentially decaying correlations in the ground state", proved here for lattice quantum systems with
  short-range interactions on a wide class of lattices. Theorem 2.8: connected correlations of bosonic
  observables decay as exp(−μ̃ dist(X, Y)) under a uniform gap (Definition 2.5).
- **Bearing.** The rigorous quantum statement of the hypothesis. *Ours, unverified:* it is proved on a
  fixed lattice with a fixed distance. Here the distance is itself dynamical, so it applies by analogy.
  The classical counterpart is a finite correlation length for *every* local observable ([KG99] Sec. II.B).

**[Tong-QFT] D. Tong, *Quantum Field Theory*, lecture notes, University of Cambridge, Ch. 3 "Interacting
Fields".** Read: Sec. 3.5.2 "The Yukawa Potential".
- Eq. (3.68): a static point source of a field with mass m has profile e^(−mr)/(4πr). Eq. (3.73):
  U(r) = −λ²e^(−mr)/(4πr); "The force has a range 1/m ... The minus sign tells us that the potential is
  attractive." "Forces arising due to the exchange of scalars: they are universally attractive ...
  different from forces due to the exchange of a spin 1 particle ... where the sign flips when we change
  the charge. However, for forces due to the exchange of a spin 2 particle -- i.e. gravity -- the force is
  again universally attractive."
- **Bearing.** A mass gap gives an exponential range; like-sources attraction needs even spin. These are
  the two textbook halves of the hypothesis.

**[HKMS03] D. A. Huse, W. Krauth, R. Moessner and S. L. Sondhi, "Coulomb and liquid dimer models in three
dimensions," arXiv:cond-mat/0305318.** Read in full (4 pp.).
- This is a classical model with hard constraints only (close-packed dimers). On the *bipartite* cubic
  lattice the constraint makes a divergence-free "magnetic field", Eq. (1), whose coarse-grained
  distribution is Gaussian, Eq. (3), giving "a critical Coulomb phase with algebraic, dipolar,
  correlations", Eq. (4), confirmed by Monte Carlo up to L = 128. "Gauge invariance explictly [sic]
  forbids any relevant operators at the fixed point ... this is the standard explanation of the
  masslessness of the photon."
- "A monomer is a monopole with charge ±1 depending on the sublattice." Two monomers feel "an attractive
  entropic force with the same form as that between oppositely charged monopoles, i.e. an inverse squared
  force."
- The non-bipartite FCC lattice gives only exponential correlations (ξ = 0.35 spacings). The authors
  "conjecture that extended critical phases are realized only on bipartite lattices".
- **Bearing (ours, unverified).** This is the closest classical, entropic, discrete relative of what the
  project wants: a hard constraint, no energy at all, and a long-range 1/r entropic potential between
  defects in three dimensions. Two lessons: the gapless mode is *protected by the constraint*, not tuned;
  and the defects carry a sublattice sign, so same-sign defects repel. This model's graphs are bipartite
  too, but they have no fixed lattice on which to define a flux.

**[Henley10] C. L. Henley, "The 'Coulomb phase' in frustrated systems," arXiv:0912.4531.** Read: Sec. 1,
Sec. 3 opening, Sec. 4 (pseudo-charge defects) and Sec. 4.2; the other sections skimmed.
- Sec. 1.1: the conditions for a Coulomb phase are (C1) each variable maps to a signed flux on a bond;
  (C2) hard constraints set the flux into each vertex to zero; (C3) "the system is in a highly disordered
  phase, without any long range ordered pattern". The correlations then go as 1/R^d.
- Sec. 4, Eq. (4.1): defects violating the constraint are Gauss-law charges with
  F_int/T = K Q₁Q₂ / (4π|r₁ − r₂|) in d = 3, so like charges repel. "The effective Coulomb interaction is
  wholly entropic." In an ordered phase the same defects are confined instead: "pulling apart the defects
  now carries a cost proportional to |r₁ − r₂|."
- Sec. 4.2: in d = 2 the entropic interaction is ln R.
- **Bearing (ours, unverified).** Condition (C3) is exactly what the flat space here lacks, since it is
  frozen. Eq. (4.1) shows that a vector-type gapless mode, which is what constraints on bonds usually
  give, makes identical relics *repel*.

### Topic 2. Emergent gapless modes, gravitons, and pulls in discrete models

**[GW06] Z.-C. Gu and X.-G. Wen, "A lattice bosonic model as a quantum theory of gravity,"
arXiv:gr-qc/0606100.** Read in full (4 pp.).
- Working definition of quantum gravity, conditions (a) to (f): among them "(d) The gapless helicity ±2
  excitations are the only low energy excitations" and "(e) ... linear dispersion". A local rotor model
  whose low-energy theory is linearized Einstein gravity, Eq. (7). The unwanted helicity 0 and ±1 modes
  are gapped by strong fluctuations of compact variables. "The compactification and the discretization of
  metric tensor are crucial." Gaplessness is "protected by the quantum order". Weinberg–Witten is evaded
  because the energy-momentum tensor "is not invariant under the linearized diffeomorphism".
- **Bearing.** The existence proof that a local lattice model can carry a *protected* massless spin-2
  mode. It is quantum, on a fixed cubic lattice, and does not show matter attracting.

**[GW09] Z.-C. Gu and X.-G. Wen, "Emergence of helicity ±2 modes (gravitons) from qubit models,"
arXiv:0907.1203 (Nucl. Phys. B 863, 90, as cited by [Pretko17]; not checked).** Read: abstract and
Sec. IV (the constrained tensor models, pp. 11 to 12).
- "In Einstein gravity, the mass of matter is coupled to gravity via a modification of constraint ...
  similar to the way that electric or magnetic charges generate electromagnetism." With only the vector
  constraint ∂ᵢE^ij = 0 "the model ... has no scaler [sic] charges ... that may correspond to mass". Adding
  a scalar constraint, "the scaler constraint can be violated at an isolated point, which corresponds to a
  mass."
- With the Hamiltonian of Eq. (32), J[(E^ij)² − ½(E^ii)²] + g a_ij R^ij, "the force between the two masses
  is proportional to m1m2/r2. Two masses with the same sign attract and two masses with opposite signs
  repel." The constraints are put on a lattice by energy penalties (Sec. V).
- **Bearing (ours, unverified).** This is the clearest published recipe for a Newtonian pull in a lattice
  model: mass is the violation of a *scalar* Gauss-law constraint of a symmetric-tensor field, and the
  quadratic form has the Einstein sign on the trace part. Note that this quadratic form is not positive
  definite. A classical Monte Carlo with that weight would meet the Euclidean conformal-mode problem,
  which [dBS97] names in its Sec. 1.

**[LW05] M. Levin and X.-G. Wen, "Colloquium: Photons and electrons as emergent phenomena,"
arXiv:cond-mat/0407140 (Rev. Mod. Phys. 77, 871 (2005), as cited in [GW06]).** Read: Secs. I, IV, VII
(simple examples) and the conclusion; the rest skimmed.
- Gapless "artificial photons" emerge from string-net condensation on a pyrochlore spin model, Eqs. (5)
  and (6), and on a cubic-lattice model, Eq. (7), in a phase that "cannot be described by Landau's
  symmetry breaking theory". On gravity: "so far we do not know how to explain ... chiral fermions and
  gravity".
- **Bearing.** Gapless gauge modes can be protected by a pattern of constraints rather than a broken
  symmetry. The graviton case was open in 2005, and [GW06] and [GW09] are the follow-ups.

**[Pretko17] M. Pretko, "Emergent gravity of fractons: Mach's principle revisited,"
arXiv:1702.07613.** Read: abstract, Secs. I, III.C (the two-body problem) and Appendix B.
- Abstract: fractons coupled to an emergent graviton attract. "This interaction between fractons is
  always attractive ... This force will generically be short-ranged, but ... the power-law behavior of
  Newtonian gravity can arise under certain conditions."
- Sec. III.C.1, Eq. (24): with gapped mediating dipoles, t(r) = b e^(−Mr) and "the orbital speed decays to
  zero exponentially as the fractons separate." Sec. III.C.2, Eqs. (26) and (27): with *gapless* dipoles,
  heuristically t ∝ 1/r, giving Keplerian orbits; the author says this "should be backed up by more
  concrete calculations." Sec. III.C.3: "The gauge potential between like charges is repulsive on very
  general grounds."
- Appendix B: Weinberg–Witten is evaded because fractons have no asymptotic momentum eigenstates.
- **Bearing.** A published model states the hypothesis almost verbatim: a gapped mediator gives an
  exponentially short attraction and a gapless one a power law. In it the attraction comes from mobility,
  a quantum effect, and the gauge potential between like charges repels.

**[PVCPN18] A. Prem, S. Vijay, Y.-Z. Chou, M. Pretko and R. M. Nandkishore, "Pinch point singularities of
tensor spin liquids," arXiv:1806.04148 (Phys. Rev. B 98, 165140 (2018), per search listing).** Read:
abstract, Sec. I, the classical-correlator passages of Sec. II (Eqs. 15 to 20).
- Classical tensor Coulomb phases exist: impose only the "spin-ice" constraint ∂ᵢ∂ⱼE^ij = 0, weight all
  constrained states equally (F ≈ −TS), and the central limit theorem gives a Gaussian F/T = K∫E^ij E^ij,
  Eq. (19), with tensor pinch-point correlations, Eq. (20).
- **Bearing (ours, unverified).** A classical, entropy-only, symmetric-tensor analog of [HKMS03] exists.
  It is the closest classical cousin of the [GW09] recipe. Its weight is positive definite, so by
  [Pretko17] Sec. III.C.3 its like charges should repel, not attract.

**[KMS08] T. Konopka, F. Markopoulou and S. Severini, "Quantum graphity: a model of emergent locality,"
arXiv:0801.0861 (Phys. Rev. D 77, 104029).** Read: abstract, Sec. on low-energy excitations (around
Figs. 4 and 5) and the conclusion.
- A reference lattice is checked to be a local minimum. Defects made by exchange moves cost finite energy
  (Figs. 4 and 5). Gapless U(1) modes appear only through Levin–Wen string-net condensation of extra
  variables. On gravity: "It has been conjectured elsewhere that such a relation can give rise to the
  Einstein equations ... It will be interesting to investigate this possibility."
- **Bearing.** In the parked Konopka model, too, the ordered graph's own excitations are gapped, and a
  gapless mode had to be added through extra degrees of freedom. No attraction is derived.

**[HMLCSM10] A. Hamma, F. Markopoulou, S. Lloyd, F. Caravelli, S. Severini and K. Markström, "A quantum
Bose-Hubbard model with evolving graph as toy model for emergent spacetime," arXiv:0911.5075.** Read:
abstract, Sec. II.B and II.C, Sec. V.
- Matter (bosons) hops on a graph whose edges are themselves quantum and are made and destroyed by
  matter: "the edges (space) tell the bosons (matter) where to go, and the bosons, by creating edges, tell
  the space how to curve." The "toy mechanism for attraction" (Sec. V) is a *trapped surface*: a highly
  connected region from which light escapes with probability of order 1/n_B (Eq. 20). It is not a force
  law that falls with distance.
- **Bearing (ours, unverified).** This is the published "matter trapped in regions of the graph" result.
  It is trapping by connectivity, the same kind of thing as the project's re-curled black hole, and says
  nothing about 1/r.

**[KL18] N. Klitgaard and R. Loll, "Introducing quantum Ricci curvature," arXiv:1712.08847 (Phys. Rev. D
97, 046008, as cited in [vdDL24]).** Read: abstract, Sec. 1, conclusions.
- A curvature built from the average distance between two spheres of radius δ, designed to be computed
  at scales much larger than the lattice spacing, because short-scale values carry "discretization
  artefacts". They contrast this with "recent implementations of Ricci curvature à la Ollivier in
  attempts to construct a theory of quantum gravity from specific statistical ensembles of random graphs
  or networks [19]", where [19] is Trugenberger.
- **Bearing (ours, unverified).** The project's per-edge Ollivier curvature is the δ = 1 case, which this
  group treats as dominated by artifacts. A correlator of it measures the lattice, not the continuum.

**[vdDL24] J. van der Duin and R. Loll, "Curvature correlators in nonperturbative 2D Lorentzian quantum
gravity," arXiv:2404.17556 (Eur. Phys. J. C 84, 759 (2024), per search listing).** Read: abstract,
Secs. 1, 2, 5.3, 6.
- Sec. 5.3, Eq. (37): the connected correlator of the quantum Ricci curvature in 2D causal dynamical
  triangulations "vanishes identically within measuring accuracy" beyond about r = 2δ (δ is the sphere
  radius of the curvature; the inequality sign is garbled in the extracted text). The "naïve"
  subtraction of Eq. (38), right for a fixed background, "does not produce a vanishing result", because on
  a dynamical geometry the disconnected part depends on r. The reason given: "because of the absence of
  local propagating degrees of freedom in two-dimensional quantum gravity, we do not expect to find
  correlations between any local geometric operators."
- **Bearing (ours, unverified).** This is the closest published relative of O31. There, too, curvature
  correlations fall to zero within about two coarse-graining lengths, and it is read as *what 2D gravity
  should do*, not as a gap. The 2D result of O31 is therefore uninformative about gravitons either way.
  The subtraction in Eq. (37) is also the one to check against [KTB19] Eq. (4.15) as the project uses it.

**[dBS95] B. V. de Bakker and J. Smit, "Correlations and binding in 4D dynamical triangulation,"
arXiv:hep-lat/9510041 (conference proceedings).** Read in full (3 pp.).
- Abstract: "In the elongated phase, curvature correlations appear to fall off like a fractional power.
  Near the transition to the crumpled phase this power is consistent with 4." Eq. (2): a disconnected part
  that depends on the geometry "is needed for the correlation to vanish at larger distances."
- **Bearing.** In a random-geometry model *with* an extended 4D regime, curvature correlations are power
  laws near the transition. That is a contrast with O31's two-step decay, but in a different dimension and
  at a phase boundary.

**[dBS97] B. V. de Bakker and J. Smit, "Gravitational binding in 4D dynamical triangulation,"
arXiv:hep-lat/9604023 (Nucl. Phys. B 484, 476 (1997), per search listing).** Read: Secs. 1, 2 and 6 in
part, Sec. 7.
- Two scalar *test* particles (quenched) propagating on the triangulations "have a positive binding energy,
  thereby showing that the model can represent gravitational attraction." Method, Sec. 2: compare the
  two-particle propagator, Eq. (13), with the square of the one-particle one, Eq. (12); binding means
  M < 2m. Sec. 1 explains why not a potential at fixed distance: "it is difficult in a fluctuating
  spacetime to keep two heavy test masses at a fixed distance." Binding energies, table (43): 0.02 to 0.08
  in lattice units at κ₂ = 1.255 and 1.259.
- **Bearing (ours, unverified).** This is the method to borrow if relics cannot be pinned. The attraction
  here comes from two probes sharing one fluctuating geometry. The average of G² is not the square of
  the average of G, so shared fluctuations bind. The binding was seen only near the transition, where the
  geometry is four-dimensional over a range of scales.

**[DLSU21] M. Dai, J. Laiho, M. Schiffer and J. Unmuth-Yockey, "Newtonian binding from lattice quantum
gravity," arXiv:2102.04492 (Phys. Rev. D 103, 114511 (2021), per search listing).** Read: abstract,
Secs. I, III.B to C, IV in part, V.
- [dBS97]'s method with a tuned measure term and extrapolation to the continuum and infinite volume. The
  binding energy fits E_b ∝ m^α with α = 4.6(9) against the Newtonian 5 of Eq. (22), E₁ = G²m⁵/4. "At
  coarser coupling and smaller lattice volumes where the effective dimension of the lattice geometries is
  around 3 or lower, the value of α drops to 2 or lower." Sec. V: "we still expect the binding of
  particles to be governed by tree-level graviton exchange."
- **Bearing.** This is the only published Newtonian-limit result from a discrete random-geometry model
  that we found. It needed a geometry that is effectively 4D at large scale, a tuning, and a continuum
  extrapolation. The *mass dependence* of a binding energy (the exponent α) is a sharper test of "Newton's
  shape" than a potential of mean force at a few separations.

**[LR25] J. Laiho and K. Ratliff, "Euclidean correlation functions in quantum gravity,"
arXiv:2510.11888 (October 2025).** Read: abstract, Sec. 1, Secs. 4.4 and 4.5 results, Sec. 5.
- Sec. 1: "The leading order curvature correlator (about flat space) ... [is] (derivatives of) a delta
  function in the source-sink separation, thus vanishing everywhere away from the origin." The first
  non-zero part is one loop: Eq. (218), ∝ G²/x_E⁸, positive. The volume correlator has a *tree-level*
  term ∝ G/x_E² plus one loop ∝ G²/x_E⁴, Eq. (223). Positivity is "a consequence of reflection positivity".
- **Bearing (ours, unverified).** The decisive reference against reading O31 as evidence of a gap. Even
  with a massless graviton, curvature correlations at separation are tiny, high-power effects, while
  volume correlations carry the massless mode directly. A graph model's gaplessness test should use ball
  volumes or distances.

**[Marolf14] D. Marolf, "Emergent gravity requires (kinematic) non-locality," arXiv:1409.2509.** Read in
full (6 pp.).
- Definition I: gravity with universal coupling to energy writes the total energy as a boundary flux, the
  gravitational Gauss law. Theorem: if a kinematically local theory is described in some limit by such
  gravity, "all local observables away from the boundary become independent of time." But "our arguments
  place no constraints on the emergence of strictly linear spin-2 degrees of freedom."
- **Bearing (ours, unverified).** (i) A directional, linear pull is not forbidden. (ii) The Gauss-law
  property that gravity needs is exactly what this model's energy lacks. H is a sum of per-edge terms,
  and O22's additivity is the absence of any flux around a relic. (iii) This model's kinematics is not
  local in Marolf's sense, because edges may join any two points, so the theorem may not apply at all.

**[Carlip12] S. Carlip, "Challenges for emergent gravity," arXiv:1207.2504.** Read: Secs. 1, 3.2, 3.3,
3.7; the others skimmed.
- Sec. 3.2: universality of free fall requires that "1. only one massless spin two field is relevant;
  2. this field couples with equal strength to all matter; 3. any spin zero or spin one components of the
  interaction are absent or strongly suppressed."
- Sec. 3.7: the Weinberg–Witten theorem, "no massless spin two field can carry a nonzero charge under the
  operator P^μ", for a Lorentz-covariant conserved T^μν. Loopholes include "4. Emergent spacetime: in
  type II models ... the basic setting of the Weinberg-Witten theorem ... is absent at the fundamental
  level." Graph models such as quantum graphity are his type II examples (Sec. 1).
- **Bearing.** Weinberg–Witten does not constrain this project, which is type II. The universality
  conditions give the "universality" leg of the gravity brief its published form. A Nordström-type scalar
  pull would pass "directional" and fail item 3 as real gravity.

### Topic 3. Entropic gravity: what an entropic pull needs

**[Verlinde11] E. Verlinde, "On the origin of gravity and the laws of Newton," arXiv:1001.0785.** Read:
Secs. 1 to 3.
- Sec. 3: the entropic force F Δx = T ΔS, Eq. (3.7), with the postulate ΔS = 2πk_B (mc/ħ) Δx,
  Eq. (3.6). Newton's law, Eq. (3.13), follows from Sec. 3.3's ingredients: "(i) there is a change of
  entropy in the emergent direction (ii) the number of degrees of freedom are proportional to the area of
  the screen, and (iii) the energy is evenly distributed over these degrees of freedom", with E = Mc²
  fixing the screen's temperature. He calls equipartition "perhaps the least obvious assumption".
- **Bearing (ours, unverified).** Stripped down, the ingredients are a Gauss law. Every closed surface
  around a mass carries a temperature set by the enclosed energy, and the number of states falls as the
  test body approaches. Neither holds in this model at a fixed coupling g: the temperature is uniform, and
  O32 found no separation-dependence in the counting at fixed wiring.

**[Kobakhidze11] A. Kobakhidze, "Gravity is not an entropic force," arXiv:1009.5414.** Read in full.
- Abstract: "experiments with ultra-cold neutrons in the gravitational field of Earth disprove recent
  speculations on the entropic origin of gravitation." An entropic translation turns pure states mixed,
  Eqs. (10) to (16), which is inconsistent with the observed quantum bound states. Footnote 1: the force
  "is not strictly conservative".
- **Bearing.** It concerns quantum coherence, which a classical Monte Carlo lacks, so it does not block a
  classical entropic pull in the model. It is a warning for any later claim about real gravity.

**[Visser11] M. Visser, "Conservative entropic forces," arXiv:1108.5240.** Read: abstract, Secs. 1,
2.1 and 2.2.
- Sec. 2: entropic forces are real (polymer elasticity, osmotic and depletion forces). For an entropic
  force to reproduce a conservative potential Φ, Eq. (2.3), −∇Φ = T∇S forces the level sets of Φ, S and T
  to coincide, Eq. (2.4), which strains Verlinde's scenario.
- **Bearing (ours, unverified).** At the fixed coupling used here, the potential of mean force
  F(r) = −g ln P(r) is automatically conservative, so Visser's constraint is met trivially. The physics is
  entirely in whether the entropy of the surroundings changes with separation as 1/r. By Topic 1 that
  needs long-range correlations in the surroundings.

### Topic 4. Nordström's scalar gravity and the Weinberg–Witten theorem

**[Giulini08] D. Giulini, "What is (not) wrong with scalar gravity?," arXiv:gr-qc/0611100.** Read:
Secs. 1 to 3, Sec. 7.
- Eq. (4): □φ = −κT with κ = 4πG/c², the source being the trace of the energy-momentum tensor. Eqs. (11)
  to (16): a consistent action with universal coupling. The theory is attractive with the Newtonian limit,
  predicts "no deflection of light", and gives "−1/6 times the right perihelion advance of Mercury".
- Sec. 7 on vector gravity: "the energy of the radiation field is negative definite due to a sign change in
  Maxwell's equations which is necessary to make like charges (i.e. masses) attract rather than repel each
  other."
- **Bearing.** The simplest attractive 1/r theory is a massless scalar sourced by energy (the trace)
  without derivatives. A positive-energy vector field cannot make like masses attract. That is the sign
  rule behind correction 3.

**Weinberg–Witten (1980): original not read.** Statement and loopholes taken from [Carlip12] Sec. 3.7;
evasions in [GW06] (last section) and [Pretko17] Appendix B. The original is Phys. Lett. B 96, 59 (1980),
as cited in [GW06]; to verify before citing directly.

### Topic 5. Trugenberger, 2024 to 2026

An arXiv author search on 2026-09-25 lists, for 2024 to 2026, two papers on this model:
arXiv:2409.09385 ([T24]) and arXiv:2512.17676 ([T25], v2 29 April 2026). The rest are condensed-matter
papers: superconductivity, Bose metals, BKT transitions, and confining strings with Diamantini, Quevedo
and Zapata (2605.13791). The nearest earlier one is arXiv:2311.17526 (November 2023, "Combinatorial
quantum gravity and emergent 3D quantum behaviour"); its text contains none of the words Newton,
attraction, gapless, massless or graviton.

**[T24] C. A. Trugenberger, "Dark matter and dark energy in combinatorial quantum gravity,"
arXiv:2409.09385v1.** Read: the dark-energy section, searched for the terms above.
- Matter is surviving random-phase domains. "Of course, one should also take into account the
  gravitational attraction between the residual matter. The details of such a computation are beyond the
  scope of the present paper".
- **Bearing.** No Newtonian limit is derived, and the attraction is assumed.

**[T25] C. A. Trugenberger, "Networks as the fundamental constituents of the universe,"
arXiv:2512.17676v2.** Read: "Cycle condensation" (text around Eq. 23) and "Emergent large-scale physics:
Matter, the Einstein equations and the cosmological constant" (Eqs. 45 to 47), plus the black-hole
section.
- Around Eq. (23): the correlation length ξ(g) "vanishes in the ground state at g = 0 and diverges at the
  critical point", and "Both phases are governed by a finite correlation length, which sets the scale for
  fluctuations and keeps the variance finite." ξ is identified with the Planck length.
- Eqs. (45) to (47): Einstein's equations are obtained as "macroscopic constitutive equations of matter",
  by a uniqueness argument for a symmetric, covariantly conserved tensor whose trace is Eq. (46). "As a
  consequence, matter particles will tend to aggregate under the influence of gravitational forces."
- The text contains no computation of a force between bubbles from the network, no Newtonian limit and
  no gapless mode.
- **Bearing (ours, unverified).** The model's author states the premise of the hypothesis himself: the
  geometric phase has a finite correlation length. Gravity at large scales is introduced by identifying
  free-energy density with curvature, not by a mediating mode. So no published source claims a
  microscopic pull between relics in this model, and by [KG99] Sec. II.B any such pull in the geometric
  phase would be confined to the scale ξ(g).

### Seen but not read (do not rely on these)
Castelnovo, Moessner and Sondhi, arXiv:0710.5515 (spin-ice monopoles); Caravelli and Markopoulou,
arXiv:1008.1340; Deruelle, arXiv:1104.4608 (Nordström). All were downloaded and none was read.

---

## (c) Design requirements for a graph model that could give a pull

Each requirement has a source and a test that can be run in this code base. The mapping to the graph
model is *ours, unverified*.

1. **The geometric phase must fluctuate at the working coupling.** A frozen crystal has no fluctuations
   to carry a force, and in an ordered phase defects are confined rather than long-range coupled
   ([Henley10] (C3) and Sec. 4). *Test:* acceptance rate and the connected correlation of a local
   observable in flat space, at the couplings of the gravity brief (λ ≈ 1.02).
2. **The long-range correlations must be protected, not tuned.** The robust classical route is a hard
   local constraint with an extensive, liquid-like ground-state set ([Henley10] C1 and C2; [HKMS03]: the
   masslessness is "forbid[den]" from being lost by gauge invariance). The quantum route is a gauge
   structure ([LW05], [GW06]). A critical point also gives power laws ([KG99] Eq. 2) but only at one
   coupling, and O31 found no growth with N. *Test:* at fixed g inside the geometric phase, a correlation
   length that grows with system size.
3. **Measure the metric, not the curvature.** Curvature correlators vanish away from contact at tree level
   even with a graviton ([LR25] Eq. 218) and vanish identically in 2D ([vdDL24] Sec. 5.3); volume
   correlators carry the massless mode ([LR25] Eq. 223). Per-edge Ollivier curvature is at the artifact
   scale ([KL18]). Subtract the r-dependent disconnected part ([vdDL24] Eq. 37; [dBS95] Eq. 2). *Test:*
   the ball volume N(r) around one relic minus the flat value, and the connected ball-volume correlator in
   flat space, both against r, fitted to an exponential and to a power law.
4. **A relic must carry a Gauss-law charge tied to its energy.** Mass as a violated scalar constraint
   ([GW09] Sec. IV); defects as charges measurable on every enclosing surface ([Henley10] Sec. 4);
   universal coupling as a boundary flux ([Marolf14] Def. I). Without this, a gapless medium gives only
   Casimir-type laws such as 1/R⁴ ([GBP93]; [KG99] Eqs. 17 and 18). *Test:* the chosen flux or field
   summed over the sphere of radius r around one relic should be independent of r.
5. **The mode must be even-spin with the attractive sign.** Vector-type (Coulomb) modes make like relics
   repel ([Henley10] Eq. 4.1; [Tong-QFT] Sec. 3.5.2; [Giulini08] Sec. 7). A massless scalar with linear,
   trace-type coupling ([Giulini08] Eq. 4) or the Einstein-signed tensor form ([GW09] Eq. 32) attract.
   In a *classical* Boltzmann weight the Einstein sign on the trace is not positive definite ([dBS97]
   Sec. 1 on the conformal mode); *ours, unverified:* the classical route to like attraction is therefore
   the scalar one. *Test:* the potential of mean force between two identical relics beside a
   relic–antirelic pair, if the model has one; opposite signs mean vector type.
6. **At least three large directions.** The same Gauss law gives ln R in 2D ([Henley10] Sec. 4.2). The
   gravity brief already fixes D = 3.
7. **Universality.** One massless mode, equal coupling to all relics, and spin-0 and spin-1 parts absent
   for real gravity ([Carlip12] Sec. 3.2). The force goes as m₁m₂ ([GW09] Sec. IV). *Test:* two relic
   types of equal energy and different wiring give the same F(r), and F scales with E₁E₂ (already in the
   gravity brief).
8. **Measure binding as well as, or instead of, a pinned potential.** Holding two defects at fixed
   separation in a fluctuating geometry is hard ([dBS97] Sec. 1). The mass dependence of a binding energy
   tests Newton's shape directly ([DLSU21] Eq. 22: E_b ∝ m⁵ in 4D; they found α = 4.6(9)). *Test:* if
   relics of several energies exist, the exponent of their binding free energy against E.
9. **Expectations if only a Casimir route is available.** If the medium becomes gapless but relics only
   change its local stiffness, identical relics should attract ([KK06]), with a fast power law ([GBP93]).
   That would pass "sign" in the gravity brief and fail "shape" (p = 1). Say so before the run.

---

## (d) Who to ask, one narrow question each (recent work checked on arXiv, 2026-09-25)

1. **Israel Klich, Department of Physics, University of Virginia.** Co-author of [KK06]. Recent work,
   checked: "Thermalization packets and optimal ice cubes" (with M. Vucelja, arXiv:2608.25141, August
   2026; UVA affiliation on the paper); "On factorizing aggregate counting distributions into independent
   latent processes" (arXiv:2607.14409, July 2026); interferometric neuromorphic learning with Pfister and
   Schwarz (arXiv:2601.18047, 2606.09756, 2026). None of them revisits fluctuation forces, so the question
   is not answered in his recent record.
   *Question:* "Your 2006 theorem with Kenneth shows that mirror-image bodies coupled to a fluctuating
   field always attract. Does the reflection-positivity argument carry over to a classical ensemble of
   discrete configurations with hard local constraints, so that the potential of mean force between two
   mirror-image defects is monotonically attractive, and is its range then bounded by the ensemble's
   correlation length?"

2. **Gia-Wei Chern, Department of Physics, University of Virginia.** Classical frustrated systems and
   constraint-induced criticality. Recent work, checked: "Phase-space networks and connectivity of the
   kagome antiferromagnet" (arXiv:2601.05933, v2 August 2026, UVA affiliation); its introduction describes
   "criticality [that] can emerge purely from local constraints rather than fine-tuned interactions".
   Also "Energy landscape of the kagome antiferromagnet" (arXiv:2604.18902, 2026) and graph neural
   networks in the Wilson-loop representation of Abelian lattice gauge theories (arXiv:2605.03901, 2026).
   *Question:* "In constraint-induced Coulomb phases, defects of like sign repel (Henley's Eq. 4.1). Is
   there a known classical constrained model in which *identical* defects attract with a power law, for
   example through a scalar or tensor-type emergent field, or is like-sign repulsion generic whenever the
   long-range entropic interaction comes from a local constraint?"

3. **Jack Laiho, Department of Physics, Syracuse University** (not UVA). Author of [DLSU21] and [LR25].
   Recent work, checked: [LR25] (arXiv:2510.11888, October 2025).
   *Question:* "Your 2025 paper finds the curvature correlator only at one loop (∝ G²/x⁸) but a tree-level
   ∝ G/x² term in the volume correlator. For a lattice geometry model with no separate matter field, would
   you treat the connected ball-volume correlator, with the geometry-dependent subtraction, as the right
   first test for a massless mode? And is a two-defect binding energy with defects that are free to move
   a sound way to see an attraction, following de Bakker and Smit?"
   *Alternate:* Renate Loll (Radboud), for the coarse-graining scale δ and the connected-correlator
   subtraction ([vdDL24]; recent work arXiv:2510.05695 and 2604.05641, checked by title only).
