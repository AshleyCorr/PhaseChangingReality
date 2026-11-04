# Reading note: space from entanglement (Cao, Carroll and Michalakis 2016; Cao and Carroll 2017), with Quantum Mereology and the locally finite Hilbert space read in part

Date: 2026-11-02. By an assistant agent (Claude Fable 5.1). The owner has not read these passages. Every quotation
is from the arXiv text (1606.08444v3, 15 December 2016; 1712.02803v1, 7 December 2017; 2005.12938v3, 3 February 2021, the version
served on 2026-11-02; 1704.00066v2, 26 May 2017), extracted from the PDFs with PyMuPDF. The published versions
(Phys. Rev. D 95, 024031 (2017); Phys. Rev. D 97, 086003 (2018); Phys. Rev. A 103, 022213 (2021), all three
journal references match the arXiv listing, checked 2026-11-02, and are still to be checked against the publisher) were not read and may differ in wording. Equation and
section numbers are the arXiv PDFs'. The figures were not seen; what is said about them is what the text says.
Everything that is not a quotation or a cited equation is ours, unverified, unless marked otherwise.

Why this was read: tier 1, items 2, 3 and 4, and tier 2, item 12, of `docs/reading/notes/2026-11-02_carroll_inventory.md`:
a graph-with-weighted-links picture of space in which a dimension is read off the wiring and a curvature responds to a
local change, which is the shape of the toy's dimension census and of piece 8 (a carrier for a pull). This note extracts
the definitions and then asks, in a section marked ours, what feeding the toy's saved wirings to their reader would
compare.

Rule 13 applies throughout: these papers are targets, not confirmations. Agreement between their estimator and the toy's
on a saved state would be a consistency between two definitions, not a result about reality.

## 1. The 2016 paper [CCM16]: definitions, in their order

### 1.1 The strategy, and the stated limit

The bullet list on p. 4 is the whole method in six lines: factor the Hilbert space, H = tensor product over p of H_p,
each factor finite-dimensional; take states that are "redundancy-constrained"; use the mutual information
I(A : B) = S(A) + S(B) - S(AB) "to define a metric on the graph connecting the factors H_p"; reconstruct flat geometries
from that graph metric; perturb the state and show the perturbation "produce[s] local curvature proportional to the
local change in entropy"; relate that to the entanglement first law and get "a geometry/energy relation reminiscent of
Einstein's equation".

Then the limit, in the next sentence:

> "We do not assume any particular Hamiltonian for the quantum dynamics of our state, nor do we explore the emergence
> of Lorentz invariance or other features necessary to claim we truly have an effective quantum theory of gravity,
> leaving that for future work." (Sec. I)

And again at Sec. II.B: "In this paper we don't assume any particular Hamiltonian, but begin by looking at states and
decompositions that give us a generalization of area-law behavior for entropy." The factorization is taken as given
(Sec. VI: "An important step in our procedure was assuming that we were given a preferred Hilbert-space decomposition
... ultimately we would like to be able to derive that decomposition rather than posit it"), which is what [CS20]
below is about.

Their D is the spatial dimension and d = D + 1 the spacetime dimension (Sec. I, last line). This note keeps their D.

### 1.2 The information graph and the cut function (Sec. II.B)

Divide the system into non-overlapping regions A_p; the system "naturally defines a weighted graph G = (V, E), on
which vertices V are the regions A_p, and the edges E between them are weighted by the mutual information". The
entropy of a union B of regions is then the sum over the cut:

> S(B) = (1/2) sum over p in B, q in the complement of B, of I(A_p : A_q).   (9)

The sentence that makes this a definition rather than an observation is the redundancy claim: "in the special case of
area-law entanglement, it suffices to specify all the mutual information between N factors", of order N^2 numbers
where a generic state needs of order 2^N (Sec. II.B).

### 1.3 Redundancy-constrained states (Sec. III.A)

> "We say that the state is redundancy-constrained (RC) if, for any subsystem B constructed as a tensor product of
> some subset of the {H_p}, its entanglement entropy is given by a cut function of the form (9), where A_p denotes
> the subsystem that lives in the Hilbert subspace H_p. Note that there is no geometric meaning associated with the
> Hilbert space at this point."

A consequence they draw at once (Eq. 11): the weighted degree of a vertex is bounded by its factor's size,
deg(A_p) = sum over q of I(A_p : A_q) <= 2 S_max(A_p) <= 2 ln D_p, with D_p = dim H_{A_p}. They say RC states "are
highly non-generic; they represent situations where entanglement is dominated by short-range effects", that a CFT
ground state with a holographic dual is not RC, and that "for the remainder of the program, we are going to focus on
simple RC states that correspond to flat space in D spatial dimensions". Appendix A says how to coarse-grain a
decomposition that is not RC into one that is (group factors into epsilon-balls of the graph metric and repeat until
the cut function holds), and admits that "there is no obvious way to choose the best coarse-graining scheme".

### 1.4 The mutual-information distance (Sec. III.B)

The ansatz, Eq. (13): for p != q with I(A_p : A_q) != 0,

> w(p, q) = l_RC Phi( I(A_p : A_q) / I_0 ),   and w(p, p) = 0,

"for some length scale l_RC, the 'redundancy-constraint scale.' No edges are drawn if I(A_p : A_q) = 0." The
normalization I_0 is "chosen such that I(A_p : A_q)/I_0 = 1 when two regions A_p, A_q are maximally entangled"; with
equal factor sizes, I_0 = 2 log(dim H_D). The conditions on Phi: "we require Phi(1) = 0 and lim x -> 0 Phi(x) =
infinity, namely, the distance is zero when two states are maximally entangled and far apart when they are
unentangled", Phi non-negative and monotonically decreasing on [0, 1]; "For definiteness it may be helpful to imagine
that Phi(x) = -log(x), as might be expected in the ground state of a gapped system". The two worked examples use
Phi(x) = -ln(x) (Sec. III.D).

The metric is then the shortest path, Eq. (14): d(p, q) = min over connected paths P of sum of w(p_n, p_{n+1}); they
note symmetry, d = 0 iff p = q, and the triangle inequality. Two further definitions in the same section:

- A fractal dimension from the entropy of a ball, Eq. (15): S(r, p) ~ r^{D_f}, with the caution "In general this
  expression may not converge to an integer D_f."
- The interface area between A_p and A_q, Eq. (16): A := I(A_p : A_q) / 2 alpha, "for some constant alpha", so that
  the area enclosing A_p is S(A_p)/alpha.

They also state the simplifying assumptions they make in passing from the information graph to the distance graph:
"the vertices and edges remain fixed, so that the graph is merely re-weighted, and furthermore that the distance weight
for any edge w(p, q) is determined solely by the corresponding mutual information ... rather than depending on the
rest of the graph" (Sec. III.B).

### 1.5 The multidimensional-scaling dimension estimator, and how it is read (Sec. III.C)

First the exact definition of dimension for a metric space. For a subset X = {v_0, ..., v_r} of vertices, Eq. (17)
R_ij = (1/2)(d(v_i, v_0)^2 + d(v_j, v_0)^2 - d(v_i, v_j)^2), and the simplicial volume Eq. (18)
vol_r(X) = (1/r!) sqrt(det R). "The dimension of a metric space, if it exists, is the largest natural number k for
which there exists a D-simplex with positive volume." (The sentence mixes k and D as printed.) Citing [59] (Morgan
1974): "the metric space can be isometrically embedded into Euclidean space with dimension d if and only if the metric
space is flat and has dimension <= D."

Then classical MDS. With N vertices and distances d(p, q), center the embedding (Eq. 22, sum over p of X_pr = 0) and
form the N x N matrix, Eq. (23),

> B_pq = -(1/2) [ d(p, q)^2 - (1/N) sum_l d(p, l)^2 - (1/N) sum_l d(l, q)^2 + (1/N^2) sum_{l,m} d(l, m)^2 ],

diagonalize B = V Lambda V^t (Eq. 24) with eigenvalues in descending order, and take the D eigenvectors of the D
non-zero eigenvalues, Eq. (25): X = (sqrt(lambda_1) v_1, ..., sqrt(lambda_d) v_d). For the inexact case:

> "there will be D dominant positive eigenvalues followed by smaller non-zero eigenvalues. We consider the
> D-dimensional embedding to be approximately valid if epsilon_D = 1 - sum_{i=1}^{D} |lambda_i| / sum_{i=1}^{N}
> |lambda_i| is sufficiently small."

and a stress function, Eq. (26), Stress = sqrt( 1 - (sum d(p,q) d_E(x_p, x_q))^2 / (sum d(p,q)^2 sum d_E(x_p,x_q)^2) ),
with d_E the Euclidean distance between embedded points. So the dimension is read off the eigenvalue spectrum of B:
count the dominant eigenvalues, and report epsilon_D. They note MDS "also works well for graph embedding in highly
symmetric surfaces (hyperbolic and spherical as well as flat)" but that embedding "generic curved manifolds" is "a
difficult problem", and that Regge calculus is the alternative (Appendix B, below).

### 1.6 The two examples (Sec. III.D), with their numbers

Both examples start "with a known vacuum state of a gapped local system". They say plainly that the mutual information
was not computed directly in the first: "we did not calculate directly the mutual information from density matrices in
the 1-d case and instead used the correlation function as a proxy", by Eq. (8), the squared connected correlator as a
lower bound on I.

- S = 1 antiferromagnetic Heisenberg chain, 100 vertices, "No coarse-graining is performed", edge weights the square of
  the correlator (a modified Bessel function K_0(r/xi)): "the distortion, as measured using eigenvalues, has
  epsilon_1 = 0.0167, from which we determine that it only slightly deviates from an embedding in 1d." Fig. 2 plots the
  eigenvalues and "a patch of approximately 50 points".
- 2-d toric code ground state, coarse-grained so that each region is homeomorphic to a plaquette, where the entropy is exact (their
  formula from [67]: S = n_1 + n_2 + n_3 - 1 in terms of the star operators with i neighbors in A); "The mutual
  information used for the network is again given by the length of overlapping boundary, up to a constant correction
  term." Result: "The distortion factor as measured by eigenvalues for an embedding in 2-d has epsilon_2 = 0.41, but
  the 2-d nature of the emergent space is evident from figure 3", with the distortion "visibly higher because
  -ln(x) is no longer an ideal ansatz for Phi(x)."

Two things to keep for the comparison below. (i) In the toric-code example the mutual information on each link is a
boundary length "up to a constant": on a uniform coarse-graining that is a constant per link, so their distance is a
constant times the graph distance, which is exactly the obvious choice for the toy. (ii) Both examples are open
patches, a line and "a patch of the 2-d toric code"; nothing in the paper applies MDS to a closed (periodic) graph.

### 1.7 Regge calculus in Appendix B

They say "the space is r-locally triangulable at p" if the vertices within a distance cutoff r form an abstract
simplicial complex that maps isometrically to a geometric one; "If K is also a simplicial manifold we can proceed to
define Regge curvature." The deficit angle at a codimension-2 hinge X is delta(X) = 2 pi - theta(X) with theta the sum
of the dihedral angles, Eq. (B1); the curvature is R_T = sum_i delta_i L_i, Eq. (B2), with L_i the hinge volumes, and
"For D = 2, the hinge is a point, and the curvature is given by the deficit angle, where we set L_i = 1." For a
perturbation that lengthens the edges at p, the deficit is linear in the length change at leading order, Eqs. (B3) to
(B5), Delta_p = N_p(D) delta l / l, with "N_p(D = 2) = 12/sqrt(3) in the case where all edges emanating from p vary by
the same amount delta l" on an equilateral triangulation; the coarse curvature is R_c = Delta_p L_p ~ N_p(D) l^{D-3}
delta l, Eq. (B6), and with the entanglement first law, Eq. (B7), R = Z delta<H>, with Z depending on "the
triangulation, strength of entanglement, and choice of coarse-graining".

This is the same object as the repository's `graphity.regge` (ASSUMPTIONS Q27: deficit angles on a triangulated toy
torus, one diagonal per square), built there for a different purpose (soft space, stage B). Their appendix is the
published statement that a deficit angle on a graph with lengths is a curvature; it does not compute one on any
example.

### 1.8 How an entanglement perturbation changes curvature (Sec. IV)

The opening caveat: "As we are only considering space rather than spacetime, we cannot directly make contact with
general relativity; in particular, we can say nothing about the emergence of dynamical fields obeying local Lorentz
invariance."

A local perturbation is a unitary on A_p and its complement; then, Eq. (27), delta I(A_p : complement) = 2 delta S_{A_p},
"since U does not change the total entropy of the system". A nonlocal perturbation entangles A_p with an ancilla B
(Eq. 28, H* = H_{A_p} x H_complement x H_B, a lossy channel); then delta I = delta S F'_Lambda(0), Eq. (31), with a
channel-dependent factor.

The curvature. Take a flat D-dimensional configuration M already obtained, a perturbation that decreases the entropy,
delta S_{A_p} < 0, Eq. (32), and Riemann normal coordinates at p, Eq. (33), following Jacobson [23]. At fixed volume the
area deficit is Eq. (34), delta A = -Omega_{D-1} r^{D+1} R_p / (2D(D+2)); the area is mutual information, Eq. (35),
delta A = (1/2 alpha) sum over neighbors of delta I ~ (1/2 alpha) delta I(A_p : complement); so, with r = gamma l_RC,

> R_p = - 2D(D+2) / (alpha Omega_{D-1} (gamma l_RC)^{D+1}) delta S_{A_p}.   (36)

"Because delta S_{A_p} < 0, the induced curvature is positive." The same by lengths: the radius excess at fixed area is
Eq. (38), delta d = r^3 R_p / 6D(D+2); the edge lengths change by Eq. (39), delta d(p, q) = -2 eta_q l_RC |Phi'| delta
S_A / I_0, with eta_q the share of the entropy change on edge q (Eq. 37, sum eta_q = 1, eta_q = 1/deg(p) for a
symmetric disentangling on a regular lattice); comparing gives Eq. (40), R_p = -12 eta_q |Phi'(i)| D(D+2) / (I_0 l_RC^2
gamma^3) delta S_{A_p}. "Both imply positive curvature for local disentangling perturbations. Similarly, entangling
perturbations with delta S_{A_p} > 0 yields negative spatial curvature." Nonlocal perturbations give Eqs. (41) and
(42) with the extra factor, and the sign result Eq. (43): "nonlocal entanglement perturbations are only able to
generate positive curvature perturbations", which they read as "a quantum proto-wormhole" (Sec. IV.B.2), with the
admission that "the entropy-curvature relation does not seem to be universal as in the local case."

### 1.9 The spatial analogue of Einstein's equation (Sec. V)

Sec. V opens with the second limit: "This is a long way from completely recovering the curved spacetime of general
relativity, both because we don't have a covariant theory with dynamics, and because we haven't related features of the
state to an effective stress-energy ... we leave the issue of dynamics to future work."

The chain: an emergence map E : H -> H_EFT(lambda_RC), Eq. (44), and an RG flow F_lambda, Eq. (45), both posited;
the posit Eq. (46), delta S = -kappa_lambda delta S_EFT(lambda), with the minus sign argued from a fixed-volume
equilibrium, Eq. (47), delta_V S_total = eta delta_V A + delta_V S_EFT = 0 ("The unperturbed state is taken to be an
equilibrium vacuum state"); the modular Hamiltonian, Eq. (49), sigma = e^{-K} / Tr(e^{-K}); relative entropy, Eq. (50);
the entanglement first law, Eq. (52), delta S = delta<K>; the effective modular energy density, Eq. (53),
epsilon_p = -delta<K_{A_p}>; and the result,

> R_p = -zeta delta S_{A_p} = zeta epsilon_p,   zeta = 2D(D+2) / (alpha Omega_{D-1} (gamma l_RC)^{D+1}).   (54), (55)

With a CFT fixed point so that K is the stress tensor, Eq. (57), this becomes Eq. (58), R = (4 pi kappa_{lambda*} / alpha)
delta<T_00^CFT>. Then, in their words, "Let us (somewhat optimistically) assume that the present framework can be adapted
to a situation with conventional time evolution, and furthermore that the dynamics are such that an approximate notion
of local Lorentz invariance holds. (Such an assumption is highly nontrivial ...)". For slices of vanishing extrinsic
curvature, R_p = 2 G_00(p), Eq. (60), so G_00 = (2 pi kappa_{lambda*} / alpha) delta<T_00^CFT>, Eq. (61), "nothing but the
00-component of the semiclassical Einstein equation" under 2 pi kappa_{lambda*} / alpha -> 8 pi G, and "If this reasoning is
approximately true for all time-like observers ... one can covariantize and arrive at the full equation", Eq. (62).
A by-product, Eq. (63): |Delta S_{A_p}| <= alpha A_p, "This resembles the holographic entropy bound."

Also recorded, because the toy has the opposite sign convention in one place: here a decrease of entanglement is
positive curvature and positive modular energy; "an entangling operation in the vicinity of A_p would give rise to a
negative 'energy' in the region, adding negative spatial curvature", which they call unphysical at large amounts.

### 1.10 What [CCM16] cites and does not

- Quantum graphity [KMS06] is cited once, as reference [30], in the sentence "There are also potential connections
  with a number of approaches to quantum gravity, including loop quantum gravity [29], quantum graphity [30],
  holographic space-time [31, 32], and random dynamics [33]; we do not investigate these directly here." (Sec. I).
- Ollivier curvature: not mentioned. The text of 1606.08444v3 was searched for "Ollivier"; no hit. Their curvature is
  Riemannian (Eq. 33) or Regge (Appendix B), never a graph curvature.
- "Spectral dimension" and "Hausdorff dimension": the first is not mentioned; "Hausdorff dimension" appears once, at the
  head of Appendix B ("For a space of fixed integral Hausdorff dimension D obtained from (15)"), naming the dimension D obtained from Eq. (15),
  which Sec. III.B sets to D = D_f + 1, one more than the entropy-of-a-ball exponent. No random-walk estimator is used.
- Garnerone, Giorda and Zanardi: not cited (searched for all three names; the only Zanardi in the reference list is
  [67], Hamma, Ionicioiu and Zanardi 2005, the toric-code entropy formula). See section 5.

## 2. The 2017 paper [CC17]: how dynamics enters, and what is assumed

### 2.1 The seven assumptions (Sec. II)

Quoted in full where short, because the whole derivation is "axiomatic" in their word:

- (A1) Factorization: "Hilbert space H comes equipped with a preferred tensor product decomposition into individual
  factors", Eq. (1); "The decomposition may be defined by the dynamics of the theory, as in [31]" ([31] is Cotler,
  Penington and Ranard, Locality from the Spectrum).
- (A2) Redundancy constraint: the cut function, Eq. (2), S(B) := (1/2) sum over i in B, j in complement of I(i : j),
  with the approximate form Eq. (3), S(B) = S_RC + S_sub.
- (A3) Area from mutual information, Eq. (4): A(B, complement) = (1/2 alpha) I(B : complement); later "we will later
  relate it to Newton's gravitational constant via alpha = 1/4 G_N" (Sec. III.A, after Eq. 10).
- (A4) Entanglement equilibrium, Eq. (5): under small perturbations "the total entropy perturbation delta S(R) of
  certain subsystems vanish", 0 = delta S_RC + delta S_sub, "a modified entanglement equilibrium condition (MEEC),
  following Jacobson [19]".
- (A5) Emergent field theory, Eq. (6): delta S_sub = delta S_EFT, with S_EFT "the vacuum-subtracted or Casini entropy".
- (A6) Dynamics: "There exists a consistent dynamical theory, e.g., a Hamiltonian or a quantum circuit, that generates
  a sequence of such configurations, each admitting an emergent spatial geometry. Furthermore, there is a way to
  organize these emergent geometries to create a consistent Lorentzian spacetime geometry via time evolution."
- (A7) Lorentz invariance: "The above assumptions hold for any constant-time slice of the emergent Minkowski space, and
  the overall theory is Lorentz-invariant in an appropriate limit."

Their own assessment, after the result: "While the number of conjectural assumptions needed to reach the result is
admittedly considerable, we find the path we've outlined to be a promising route" (Sec. IV.C). And on time, in the
Discussion: "There simply is a quantum state, represented by a vector in Hilbert space, evolving according to the
Schrödinger equation with some particular Hamiltonian. (For these purposes we take time as fundamental, but it is also
conceivable that time itself is emergent, arising through the entanglement of 'system' and 'clock' factors of Hilbert
space.)" So time is assumed, not derived, and Lorentz invariance is assumed in (A7) and then used once, to pass from the
00 component to the full equation.

### 2.2 From areas to a metric: the Radon transform (Sec. III)

The change from 2016: "Rather than deriving distances from mutual information, it is more natural to derive areas of
surfaces using the entanglement entropy across them" (Sec. I); the 2016 distance function is called "an ad hoc distance
function" (Sec. III.A). The procedure is two-part: "use MDS to emerge a symmetric 'background' geometry, and then
recover the metric perturbation from an inverse tensor Radon transform" (Sec. III.B). The scalar Radon transform is
Eq. (11), R[f](S) = integral over a totally geodesic codimension-1 surface S of f; the longitudinal tensor transform is
Eq. (12), R_par[s_ij] = integral of w^ij s_ij, with w the induced metric. The area change across a cut C is the mutual
information change, Eq. (17), delta I(C) = 2 alpha delta A(C); the area change is the transform of the metric
perturbation, Eq. (22), delta A = (1/2) R_par[delta w_ij]; hence Eq. (23), delta I(C) = alpha R_par[delta h_ij],
inverted over all cuts to recover delta h_ij, up to the gauge freedom Eq. (24), delta h_ij -> delta h_ij + d_i xi_j +
d_j xi_i. They state the mathematical status plainly: invertibility is known in two dimensions for simple
(boundary-rigid) manifolds and near flat or hyperbolic space in higher dimensions, "but an explicit inversion formula
and numerical results are still unknown, to the best of our knowledge. Henceforth we will simply assume that the
appropriate tensor-transform inversion can be performed." For discrete systems "we will also assume a continuum limit or
a smoothing process over the data delta S".

### 2.3 How dynamics enters: the Hamiltonian constraint, the modular Hamiltonian, the first law (Sec. IV)

Classical side first (Sec. IV.A): the Hamiltonian constraint G_mu nu t^mu t^nu = 8 pi G_N T_mu nu t^mu t^nu, Eq. (25),
linearized about Minkowski space on slices of vanishing extrinsic curvature, where R = 2 G_tt, so Eq. (26),
delta R = 16 pi G_N delta T_tt. Radon-transformed along hyperplanes, with the intertwining relation Eq. (29),
R[nabla^2 f] = d^2/dp^2 R[f], and R_par[delta h] = 2 delta A, this is Eq. (30), R[delta R] = -2 d^2 delta A / dp^2, and
Eq. (31), -d^2 delta A / dp^2 = 8 pi G_N R[delta T_tt]; solved with the Green function G(p, q) = (q - p) theta(q - p),
Eq. (32), giving Eq. (34), delta A = -8 pi G_N integral over the half space beyond the cut of (q - p) delta T_tt.
Footnote 6 fixes the two free constants of the Green function "by requiring the solution delta A matches to the
Rindler Hamiltonian of the bulk matter fields".

Quantum side (Sec. IV.B): the left of (34) is mutual information along the cut, delta I = 2 delta S(Sigma_C) for a
pure state, Eq. (35); the right is recognized as the Rindler modular Hamiltonian, Eq. (36), H_mod = 2 pi integral over
x > 0 of x T_tt, so delta T_tt = Tr[delta rho_EFT T_tt], Eq. (37), and by the entanglement first law, Eq. (38),
delta<H_mod> = delta S_EFT(C). Substituting:

> (1/2) delta I(Sigma_C : complement) + 4 G alpha delta S_EFT(C) = 0 (pure state),   (39)
> delta S(Sigma_C) + 4 G_N alpha delta S_EFT(C) = 0.   (40)

This is their MEEC with the dictionary Eqs. (42), (43): delta S(Sigma_C) is the UV, RC, geometric term; 4 G alpha
delta S_EFT is the IR, subleading, matter term. The differences from Jacobson [19] that they claim: the condition holds
"across all cuts C made by (n - 1)-dimensional totally-geodesic submanifolds in the background space" instead of
small balls, and "one no longer has to rely on CFT modular Hamiltonians", since the Rindler Hamiltonian exists for a
generic QFT. Identifying delta S_sub = delta S_EFT "fixes the constant alpha = 1/4 G_N", so Eq. (44),
0 = delta S_total = delta S_RC + delta S_sub = alpha delta A + delta S_EFT.

### 2.4 What they get (Sec. IV.C)

Under (A1) to (A5): the Radon transform of the linearized constraint, Eq. (46), R[delta R] = 16 pi G_N R[delta T_tt],
on every cut, hence Eq. (47), delta R = 16 pi G_N delta T_tt, "provided the inverse is well-defined". Under (A6): a
sequence of states |Psi(t)> "labeled by a single parameter t" in synchronous gauge, so Eq. (48),
delta G_mu nu t^mu t^nu = 8 pi G_N delta T_mu nu t^mu t^nu. Under (A7): "this must be valid for arbitrary normal t^mu",
so Eq. (49), delta G_mu nu = 8 pi G_N delta T_mu nu, the full linearized Einstein equation. Weak field only ("quantum
states corresponding to emergent spacetimes in the weak-field regime, small perturbations of Minkowski space",
Sec. I); the UV/IR split is then tied to quantum error correction (Sec. V, "tentative (even by the standards of the
rest of the paper)").

### 2.5 The paper's own list of what is missing (Sec. VI)

Five "biggest looming questions": a specific theory "for example by specifying an explicit Hamiltonian, but perhaps by
less direct means"; reconstructing more general geometries; "the hope that Lorentz symmetry can be emergent ... a clear
understanding of its feasibility is still lacking"; and this one, which is the nearest to the toy:

> "Given the recent interest in emergent gauge theories in condensed matter models, it may be possible to understand
> if certain condensed matter models can be 'gravitized' by emerging the geometry through entanglement of a state,
> instead of using the pre-existing geometry provided by the lattice structure or Hamiltonian. This may yield
> interesting toy models that exhibit (analogous) features of gravity."

[CC17] cites neither Ollivier nor Garnerone, Giorda and Zanardi (the text was searched; no hits for "Ollivier",
"Garnerone", "Zanardi", "spectral dimension" or "Regge").

## 3. [CS20], Quantum Mereology, read in part: how a factorization is chosen

Read: the abstract, Sec. I, Sec. II, the opening of Sec. III, Sec. V (the algorithm) and Sec. VIII (Discussion). Not
read: Secs. III.B to IV, VI, VII and the appendices.

The question is the one [CCM16] left as its first open step: given only "a Hilbert space of some dimensionality, the
Hamiltonian, and an initial state, what is the best way to factorize Hilbert space into subsystems? Since we are not
given a preferred factorization to begin with, there is no preferred basis other than the eigenstates of the
Hamiltonian" (Sec. I). A factorization is a bijection lambda : H -> H_A x H_B, Eq. (1), and any unitary gives another,
Eq. (2). The Hamiltonian splits as Eq. (3), H = H_A x I + I x H_B + H_int, with the interaction expanded in a product
basis, Eq. (5), H_int = sum over alpha of lambda_alpha A_alpha x B_alpha. A quasi-classical factorization is one with
(Sec. II) "1. Robustness: There exist preferred pointer states of the system ... that, if initially unentangled with
the environment, typically remain unentangled under evolution by H. 2. Predictability: For states with near definite
value of the pointer observable, it will serve as a predictable quasi-classical variable, with minimal spreading".

The algorithm (Sec. V.B), for fixed subsystem sizes d_A, d_B: (1) find the Candidate Pointer Observable, Eq. (24),
O_CPO = O_A x O_B, "the best tensor product observable compatible with the interaction Hamiltonian", by minimizing the
Frobenius norm of [H_int, O_CPO], Eq. (25), with the factors expanded in traceless generalized Gell-Mann matrices,
Eq. (26); (2) build d_A product states peaked on its eigenstates, Eq. (27); (3) for each, compute the second time
derivative at t = 0 of the linear entropy of the reduced state (robustness, Eq. 28) and of the pointer entropy
(predictability, Eq. 29); (4) take the Schwinger entropy, Eq. (30), S''_Schwinger = max(S''_lin(0), S''_pointer(0)),
averaged over the d_A states; (5) "Find the factorization that minimizes S''_Schwinger. This will be the quasi-classical
factorization."

Their own caveats (Sec. VIII): "it is unclear how feasible our algorithm is in practice. Given nothing but the spectrum
of a finite-dimensional Hamiltonian, it could take a long time to sift through the space of factorizations"; only the
bipartite split is treated, "ultimately we would want to continue to factorize the system into appropriate subsystems
... though locality and other considerations may come into play"; the split of dimensions D = d_A d_B is fixed by hand.
The link to the geometry papers is one sentence: "as we consider more abstract theories, including those of emergent
spacetime in quantum gravity [8, 9], our laboratory intuition may no longer be relevant, and an algorithm of the sort
presented here can become important."

*Ours.* For the toy the factorization is not a question: the points are given by construction, one factor per point
would be the obvious reading, and the renaming count of ASSUMPTIONS Q15 (which points are interchangeable) is a
symmetry of a fixed wiring, not a choice among tensor-product structures. The inventory's remark that Update 12 "is a
factorization question in his language" survives only loosely: his factorization is chosen by a dynamics the toy does not
have.

## 4. [BCCS17], locally finite-dimensional Hilbert space, read in part

Read: the whole essay (six pages); it has no numbered sections. The claim: "the density operator describing the state
corresponding to a small region of space, when such a notion makes sense, is defined on a finite-dimensional factor of a
larger Hilbert space" (abstract), argued from the finite entropy of black holes and of a de Sitter patch. The structure
is Eq. (1), H_QG = H_sys x H_env, decoherence into pointer states, Eq. (2), and an emergence map Phi from a decohered
branch to a spatial manifold with a 3-metric, its conjugate momentum and an effective-field-theory state, Eq. (3). The
caveat the toy shares: "There are well-known obstacles to constructing phenomenological acceptable theories with
finite-dimensional Hilbert spaces, including a lack of exact Lorentz invariance." And the position: "quantum field theory
cannot be a fundamental description of Nature" (abstract). For this note its use is one line: a finite local state space
is, in this group's view, a feature to argue for; the toy's finite graphs are finite by construction, which is a
different reason, and nothing here makes the toy's finiteness a point in its favor (rule 13).

## 5. Does "any graph can be read as an entanglement pattern" appear?

No, in neither [CCM16] nor [CC17]. What they say is the converse direction only: an RC state in a given factorization
defines an information graph (Sec. II.B and III.A of [CCM16]; Sec. III.A of [CC17], "Note that we are not given the graph
as a fundamental piece of information; it is derived from the quantum state in this particular factorization"). Neither
paper asks which weighted graphs arise as the information graph of some RC state, and neither cites Garnerone, Giorda
and Zanardi. The one constraint they state in that direction is Eq. (11) of [CCM16], the weighted degree bounded by
2 ln of the factor's dimension, which any finite graph meets with large enough factors.

The published statement the task named: Garnerone, Giorda and Zanardi, "Bipartite quantum states and random complex
networks", New J. Phys. 14, 013011 (2012), arXiv:1103.5989, DOI 10.1088/1367-2630/14/1/013011. Abstract only, read from
the arXiv API on 2026-11-02: "We introduce a mapping between graphs and pure quantum bipartite states and show that the
associated entanglement entropy conveys non-trivial information about the structure of the graph." Two cautions, ours.
First, that mapping sends a graph to one bipartite pure state whose entanglement entropy is a number that characterizes
the graph; it is not the [CCM16] construction, in which a state on many factors yields a graph weighted by pairwise
mutual information. So "any graph can be read as an entanglement pattern" is true in the [GGZ12] sense (a state is
built from the graph) and open in the [CCM16] sense (whether a given weighted graph is the information graph of an RC
state). Second, [GGZ12] has not been read beyond the abstract; whether its map is one-to-one, or what its state's mutual
informations between vertex factors are, is not known to this note. If the sentence is to be written anywhere, it needs
[GGZ12] read in full first and the sense stated (rule 15).

## 6. The toy's two obvious counterparts

### 6.1 Dimension: their MDS estimator against the toy's local count and the published spectral dimension

Three different things are called dimension here, and the record should keep them apart.

| Estimator | What it reads | Local or global | Scale | Source |
|---|---|---|---|---|
| [CCM16] MDS, Eqs. (23) to (26) | the number of dominant eigenvalues of B built from all pairwise distances; epsilon_D says how well D fits | global, one number for the whole patch | none; all pairs enter | [CCM16] Sec. III.C |
| [CCM16] Eq. (15) | the exponent of the entropy of a ball against its radius (the dimension is D_f + 1, Sec. III.B) | local at p, from the ball | r | [CCM16] Sec. III.B |
| toy d(v) (`graphity.dimension.local_dimension`) | the number of link pairs at v that close no square: sheet 2, tube 1, 4-cube 0 | local, one point | one step | ASSUMPTIONS, dimension ladder; `dimension.py` docstring: "a square-count signature, not a certified geometric dimension" |
| toy D_s(t) (`spectral_dimension`) and [T22] Eqs. (15) to (17) (`heat_kernel_dimension`) | the fall of a random walker's return probability | global average, a function of walk length | t | [Carlip17] Sec. 2.1; [T22]; T80 draft |

The published spectral dimension of the model's own graphs ([T22], its Fig. 5 at N = 2000, quoted in the
`heat_kernel_dimension` docstring) rises, falls and then goes to zero with finite size; [CCM16] never computes a
spectral dimension, and the toy never computes MDS. The nearest exact bridge is that the first step of the toy's D_s
depends only on the degree and the square count (docstring of `spectral_dimension`), while [CCM16]'s MDS depends on the
whole distance matrix: the two cannot be compared except on the same graph, which is what section 7 proposes.

### 6.2 Curvature: their curvature against Ollivier curvature

The toy's energy is the total Ollivier-Ricci curvature of the graph at unit link lengths, H = -2D sum over links of
kappa, [T25] Eq. (21), with kappa a function of the squares on the link ([T25] Eq. (8); `cqg.py` docstring, Q1); the
curvature is a property of the wiring, and a missing square costs 16. [CCM16]'s curvature is a property of lengths: the
wiring is fixed ("the vertices and edges remain fixed, so that the graph is merely re-weighted", Sec. III.B) and a
perturbation of the state lengthens edges, which the embedding (Eq. 36) or the deficit angle (Eq. B6) reads as
curvature. Neither paper mentions Ollivier (searched). So the two curvatures live on different variables: squares
against lengths. Where they could meet is exactly where the repository's `graphity.regge` already stands (Q27): a toy
torus cut into triangles with a length on each side, whose deficit angles are [CCM16]'s Appendix B with D = 2 and
L_i = 1. Ours, unverified: at unit lengths every triangle of that triangulation is a right isosceles one, the flat torus
has zero deficit everywhere (Q27, tested), and a knot in an opened sheet is a place where the triangulation itself
changes, since a point with d(v) = 1 or 0 has a different set of squares; whether their reader would see a knot as a
length excess (positive curvature by Eq. 40) or as a missing simplex is not settled by either paper.

## 7. Ours, unverified: feeding the toy's saved wirings to their reader

Nothing in this section has been run. It says what would have to be chosen and what the first exact look would compare.

### 7.1 The inputs that exist

The toy saves wirings as `(N, degree)` neighbor arrays in `results/*_adj/*.npz` (keys `adj`, `part`, `lam`, `g`, `h0`
on the T23 and T24 decays; other runs may carry other keys; 708 such directories on 2026-11-02). The built states need
no run: `cqg.torus(lx, ly)` gives the flat sheet and, with one side of length 4, the tube; the 4-cube is the 4 x 4 torus
(Q4). Saved states that would make the first look: the T23 and T24 end states (opened sheets, some perfectly flat, some
resting on a four-point remnant or a rarer knot, N = 64 to 192); the T37 and T51 large opened sheets with their knots
(4 x 1024 and 128 x 4 tubes opened from several seeds); T27's leak 0 control (a melted sheet, N = 144); the lambda = 0
cold states of Gate A for shattered cubes (whether a wiring was saved for those is to be checked; the CSV runs
`cqg_lam0_*` carry sweep averages, not wirings). Flat sheet and tube are built; the others are measured.

### 7.2 What has to be chosen

1. **A factor per point.** Their vertices are regions A_p large enough for the cut function to hold; the toy's points
   would be taken as the factors directly, with no coarse-graining. This is a choice, and it is the one their toric-code
   example makes at the plaquette scale.
2. **A mutual information per link.** The toy has no state and no entropy; its links are all alike. The obvious choice
   is a constant, i(p : q) = x_0 on every link and 0 off-link. With Phi = -ln, w = l_RC (-ln x_0), a constant, so
   their metric Eq. (14) is a constant times the graph distance; choosing x_0 = 1/e makes w = l_RC. (Not x_0 = 1: then
   w = 0 everywhere and the graph collapses to a point by their own Phi(1) = 0.) Any other choice that gives every link
   the same weight gives the same embedding up to scale, so the first look is free of this choice; a choice that weights
   links by their square count would put the toy's energy into the lengths and is a second, different look.
3. **Open patches, not the whole torus.** Their examples are open patches (Sec. 1.6 above). The toy's states are tori.
   Ours, exact (computed by the second agent with numpy and rechecked by the session, 2026-11-02): each wrapped cycle
   takes two eigenvalues of B. An 8 x 8 torus gives four equal at 218.5, then four at 37.5; a 4 x 16 tube gives
   525.5, 525.5 (the long cycle), then 160, 160 (the curled direction, about 0.3 of the long pair and growing with the
   number of points, not of order 1), then the long cycle's harmonics; the 4-cube gives exactly four positive
   eigenvalues of 16, the rest zero or −4. So a whole-torus reading calls the sheet, the tube and the 4-cube all
   four-dimensional in the count of large eigenvalues, telling them apart only by ratios, for the wrap and not for
   the wiring, and never calls the 4-cube zero-dimensional, where the toy's d(v) does. The first look therefore uses a window: a patch
   of the saved wiring smaller than half the torus in each direction, so that no shortest path wraps, with the built
   lattice patch of the same size as the control. (The tube's short direction wraps inside any window wider than two
   columns; a 4 x k window of the tube is the one case where the wrap cannot be cut out, and its reading has to be
   compared with the built tube's.)
4. **The reading rule, fixed in advance.** Report the ordered eigenvalues of B, epsilon_1, epsilon_2, epsilon_3 and
   epsilon_4 (Sec. 1.5), and the stress of the 2-d embedding, for the saved patch and for the built control of the same
   size and shape; no threshold is set here, because their own examples give epsilon_1 = 0.0167 for a clean line and
   epsilon_2 = 0.41 for a clean plane, so "small" depends on the case and is to be set from the built control.

### 7.3 What the exact first look would compare

- Built flat L x L patch, built 4 x k tube patch, built 4-cube: the controls. Their numbers (eigenvalues, epsilon_D)
  are exact for the graph metric and need no sampling.
- A perfectly flat opened sheet (T23 or T24 end state with H = 0): by construction it is a flat 4-regular bipartite
  graph with two squares on every link, so it is a lattice torus, possibly of another shape than the built one; it
  cannot differ from a built control of the same shape. This case checks the pipeline, not the toy.
- An opened sheet with a knot (the four-point remnant, O16; the 20-unit twist, O15; the double column, O107): the
  patch round the knot against the flat control. The question is whether MDS shows a local distortion (a rise in
  epsilon_2 and in the stress, confined to the window that holds the knot) and whether the Regge deficits of the
  triangulated patch (`graphity.regge`, with the knot's own squares triangulated by the same rule) put a positive
  deficit at the knot, which is the sign Eq. (40) gives for a disentangling perturbation.
- The melt (T27 leak 0): a random 4-regular graph has shortest paths of order log N between most pairs; MDS on such a
  metric has no small set of dominant eigenvalues, so epsilon_D stays large for every small D. [T22] calls the random
  phase's Hausdorff dimension infinite (VISION Update 5); their reader would call it "no geometry". The comparison is
  between those two statements on one graph.
- Shattered cubes (lambda = 0 cold state, if a wiring is saved): disconnected pieces, each a 4-cube or the 14-point
  biplane (Q8); their procedure runs "separately for each connected component" (Sec. III.B), and the reading per piece
  is the 4-cube's.

### 7.4 What would count against the toy, and what for the reader

- Against the toy: a flat opened sheet, or a built flat patch, reading far from two dominant eigenvalues (epsilon_2
  large against epsilon_1 and epsilon_3 small) would say the toy's "flat sheet" is not what a mutual-information reader
  calls two-dimensional. For the built lattice this is a statement about the graph (Manhattan) metric under MDS, not
  about the toy, and it is the first number to compute: if a clean lattice patch already scores badly, the reader is
  not a test of the toy's states at all and the exercise stops there.
- Against the toy's knot reading: a knot that leaves no mark in the MDS window or in the deficits, while the toy charges
  it 8 to 45 units (O13, O15, O107), would say the toy's leftover is invisible to a length-based curvature.
- For the reader: agreement on the built controls (two for the sheet patch, one for a tube patch read along its length,
  "no geometry" for the melt), and a localized positive distortion at each knot. Agreement is a consistency between
  the toy's square count and their length-based reader on the same graph; it is not evidence for either about reality
  (rule 13), and it would not touch any pre-registered verdict.
- Either way, the 4-cube is where the two conventions part: the toy's d(v) calls it zero-dimensional (fully curled); a
  metric embedding of the Hamming cube is not zero-dimensional in any sense of [CCM16]. That is a difference of
  definitions, and section 8's question is about it.

### 7.5 What is lost and what is gained if the toy's dynamics were rewritten in their formalism

Lost:
- The toy's dynamics is a Metropolis chain on wirings under an energy that counts squares; [CCM16] has no dynamics ("We
  do not assume any particular Hamiltonian", Sec. I) and its only change is a perturbation of a state at fixed wiring.
  Nothing in it moves a link. The toy's whole result, a sharp opening from a seed with a front and a release (T7, T9,
  T10), has no counterpart.
- [CC17]'s route needs a quantum state in a factorized Hilbert space, a modular Hamiltonian for each cut (Eq. 36, a
  Rindler Hamiltonian of an emergent field theory), an entanglement first law (Eq. 38), a time parameter (A6) and Lorentz
  invariance (A7). The toy has none of these: no state, no entropy, no time beyond the count of moves, and no Lorentz
  symmetry (VISION, "The target", stated as out of reach). The curvature-energy relation they get is linearized about
  flat space; the toy's changes (tube to sheet) are not small perturbations of a flat state.
- The toy's energy is a count with no units; their alpha is fixed to 1/4 G_N (Sec. 2.3 above). The unit question of
  `docs/reading/notes/2026-11-01_units_for_the_toy.md` would come back in their language as the value of l_RC and alpha.

Gained:
- A route from a local change to a curvature to Einstein's equation, which the programme's piece 8 lacks: the record
  says the toy's flat space is gapped and can pass on no pull at a distance (O60 (a)), and the owner's decision of
  Update 40 adds a quantity on the points as a carrier. In [CCM16] and [CC17] the carrier is the length on each link
  set by mutual information, and the pull's equation is the linearized Hamiltonian constraint. Whether a quantity on the
  points of the toy could be read as their delta I per link, so that their Eq. (40) gives the curvature it produces, is
  a design question for the gravity-field brief (not written), not a reading result.
- An independent dimension estimator (MDS) for the toy's saved states, which the T80 draft does not include; it can be
  run exactly, at no cost beyond an eigendecomposition.

## 8. One narrow question for a letter

Their papers leave open what the mutual-information reader does with a direction that is closed on a short loop. In
[CCM16] a curled direction of length 4 is, with equal mutual information on every link, a 4-cycle in the distance
graph; a 4-cycle's shortest-path metric is not Euclidean (opposite corners at distance 2, not the square's root 2), and
classical MDS gives eigenvalues 2, 2, 0, −1: two positive and one negative. So a graph that the toy calls
one-dimensional (a 4 x L tube, one open direction) carries two nonzero MDS eigenvalues for its curled direction that
are not small (160 against 525 for a 4 x 16 tube), and a fully curled 4-cube, which the toy calls zero-dimensional, embeds as a four-dimensional object. The
question, in one sentence: in the redundancy-constrained reading, is a direction curled to a few sites a dimension
(because its distances are Euclidean-embeddable) or not (because its entropy-of-a-ball exponent, Eq. (15), never
grows along it), and which of the two estimators is the one the program intends for a state with small closed
directions? Everything in the toy's ladder (sheet 2, tube 1, cube 0) depends on the answer, and neither paper contains
a closed direction.

## 9. Read-status table

| Key | Paper | arXiv | Read | Notes |
|---|---|---|---|---|
| CCM16 | Cao, Carroll, Michalakis, Space from Hilbert Space: Recovering Geometry from Bulk Entanglement, Phys. Rev. D 95, 024031 (2017) | 1606.08444v3 | read in full (text; figures not seen) | Sections I to VI, Appendices A and B, reference list |
| CC17 | Cao, Carroll, Bulk Entanglement Gravity without a Boundary: Towards Finding Einstein's Equation in Hilbert Space, Phys. Rev. D 97, 086003 (2018) | 1712.02803v1 | read in full (text; figures not seen) | journal reference from the arXiv listing, to verify |
| CS20 | Carroll, Singh, Quantum Mereology: Factorizing Hilbert Space into Subsystems with Quasi-Classical Dynamics, Phys. Rev. A 103, 022213 (2021) | 2005.12938 | read in part | abstract, Secs. I, II, III opening, V, VIII; journal reference from the arXiv listing, to verify |
| BCCS17 | Bao, Carroll, Singh, The Hilbert Space of Quantum Gravity Is Locally Finite-Dimensional | 1704.00066v2 | read in full (an essay of six pages) | Gravity Research Foundation essay; the arXiv listing gives DOI 10.1142/S0218271817430131 (probably Int. J. Mod. Phys. D 26, 1743013 (2017); to verify) |
| GGZ12 | Garnerone, Giorda, Zanardi, Bipartite quantum states and random complex networks, New J. Phys. 14, 013011 (2012) | 1103.5989 | abstract only | title, authors and DOI from the arXiv abstract page; the journal reference is from the DOI and is to verify |

## 10. REFERENCES.bib entries owed

To be added by the session that commits this note (the agent writes one file only). Reading status in the `note` field
as the file's header requires.

```
@article{CCM16,
  author  = {Cao, ChunJun and Carroll, Sean M. and Michalakis, Spyridon},
  title   = {Space from Hilbert Space: Recovering Geometry from Bulk Entanglement},
  journal = {Phys. Rev. D},
  volume  = {95},
  pages   = {024031},
  year    = {2017},
  eprint  = {1606.08444},
  archivePrefix = {arXiv},
  note    = {read-full (arXiv v3 text, by an assistant agent, 2026-11-02; figures not seen; docs/reading/notes/2026-11-02_carroll_space_from_entanglement.md)}
}

@article{CC17,
  author  = {Cao, ChunJun and Carroll, Sean M.},
  title   = {Bulk Entanglement Gravity without a Boundary: Towards Finding Einstein's Equation in Hilbert Space},
  journal = {Phys. Rev. D},
  volume  = {97},
  pages   = {086003},
  year    = {2018},
  eprint  = {1712.02803},
  archivePrefix = {arXiv},
  note    = {read-full (arXiv v1 text, by an assistant agent, 2026-11-02; figures not seen; journal reference from the arXiv listing, to verify; same note)}
}

@article{CS20,
  author  = {Carroll, Sean M. and Singh, Ashmeet},
  title   = {Quantum Mereology: Factorizing Hilbert Space into Subsystems with Quasi-Classical Dynamics},
  journal = {Phys. Rev. A},
  volume  = {103},
  pages   = {022213},
  year    = {2021},
  eprint  = {2005.12938},
  archivePrefix = {arXiv},
  note    = {read in part (abstract, Secs. I, II, V, VIII, by an assistant agent, 2026-11-02; journal reference from the arXiv listing, to verify; same note)}
}

@misc{BCCS17,
  author  = {Bao, Ning and Carroll, Sean M. and Singh, Ashmeet},
  title   = {The Hilbert Space of Quantum Gravity Is Locally Finite-Dimensional},
  year    = {2017},
  eprint  = {1704.00066},
  archivePrefix = {arXiv},
  howpublished = {Essay for the Gravity Research Foundation 2017 Awards},
  note    = {read-full (arXiv v2, by an assistant agent, 2026-11-02; same note)}
}

@article{GGZ12,
  author  = {Garnerone, Silvano and Giorda, Paolo and Zanardi, Paolo},
  title   = {Bipartite quantum states and random complex networks},
  journal = {New J. Phys.},
  volume  = {14},
  pages   = {013011},
  year    = {2012},
  eprint  = {1103.5989},
  archivePrefix = {arXiv},
  doi     = {10.1088/1367-2630/14/1/013011},
  note    = {abstract-only (arXiv abstract page and API, 2026-11-02); the journal reference is from the arXiv listing and the DOI; not cited by CCM16 or CC17}
}
```

Verified against the sources by a second agent on 2026-11-02 (about sixty quotations, equations and numbers; all found); its corrections applied the same day, including two of our own statements about multidimensional scaling that were wrong and were recomputed exactly.
