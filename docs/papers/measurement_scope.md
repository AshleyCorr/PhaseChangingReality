# Scope of the curled-torus evidence

This reference states the interpretation used across the repository as of 9 October 2026.
Derivations and retrospective analysis are **Ours, unverified by an independent physicist**.
The [paper](curled_torus/paper.pdf), [methods supplement](curled_torus/supplement.md), and
[O108](../../ASSUMPTIONS.md) give the detailed evidence. Historical predictions and verdicts
remain records of what was tested; they do not establish stronger physical interpretations.

| Observable or calculation | What it establishes | What it does not establish |
|---|---|---|
| d(v)=6−q(v), f=mean(d)−1 in the four-link model | Local and mean square counts | Geometric dimension, two coexisting geometric phases, or macroscopic torus periods |
| Largest component of selected vertices | Connectivity of that selected set at a snapshot | A unique seed, a unique moving front, or its shape |
| T8 independent surface checks | All 120 saved λ=1.05 endpoints checked are tori | That every other endpoint is a torus, or that both periods grew with N |
| Initial energy gap εN, ε=4(λ−1) | Energy above zero of the starting 4×L torus, even L>4 | That all of it was released: Q=εN−H_final; a time-window mean differs from endpoint energy |
| Neutral-basin enumeration at λ=1.25 | Minimum nonneutral cost 12 at N=48,64,96,144,192,288; all neutral targets enumerated up to isomorphism | An arbitrary-size theorem; the N=32 control has different full censuses across its three classes |
| Seed scan, 12,000 sweeps, eight replicas per cell | At the sampled seed energies, 0/8 escape below 12 and 8/8 at or above 12 at N=48–192 | Guaranteed conversion, probability exactly zero or one, or localized energy delivery; the seed is globally available |
| Counted two-channel escape rate | Approximation to first exit for the specified proposal clock | A common clock for detected departure, commitment, conversion and release |
| CV or exponential goodness-of-fit check | A specified finite-sample diagnostic | Proof of memorylessness; long detected waits in T38 require a broader account |
| T38 TWO POPULATIONS | An observed excess of long detected waits in some cells | A diagnosed mechanism; all 190 missing detector crossings still reach f≥0.75 and are not ordinary right-censored survivors |
| 200-sweep persistence edge at g=1.5 | Operational boundary between λ=1.35 and 1.40 on this protocol | An energetic spinodal or loss of all activation barriers |
| bath_T in sealed CSVs | Mean store energy u | A calibrated temperature g |
| Conservation of graph plus stores | Energy bookkeeping | Equilibrium, mixing, coexistence or a required residual graph defect |

For a canonical distribution over accessible levels 0,δ,2δ,…, the conditional thermometer is
u=δ/expm1(δ/g), or g=δ/log1p(δ/u). At λ=1.25 the known increments 12 and 25 have gcd 1;
this is not a proof of full level accessibility or equilibration. For δ=1, u(3.5)=3.023777.
The capacity estimate is C*≈[εN+s−U_graph(g_m)]/u(g_m). Dropping the seed and graph-energy
terms gives N/3.024 at λ=1.25, only as an approximation. The T9 crossover measured on the
grid is (N/4,N/2]. Its absence of stalled classifications does not exclude coexistence.
T10's historical mean-energy-below-one check is preserved; conditional temperature proxies
span approximately 0.8406–1.0128, so “every bath below g=1” is unsupported.

For the alternative uniform-unlabelled-class ensemble, Metropolis–Hastings acceptance is
min(1, A(G′)/A(G) exp(−ΔH/g)). The automorphism ratio is inside the minimum.
A rate suppression by cN corresponds to a barrier increment g log(cN), not one proportional
to N. A first-move ratio alone does not determine the complete lifetime. This classical
ensemble is a modeling choice; quantum indistinguishability does not derive it here.

The T6 disorder-to-order verdict remains INCONCLUSIVE at λ=1,1.25,1.5. Finite sampled
histograms do not justify a general thermodynamic bound on latent heat. T7 and the three
lambda maps retain all their original failures. Retrospective amendments are not prospective
predictions. The pooled 432-wait p-value is nominal because the original exploratory runs
reuse seeds across couplings whereas its bootstrap treats cells independently.

Current summaries and companion manuscripts use these limits. Older dated log entries and
preregistrations retain the original claims with correction pointers. Raw results and configs
are unchanged. Public HTML changes here are repository source changes, not a website deployment.

The sealed-sheet budget runner formerly passed 4λ as the demon level spacing; at λ=1.25
this incorrectly used 5. Old `sealed_sheet_budget_*` result temperature columns are therefore
unvalidated proxies and must not be used as measured temperatures. Future outputs retain the
legacy column, leave it NaN without an explicitly justified `thermometer_step`, and always
record `demon_mean`. Supplying a step does not establish equilibrium.
