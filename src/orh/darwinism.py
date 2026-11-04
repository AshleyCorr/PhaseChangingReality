"""Closed forms for the controlled-tilt record model: the exact benchmarks the Stage B simulator is checked against.

The model (Zurek's c-not model generalized to imperfect records; see docs/reading/notes/2026-10-12_orh_quantum_
darwinism_records.md for the published forms and their equation numbers, which this file does not repeat). A system
qubit S starts in sqrt(1-p)|0> + sqrt(p)|1>. Each of N environment qubits E_i starts in |0> and is written on by a
controlled tilt of angle theta_i: if S is |1>, E_i goes to |theta_i> = cos(theta_i)|0> + sin(theta_i)|1>. The global
state is then

    |Psi> = sqrt(1-p) |0>_S |0...0>_E  +  sqrt(p) |1>_S |theta_1 ... theta_N>_E,

and every quantity below follows from the overlaps of the two environment branch states on a set of qubits F,

    c_F = <0_F | theta_F> = prod_{i in F} cos(theta_i).

Derivations (ours, elementary; each is checked against the simulator in tests/test_orh_darwinism.py):

* Reduced state of S: diagonal (1-p, p), off-diagonal sqrt(p(1-p)) c_all. Visibility V = 2 sqrt(p(1-p)) c_all, which
  for p = 1/2 is |<E_1|E_0>| = c_all (handoff section 3).
* Reduced state of a fragment F (tracing S kills the cross terms because |0>_S and |1>_S are orthogonal):
  (1-p)|0_F><0_F| + p|theta_F><theta_F|, a mixture of two pure states with overlap c_F. Its eigenvalues are
  (1 +- sqrt(1 - 4p(1-p)(1 - c_F^2)))/2, so its entropy is f(c_F) with f defined below.
* H(S) = f(c_all) (same eigenvalue formula). H(SF) = H(rest) = f(c_rest) because the global state is pure.
* I(S:F) = H(S) + H(F) - H(SF) = f(c_all) + f(c_F) - f(c_rest).
  Limits: perfect records (every theta = pi/2) give I(S:F) = h(p) for every nonempty F smaller than the whole
  environment (the classical plateau) and 2h(p) for F = all (the quantum rise at the end of the plot).
* Two observers each copying a different environment qubit (i, j) with a controlled-NOT report different values with
  probability p (cos^2 theta_i sin^2 theta_j + sin^2 theta_i cos^2 theta_j), which is zero for perfect records.
"""
from __future__ import annotations

import numpy as np

from .statevector import binary_entropy


def overlap(thetas):
    """c_F = prod cos(theta_i) over the given angles (1.0 for an empty set)."""
    return float(np.prod(np.cos(np.asarray(thetas, dtype=float)))) if len(thetas) else 1.0


def visibility(p, thetas):
    """Fringe visibility of S after the environment has recorded with angles `thetas`."""
    return float(2 * np.sqrt(p * (1 - p)) * overlap(thetas))


def mixture_entropy(p, c):
    """f(c): entropy in bits of (1-p)|u><u| + p|v><v| with |<u|v>| = c."""
    disc = max(0.0, 1 - 4 * p * (1 - p) * (1 - c * c))
    lam = 0.5 * (1 + np.sqrt(disc))
    return binary_entropy(lam)


def mutual_information(p, thetas, fragment):
    """I(S:F) in bits for the fragment made of the environment qubits with indices `fragment`."""
    thetas = np.asarray(thetas, dtype=float)
    fragment = sorted(set(int(i) for i in fragment))
    rest = [i for i in range(len(thetas)) if i not in fragment]
    return (mixture_entropy(p, overlap(thetas)) + mixture_entropy(p, overlap(thetas[fragment]))
            - mixture_entropy(p, overlap(thetas[rest])))


def partial_information_plot(p, theta, n_env):
    """I(S:F_m) for fragments of the first m environment qubits, m = 0 .. n_env, all written with the same angle."""
    thetas = np.full(n_env, float(theta))
    return np.array([mutual_information(p, thetas, range(m)) for m in range(n_env + 1)])


def redundancy(p, theta, n_env, delta):
    """R_delta = n_env / m_delta, where m_delta is the smallest fragment that carries at least (1 - delta) of
    H(S) = h(p) about S; infinity if no fragment short of the whole environment does (then the information is not
    redundant). Zurek's definition; the reading note gives the equation."""
    target = (1 - delta) * binary_entropy(p)
    plot = partial_information_plot(p, theta, n_env)
    for m in range(1, n_env):
        if plot[m] >= target - 1e-12:
            return n_env / m
    return float("inf")


def disagreement_rate(p, theta_i, theta_j):
    """Probability that two observers who each copy a different environment qubit (angles theta_i, theta_j) by a
    controlled-NOT hold different values."""
    ci, si = np.cos(theta_i), np.sin(theta_i)
    cj, sj = np.cos(theta_j), np.sin(theta_j)
    return float(p * (ci * ci * sj * sj + si * si * cj * cj))
