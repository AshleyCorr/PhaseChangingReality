"""Soft space, stage A: one number phi on each point, with waves, sourced by the relics (VISION Update 77).

Naming. The wiring's energy is the toy's, H = 16 (N - S) + 4 lam X (cqg.py, Q1), four links per point, no cap. At
lam != 1 this is our family around the published model, never combinatorial quantum gravity (VISION Update 22).

THE ENERGY. A field phi (one real number per point) is added, with

    E_phi = (kappa / 2) sum over links (phi_p - phi_q)^2  +  q sum_p e_p phi_p,

e_p the curling energy at point p: each link with S_e > 2 carries 4 (lam - 1)(S_e - 2) (its share of H above the flat
value, Q1 written link by link), split equally between its two points. So a frozen relic (a knot or a scrap: curled
links) is a source and phi dips where it sits for q > 0; a missing square (S_e < 2, the network's flickering flaws) is
not a source (the owner's answer of 20 October, VISION Update 76). The total sampled is exp(-(H + E_phi) / g).

THE MEAN OF PHI. Only differences of phi are physical (shift symmetry, the brief's declaration 3), and on a closed
space the source term would otherwise drive the uniform mode without bound. phi is therefore kept on the plane
sum_p phi_p = 0: every momentum is drawn there and every force is projected onto it. On that plane the source term is
q sum_p (e_p - mean e) phi_p, the closed-space form of Poisson's equation (sources balanced by a uniform background).

THE MOVES, and why each leaves exp(-(H + E_phi)/g) unchanged (detailed balance).
1. Switches of the wiring (`soft_attempts`): cqg.run_chain's switch, line for line (the same four integer draws per
   attempt, a uniform drawn only when the energy rises), with phi held fixed and the energy change taken as
   dH + dE_phi. dE_phi has two parts: the gradient terms of the two links removed and the two added, and the change of
   the source term over the links whose square count changed, which are exactly the links cqg already notes for X
   (`_note_square_edges`). The proposal is cqg's, symmetric (Q4); Metropolis on the full energy change then gives
   detailed balance with respect to exp(-(H + E_phi)/g) at fixed phi.
2. Field trajectories (`hmc_trajectory`): hybrid Monte Carlo at fixed wiring. Momenta pi are drawn from exp(-K/g),
   K = sum pi^2 / 2, on the plane sum pi = 0; a leapfrog trajectory of n_leap steps of size eps integrates
   d phi/dt = pi, d pi/dt = F, F the projected force -dE_phi/d phi; the end is accepted with probability
   min(1, exp(-(dK + dE_phi)/g)). Leapfrog is reversible and keeps volume, so this is exact detailed balance in phi at
   fixed wiring (Duane, Kennedy, Pendleton and Roweth, 1987; general knowledge). The driver draws each trajectory's
   number of steps uniformly between half its maximum and the maximum, independently of the state, which keeps detailed
   balance and stops a fixed length from resonating with a wave mode (tested: unbiased either way, faster mixing). Between refreshes the trajectory is
   the wave equation phi'' = -kappa L phi - q (e - mean e) on the current wiring (L its Laplacian), so a disturbance
   travels; at long wavelengths on the flat sheet at speed sqrt(kappa) links per unit of trajectory time.
Alternating the two kernels keeps the joint distribution (each preserves it).

The clocks. One clock for both ends of a move (the owner's answer to the brief's question 4): phi does not change
which switches are offered or how often. phi is read as clocks and, oppositely, link lengths only in what is measured.

Random streams. The switch block uses numba's stream, seeded once and carried on (seed < 0), as in cqg.run_chain;
the field uses a numpy Generator. With q = 0 and phi = 0 a switch block is row_growth._attempts draw for draw (tested).
"""
import numpy as np
from numba import njit

from graphity.cqg import (ENERGY_PER_SQUARE, ENERGY_PER_SURPLUS, NO_CAP, _new_edge_ok, _note_square_edges, _replace,
                          _surplus_on, _switch, hamiltonian)
from graphity.squares import has_edge, squares_on_edge


# ---------------------------------------------------------------------------------------------- energies

@njit(cache=True)
def curl_sources(adj, lam):
    """e_p: the curling energy 4 (lam - 1)(S_e - 2) of every link with S_e > 2, split equally between its points."""
    n = adj.shape[0]
    e = np.zeros(n)
    c = 4.0 * (lam - 1.0)
    for u in range(n):
        for k in range(4):
            v = adj[u, k]
            if v > u:
                s_e = squares_on_edge(adj, u, v)
                if s_e > 2:
                    e[u] += 0.5 * c * (s_e - 2)
                    e[v] += 0.5 * c * (s_e - 2)
    return e


def laplacian_apply(adj, phi):
    """(L phi)_p = 4 phi_p - sum of phi over p's four neighbours."""
    return 4.0 * phi - phi[adj].sum(axis=1)


def field_energy(adj, phi, kappa, q, lam):
    """E_phi = (kappa/2) sum over links (phi_p - phi_q)^2 + q sum_p e_p phi_p."""
    grad = 0.5 * kappa * float(np.dot(phi, laplacian_apply(adj, phi)))
    return grad + q * float(np.dot(curl_sources(adj, lam), phi))


def total_energy(adj, phi, kappa, q, lam):
    """H + E_phi."""
    return float(hamiltonian(adj, lam)) + field_energy(adj, phi, kappa, q, lam)


# ---------------------------------------------------------------------------------------------- switches

@njit(cache=True)
def _source_on(adj, edges, n, phi):
    """sum over the noted edges that exist of (S_e - 2)_+ (phi_a + phi_b) / 2."""
    s = 0.0
    for i in range(n):
        a, b = edges[i, 0], edges[i, 1]
        if has_edge(adj, a, b):
            s_e = squares_on_edge(adj, a, b)
            if s_e > 2:
                s += (s_e - 2) * 0.5 * (phi[a] + phi[b])
    return s


@njit(cache=True)
def soft_attempts(adj, side_u, inv_g, n_attempts, seed, lam, cap, phi, kappa, q):
    """n_attempts switches at fixed phi, in place (Metropolis on dH + dE_phi). seed < 0 carries numba's stream on.
    Returns (accepted, the summed energy change of the accepted switches). The body is row_growth._attempts (itself
    cqg.run_chain's loop) with the field's change added."""
    if seed >= 0:
        np.random.seed(seed)
    nu = side_u.shape[0]
    track_x = cap > 2
    edges = np.empty((64, 2), dtype=np.int64)
    accepted = 0
    total = 0.0
    qc = q * 4.0 * (lam - 1.0)
    for _ in range(n_attempts):
        u1 = side_u[np.random.randint(0, nu)]
        v1 = adj[u1, np.random.randint(0, 4)]
        u2 = side_u[np.random.randint(0, nu)]
        v2 = adj[u2, np.random.randint(0, 4)]
        if u1 == u2 or v1 == v2 or has_edge(adj, u1, v2) or has_edge(adj, u2, v1):
            continue
        n_edges = 0
        if track_x:
            n_edges = _note_square_edges(adj, u1, v1, edges, n_edges)
            n_edges = _note_square_edges(adj, u2, v2, edges, n_edges)
        lost = squares_on_edge(adj, u1, v1)
        _replace(adj, u1, v1, -1)
        _replace(adj, v1, u1, -1)
        lost += squares_on_edge(adj, u2, v2)
        _replace(adj, u2, v2, -1)
        _replace(adj, v2, u2, -1)
        _replace(adj, u1, -1, v2)
        _replace(adj, v2, -1, u1)
        gained = squares_on_edge(adj, u1, v2)
        _replace(adj, u2, -1, v1)
        _replace(adj, v1, -1, u2)
        gained += squares_on_edge(adj, u2, v1)
        d_s = gained - lost
        d_x = 0
        d_src = 0.0
        ok = _new_edge_ok(adj, u1, v2, cap) and _new_edge_ok(adj, u2, v1, cap)
        if ok and track_x:
            n_edges = _note_square_edges(adj, u1, v2, edges, n_edges)
            n_edges = _note_square_edges(adj, u2, v1, edges, n_edges)
            after = _surplus_on(adj, edges, n_edges)
            src_after = _source_on(adj, edges, n_edges, phi)
            _switch(adj, u1, v2, u2, v1)
            before = _surplus_on(adj, edges, n_edges)
            src_before = _source_on(adj, edges, n_edges, phi)
            _switch(adj, u1, v1, u2, v2)
            d_x = after - before
            d_src = src_after - src_before
        if ok:
            d_h = -ENERGY_PER_SQUARE * d_s + ENERGY_PER_SURPLUS * lam * d_x
            d_f = 0.5 * kappa * ((phi[u1] - phi[v2]) ** 2 + (phi[u2] - phi[v1]) ** 2
                                 - (phi[u1] - phi[v1]) ** 2 - (phi[u2] - phi[v2]) ** 2) + qc * d_src
            d_h += d_f
            if d_h > 0.0 and np.random.random() >= np.exp(-inv_g * d_h):
                ok = False
            elif ok:
                total += d_h
        if ok:
            accepted += 1
        else:
            _switch(adj, u1, v2, u2, v1)
    return accepted, total


# ---------------------------------------------------------------------------------------------- the field

def _project(x):
    return x - x.mean()


def field_force(adj, phi, kappa, q, src):
    """The projected force -dE_phi/d phi on the plane sum phi = 0; src = e (the curl sources) on this wiring."""
    return _project(-kappa * laplacian_apply(adj, phi) - q * src)


def leapfrog(adj, phi, pi, kappa, q, src, n_leap, eps):
    """n_leap leapfrog steps of size eps (copies; the inputs are not changed). Returns (phi, pi)."""
    phi = phi.copy()
    pi = pi + 0.5 * eps * field_force(adj, phi, kappa, q, src)
    for k in range(n_leap):
        phi = phi + eps * pi
        f = field_force(adj, phi, kappa, q, src)
        pi = pi + (eps if k < n_leap - 1 else 0.5 * eps) * f
    return phi, pi


def hmc_trajectory(adj, phi, g, kappa, q, lam, n_leap, eps, rng):
    """One hybrid Monte Carlo trajectory at fixed wiring. Returns (phi after, accepted). phi must have mean 0."""
    src = curl_sources(adj, lam)
    pi0 = _project(rng.normal(0.0, np.sqrt(g), size=phi.shape[0]))
    e0 = 0.5 * kappa * float(np.dot(phi, laplacian_apply(adj, phi))) + q * float(np.dot(src, phi))
    phi1, pi1 = leapfrog(adj, phi, pi0, kappa, q, src, n_leap, eps)
    e1 = 0.5 * kappa * float(np.dot(phi1, laplacian_apply(adj, phi1))) + q * float(np.dot(src, phi1))
    dh = (0.5 * float(np.dot(pi1, pi1)) + e1) - (0.5 * float(np.dot(pi0, pi0)) + e0)
    if dh <= 0.0 or rng.random() < np.exp(-dh / g):
        return phi1, True
    return phi, False


# ---------------------------------------------------------------------------------------------- the driver

def run_soft(adj, part, g, lam, kappa, q, n_blocks, attempts_per_block, n_leap, eps, seed, phi=None, cap=NO_CAP):
    """Alternate a block of switches and one field trajectory, n_blocks times, in place on adj. Returns
    (phi, readings): one reading per block, with the wiring's H, the field's energy, the switch and trajectory
    acceptances. seed: anything SeedSequence takes; the switch stream and the field's Generator are both drawn from it."""
    adj = np.ascontiguousarray(adj, dtype=np.int64)
    side_u = np.flatnonzero(np.asarray(part) == 0).astype(np.int64)
    chain, field = np.random.SeedSequence(seed).spawn(2)
    chain_seed = int(chain.generate_state(1)[0])
    rng = np.random.default_rng(field)
    phi = np.zeros(adj.shape[0]) if phi is None else _project(np.asarray(phi, dtype=float))
    inv_g = 1.0 / float(g)
    out = []
    for b in range(n_blocks):
        acc, _ = soft_attempts(adj, side_u, inv_g, int(attempts_per_block), chain_seed if b == 0 else -1, float(lam),
                               int(cap), phi, float(kappa), float(q))
        steps = int(rng.integers(max(1, int(n_leap) // 2), int(n_leap) + 1))
        phi, took = hmc_trajectory(adj, phi, float(g), float(kappa), float(q), float(lam), steps, float(eps), rng)
        out.append(dict(block=b + 1, H=float(hamiltonian(adj, lam)), E_phi=field_energy(adj, phi, kappa, q, lam),
                        switch_acceptance=acc / attempts_per_block if attempts_per_block else None,
                        field_accepted=bool(took)))
    return phi, out
