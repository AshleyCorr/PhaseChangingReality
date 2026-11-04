"""The owner's square-count rule in its soft form, at any number of links: H_soft = H + mu (S - S0)^2.

Decided by the owner on 27 October 2026 ("proceed w soften", VISION Update 133), in place of her hard rule of 25 October
(S held at exactly S0 = D(D-1)/2 N, which is N at four links and 3N at six; a move allowed only if it keeps S). The hard
rule froze clean space: no count-keeping move out of flat six-link space has been found at up to three switches (Update
128, O159, O163). The soft form lets S drift from S0 at a price that grows with the square of the drift. It is a new
energy term (S1): mu is scanned, never tuned, and every value run is published. Whether mu is held fixed or scales as
mu0 / N is fixed in each registration, not here: this module takes mu as a number.

H is cqg_d's energy, unchanged: H = 16 (S0 - S) + 4 lam X (ASSUMPTIONS Q21). Written per link it is
H = 4 uncurl + 4 (lam - 1) curl, with curl = X, the squares above 2D - 2 on a link, and uncurl the squares missing below
it, so S - S0 = (curl - uncurl) / 4 (ours, arithmetic; checked on the flat, slab and rod tori in tests/test_soft_d.py).

Move and acceptance: cqg_d.run_chain's, unchanged, with dH gaining the penalty's change. The penalty depends on the
state alone, so dH_soft = dH + mu ((S + dS - S0)^2 - (S - S0)^2) = dH + mu dS (2 (S - S0) + dS) is exact, and with the
same symmetric switch proposal Metropolis and Glauber acceptance are detailed-balanced for exp(-H_soft / g) by the
argument of cqg.py's docstring. At mu = 0 the chain is cqg_d.run_chain draw for draw (tests/test_soft_d.py).
cqg_d.py is left untouched, so every result already run on it is safe by construction.
"""
import numpy as np
from numba import njit

from graphity.cqg_d import (ENERGY_PER_SQUARE, ENERGY_PER_SURPLUS, _new_edge_ok, _note_square_edges, _replace,
                            _surplus_on, _switch, has_edge, hamiltonian, squares_on_edge, surplus, total_squares)


@njit(cache=True)
def target(adj):
    """S0 = D(D-1)/2 N: the squares of flat space, N at four links and 3N at six."""
    dim = adj.shape[1] // 2
    return dim * (dim - 1) // 2 * adj.shape[0]


@njit(cache=True)
def hamiltonian_soft(adj, lam, mu):
    """H + mu (S - S0)^2, computed from scratch."""
    d = total_squares(adj) - target(adj)
    return hamiltonian(adj, lam) + mu * d * d


@njit(cache=True)
def run_chain(adj, side_u, inv_g, n_equil, n_meas, seed, lam=1.0, mu=0.0, glauber=False):
    """Markov chain for H_soft at coupling g = 1 / inv_g. Modifies adj in place.

    As cqg_d.run_chain: seed < 0 carries on the stream; one sweep = 2N attempted switches. Returns (S after each
    measurement sweep, X after each, acceptance rate).
    """
    if seed >= 0:
        np.random.seed(seed)
    n, deg = adj.shape
    nu = side_u.shape[0]
    edges = np.empty((4 * (1 + 3 * (deg - 1)) + 8, 2), dtype=np.int64)
    s0 = target(adj)
    s = total_squares(adj)
    x = surplus(adj)
    out_s = np.zeros(n_meas, dtype=np.int64)
    out_x = np.zeros(n_meas, dtype=np.int64)
    attempted = 0
    accepted = 0
    for sweep in range(n_equil + n_meas):
        for _ in range(2 * n):
            attempted += 1
            u1 = side_u[np.random.randint(0, nu)]
            v1 = adj[u1, np.random.randint(0, deg)]
            u2 = side_u[np.random.randint(0, nu)]
            v2 = adj[u2, np.random.randint(0, deg)]
            if u1 == u2 or v1 == v2 or has_edge(adj, u1, v2) or has_edge(adj, u2, v1):
                continue
            n_edges = 0
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
            ok = _new_edge_ok(adj, u1, v2) and _new_edge_ok(adj, u2, v1)
            if ok:
                n_edges = _note_square_edges(adj, u1, v2, edges, n_edges)
                n_edges = _note_square_edges(adj, u2, v1, edges, n_edges)
                after = _surplus_on(adj, edges, n_edges)
                _switch(adj, u1, v2, u2, v1)            # back to the old graph
                before = _surplus_on(adj, edges, n_edges)
                _switch(adj, u1, v1, u2, v2)            # forward again
                d_x = after - before
                d_h = (-ENERGY_PER_SQUARE * d_s + ENERGY_PER_SURPLUS * lam * d_x
                       + mu * d_s * (2 * (s - s0) + d_s))
                if glauber:
                    if np.random.random() >= 1.0 / (1.0 + np.exp(inv_g * d_h)):
                        ok = False
                elif d_h > 0.0 and np.random.random() >= np.exp(-inv_g * d_h):
                    ok = False
            if ok:
                s += d_s
                x += d_x
                accepted += 1
            else:
                _switch(adj, u1, v2, u2, v1)
        if sweep >= n_equil:
            out_s[sweep - n_equil] = s
            out_x[sweep - n_equil] = x
    return out_s, out_x, accepted / max(attempted, 1)
