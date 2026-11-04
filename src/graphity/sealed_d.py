"""Sealed runs at any dimension D: the bath of stores of sealed.py on the 2D-regular graphs of cqg_d.py.

WHY. sealed.py holds the energy-conserving dynamics (Creutz demons [Creutz83]; ASSUMPTIONS Q12, Q22) with four
links per point wired in through cqg.py's helpers. The six-link work of VISION Update 22 needs the same dynamics
on the graphs of cqg_d.py. This module is sealed.run_sealed_bath transcribed onto cqg_d's helpers, with the
number of links read from the array, and nothing else changed: the same draws in the same order, the same
store rule, the same conservation. tests/test_sealed_d.py checks that at four links it makes exactly the chain
of sealed.run_sealed_bath without the cap, draw for draw, and that at six links H + stores is conserved to the
last unit.

THE RULE, as sealed.run_sealed_bath. A proposed switch picks one store (at random, or the store of u1 when
by_vertex is set), is refused if that store cannot pay dH, and otherwise pays or is paid. seed < 0 carries the
random stream on from the previous call (Q14).
"""
import numpy as np
from numba import njit

from graphity.cqg_d import (ENERGY_PER_SQUARE, ENERGY_PER_SURPLUS, _new_edge_ok, _note_square_edges, _replace,
                            _surplus_on, _switch, has_edge, squares_on_edge, surplus, total_squares)


@njit(cache=True)
def run_sealed_bath_d(adj, side_u, demons, n_sweeps, seed, lam=1.0, by_vertex=False):
    """Sealed run with a bath of stores at the dimension of adj (adj.shape[1] = 2D). Modifies adj and demons.

    Returns (S after each sweep, X after each sweep, mean store energy, total store energy, acceptance rate).
    One sweep = 2N attempted switches.
    """
    if seed >= 0:
        np.random.seed(seed)
    n, deg = adj.shape
    nu = side_u.shape[0]
    edges = np.empty((4 * (1 + 3 * (deg - 1)) + 8, 2), dtype=np.int64)
    s = total_squares(adj)
    x = surplus(adj)
    c = demons.shape[0]
    out_s = np.zeros(n_sweeps, dtype=np.int64)
    out_x = np.zeros(n_sweeps, dtype=np.int64)
    out_mean = np.zeros(n_sweeps, dtype=np.float64)
    out_sum = np.zeros(n_sweeps, dtype=np.float64)
    attempted = 0
    accepted = 0
    for sweep in range(n_sweeps):
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
                _switch(adj, u1, v2, u2, v1)
                before = _surplus_on(adj, edges, n_edges)
                _switch(adj, u1, v1, u2, v2)
                d_x = after - before
                d_h = -ENERGY_PER_SQUARE * d_s + ENERGY_PER_SURPLUS * lam * d_x
                if by_vertex:
                    k = u1
                else:
                    k = 0 if c == 1 else np.random.randint(0, c)
                if d_h > demons[k]:
                    ok = False
                else:
                    demons[k] -= d_h
            if ok:
                s += d_s
                x += d_x
                accepted += 1
            else:
                _switch(adj, u1, v2, u2, v1)
        out_s[sweep] = s
        out_x[sweep] = x
        out_sum[sweep] = demons.sum()
        out_mean[sweep] = out_sum[sweep] / c
    return out_s, out_x, out_mean, out_sum, accepted / max(attempted, 1)


def energy_d(adj, lam):
    """H = 16 (D(D-1)/2 N - S) + 4 lam X at the dimension of adj."""
    n, deg = adj.shape
    d = deg // 2
    return ENERGY_PER_SQUARE * (d * (d - 1) // 2 * n - total_squares(adj)) + ENERGY_PER_SURPLUS * lam * surplus(adj)
