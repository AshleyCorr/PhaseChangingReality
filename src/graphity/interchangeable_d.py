"""Interchangeable points at any number of links (the six-link black-hole test; VISION Update 27; ASSUMPTIONS O55).

WHAT IT IS. `interchangeable.py`'s chain, written for the kernel of `cqg_d.py` (any 2D links per point), with the move
priced incrementally instead of by full recomputation. With interchangeable points every arrangement (class of graphs up
to side-preserving renaming) has weight exp(-H/g), so per named graph the target weight is A(G) exp(-H/g), A the number of
side-preserving symmetries (ASSUMPTIONS Q15). The move is the kernel's switch (u1,v1),(u2,v2) -> (u1,v2),(u2,v1), drawn
exactly as cqg_d.run_chain draws it, so the proposal is symmetric; the acceptance gains the factor A(G')/A(G) (Betre and
Lewis, Eq. (72) [DQM25]):
    canonical:  accept with probability min(1, exp(-dH/g) A(G')/A(G));
    sealed:     pick one of the C demons at random; refuse if it cannot pay dH; otherwise accept with probability
                min(1, A(G')/A(G)), and the demon pays.
Detailed balance for A(G) exp(-H/g) (canonical) or A(G) on the energy shell H + sum(demons) = E (sealed) follows from the
symmetric proposal, as in `interchangeable.py` and `sealed.py`.

COST. The symmetry count (igraph, `graphity.symmetry`) is made only for a proposal that keeps the hard-core rule and, when
sealed, that the drawn demon can pay; everything else is priced in compiled code. In a cold sealed box almost every
proposal fails the demon, so the counts are few.

VALIDATION (tests/test_interchangeable_d.py). At four links it reproduces the exact interchangeable averages at N = 18
(which differ from the named ones there, so the test can fail); energy bookkeeping is exact against full recomputation;
a sealed run conserves H + demons to the last bit; the hard-core rule, the degrees and the sides are preserved; the same
seed gives the same run. Random numbers: one numpy Generator per run, seeded once (ASSUMPTIONS Q14).
"""
import numpy as np
from numba import njit

from graphity import symmetry
from graphity.cqg_d import (ENERGY_PER_SQUARE, ENERGY_PER_SURPLUS, _new_edge_ok, _note_square_edges, _replace,
                            _surplus_on, _switch, has_edge, hamiltonian, squares_on_edge, surplus, total_squares)


@njit(cache=True)
def try_switch(adj, u1, v1, u2, v2, lam, edges):
    """Apply the switch if it keeps the hard-core rule and return (ok, dS, dX, dH).

    ok False: adj is unchanged. ok True: adj holds the new graph; `undo(adj, u1, v1, u2, v2)` restores the old one.
    The pricing is cqg_d.run_chain's, line for line.
    """
    if u1 == u2 or v1 == v2 or has_edge(adj, u1, v2) or has_edge(adj, u2, v1):
        return False, 0, 0, 0.0
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
    if not (_new_edge_ok(adj, u1, v2) and _new_edge_ok(adj, u2, v1)):
        _switch(adj, u1, v2, u2, v1)
        return False, 0, 0, 0.0
    n_edges = _note_square_edges(adj, u1, v2, edges, n_edges)
    n_edges = _note_square_edges(adj, u2, v1, edges, n_edges)
    after = _surplus_on(adj, edges, n_edges)
    _switch(adj, u1, v2, u2, v1)
    before = _surplus_on(adj, edges, n_edges)
    _switch(adj, u1, v1, u2, v2)
    d_x = after - before
    return True, d_s, d_x, -ENERGY_PER_SQUARE * d_s + ENERGY_PER_SURPLUS * lam * d_x


@njit(cache=True)
def undo(adj, u1, v1, u2, v2):
    _switch(adj, u1, v2, u2, v1)


def run(adj, part, n_sweeps, seed, lam, g=None, demons=None, weighted=True, on_sweep=None, rng=None):
    """Run in place. Exactly one of g (canonical) or demons (sealed; float array, modified) is given.

    weighted=False drops the symmetry factor (named points, the same chain otherwise), for controls and tests.
    rng: a numpy Generator to carry on (a run read in blocks passes the same one each time); otherwise one is seeded.
    Returns per sweep: S, X, A (the symmetry count; 0 when unweighted) and the demons' total.
    One sweep is 2N attempted switches, as in cqg_d.run_chain.
    """
    if (g is None) == (demons is None):
        raise ValueError("give g for a canonical run or demons for a sealed one, not both")
    if rng is None:
        rng = np.random.default_rng(seed)
    n, deg = adj.shape
    part = np.asarray(part)
    side_u = np.flatnonzero(part == 0)
    edges = np.empty((4 * (1 + 3 * (deg - 1)) + 8, 2), dtype=np.int64)
    a = symmetry.count(adj, part) if weighted else 0
    s, x = int(total_squares(adj)), int(surplus(adj))
    out_s, out_x, out_a = (np.zeros(n_sweeps, dtype=np.int64) for _ in range(3))
    out_d = np.zeros(n_sweeps)
    for sweep in range(n_sweeps):
        for _ in range(2 * n):
            u1 = int(side_u[rng.integers(len(side_u))])
            v1 = int(adj[u1, rng.integers(deg)])
            u2 = int(side_u[rng.integers(len(side_u))])
            v2 = int(adj[u2, rng.integers(deg)])
            ok, d_s, d_x, d_h = try_switch(adj, u1, v1, u2, v2, lam, edges)
            if not ok:
                continue
            if demons is not None:
                k = rng.integers(len(demons))
                if d_h > demons[k]:
                    undo(adj, u1, v1, u2, v2)
                    continue
                a_new = symmetry.count(adj, part) if weighted else 0
                if weighted and rng.random() >= min(1.0, a_new / a):
                    undo(adj, u1, v1, u2, v2)
                    continue
                demons[k] -= d_h
            else:
                a_new = symmetry.count(adj, part) if weighted else 0
                p = np.exp(-d_h / g) * (a_new / a if weighted else 1.0)
                if rng.random() >= min(1.0, p):
                    undo(adj, u1, v1, u2, v2)
                    continue
            s += d_s
            x += d_x
            a = a_new
        out_s[sweep], out_x[sweep], out_a[sweep] = s, x, a
        out_d[sweep] = demons.sum() if demons is not None else 0.0
        if on_sweep is not None:
            on_sweep(sweep, adj)
    return out_s, out_x, out_a, out_d


def energy(adj, lam):
    return float(hamiltonian(adj, lam))
