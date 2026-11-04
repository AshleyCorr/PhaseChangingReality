"""A chain in which the points are interchangeable (VISION Update 12; series papers 4 and 6).

WHAT CHANGES. With named points every graph has weight exp(-H/g). With interchangeable points every
*arrangement* (class of graphs up to renaming) has weight exp(-H/g), and a class of A symmetries holds
(n!)^2 / A named graphs (renamings within each side). So, per named graph, the target weight is
A(G) exp(-H/g). The move is the kernel's own (cqg.run_chain: two points of side 0 each swap one
partner; the proposal is symmetric), and the acceptance gains the factor A(G')/A(G), as in Betre and
Lewis, Eq. (72) [DQM25]:
    accept with probability min(1, exp(-dH/g) A(G')/A(G)).
Detailed balance for the target A(G) exp(-H/g) follows because the proposal is symmetric.

SEALED. With a bath of C demons (sealed.run_sealed_bath) the named target is uniform over the energy
shell H + sum(demons) = E; here it is A(G) on that shell. A move picks one demon at random, is refused
if that demon cannot pay dH, and otherwise is accepted with probability min(1, A(G')/A(G)).

COST. The symmetry count is igraph's (graphity.symmetry, about a millisecond at N <= 64) and is made for
every valid proposal, so this chain is slow and meant for N up to about 64. Energies are recomputed in
full each move (O(N)), which is simple and exact. Validated against the exact interchangeable averages
at N = 18 (tests/test_interchangeable.py).
"""
import numpy as np

from graphity import symmetry
from graphity.cqg import ENERGY_PER_SQUARE, ENERGY_PER_SURPLUS, _switch, has_edge, is_valid, surplus, total_squares


def energy(adj, lam):
    n = adj.shape[0]
    return ENERGY_PER_SQUARE * (n - total_squares(adj)) + ENERGY_PER_SURPLUS * lam * surplus(adj)


def run(adj, part, n_sweeps, seed, lam, cap, g=None, demons=None, on_sweep=None):
    """Run in place. Exactly one of g (canonical) or demons (sealed; float array, modified) is given.

    Returns per sweep: S, X, the symmetry count A, and (sealed) the demons' total. One sweep is 2N
    attempted switches, as in cqg.run_chain. `on_sweep(sweep, adj)`, if given, is called after every
    sweep and must only read the graph (T15 rung 1 identifies the arrangement with it).
    """
    if (g is None) == (demons is None):
        raise ValueError("give g for a canonical run or demons for a sealed one, not both")
    rng = np.random.default_rng(seed)
    n = adj.shape[0]
    side_u = np.flatnonzero(np.asarray(part) == 0)
    h = energy(adj, lam)
    a = symmetry.count(adj, part)
    out_s, out_x, out_a = (np.zeros(n_sweeps, dtype=np.int64) for _ in range(3))
    out_d = np.zeros(n_sweeps)
    for sweep in range(n_sweeps):
        for _ in range(2 * n):
            u1, u2 = side_u[rng.integers(len(side_u))], side_u[rng.integers(len(side_u))]
            v1, v2 = adj[u1, rng.integers(4)], adj[u2, rng.integers(4)]
            if u1 == u2 or v1 == v2 or has_edge(adj, u1, v2) or has_edge(adj, u2, v1):
                continue
            trial = adj.copy()
            _switch(trial, int(u1), int(v1), int(u2), int(v2))
            if not is_valid(trial, cap):
                continue
            h_new = energy(trial, lam)
            d_h = h_new - h
            if demons is not None:
                k = rng.integers(len(demons))
                if d_h > demons[k]:
                    continue
                a_new = symmetry.count(trial, part)
                if rng.random() < min(1.0, a_new / a):
                    adj[:] = trial
                    demons[k] -= d_h
                    h, a = h_new, a_new
            else:
                a_new = symmetry.count(trial, part)
                if rng.random() < min(1.0, np.exp(-d_h / g) * a_new / a):
                    adj[:] = trial
                    h, a = h_new, a_new
        out_s[sweep] = total_squares(adj)
        out_x[sweep] = surplus(adj)
        out_a[sweep] = a
        out_d[sweep] = demons.sum() if demons is not None else 0.0
        if on_sweep is not None:
            on_sweep(sweep, adj)
    return out_s, out_x, out_a, out_d
