"""Metropolis Monte Carlo on v0-regular graphs, optionally inside a menu.

Algorithm: Metropolis [K08, Sec. IV; NB99]. Acceptance min(1, exp(-beta dE)).

Move (ASSUMPTION A5). [K08, Sec. IV] uses the two degree-preserving moves
of its Fig. 1(a),(b), which "preserve the connectedness of a graph state".
We could not read the figure, so we use the standard local edge switch that
has both properties. Take a path a - b - c - d and replace

    edges (a,b), (c,d)   ->   edges (a,c), (b,d),     edge (b,c) stays.

Degrees are unchanged. Connectedness is preserved because a, b, c, d remain
mutually connected through (a,c), (c,b), (b,d).

Detailed balance: the path is chosen by picking a directed edge (b,c)
uniformly, then a uniformly from N(b)\\{c} and d uniformly from N(c)\\{b}.
On a v0-regular graph every specific (a,b,c,d) therefore has probability
1 / (N v0 (v0-1)^2). The reverse move is the path a - c - b - d in the new
graph, which has the same probability because the graph is still regular.
The proposal is symmetric, so plain Metropolis acceptance is correct [NB99].

Menu (ASSUMPTION B1). With a menu, a move is rejected when (a,c) or (b,d)
is not an allowed edge. The reverse move only re-creates edges that were
present, hence allowed, so symmetry is kept. Whether the move set stays
ERGODIC inside a sparse menu is NOT established (open issue O2).

Sweep (ASSUMPTION A6): one sweep = N*v0/2 attempted moves, i.e. one attempt
per edge, following [CP12, Sec. IV]. [K08] does not define its time unit.
"""
import numpy as np
from numba import njit

from .energy import edge_energy, total_energy


@njit(cache=True)
def _has_edge(adj, u, v):
    for k in range(adj.shape[1]):
        if adj[u, k] == v:
            return True
    return False


@njit(cache=True)
def _replace(adj, u, old, new):
    for k in range(adj.shape[1]):
        if adj[u, k] == old:
            adj[u, k] = new
            return


@njit(cache=True)
def _in_menu(menu, u, v):
    for k in range(menu.shape[1]):
        if menu[u, k] == v:
            return True
    return False


@njit(cache=True)
def run_chain(adj, menu, use_menu, weights, beta, n_equil, n_meas, seed):
    """Run one Markov chain at inverse temperature beta. Modifies adj in place.

    Returns (energies, acceptance): total energy after each measurement
    sweep, and the fraction of attempted moves that were accepted.
    """
    np.random.seed(seed)
    n, deg = adj.shape
    l_max = weights.shape[0] - 1
    visited = np.zeros(n, dtype=np.bool_)
    path = np.zeros(l_max + 1, dtype=np.int64)
    ptr = np.zeros(l_max + 1, dtype=np.int64)
    out = np.zeros(l_max + 1, dtype=np.int64)

    energy = total_energy(adj, weights)
    energies = np.zeros(n_meas, dtype=np.float64)
    moves_per_sweep = (n * deg) // 2
    attempted = 0
    accepted = 0

    for sweep in range(n_equil + n_meas):
        for _ in range(moves_per_sweep):
            attempted += 1
            b = np.random.randint(0, n)
            kc = np.random.randint(0, deg)
            c = adj[b, kc]
            ka = np.random.randint(0, deg - 1)
            if ka >= kc:
                ka += 1
            a = adj[b, ka]
            kb = np.random.randint(0, deg - 1)
            for k in range(deg):           # skip the slot of c that holds b
                if adj[c, k] == b:
                    if kb >= k:
                        kb += 1
                    break
            d = adj[c, kb]

            if a == d or a == c or b == d:
                continue
            if _has_edge(adj, a, c) or _has_edge(adj, b, d):
                continue
            if use_menu and not (_in_menu(menu, a, c) and _in_menu(menu, b, d)):
                continue

            # energy lost: cycles through (a,b), then through (c,d) once (a,b) is gone
            loss = edge_energy(adj, a, b, weights, visited, path, ptr, out)
            _replace(adj, a, b, -1)
            _replace(adj, b, a, -1)
            loss += edge_energy(adj, c, d, weights, visited, path, ptr, out)
            _replace(adj, c, d, -1)
            _replace(adj, d, c, -1)
            # energy gained: cycles through (a,c), then through (b,d)
            _replace(adj, a, -1, c)
            _replace(adj, c, -1, a)
            gain = edge_energy(adj, a, c, weights, visited, path, ptr, out)
            _replace(adj, b, -1, d)
            _replace(adj, d, -1, b)
            gain += edge_energy(adj, b, d, weights, visited, path, ptr, out)

            d_e = gain - loss
            if d_e <= 0.0 or np.random.random() < np.exp(-beta * d_e):
                energy += d_e
                accepted += 1
            else:                          # undo
                _replace(adj, a, c, b)
                _replace(adj, c, a, d)
                _replace(adj, b, d, a)
                _replace(adj, d, b, c)
        if sweep >= n_equil:
            energies[sweep - n_equil] = energy
    return energies, accepted / max(attempted, 1)


def dummy_menu():
    """Placeholder passed when the menu is the complete graph."""
    return np.full((1, 1), -1, dtype=np.int64)
