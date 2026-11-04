"""Cycle energy of the basic graphity model, restricted to v0-regular graphs.

Model definition [K08, Sec. II A, Eqs. (9)-(12)]:

    E_B = sum_a E_B(a),   E_B(a) = - sum_{L=3}^{Lmax} g_B(L) P(a, L),
    g_B(L) = g_B * r**L / L!

P(a, L) is the number of cycles of length L through vertex a.

ASSUMPTION A3 (cycle-counting convention; see ASSUMPTIONS.md).
[K08] defines a cycle as a closed walk that repeats no edge. On a 3-regular
graph such a walk cannot revisit a vertex (that would need degree >= 4), so
cycles are simple cycles. [K08] does not state whether the two orientations
of a cycle count separately. We count them separately (multiplicity 2),
because only that choice reproduces the published ground-state energy per
node, epsilon_0 = -12.2 at r = -2.5 [K08, Sec. IV A 2]: the ring of
(K_{3,3} minus one edge) units gives -12.207 with multiplicity 2 and -6.104
with multiplicity 1. This is checked in tests/test_energy.py.

With that convention each simple cycle of length L contributes

    w(L) = - ORIENTATIONS * L * g_B * r**L / L!

to the total energy (it passes through L vertices, in 2 orientations).
"""
import math

import numpy as np
from numba import njit

ORIENTATIONS = 2  # ASSUMPTION A3


def cycle_weights(r: float, l_max: int, g_b: float = 1.0) -> np.ndarray:
    """Energy contributed by ONE simple cycle of each length L (index = L)."""
    w = np.zeros(l_max + 1, dtype=np.float64)
    for L in range(3, l_max + 1):
        w[L] = -ORIENTATIONS * L * g_b * r**L / math.factorial(L)
    return w


@njit(cache=True)
def count_cycles_through_edge(adj, u, v, l_max, visited, path, ptr, out):
    """Count simple cycles, by length, that contain the edge (u, v).

    adj      : (N, v0) int array of neighbours; -1 marks an empty slot.
    out[L]   : number of simple cycles of length L (3 <= L <= l_max) through
               (u, v). Each undirected cycle is counted exactly once.
    visited, path, ptr : preallocated scratch arrays (N, l_max+1, l_max+1).

    Method: depth-first enumeration of simple paths v -> u of length L-1 that
    do not use the edge (v, u) itself.
    """
    for L in range(l_max + 1):
        out[L] = 0
    deg = adj.shape[1]
    depth = 0
    path[0] = v
    ptr[0] = 0
    visited[v] = True
    while depth >= 0:
        if ptr[depth] == deg:
            visited[path[depth]] = False
            depth -= 1
            continue
        w = adj[path[depth], ptr[depth]]
        ptr[depth] += 1
        if w < 0:
            continue
        if w == u:
            if depth > 0:            # depth == 0 is the edge (v, u) itself
                out[depth + 2] += 1  # depth+1 path edges, plus the edge (u, v)
            continue
        if visited[w]:
            continue
        if depth + 3 <= l_max:       # room for w and at least one more step
            depth += 1
            path[depth] = w
            ptr[depth] = 0
            visited[w] = True


@njit(cache=True)
def edge_energy(adj, u, v, weights, visited, path, ptr, out):
    """Total energy of all cycles through edge (u, v)."""
    l_max = weights.shape[0] - 1
    count_cycles_through_edge(adj, u, v, l_max, visited, path, ptr, out)
    e = 0.0
    for L in range(3, l_max + 1):
        e += weights[L] * out[L]
    return e


@njit(cache=True)
def total_energy(adj, weights):
    """Total cycle energy, computed from scratch.

    Every cycle of length L is seen once from each of its L edges, so we sum
    edge contributions divided by L.
    """
    n, deg = adj.shape
    l_max = weights.shape[0] - 1
    visited = np.zeros(n, dtype=np.bool_)
    path = np.zeros(l_max + 1, dtype=np.int64)
    ptr = np.zeros(l_max + 1, dtype=np.int64)
    out = np.zeros(l_max + 1, dtype=np.int64)
    e = 0.0
    for u in range(n):
        for k in range(deg):
            v = adj[u, k]
            if v > u:
                count_cycles_through_edge(adj, u, v, l_max, visited, path, ptr, out)
                for L in range(3, l_max + 1):
                    e += weights[L] * out[L] / L
    return e


@njit(cache=True)
def cycle_census(adj, l_max):
    """Number of simple cycles of each length in the whole graph."""
    n, deg = adj.shape
    visited = np.zeros(n, dtype=np.bool_)
    path = np.zeros(l_max + 1, dtype=np.int64)
    ptr = np.zeros(l_max + 1, dtype=np.int64)
    out = np.zeros(l_max + 1, dtype=np.int64)
    tot = np.zeros(l_max + 1, dtype=np.int64)
    for u in range(n):
        for k in range(deg):
            v = adj[u, k]
            if v > u:
                count_cycles_through_edge(adj, u, v, l_max, visited, path, ptr, out)
                for L in range(3, l_max + 1):
                    tot[L] += out[L]
    for L in range(3, l_max + 1):
        tot[L] //= L
    return tot
