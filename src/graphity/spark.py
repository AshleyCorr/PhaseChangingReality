"""Delivering energy to one place in a flat sheet: the local spark (PREREGISTRATION T26; ASSUMPTIONS Q22).

WHY. Every sealed run that gave a sheet energy shared it through a bath: one demon or a bath of stores that
any move anywhere could draw on (O20, T21). Heating a whole sheet at equilibrium randomizes it by
construction, so those runs could not ask the author's question, which is what energy packed into one
place does to cold space around it: re-curl it (her black hole) or melt it ([T25]'s). Two ways to pack
energy into a place, both protocols and neither a change to the energy:

  stores  the sheet starts perfect, and the energy sits in the stores of the vertices near a center,
          under the per-vertex bath (sealed.run_sealed_bath with by_vertex=True): only a move made
          from one of those vertices can spend it. Spendable heat, put in one place.
  patch   the energy is put into the wiring itself: uphill switches confined to the vertices near the
          center are applied until the patch holds the budget, and the run then starts in a cold,
          empty, shared bath. Disorder put in one place, with nothing loose anywhere.

Distances are graph distances in the starting flat torus, so "near the center" means the same thing for
both protocols.
"""
import numpy as np

from graphity.cqg import ENERGY_PER_SQUARE, ENERGY_PER_SURPLUS, _switch, hamiltonian, is_valid, surplus, total_squares


def distances_from(adj, center):
    """Graph distance from `center` to every vertex (breadth-first search)."""
    n = adj.shape[0]
    dist = np.full(n, -1, dtype=np.int64)
    dist[center] = 0
    frontier = [int(center)]
    while frontier:
        nxt = []
        for v in frontier:
            for w in adj[v]:
                w = int(w)
                if w >= 0 and dist[w] < 0:
                    dist[w] = dist[v] + 1
                    nxt.append(w)
        frontier = nxt
    return dist


def patch_vertices(adj, center, radius):
    """Vertices within `radius` of the center; every vertex when radius is None."""
    if radius is None:
        return np.arange(adj.shape[0])
    return np.flatnonzero(distances_from(adj, center) <= radius)


def local_stores(adj, part, center, radius, energy):
    """One store per vertex, the energy shared equally among the side-0 vertices of the patch.

    Only side-0 stores are ever drawn on by the per-vertex bath (a move pays from u1's store, and u1
    is always in side 0), so the energy is placed there.
    """
    patch = patch_vertices(adj, center, radius)
    patch = patch[np.asarray(part)[patch] == 0]
    stores = np.zeros(adj.shape[0])
    stores[patch] = float(energy) / len(patch)
    return stores


def hot_patch(adj, part, center, radius, budget, lam, cap, rng, max_tries=200000):
    """Put `budget` units of energy into the wiring near the center. Modifies adj in place.

    Uphill switches whose four vertices all lie within `radius` of the center (distances in the
    starting graph) are proposed at random and applied when valid and when the patch's energy after
    the move does not exceed the budget. Stops when the energy equals the budget exactly or when
    `max_tries` proposals in a row have not raised it. Returns the energy actually delivered, which is
    the budget unless no allowed move fits the remaining gap.
    """
    part = np.asarray(part)
    inside = patch_vertices(adj, center, radius)
    side0 = inside[part[inside] == 0]
    if len(side0) < 2:
        raise ValueError("the patch holds fewer than two side-0 vertices")
    allowed = np.zeros(adj.shape[0], dtype=bool)
    allowed[inside] = True
    h0 = hamiltonian(adj, lam)
    h = h0
    idle = 0
    while h - h0 < budget and idle < max_tries:
        idle += 1
        u1, u2 = int(side0[rng.integers(len(side0))]), int(side0[rng.integers(len(side0))])
        if u1 == u2:
            continue
        v1, v2 = int(adj[u1, rng.integers(4)]), int(adj[u2, rng.integers(4)])
        if v1 == v2 or not (allowed[v1] and allowed[v2]):
            continue
        if v2 in adj[u1] or v1 in adj[u2]:
            continue
        trial = adj.copy()
        _switch(trial, u1, v1, u2, v2)
        if not is_valid(trial, cap):
            continue
        h_new = hamiltonian(trial, lam)
        if h_new <= h or h_new - h0 > budget:
            continue
        adj[:] = trial
        h = h_new
        idle = 0
    return h - h0


def energy_of(adj, lam):
    n = adj.shape[0]
    return ENERGY_PER_SQUARE * (n - total_squares(adj)) + ENERGY_PER_SURPLUS * lam * surplus(adj)
