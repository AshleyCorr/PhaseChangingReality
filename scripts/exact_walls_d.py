"""EXACT: the cheapest single moves out of a curled torus (or a gas of them) at any dimension, by brute force. Usage:

    python scripts/exact_walls_d.py 1.25 4,4,18 4,4,32 4,4,4x8 4,4,4,36 4,4,4,4x9 4,4,12,12 4,6,8,12

Every side-0 vertex of a torus is equivalent under translation (and every cube of a gas under permutation), so
one u1 suffices; u2 runs over every side-0 vertex, and v1, v2 over the neighbor slots. Each valid switch is
priced at -16 dS + 4 lam dX (ASSUMPTIONS Q21) and the cheapest kinds are listed with how many ways each is
offered from that one u1. Each switch is seen from both of its side-0 vertices, so the total number of switches of a
kind is (ways from one u1) x (side-0 vertices) / 2: paper 1's move A shows 12 from one u1 of the 16 x 4 torus, 3N in all. "AxBxC x k" means a gas of
k separate tori. Written 2026-09-25 for T30's and T33's pre-registrations (O49, O50); checked against O41's
4 x 4 x 6 and paper 1's 16 x 4 (tests/test_exact_walls_d.py).
"""
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
from graphity.cqg_d import _switch, is_valid, surplus, torus, total_squares   # noqa: E402


def gas(dims, copies):
    a, p = torus(dims)
    n = a.shape[0]
    return np.concatenate([a + k * n for k in range(copies)]), np.concatenate([p] * copies)


def within(adj, center, radius):
    """Vertices within graph distance `radius` of `center` (breadth-first search)."""
    dist = {int(center): 0}
    frontier = [int(center)]
    for _ in range(radius):
        nxt = []
        for v in frontier:
            for w in adj[v]:
                w = int(w)
                if w not in dist:
                    dist[w] = dist[v] + 1
                    nxt.append(w)
        frontier = nxt
    return np.array(sorted(dist))


def walls(adj, part, lam, u1=None, top=5, near=None):
    """[(cost, dS, dX, ways from one u1)] sorted by cost.

    near: if given, u2 runs only over side-0 vertices within that graph distance of u1. A switch whose two
    edges share no square loses every square on both, the dearest kind, so the cheapest moves are always among
    near partners; near = 4 is safe for the walls (checked against the full search on the 3D tori)."""
    n, deg = adj.shape
    s0, x0 = int(total_squares(adj)), int(surplus(adj))
    side0 = np.flatnonzero(part == 0)
    u1 = int(side0[0]) if u1 is None else int(u1)
    if near is not None:
        close = set(within(adj, u1, near).tolist())
        side0 = np.array([v for v in side0 if int(v) in close])
    kinds = Counter()
    nbr1 = set(int(v) for v in adj[u1])
    for u2 in side0:
        u2 = int(u2)
        if u2 == u1:
            continue
        nbr2 = set(int(v) for v in adj[u2])
        for v1 in adj[u1]:
            v1 = int(v1)
            if v1 in nbr2:
                continue
            for v2 in adj[u2]:
                v2 = int(v2)
                if v2 == v1 or v2 in nbr1:
                    continue
                t = adj.copy()
                _switch(t, u1, v1, u2, v2)
                if not is_valid(t):
                    continue
                kinds[(int(total_squares(t)) - s0, int(surplus(t)) - x0)] += 1
    out = sorted(((-16.0 * ds + 4.0 * lam * dx, ds, dx, c) for (ds, dx), c in kinds.items()))
    return out[:top]


def parse(spec):
    if "x" in spec:
        dims, k = spec.split("x")
        return gas([int(d) for d in dims.split(",")], int(k)), spec
    return torus([int(d) for d in spec.split(",")]), spec


def main(argv):
    near = None
    if argv and argv[0].startswith("--near="):
        near = int(argv[0].split("=")[1])
        argv = argv[1:]
    lam = float(argv[0])
    for spec in argv[1:]:
        (adj, part), label = parse(spec)
        n = adj.shape[0]
        w = walls(adj, part, lam, near=near)
        print("%-14s N=%-5d D=%d  cheapest at lambda=%.2f: %s" % (label, n, adj.shape[1] // 2, lam,
              "; ".join("%.1f (dS %d, dX %d, %d ways)" % (c, ds, dx, k) for c, ds, dx, k in w)))


if __name__ == "__main__":
    main(sys.argv[1:])
