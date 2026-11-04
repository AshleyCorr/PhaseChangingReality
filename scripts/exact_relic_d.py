"""EXACT: the smallest curled relic inside flat six-link space, its cost, whether it is a dip, and the counting pull
between two of them at fixed wiring. Usage:

    python scripts/exact_relic_d.py [L] [lambda]

Written 2026-09-25 for the gravity design (series paper 5; VISION Update 28). A relic is made by one bipartite switch
on the flat L x L x L torus: the edges (v+3x, v+4x) and (v-x, v) become (v+3x, v) and (v-x, v+4x), which closes the
line v .. v+3x into a 4-cycle along x (a column curled to length 4) and lets the line jump from v-x to v+4x. Its
energy above flat space is read exactly; it is a DIP if every single switch out of it costs energy (listed by the
nearby-partner search of exact_walls_d.py, which is exact for the walls). Then two such relics are placed at every
separation along x, y and a diagonal, and the number of symmetries of each arrangement is counted (graphity.symmetry,
igraph): with interchangeable points that count is the arrangement's weight, so a count that rises as the relics
approach is a pull from counting at fixed wiring, and a flat count is none (ASSUMPTIONS O32 asked this in 2D).
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graphity import symmetry                                       # noqa: E402
from graphity.cqg_d import _switch, is_valid, torus                 # noqa: E402
from graphity.sealed_d import energy_d                              # noqa: E402
from exact_walls_d import walls                                     # noqa: E402


def index(dims):
    n = int(np.prod(dims))
    coords = np.array(np.unravel_index(np.arange(n), dims)).T
    look = {tuple(int(c) for c in cc): i for i, cc in enumerate(coords)}
    return coords, look


def at(look, dims, c):
    return look[tuple(int(x) % d for x, d in zip(c, dims))]


def make_relic(adj, dims, look, origin, axis=0):
    """Curl the column origin .. origin+3 along `axis` into a 4-cycle. Modifies adj; returns the four vertices."""
    o = list(origin)
    def shift(k):
        c = list(o); c[axis] += k; return at(look, dims, c)
    v, v3, v4, vm = shift(0), shift(3), shift(4), shift(-1)
    # switch (v3, v4), (vm, v) -> (v3, v), (vm, v4): u1 = v3, v1 = v4, u2 = vm, v2 = v   (v3 and vm on one side)
    _switch(adj, v3, v4, vm, v)
    return [shift(k) for k in range(4)]


def main(L=8, lam=1.02):
    dims = [L, L, L]
    coords, look = index(dims)
    adj, part = torus(dims)
    n = adj.shape[0]
    base = energy_d(adj, lam)
    one = adj.copy()
    col = make_relic(one, dims, look, (0, 0, 0))
    assert is_valid(one), "the relic switch breaks the hard-core rule"
    e1 = energy_d(one, lam) - base
    print("flat %d x %d x %d, lambda = %.3f. One relic (a column curled along x): energy above flat %.3f "
          "(= %s at this lambda)" % (L, L, L, lam, e1, "24 lam - 16" if abs(e1 - (24 * lam - 16)) < 1e-9 else "not 24 lam - 16"))
    # is it a dip? cheapest moves out, from each of the four relic vertices and their neighbors
    cheapest = None
    side0 = np.flatnonzero(part == 0)
    seen = set()
    for v in col + [int(w) for v in col for w in one[v]]:
        if part[v] != 0 or v in seen:
            continue
        seen.add(int(v))
        w = walls(one, part, lam, u1=int(v), top=1, near=3)
        if w and (cheapest is None or w[0][0] < cheapest[0]):
            cheapest = w[0]
    print("cheapest single move out of the relic (from its vertices or their neighbors): cost %.3f (dS %d, dX %d) -> %s"
          % (cheapest[0], cheapest[1], cheapest[2], "a DIP" if cheapest[0] > 0 else "NOT a dip"))
    a1 = symmetry.count(one, part)
    print("symmetries: flat %d, one relic %d" % (symmetry.count(adj, part), a1))
    # two relics at separations along x, y, and the (1,1,0) diagonal
    print("two relics: separation, energy above flat, symmetries (interchangeable weight); additivity means 2 x %.3f = %.3f" % (e1, 2 * e1))
    for label, step in (("along x", (1, 0, 0)), ("along y", (0, 1, 0)), ("along z", (0, 0, 1)), ("diagonal xy", (1, 1, 0))):
        rows = []
        for r in range(1, L):
            two = adj.copy()
            make_relic(two, dims, look, (0, 0, 0))
            origin = tuple(r * s for s in step)
            try:
                make_relic(two, dims, look, origin)
            except Exception:
                continue
            if not is_valid(two):
                rows.append((r, None, None)); continue
            rows.append((r, energy_d(two, lam) - base, symmetry.count(two, part)))
        print("  %-12s " % label + "  ".join(("r=%d: %s" % (r, "invalid" if e is None else "%.2f, A=%d" % (e, a))) for r, e, a in rows))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8, float(sys.argv[2]) if len(sys.argv) > 2 else 1.02)
