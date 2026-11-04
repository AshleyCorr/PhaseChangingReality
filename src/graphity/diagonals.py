"""Body diagonals derived from the cube edges, and light on them (docs/design/diagonals_brief_2026-10-28.md).

The owner's decisions of 28 October (VISION Updates 142, 143): gravity travels through the cube edges, whose lengths
relax from curl (graphity.lengths); light travels through the body diagonals; a body diagonal's length is DERIVED from its
cube's edges by geometry, sqrt(lx^2 + ly^2 + lz^2), with no new constant; a photon spends one tick per unit of a
diagonal's length. Here, ours:

* A cube is named by its lower corner c. Its edges along axis a are the four axis links at the lower points c + {0, e_b}
  + {0, e_c} (b, c the other two axes). Its edge length along a is the MEAN of those four, 1 + mean(delta).
* The eight body diagonals of a cube all have the length sqrt(lx^2 + ly^2 + lz^2) of that cube. A body-diagonal step
  from point p with signs s belongs to the cube with lower corner p + min(s, 0).
* First order: one edge lengthened by d lengthens each of the four cubes it bounds by d/4 along its axis, and their
  diagonals by (d/4) / sqrt(3)  (d sqrt(sum l^2) / d lx = lx / sqrt(3) at the flat point). A test checks this.
* The light reader: the shortest path in ticks over body-diagonal steps between two points (Dijkstra), ticks being the
  summed diagonal lengths; the delay past a knot is the difference between that path and the same path far from it.
"""
import heapq
from itertools import product

import numpy as np

SIGNS = [np.array(s) for s in product((1, -1), repeat=3)]


def cube_edge_lengths(delta):
    """(3, L, L, L): for each cube (by lower corner) and axis a, 1 + the mean of delta over its four edges along a."""
    out = np.empty_like(delta)
    for a in range(3):
        b, c = [x for x in range(3) if x != a]
        d = delta[a]
        out[a] = 1.0 + (d + np.roll(d, -1, b) + np.roll(d, -1, c) + np.roll(np.roll(d, -1, b), -1, c)) / 4.0
    return out


def diagonal_lengths(delta):
    """(L, L, L): the body-diagonal length of each cube, sqrt(lx^2 + ly^2 + lz^2)."""
    l = cube_edge_lengths(delta)
    return np.sqrt((l ** 2).sum(axis=0))


def light_ticks(diag, src, dst):
    """Shortest path in ticks over body-diagonal steps from src to dst (tuples), on the torus of diag.shape."""
    dims = np.array(diag.shape)
    src, dst = tuple(np.mod(src, dims)), tuple(np.mod(dst, dims))
    dist = {src: 0.0}
    heap = [(0.0, src)]
    while heap:
        d, p = heapq.heappop(heap)
        if p == dst:
            return d
        if d > dist[p]:
            continue
        pa = np.array(p)
        for s in SIGNS:
            cube = tuple(np.mod(pa + np.minimum(s, 0), dims))
            q = tuple(np.mod(pa + s, dims))
            nd = d + diag[cube]
            if nd < dist.get(q, np.inf):
                dist[q] = nd
                heapq.heappush(heap, (nd, q))
    raise RuntimeError("unreachable")
