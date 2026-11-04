"""What can a second curl do above lambda = 1? Exact, and no chain is run. Usage:

    python scripts/exact_knot_descent.py [lambda]

EXACT and exploratory (ASSUMPTIONS O110, written 2026-10-10 after the owner's idea that the long-waiting tubes of
T38 had curled a second direction and stored energy there). In the four-link model a second curl of a stretch of
tube is a knot: sixteen points forming a 4-cube, every point with both directions closing squares (d = 0). Two
questions, both answered by listing switches:

1. Beside a tube. The state "12 x 4 tube + one 4-cube" (64 points) is what a 16 x 4 tube would be had sixteen of
   its points curled a second time. Every valid single switch out of it is priced and sorted by where its two
   links sit: inside the knot, inside the tube, or one in each (which joins the two pieces again).
2. Alone. From a lone 4-cube, every switch that lowers the energy is followed until none is left, and the levels
   passed through are listed with their square counts and their points at each d.

Energy H = 16 (N - S) + 4 lam X as everywhere (ASSUMPTIONS Q1); the move and the hard-core rule are the kernel's,
through scripts/exact_exits_d.py, whose D = 2 census is checked against paper 1's exits in its own tests.
"""
import sys
from collections import Counter, deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from exact_exits_d import _has, _locally_valid, _replace, disjoint, s_and_x, torus   # noqa: E402
from graphity.connectivity import connectivity                                        # noqa: E402
from graphity.dimension import local_dimension                                        # noqa: E402

SAT = 2                                             # squares an edge carries before any counts as surplus


def cost(ds, dx, lam):
    """Energy change of a switch that changes the squares by ds and the surplus squares by dx."""
    return -16.0 * ds + 4.0 * lam * dx


def switches(adj, side_u):
    """(u1, u2, new graph) for every valid switch, one per unordered switch."""
    pts = np.empty(4, dtype=np.int64)
    for i in range(len(side_u)):
        u1 = int(side_u[i])
        for j in range(i + 1, len(side_u)):
            u2 = int(side_u[j])
            for a in range(4):
                v1 = int(adj[u1, a])
                for b in range(4):
                    v2 = int(adj[u2, b])
                    if v1 == v2 or _has(adj, u1, v2) or _has(adj, u2, v1):
                        continue
                    new = adj.copy()
                    new[u1, a] = v2
                    new[u2, b] = v1
                    _replace(new, v1, u1, u2)
                    _replace(new, v2, u2, u1)
                    pts[:] = (u1, u2, v1, v2)
                    if _locally_valid(new, pts):
                        yield u1, u2, new


def beside_a_tube():
    """Counter of (where, dS, dX, pieces afterwards) over every switch out of 12 x 4 tube + 4-cube, and the
    (S, X) of that state and of the 16 x 4 tube."""
    tube, side_t = torus((12, 4))
    knot, side_k = torus((4, 4))
    adj, side = disjoint([(tube, side_t), (knot, side_k)])
    n_tube = tube.shape[0]
    s0, x0 = s_and_x(adj, SAT)
    kinds = Counter()
    for u1, u2, new in switches(adj, np.flatnonzero(side == 0)):
        where = "knot" if min(u1, u2) >= n_tube else ("tube" if max(u1, u2) < n_tube else "join")
        s1, x1 = s_and_x(new, SAT)
        kinds[(where, int(s1 - s0), int(x1 - x0), int(connectivity(new)[0]))] += 1
    whole, _ = torus((16, 4))
    return kinds, (int(s0), int(x0)), tuple(int(v) for v in s_and_x(whole, SAT))


def downhill_from_a_lone_knot(lam):
    """Levels (S, X, points at d = 0..3) reached from a 4-cube by switches that lower the energy, each with the
    number of labelled states reached and how many of them have no downhill switch left."""
    knot, side = torus((4, 4))
    side_u = np.flatnonzero(side == 0)

    def describe(a):
        s, x = s_and_x(a, SAT)
        return int(s), int(x), tuple(np.bincount(local_dimension(a), minlength=4)[:4].tolist())

    def key(a):
        return tuple(map(tuple, np.sort(a, axis=1).tolist()))

    seen, queue = {key(knot)}, deque([knot.copy()])
    levels, resting = Counter(), Counter()
    while queue:
        a = queue.popleft()
        here = describe(a)
        e_here = 16.0 * (16 - here[0]) + 4.0 * lam * here[1]
        downhill = 0
        for _u1, _u2, new in switches(a, side_u):
            s1, x1 = s_and_x(new, SAT)
            if 16.0 * (16 - s1) + 4.0 * lam * x1 < e_here - 1e-9:
                downhill += 1
                k = key(new)
                if k not in seen:
                    seen.add(k)
                    queue.append(new)
        levels[here] += 1
        resting[here] += int(downhill == 0)
    return levels, resting


def main(lam=1.30):
    kinds, (s0, x0), (sw, xw) = beside_a_tube()
    print("lambda = %.2f" % lam)
    print("the 16 x 4 tube has S = %d, X = %d; the 12 x 4 tube beside a knot has S = %d, X = %d: %+.1f above the tube"
          % (sw, xw, s0, x0, cost(s0 - sw, x0 - xw, lam)))
    print("\n1. every single switch out of tube + knot")
    names = {"knot": "inside the knot", "tube": "inside the tube", "join": "one link from each (joins them)"}
    for (where, ds, dx, pieces), ways in sorted(kinds.items(), key=lambda kv: (kv[0][0], cost(kv[0][1], kv[0][2], lam))):
        print("   %-32s dS %+d dX %+3d  cost %+6.1f  -> %d piece(s)   %4d ways"
              % (names[where], ds, dx, cost(ds, dx, lam), pieces, ways))
    levels, resting = downhill_from_a_lone_knot(lam)
    print("\n2. a lone knot, following every switch that lowers the energy")
    for (s, x, hist), count in sorted(levels.items(), key=lambda kv: -kv[0][0]):
        print("   S %2d X %2d  energy %5.1f  points at d = 0,1,2,3: %s   %5d labelled states%s"
              % (s, x, 16.0 * (16 - s) + 4.0 * lam * x, list(hist), count,
                 "   <- no downhill switch from %d of them" % resting[(s, x, hist)] if resting[(s, x, hist)] else ""))


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 1.30)
