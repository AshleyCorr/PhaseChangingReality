"""EXACT: is there a small curled object that is a dip inside flat six-link space? A search over one- and two-switch
constructions near one point of the flat 6 x 6 x 6 torus. Usage:

    python scripts/exact_relic_search_d.py [lambda] [radius] [top]

Written 2026-09-25 (gravity brief, step 1; VISION Update 29). Every bipartite switch whose four points lie within
`radius` of a center is applied to the flat torus (one switch), and every second switch within the same radius on top
of each valid first one (two switches). Each valid arrangement is priced exactly above flat space, and the ones that add
surplus squares (dX > 0: some edge now carries a fifth square, the mark of curling) are kept, lowest energy first. The
`top` lowest are then tested for being a DIP: every single switch out of them, by the nearby-partner wall search, must
cost energy. A dip with dX > 0 is a relic; if none is found the report says so. One center suffices by translation
symmetry. Arrangements are compared up to renaming with graphity.symmetry.canonical_key, so the same object reached by
two routes is listed once.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graphity import symmetry                                                        # noqa: E402
from graphity.cqg_d import _switch, is_valid, surplus, torus, total_squares         # noqa: E402
from graphity.sealed_d import energy_d                                               # noqa: E402
from exact_walls_d import walls, within                                              # noqa: E402


def switches_near(adj, part, allowed, side0_near):
    """Every valid bipartite switch (u1, v1, u2, v2) with all four points in `allowed`, u1 < u2 on side 0."""
    out = []
    for i, u1 in enumerate(side0_near):
        n1 = set(int(v) for v in adj[u1])
        for u2 in side0_near[i + 1:]:
            n2 = set(int(v) for v in adj[u2])
            for v1 in adj[u1]:
                v1 = int(v1)
                if not allowed[v1] or v1 in n2:
                    continue
                for v2 in adj[u2]:
                    v2 = int(v2)
                    if v2 == v1 or not allowed[v2] or v2 in n1:
                        continue
                    out.append((int(u1), v1, int(u2), v2))
    return out


def main(lam=1.02, radius=3, top=30):
    t0 = time.time()
    dims = [6, 6, 6]
    adj, part = torus(dims)
    n = adj.shape[0]
    base = energy_d(adj, lam)
    s0, x0 = int(total_squares(adj)), int(surplus(adj))
    center = 0
    inside = within(adj, center, radius)
    allowed = np.zeros(n, dtype=bool)
    allowed[inside] = True
    side0_near = [int(v) for v in inside if part[v] == 0]
    first = switches_near(adj, part, allowed, side0_near)
    print("flat 6 x 6 x 6, lambda %.3f, radius %d: %d side-0 points near the center, %d candidate first switches"
          % (lam, radius, len(side0_near), len(first)))
    found = {}          # canonical key -> (energy, dS, dX, description)
    n_valid1 = 0
    for (u1, v1, u2, v2) in first:
        g1 = adj.copy()
        _switch(g1, u1, v1, u2, v2)
        if not is_valid(g1):
            continue
        n_valid1 += 1
        ds, dx = int(total_squares(g1)) - s0, int(surplus(g1)) - x0
        e = energy_d(g1, lam) - base
        if dx > 0:
            found.setdefault(symmetry.canonical_key(g1, part), (e, ds, dx, "1 switch"))
        second = switches_near(g1, part, allowed, side0_near)
        for (a, b, c, d) in second:
            g2 = g1.copy()
            _switch(g2, a, b, c, d)
            if not is_valid(g2):
                continue
            dx2 = int(surplus(g2)) - x0
            if dx2 <= 0:
                continue
            e2 = energy_d(g2, lam) - base
            key = symmetry.canonical_key(g2, part)
            if key not in found or found[key][0] > e2:
                found[key] = (e2, int(total_squares(g2)) - s0, dx2, "2 switches")
        # keep memory bounded: only the two-switch states we need are stored by key
    print("valid first switches %d; distinct arrangements with dX > 0 found: %d (%.0f s)" % (n_valid1, len(found), time.time() - t0))
    ranked = sorted(found.items(), key=lambda kv: kv[1][0])[:top]
    print("lowest-energy curled arrangements, tested for being a dip (every single move out costs energy):")
    dips = 0
    for key, (e, ds, dx, how) in ranked:
        # rebuild the state from its key is not possible; re-derive by searching again is costly, so we test dip-ness
        # by re-generating: keys map to the first construction found, which we recompute below.
        pass
    # second pass: rebuild the ranked states (cheap relative to the search) and test dips
    wanted = {key for key, _ in ranked}
    rebuilt = {}
    for (u1, v1, u2, v2) in first:
        g1 = adj.copy()
        _switch(g1, u1, v1, u2, v2)
        if not is_valid(g1):
            continue
        k1 = symmetry.canonical_key(g1, part)
        if k1 in wanted and k1 not in rebuilt:
            rebuilt[k1] = g1.copy()
        if len(rebuilt) == len(wanted):
            break
        second = switches_near(g1, part, allowed, side0_near)
        for (a, b, c, d) in second:
            g2 = g1.copy()
            _switch(g2, a, b, c, d)
            if not is_valid(g2) or int(surplus(g2)) - x0 <= 0:
                continue
            k2 = symmetry.canonical_key(g2, part)
            if k2 in wanted and k2 not in rebuilt:
                rebuilt[k2] = g2.copy()
        if len(rebuilt) == len(wanted):
            break
    for key, (e, ds, dx, how) in ranked:
        g = rebuilt.get(key)
        if g is None:
            print("  E=%7.2f dS=%3d dX=%3d %-10s (not rebuilt)" % (e, ds, dx, how)); continue
        changed = [int(v) for v in np.flatnonzero((g != adj).any(axis=1))]
        cheapest = None
        for v in changed:
            if part[v] != 0:
                continue
            w = walls(g, part, lam, u1=v, top=1, near=3)
            if w and (cheapest is None or w[0][0] < cheapest[0]):
                cheapest = w[0]
        dip = cheapest is not None and cheapest[0] > 0
        dips += dip
        print("  E=%7.2f dS=%3d dX=%3d %-10s symmetries %6d | cheapest move out %7.2f -> %s"
              % (e, ds, dx, how, symmetry.count(g, part), cheapest[0] if cheapest else float("nan"), "DIP" if dip else "not a dip"))
    print("dips among the %d lowest: %d (%.0f s)" % (len(ranked), dips, time.time() - t0))


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 1.02, int(sys.argv[2]) if len(sys.argv) > 2 else 3,
         int(sys.argv[3]) if len(sys.argv) > 3 else 30)
