"""EXACT: rebuild the dips found by exact_relic_search_d.py and test them completely. Usage:

    python scripts/exact_relic_confirm_d.py [lambda] [out_dir]

Written 2026-09-25 (gravity brief, step 1; ASSUMPTIONS O58). The search tested dip-ness only from switches whose first
side-0 point was one the construction changed; a switch touching the object only through side-1 points was not
listed. Here each candidate (a two-switch construction with the given (dS, dX) near vertex 0 of the flat 6 x 6 x 6
torus) is rebuilt, and every switch whose first side-0 point lies within distance 3 of any changed point, with its
partner within distance 4 of that point, is priced (moves farther away touch only flat space and cost at least 64).
A dip is confirmed if all cost more than zero. Confirmed states are saved as .npz for the gravity test.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graphity import symmetry                                                        # noqa: E402
from graphity.cqg_d import _switch, is_valid, surplus, torus, total_squares         # noqa: E402
from graphity.sealed_d import energy_d                                               # noqa: E402
from exact_relic_search_d import switches_near                                       # noqa: E402
from exact_walls_d import walls, within                                              # noqa: E402

TARGETS = [(-4, 4), (-7, 4)]


def rebuild(lam, radius=3):
    adj, part = torus([6, 6, 6])
    n = adj.shape[0]
    s0, x0 = int(total_squares(adj)), int(surplus(adj))
    inside = within(adj, 0, radius)
    allowed = np.zeros(n, dtype=bool)
    allowed[inside] = True
    side0_near = [int(v) for v in inside if part[v] == 0]
    found = {}
    for (u1, v1, u2, v2) in switches_near(adj, part, allowed, side0_near):
        g1 = adj.copy()
        _switch(g1, u1, v1, u2, v2)
        if not is_valid(g1):
            continue
        for (a, b, c, d) in switches_near(g1, part, allowed, side0_near):
            g2 = g1.copy()
            _switch(g2, a, b, c, d)
            if not is_valid(g2):
                continue
            key_ds = (int(total_squares(g2)) - s0, int(surplus(g2)) - x0)
            if key_ds in TARGETS:
                k = symmetry.canonical_key(g2, part)
                found.setdefault(key_ds, {}).setdefault(k, g2)
    return adj, part, found


def full_dip(g, flat, part, lam):
    changed = set(int(v) for v in np.flatnonzero((g != flat).any(axis=1)))
    near = set()
    for v in changed:
        near |= set(int(w) for w in within(g, v, 3))
    cheapest = None
    for u in sorted(near):
        if part[u] != 0:
            continue
        w = walls(g, part, lam, u1=u, top=1, near=4)
        if w and (cheapest is None or w[0][0] < cheapest[0]):
            cheapest = w[0]
    return cheapest, len(changed)


def main(lam=1.02, out_dir="results/relic_6link"):
    flat, part, found = rebuild(lam)
    base = energy_d(flat, lam)
    out = Path(out_dir)
    for key in TARGETS:
        states = found.get(key, {})
        print("(dS, dX) = %s: %d distinct arrangements" % (key, len(states)))
        for i, g in enumerate(states.values()):
            cheapest, n_changed = full_dip(g, flat, part, lam)
            e = energy_d(g, lam) - base
            ok = cheapest is not None and cheapest[0] > 0
            print("  #%d E = %.2f, %d points changed, symmetries %d, cheapest move out %.2f (dS %d, dX %d) -> %s"
                  % (i, e, n_changed, symmetry.count(g, part), cheapest[0], cheapest[1], cheapest[2], "DIP, confirmed" if ok else "not a dip"))
            if ok:
                out.mkdir(parents=True, exist_ok=True)
                np.savez(out / ("relic_dS%d_dX%d_%d.npz" % (key[0], key[1], i)), adj=g, part=part, lam=lam, energy=e)


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 1.02, sys.argv[2] if len(sys.argv) > 2 else "results/relic_6link")
