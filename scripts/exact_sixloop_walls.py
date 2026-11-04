"""EXACT, by complete listing (12 October 2026): what a price on loops of six does to the tube. Usage:

    python scripts/exact_sixloop_walls.py [LENGTH] [LAMBDA]

The rule is graphity.sixloop's (VISION Update 56; ASSUMPTIONS O124; ours, not the published model):
H6 = 16 (N - S) + 4 lam X + mu E6, with E6 the sum over links of (loops of six through the link - 6) where
positive. For the 4 x LENGTH tube with its seam shifted by 0, 2 and 4 places (four, six and eight around) this
script lists, by trying every one:

- E6 per point;
- every valid single switch out of it, grouped by what it does to the squares, the surplus and E6, with how many
  distinct switches there are of each kind (a switch is a pair of links; the chain proposes each pair two ways);
- the price below which the arrangement is stuck: every switch out of it costs energy;
- for the tube six around, the push: the switch that breaks four squares, adds no surplus and removes the most
  E6; and every switch out of the pushed state.

Nothing is run. It writes nothing.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graphity.cqg import NO_CAP, _new_edge_ok, _switch, surplus, total_squares      # noqa: E402
from graphity.sixloop import excess6                                               # noqa: E402
from read_endpoint_width import twisted_tube                                       # noqa: E402

NAMES = {0: "the tube, four around", 2: "re-glued, six around", 4: "re-glued again, eight around"}


def side_of(length):
    return np.array([x * length + y for x in range(4) for y in range(length) if (x + y) % 2 == 0], dtype=np.int64)


def switches(adj, side):
    """[((u1, v1, u2, v2), dS, dX, dE6)] for every valid distinct switch out of `adj`."""
    s0, x0, e0 = total_squares(adj), surplus(adj), excess6(adj, side)
    out = []
    for i, u1 in enumerate(side):
        for v1 in adj[u1]:
            for u2 in side[i + 1:]:                         # each pair of links once
                for v2 in adj[u2]:
                    if v1 == v2 or v2 in adj[u1] or v1 in adj[u2]:
                        continue
                    trial = adj.copy()
                    _switch(trial, int(u1), int(v1), int(u2), int(v2))
                    if not (_new_edge_ok(trial, int(u1), int(v2), NO_CAP) and _new_edge_ok(trial, int(u2), int(v1), NO_CAP)):
                        continue
                    out.append(((int(u1), int(v1), int(u2), int(v2)), int(total_squares(trial) - s0),
                                int(surplus(trial) - x0), int(excess6(trial, side) - e0)))
    return out


def kinds(moves):
    """{(dS, dX, dE6): number of distinct switches}."""
    out = {}
    for _move, ds, dx, de in moves:
        out[(ds, dx, de)] = out.get((ds, dx, de), 0) + 1
    return out


def cost(kind, lam, mu):
    ds, dx, de = kind
    return -16.0 * ds + 4.0 * lam * dx + mu * de


def stuck_below(table, lam):
    """The price below which every switch costs energy: None if some switch is free or downhill at every price,
    infinity if no price unsticks it."""
    limit = float("inf")
    for (ds, dx, de) in table:
        base = cost((ds, dx, de), lam, 0.0)
        if de == 0 and base == 0.0:
            continue                                        # a relabelling: the same arrangement
        if de >= 0:
            if base <= 0.0 and not (base == 0.0 and de > 0):
                return None
            continue
        if base <= 0.0:
            return None
        limit = min(limit, base / -de)
    return limit


def pushed(length):
    """(the tube six around after the push, the push's (dS, dX, dE6))."""
    adj, side = twisted_tube(length, 2), side_of(length)
    best = min((m for m in switches(adj, side) if m[1] == -4 and m[2] == 0), key=lambda m: m[3])
    _switch(adj, *best[0])
    return adj, best[1:]


def main(length=12, lam=1.25):
    n = 4 * length
    for shift in (0, 2, 4):
        adj, side = twisted_tube(length, shift), side_of(length)
        table = kinds(switches(adj, side))
        e6 = excess6(adj, side)
        limit = stuck_below(table, lam)
        print("%s (N = %d): E6 = %d, %.0f per point; %d distinct switches of %d kinds; stuck for mu %s"
              % (NAMES[shift], n, e6, e6 / n, sum(table.values()), len(table),
                 "at no price" if limit is None else ("at every price" if limit == float("inf") else "< %.4f" % limit)))
        for mu in (0.0, 0.05, 0.10, 0.20, 0.30):
            ranked = sorted((cost(k, lam, mu), k, c) for k, c in table.items() if cost(k, lam, 0.0) != 0.0 or k[2] != 0)
            print("   mu = %.2f: cheapest (cost; squares, surplus, E6; how many): %s"
                  % (mu, " | ".join("%.1f; %+d, %+d, %+d; %d" % (c, k[0], k[1], k[2], m) for c, k, m in ranked[:3])))
    adj, push = pushed(length)
    side = side_of(length)
    print("\nThe push out of six around: squares %+d, surplus %+d, E6 %+d; it costs 64 - %d mu." % (push + (-push[2],)))
    table = kinds(switches(adj, side))
    back = [k for k in table if k == (4, 0, -push[2])]
    keep = sorted((k[2], table[k]) for k in table if k[0] == 0 and k[1] == 0 and k[2] < 0)
    print("From the pushed state: %d switch(es) straight back; switches that keep every square and remove E6: %s"
          % (sum(table[k] for k in back), ", ".join("%d (x%d)" % kc for kc in keep)))
    for mu in (0.05, 0.10, 0.20, 0.30):
        down = sum(c for k, c in table.items() if cost(k, lam, mu) < 0 and k not in back)
        print("   mu = %.2f: the push costs %.1f; switches downhill from there, other than straight back: %d"
              % (mu, cost(push, lam, mu), down))
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    sys.exit(main(int(a[0]) if a else 12, float(a[1]) if len(a) > 1 else 1.25))
