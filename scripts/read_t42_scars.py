"""What T42's and T34's "melted" replicas are, read from the census over time and from the saved final wiring.

WHY. The owner asked (27 Sep) what "melted" means and how we know it melted rather than curled. T34's rule, reused by
T42, calls a replica MELTED when its last reading has at least 4 points with more open directions than flat space
allows (d > 3) and more of those than curled points (d < 3). This script reads what those points are, without changing
any verdict: per cell, the damaged count at the end of each replica, the largest damaged count at any reading, the
largest curled count at any reading, the share of readings with any damage and how often it switched on or off; and,
for T42's smallest push, the saved final graphs compared link by link with the starting 8 x 8 x 8 torus, with the
energy after the one switch that trades the changed links back.

Reads results/ only; writes nothing.
"""
import csv
import glob
import os
from collections import defaultdict

import numpy as np

from graphity import cqg_d
from graphity.dimension import local_dimension_d

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def cells(path):
    """{push: [rows of one replica, in sweep order], ...} grouped by (push, replica)."""
    runs = defaultdict(list)
    for r in csv.DictReader(open(path)):
        runs[(float(r["spark"]), int(r["replica"]))].append(r)
    for v in runs.values():
        v.sort(key=lambda r: int(r["sweep"]))
    return runs


def curled(r):
    return int(r["d0"]) + int(r["d1"]) + int(r["d2"])


def summary(path):
    by_push = defaultdict(list)
    for (push, _), rows in cells(path).items():
        m = [int(r["melted"]) for r in rows if int(r["sweep"]) > 0]
        by_push[push].append(dict(end=m[-1], peak=max(m), curled=max(curled(r) for r in rows),
                                  share=sum(x > 0 for x in m) / len(m),
                                  switches=sum((a > 0) != (b > 0) for a, b in zip(m, m[1:]))))
    return by_push


def edges(adj):
    return {(min(u, int(v)), max(u, int(v))) for u in range(len(adj)) for v in adj[u] if v >= 0}


def undo(adj, removed, added, lam):
    """Energy after the switch that trades the two added links for the two removed ones (None if not one switch)."""
    if len(removed) != 2 or len(added) != 2:
        return None
    a = adj.copy()
    for u, v in added:
        a[u][list(a[u]).index(v)] = -1
        a[v][list(a[v]).index(u)] = -1
    for u, v in removed:
        a[u][list(a[u]).index(-1)] = v
        a[v][list(a[v]).index(-1)] = u
    return cqg_d.hamiltonian(a, lam) if cqg_d.is_valid(a) else "invalid"


def main():
    for path in sorted(glob.glob(os.path.join(RESULTS, "t42_*.csv"))) + sorted(glob.glob(os.path.join(RESULTS, "t34_*.csv"))):
        n = int(next(csv.DictReader(open(path)))["N"])
        print(os.path.basename(path)[:-4], "N =", n)
        for push, reps in sorted(summary(path).items()):
            print("  push %-6g end damaged %-28s peak %-4d most curled %d | share with damage %.2f, on/off %d (means)"
                  % (push, [r["end"] for r in reps], max(r["peak"] for r in reps), max(r["curled"] for r in reps),
                     np.mean([r["share"] for r in reps]), round(np.mean([r["switches"] for r in reps]))))
    start = edges(cqg_d.torus((8, 8, 8))[0])
    for path in sorted(glob.glob(os.path.join(RESULTS, "t42_named_lam102_n512_e64_adj", "*.npz"))):
        z = np.load(path)
        adj, lam = z["adj"], float(z["lam"])
        now = edges(adj)
        removed, added = sorted(start - now), sorted(now - start)
        d = np.asarray(local_dimension_d(adj))
        print("t42 named E=64 %s: H %.0f, damaged %d, curled %d, links changed %d, energy after the one undo switch %s"
              % (os.path.basename(path)[:-4], cqg_d.hamiltonian(adj, lam), (d > 3).sum(), (d < 3).sum(), len(removed),
                 undo(adj, removed, added, lam)))


if __name__ == "__main__":
    main()
