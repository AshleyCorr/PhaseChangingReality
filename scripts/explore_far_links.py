"""EXPLORATORY, read only: are any far-reaching links left in the saved end states of T37? Usage:

    python scripts/explore_far_links.py

Written 2026-10-05 after the owner asked whether her runs touch "disordered locality" ([PWS09]: a network that is
mostly local but keeps a few links between points that are far apart in the space it makes). For every link of a
saved end state this reads its WAY ROUND: the graph distance between its two ends once the link itself is removed.
On a flat sheet every link lies on two squares, so its way round is 3. A link with a way round of 7 or more joins two
points that would otherwise be at least 7 steps apart, and is called FAR here.

Not pre-registered, a first look and not a finding. Positions were not read, so nothing here says what a far link
is (a stitch across a tear between two patches, a shortcut, part of a leftover); only how many there are, how far
they reach, and whether their ends are flat points.
"""
import sys
from glob import glob
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphity.dimension import local_dimension    # noqa: E402

FAR, CAP, CLEAN = 7, 60, 0.90
CELLS = (("g = 1.5,  L = 64", "results/t37_lam125_g150_L64_adj/*.npz"),
         ("g = 1.5,  L = 128", "results/t37_lam125_g150_L128_adj/*.npz"),
         ("g = 1.5,  L = 256", "results/t37_lam125_g150_L256_*_adj/*.npz"),
         ("g = 1.25, L = 1024", "results/t37_lam125_g125_L1024_*_adj/*.npz"))


def way_round(adj, u, v, cap=CAP):
    """The graph distance from u to v without using the link (u, v); cap + 1 if it is more than cap."""
    seen, edge, steps = {u}, [u], 0
    while edge and steps < cap:
        steps += 1
        nxt = []
        for a in edge:
            for b in adj[a]:
                b = int(b)
                if a == u and b == v:
                    continue
                if b == v:
                    return steps
                if b not in seen:
                    seen.add(b)
                    nxt.append(b)
        edge = nxt
    return cap + 1


def census(adj):
    """(every link's way round, how many ends of far links are at flat points, the share of flat points)."""
    d = local_dimension(adj)
    ways, flat_ends = [], 0
    for u in range(adj.shape[0]):
        for v in adj[u]:
            v = int(v)
            if u < v:
                w = way_round(adj, u, v)
                ways.append(w)
                if w >= FAR:
                    flat_ends += int(d[u] == 2) + int(d[v] == 2)
    return ways, flat_ends, float((d == 2).mean())


def main():
    print("EXPLORATORY. A far link has a way round of %d steps or more. Clean tubes only (at least %.0f %% of points flat)."
          % (FAR, 100 * CLEAN))
    for label, pattern in CELLS:
        rows = [(np.load(f)["adj"].shape[0],) + census(np.load(f)["adj"]) for f in sorted(glob(pattern))]
        clean = [r for r in rows if r[3] >= CLEAN]
        columns = clean[0][0] / 4
        far = [[w for w in r[1] if w >= FAR] for r in clean]
        every = [w for f in far for w in f]
        print("%-19s clean %2d of %2d | far links per tube %.2f, in %d tubes | per 1,000 columns %.1f | of all links %.5f | "
              "longest way round %s | ends at flat points %d of %d"
              % (label, len(clean), len(rows), np.mean([len(f) for f in far]), sum(1 for f in far if f),
                 1000 * np.mean([len(f) for f in far]) / columns, len(every) / sum(len(r[1]) for r in clean),
                 max(every) if every else "-", sum(r[2] for r in clean), 2 * len(every)))


if __name__ == "__main__":
    main()
