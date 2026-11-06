"""Sandbox look (4 November 2026, her request "Count in small sheet"): from one made state on a small flat torus, count
every way the pairs could resolve under her clock (room judged once on the made state). A resolution is a set of
disjoint links, each a cancellation (opposite signs, room beside either) or a join (same sign, no room beside either),
that leaves no further such link with both squares free. For each resolution, the number of stores (joins). Her "max
entropy size" is the store count with the most resolutions. Exact enumeration with a cap; a look, not a test."""
import sys
from collections import Counter

import numpy as np

from graphity.cqg import NO_CAP, torus
from graphity.curl_pair_moves import PairMoves
from graphity.curl_pairs import CurlState

sys.setrecursionlimit(10000)
CAP = 3_000_000


def candidates(m):
    c = m.state.c
    room = [m.room(i) for i in range(len(c))]
    cand = []
    for i, j in m.links:
        if c[i] == 0.0:
            continue
        if c[i] == -c[j] and (room[i] or room[j]):
            cand.append((i, j, "c"))
        elif c[i] == c[j] and not room[i] and not room[j]:
            cand.append((i, j, "j"))
    return cand


def enumerate_resolutions(cand, n):
    """Counter {joins: number of maximal matchings with that many joins}, and whether the cap was hit."""
    by_v = [[] for _ in range(n)]
    for k, (i, j, _) in enumerate(cand):
        by_v[i].append(k)
        by_v[j].append(k)
    decided = [False] * n          # this vertex's fate is fixed (matched, or left out)
    matched = [False] * n
    out = Counter()
    count = [0]

    def partner(k, v):
        i, j, _ = cand[k]
        return j if i == v else i

    def rec(v, joins):
        if count[0] >= CAP:
            return
        while v < n and (decided[v] or not by_v[v]):
            v += 1
        if v == n:
            for u in range(n):           # maximal: no candidate link with both squares unmatched
                if not matched[u]:
                    for k in by_v[u]:
                        if not matched[partner(k, u)]:
                            return
            out[joins] += 1
            count[0] += 1
            return
        for k in by_v[v]:
            w = partner(k, v)
            if decided[w]:
                continue
            decided[v] = decided[w] = matched[v] = matched[w] = True
            rec(v + 1, joins + (cand[k][2] == "j"))
            decided[v] = decided[w] = matched[v] = matched[w] = False
        decided[v] = True             # left out
        rec(v + 1, joins)
        decided[v] = False

    rec(0, 0)
    return out, count[0] >= CAP


if __name__ == "__main__":
    sides = [int(a) for a in sys.argv[1:]] or [4, 6]
    for side in sides:
        for lam in (1.05, 1.1, 1.25, 1.5, 2.0):
            for seed in (1, 2, 3):
                adj, part = torus(side, side, NO_CAP)
                m = PairMoves(CurlState(adj, part), lam=lam, barrier=0.0, seed=seed, price_from="gravity",
                              clock="simultaneous")
                m.make_pairs()
                cand = candidates(m)
                nc = sum(1 for x in cand if x[2] == "c")
                nj = sum(1 for x in cand if x[2] == "j")
                dist, capped = enumerate_resolutions(cand, len(m.state.c))
                total = sum(dist.values())
                mode = max(dist.items(), key=lambda kv: kv[1])[0] if dist else None
                spread = " ".join(f"{k}:{v}" for k, v in sorted(dist.items()))
                print(f"side {side} lam {lam} seed {seed}: full {int(np.sum(m.state.c != 0))}/{len(m.state.c)}, "
                      f"cancel links {nc}, join links {nj}, resolutions {total}{' (capped)' if capped else ''}, "
                      f"most copies at {mode} stores; stores:copies {spread}", flush=True)
