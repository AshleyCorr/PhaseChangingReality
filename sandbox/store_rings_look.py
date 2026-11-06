"""Sandbox look (6 November 2026, her words of the night before: stores form "by combining into a ring of same handed
Klein bottles"). T105's cell under her clock (room judged once on the made state, price from gravity), the stores at
the freeze grouped by sign into clusters (store squares joined through a shared link), and each cluster read for its
shape: a chain (no cycle), a closed ring (every member with exactly two same-sign store neighbours and a cycle), or a
clump (a cycle but not a ring). Asked: do same-sign stores sit in rings. A look, not a test; T105's states, re-made
with the same rules at a fresh seed."""
import sys

import numpy as np

from graphity.cqg import NO_CAP, torus
from graphity.curl_pair_moves import PairMoves
from graphity.curl_pairs import CurlState


def clusters(pm, sign):
    members = set(int(i) for i in np.flatnonzero(pm.store == sign))
    seen, out = set(), []
    for start in members:
        if start in seen:
            continue
        stack, nodes = [start], []
        seen.add(start)
        while stack:
            i = stack.pop()
            nodes.append(i)
            for j in pm.nb[i]:
                if j in members and j not in seen:
                    seen.add(j)
                    stack.append(j)
        degs = [sum(1 for j in pm.nb[i] if j in members) for i in nodes]
        edges = sum(degs) // 2
        cycle = edges >= len(nodes)
        ring = cycle and all(d == 2 for d in degs)
        out.append(dict(size=len(nodes), edges=edges, cycle=cycle, ring=ring))
    return out


if __name__ == "__main__":
    side = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    for lam in (1.25, 1.5, 2.0):
        for barrier in (0, 12):
            tally = dict(clusters=0, singles=0, chains=0, rings=0, clumps=0, largest=0, stores=0)
            for rep in range(reps):
                adj, part = torus(side, side, NO_CAP)
                st = CurlState(adj, part)
                pm = PairMoves(st, lam=lam, barrier=barrier, seed=20261106 + 7919 * rep + 101 * int(lam * 100) + barrier,
                               price_from="gravity", clock="simultaneous")
                pm.run()
                for sign in (1, -1):
                    for c in clusters(pm, sign):
                        tally["clusters"] += 1
                        tally["stores"] += c["size"]
                        tally["largest"] = max(tally["largest"], c["size"])
                        if c["size"] == 1:
                            tally["singles"] += 1
                        elif c["ring"]:
                            tally["rings"] += 1
                        elif c["cycle"]:
                            tally["clumps"] += 1
                        else:
                            tally["chains"] += 1
            print(f"side {side} lam {lam} B {barrier} ({reps} replicas): stores {tally['stores']}, same-sign clusters "
                  f"{tally['clusters']}: singles {tally['singles']}, chains {tally['chains']}, rings {tally['rings']}, "
                  f"clumps {tally['clumps']}; largest {tally['largest']}", flush=True)
