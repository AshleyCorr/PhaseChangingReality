"""The ways out of the curled torus, counted the way the chain proposes them. Usage:

    python scripts/exact_torus_level.py

Exact move censuses at individual states, with sampled neutral walks (ASSUMPTIONS Q13 and O42;
paper 1, Eq. (2)). Exhaustive neutral-class closure is in analyse_curled_revision.py --neutral.
A review of Eq. (2) (generated with ChatGPT) asked
to see made explicit (2026-09-24):

1. Ordered proposals. cqg.run_chain draws a vertex of side 0 and one of its four slots, twice, so it makes
   (2N)^2 = 4N^2 equally likely ordered proposals, and the same two edges drawn in the other order give the
   same switch. Counted here from the perfect 4 x L torus: every distinct switch, sorted by (dS, dX), and how
   often the chain offers each kind per sweep of 2N attempts.
2. The level. Some switches change neither S nor X. They reconnect two neighbouring columns (the short
   4-cycles) at two opposite points, which is a reflection of the square: the torus is twisted globally, and
   is no longer isomorphic to the perfect one, but every neighbourhood looks the same. So the chain, while it
   waits, wanders a family of arrangements at the torus's energy. Eq. (2) counts exits from the perfect torus;
   this checks that every arrangement met along a walk of such switches has the same counts, and counts their
   symmetries (what interchangeable vertices would change; paper 1, Discussion).
3. What Eq. (2) leaves out. Every exit other than A and B, counted at N = 64 to 192. Those that break two
   distant edges at once are offered a number of times per sweep that grows with N; their exact contribution
   to the exit rate is printed for every condition the paper quotes a waiting time for.
"""
import sys
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphity.cqg import NO_CAP, _switch, has_edge, is_valid, surplus, torus, total_squares   # noqa: E402

A, B, NEUTRAL = (-2, -4), (-4, -10), (0, 0)


def census(adj, side_u):
    """(Counter of distinct switches by (dS, dX), list of the arrangements the neutral ones lead to).
    Distinct: each unordered pair of edges once. The chain proposes each of them in two orders."""
    s0, x0 = total_squares(adj), surplus(adj)
    counts, neutral = Counter(), []
    for i, u1 in enumerate(side_u):
        for u2 in side_u[i + 1:]:
            for v1 in adj[u1]:
                for v2 in adj[u2]:
                    if v1 == v2 or has_edge(adj, u1, v2) or has_edge(adj, u2, v1):
                        continue
                    trial = adj.copy()
                    _switch(trial, int(u1), int(v1), int(u2), int(v2))
                    if not is_valid(trial, NO_CAP):
                        continue
                    key = (total_squares(trial) - s0, surplus(trial) - x0)
                    counts[key] += 1
                    if key == NEUTRAL:
                        neutral.append((trial, (int(u1), int(v1), int(u2), int(v2))))
    return counts, neutral


def ordered_census(adj, side_u):
    """Counter of the chain's ordered proposals by (dS, dX), enumerating (u1, slot, u2, slot) as it does."""
    s0, x0 = total_squares(adj), surplus(adj)
    counts = Counter()
    for u1 in side_u:
        for v1 in adj[u1]:
            for u2 in side_u:
                for v2 in adj[u2]:
                    if u1 == u2 or v1 == v2 or has_edge(adj, u1, v2) or has_edge(adj, u2, v1):
                        continue
                    trial = adj.copy()
                    _switch(trial, int(u1), int(v1), int(u2), int(v2))
                    if is_valid(trial, NO_CAP):
                        counts[(total_squares(trial) - s0, surplus(trial) - x0)] += 1
    return counts


def is_column_reflection(move, length):
    """True when the switch joins columns y and y + 1 at two opposite points x and x + 2 (torus(4, L) numbers
    vertex (x, y) as x * L + y, with x round the short side)."""
    (xa, ya), (xb, yb), (xc, yc), (xd, yd) = [divmod(v, length) for v in move]
    same_pair = ya == yc and yb == yd and (yb - ya) % length in (1, length - 1)
    return same_pair and xa == xb and xc == xd and (xc - xa) % 4 == 2


def as_graph(adj):
    return nx.Graph([(i, int(j)) for i in range(len(adj)) for j in adj[i]])


def cost(key, lam):
    return -16.0 * key[0] + 4.0 * lam * key[1]


def rate_per_sweep(counts, n, lam, g):
    """Metropolis exits per sweep of 2N attempts: each distinct switch is proposed twice out of 4N^2."""
    return sum(2 * n * 2 * c / (4 * n * n) * np.exp(-max(cost(k, lam), 0.0) / g)
               for k, c in counts.items() if k != NEUTRAL)


# Every condition the paper quotes a waiting time for: (lambda, g, sizes).
CONDITIONS = ([(1.25, g, (64, 144)) for g in (1.4, 1.5, 1.6, 1.75, 2.0, 2.5)]     # Fig. 2(a), committed config
              + [(lam, 1.5, (64, 96, 144, 192)) for lam in (1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35)])


def correction_table(sizes=(64, 96, 144, 192)):
    """How much every exit other than A and B adds to the exit rate of Eq. (2), at each condition quoted."""
    counts = {}
    for n in sizes:
        adj, part = torus(4, n // 4, cap=NO_CAP)
        counts[n], _ = census(adj, np.flatnonzero(part == 0))
        other = {k: v for k, v in counts[n].items() if k not in (A, B, NEUTRAL)}
        print("N = %d: distinct exits other than A and B: %s" % (n, ", ".join(
            "(%+d, %+d) %d = %.2f N" % (k[0], k[1], v, v / n) for k, v in sorted(other.items(), key=lambda kv: cost(kv[0], 1.25)))))
    worst = []
    for lam, g, ns in CONDITIONS:
        for n in ns:
            ab = {k: counts[n][k] for k in (A, B)}
            extra = rate_per_sweep(counts[n], n, lam, g) / rate_per_sweep(ab, n, lam, g) - 1.0
            worst.append((extra, lam, g, n))
    worst.sort(reverse=True)
    print("Largest additions to the exit rate of Eq. (2) from every other exit, over the conditions quoted:")
    for extra, lam, g, n in worst[:6]:
        print("   lambda = %.2f, g = %.1f, N = %d: %+.2f %%" % (lam, g, n, 100 * extra))
    at15 = [w for w in worst if w[2] == 1.5]
    print("   at g = 1.5, the largest: %+.3f %% (lambda = %.2f, N = %d)" % (100 * at15[0][0], at15[0][1], at15[0][3]))
    return worst


def symmetry_along_walk(length, walk=40, seed=0):
    """Side-preserving symmetry counts (graphity.symmetry) of the arrangements met along a walk of neutral
    switches, and of every arrangement one move A away from the last of them. With interchangeable vertices
    the symmetry ratio multiplies exp(-dH/g) BEFORE clipping at one (paper 1, Discussion;
    the old multiply-an-acceptance wording in ASSUMPTIONS Q20 is corrected by O108)."""
    from graphity import symmetry
    adj, part = torus(4, length, cap=NO_CAP)
    side_u = np.flatnonzero(part == 0)
    rng = np.random.default_rng(seed)
    met, classes, cur = Counter(), set(), adj.copy()
    for _ in range(walk):
        met[symmetry.count(cur, part)] += 1
        classes.add(symmetry.canonical_key(cur, part))
        _, nb = census(cur, side_u)
        cur = nb[rng.integers(len(nb))][0]
    s0, x0 = total_squares(cur), surplus(cur)
    after = Counter()
    for i, u1 in enumerate(side_u):
        for u2 in side_u[i + 1:]:
            for v1 in cur[u1]:
                for v2 in cur[u2]:
                    if v1 == v2 or has_edge(cur, u1, v2) or has_edge(cur, u2, v1):
                        continue
                    t = cur.copy()
                    _switch(t, int(u1), int(v1), int(u2), int(v2))
                    if is_valid(t, NO_CAP) and (total_squares(t) - s0, surplus(t) - x0) == A:
                        after[symmetry.count(t, part)] += 1
    return met, len(classes), symmetry.count(cur, part), after


def main(length=16, walk=60, seed=1):
    for ell in (8, 12, length):
        adj, part = torus(4, ell, cap=NO_CAP)
        n = len(adj)
        side_u = np.flatnonzero(part == 0)
        c = ordered_census(adj, side_u)
        print("N = %d: %d ordered proposals; per sweep of 2N attempts the chain offers" % (n, 4 * n * n))
        for key in sorted(c, key=lambda k: -16 * k[0] + 5 * k[1]):
            print("   dS = %+d, dX = %+3d: %5d ordered (%4d distinct = %.2f N), %.2f per sweep"
                  % (key[0], key[1], c[key], c[key] // 2, c[key] / 2 / n, 2 * n * c[key] / (4 * n * n)))

    adj, part = torus(4, length, cap=NO_CAP)
    n = len(adj)
    side_u = np.flatnonzero(part == 0)
    base, neutral = census(adj, side_u)
    g0 = as_graph(adj)
    reflections = sum(is_column_reflection(m, length) for _, m in neutral)
    isomorphic = sum(nx.is_isomorphic(g0, as_graph(t)) for t, _ in neutral)
    print("\nN = %d, the level: %d neutral switches out of the perfect torus; %d join two neighbouring columns at"
          " opposite points; %d lead to an arrangement isomorphic to the perfect torus"
          % (n, len(neutral), reflections, isomorphic))
    rng = np.random.default_rng(seed)
    cur, same = adj.copy(), 0
    for step in range(walk):
        counts, nb = census(cur, side_u)
        same += (counts[A], counts[B], counts[NEUTRAL]) == (base[A], base[B], base[NEUTRAL])
        cheaper = [k for k in counts if k not in (A, B, NEUTRAL) and -16 * k[0] + 4 * k[1] <= 24]
        if cheaper:
            print("   step %d: other exits within 24 at lambda = 1: %s" % (step, cheaper))
        cur = nb[rng.integers(len(nb))][0]
    print("   along a walk of %d neutral switches: %d of %d arrangements have exactly %d A, %d B and %d neutral"
          " switches, as the perfect torus does" % (walk, same, walk, base[A], base[B], base[NEUTRAL]))
    for ell in (16, 24):
        met, n_classes, last, after = symmetry_along_walk(ell)
        print("   N = %d: symmetry counts met along a 40-step neutral walk %s (%d classes); one move A away from"
              " the last (%d symmetries): %s" % (4 * ell, dict(met), n_classes, last, dict(after)))
    print()
    correction_table()


if __name__ == "__main__":
    main()
