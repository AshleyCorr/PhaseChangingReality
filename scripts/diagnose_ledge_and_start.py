"""Two questions T7 raised, answered on N = 64 tubes where a decay takes minutes. Usage:

    python scripts/diagnose_ledge_and_start.py [replicas]

1. WHAT IS THE LEDGE. Half the decays stop at 0.78 of the release and sit there. At N = 64
   that is 14 energy units. With H = 16(N - S) + 5X the simplest way to make 14 is one extra
   square and six surplus units (-16 + 30), i.e. a sheet with a single localised wrinkle. When
   a decay is seen to rest on the ledge (released energy steady within 2 % over two 600-sweep
   windows and below 0.95 of the full release) the graph is read: S, X, the local-dimension
   histogram, the vertices that are not sheet-like and whether they form one cluster, and the
   edges carrying surplus. Reported per decay so that "the same state every time" can be checked
   rather than assumed.

2. WHERE THE FRONT STARTS. The mean wait does not fall with size, which rules out "any vertex
   can start it". At first departure (phi below the tube's value by three resting standard
   deviations) the vertices that are no longer tube-like are read and mapped onto the 16 x 4
   lattice: (x, y) with x = i // 4 (the long direction) and y = i % 4 (the curled one). Over
   many replicas, a histogram of where it starts. Uniform in x means no special place.

Exploratory (not pre-registered); writes nothing to results/.
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphity.cqg import (ENERGY_PER_SQUARE, ENERGY_PER_SURPLUS, NO_CAP,   # noqa: E402
                          run_chain, surplus, torus, total_squares, squares_on_edge)
from graphity.dimension import local_dimension, pieces_of                  # noqa: E402

LX, LY, LAM, G, BLOCK = 16, 4, 1.25, 1.5, 5
N = LX * LY
EXPECT = 4.0 * (LAM - 1.0)


def h_per_point(adj):
    return (ENERGY_PER_SQUARE * (N - total_squares(adj)) + ENERGY_PER_SURPLUS * LAM * surplus(adj)) / N


def surplus_edges(adj):
    out = []
    for u in range(N):
        for k in range(4):
            v = adj[u, k]
            if v > u:
                s = squares_on_edge(adj, u, v)
                if s > 2:
                    out.append((u, v, s - 2))
    return out


def describe(adj, label):
    d = local_dimension(adj)
    odd = d != 2
    pieces = pieces_of(adj, odd)
    se = surplus_edges(adj)
    xs = sorted({u // LY for u, v, _ in se} | {v // LY for u, v, _ in se})
    print("   %s: S=%d (N=%d)  X=%d  h=%.4f  d-hist=%s  non-sheet vertices=%d in %d cluster(s) %s  "
          "surplus edges=%d at x in %s"
          % (label, total_squares(adj), N, surplus(adj), h_per_point(adj), np.bincount(d, minlength=5).tolist(),
             int(odd.sum()), len(pieces), pieces[:4], len(se), xs))
    return (int(total_squares(adj)), int(surplus(adj)), tuple(np.bincount(d, minlength=5).tolist()),
            int(odd.sum()), len(pieces))


def main(reps=30):
    starts, ledges, signatures = [], 0, Counter()
    for rep in range(reps):
        adj, part = torus(LX, LY, NO_CAP)
        side_u = np.flatnonzero(part == 0)
        h0 = h_per_point(adj)
        phis, done, first, thresh, departed = [], 0, True, None, False
        while done < 30000 and not departed:
            s, x, _ = run_chain(adj, side_u, 1.0 / G, 0, BLOCK, (5000 + rep) if first else -1, LAM, NO_CAP, False)
            first = False; done += BLOCK
            phi = s[-1] / N; phis.append(phi)
            if done <= 200:
                continue
            if thresh is None:
                rest = np.array(phis[: 200 // BLOCK]); thresh = 1.25 - 3 * max(rest.std(), 1e-6) - 1e-9
            if phi < thresh:
                departed = True
                d = local_dimension(adj)
                moved = np.flatnonzero(d != 1)
                xs = [int(v // LY) for v in moved]
                starts.append((done, xs))
        if not departed:
            print("rep %2d: never departed" % rep); continue
        # run on to conversion, then watch for a ledge
        while done < 30000:
            s, x, _ = run_chain(adj, side_u, 1.0 / G, 0, BLOCK, -1, LAM, NO_CAP, False); done += BLOCK
            if (1.25 - s[-1] / N) / 0.25 >= 0.98:
                break
        windows = []
        for w in range(12):
            hs = []
            for _ in range(600 // BLOCK):
                s, x, _ = run_chain(adj, side_u, 1.0 / G, 0, BLOCK, -1, LAM, NO_CAP, False)
                hs.append((ENERGY_PER_SQUARE * (N - s[-1]) + ENERGY_PER_SURPLUS * LAM * x[-1]) / N)
            windows.append(h0 - float(np.mean(hs)))
            if len(windows) >= 2 and abs(windows[-1] - windows[-2]) < 0.02 * EXPECT:
                if windows[-1] < 0.95 * EXPECT:
                    print("rep %2d: departed at %d (started at x=%s), ON THE LEDGE at %.3f of the release" % (rep, starts[-1][0], sorted(set(starts[-1][1])), windows[-1] / EXPECT))
                    signatures[describe(adj, "ledge")] += 1
                    ledges += 1
                else:
                    print("rep %2d: departed at %d (started at x=%s), full release %.3f" % (rep, starts[-1][0], sorted(set(starts[-1][1])), windows[-1] / EXPECT))
                break
    print("\n=== WHERE IT STARTS: x-position (0..%d along the long axis) of the first non-tube vertices, over %d departures" % (LX - 1, len(starts)))
    c = Counter(x for _, xs in starts for x in set(xs))
    print("   " + " ".join("x=%2d:%2d" % (x, c.get(x, 0)) for x in range(LX)))
    print("   first-departure clusters had %s non-tube vertices (median %.0f)"
          % (sorted(len(xs) for _, xs in starts)[:8], np.median([len(xs) for _, xs in starts])))
    print("\n=== THE LEDGE: %d of %d decays rested on it. Distinct (S, X, d-hist, non-sheet count, clusters) signatures:" % (ledges, reps))
    for sig, k in signatures.most_common():
        print("   %2d x  S=%d X=%d d-hist=%s non-sheet=%d clusters=%d" % (k, *sig))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
