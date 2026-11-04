"""EXACT, exploratory: the owner's ledger (VISION Update 33) inside one fully curled 6-cube's worth of wiring. Usage:

    python scripts/exact_ledger_per_cube.py

Her question (26 September): is there a λ at which the first opening costs 44 and releases 66, and then what does the
second cost, and is the third free? A 6-cube has 64 points and each curled direction releases 4(λ − 1) per point, so a
release of 66 per direction per cube fixes λ = 1 + 66/256 = 1.2578. The walls are the cheapest single moves out of the
fully curled gas (4,4,4x8), the two-curled rung (4,4,18) and the one-curled rung (4,12,12), priced exactly by
`exact_walls_tie_d.kinds` (walls are local, so a torus with a long open side prices the patch). Three ties are tried:
none; the follow form of VISION Update 30, f(d) = κ(D − d), at κ = 2.03 (first wall 44) and 2.25; and every two-constant
tie f(1), f(2) on a grid from −2 to 8, searched for the walls closest to 44, 12, 0. Recorded in ASSUMPTIONS O71's addendum.

Added the same morning (O73; the owner's follow-up questions): (i) the order 44, none, 12, solved for (λ, κ) under the
follow tie and checked by exact pricing; (ii) the same walls on short tori (4,4,6 and 4,6,6), where a short open side
changes the cheapest move (O49), to see whether another arrangement reaches 44, 12, 0; (iii), the counting cost of each opening
with interchangeable points, is arithmetic on O55's exact renaming counts and is written out in O73, not computed here.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exact_walls_tie_d as tie    # noqa: E402

LAM = 1 + 66 / 256
SPECS = ["4,4,4x8", "4,4,18", "4,12,12"]


def main():
    kinds = {}
    for sp in SPECS:
        (adj, part), _ = tie.parse(sp)
        kinds[sp] = [k for k in tie.kinds(adj, part) if not tie.is_null(k)]

    def walls(f1, f2):
        f = {1: f1, 2: f2}
        return [min(-16 * k[0] + 4 * LAM * k[1] + sum(f.get(d, 0) * c for d, c in k[4]) for k in kinds[sp])
                for sp in SPECS]

    a = 4 * (LAM - 1)
    print("lambda = %.4f; release per direction per 6-cube without a tie: %.1f" % (LAM, 64 * a))
    print("walls (X, then two-curled, then one-curled), no tie: %s" % [round(w, 1) for w in walls(0, 0)])
    for kap in (2.03, 2.25):
        print("follow tie kappa=%.2f: walls %s; per-cube releases %s" % (
            kap, [round(w, 1) for w in walls(2 * kap, kap)],
            [round(64 * x, 1) for x in (a - 2 * kap, a + kap, a + kap)]))
    best = None
    grid = np.arange(-2, 8.01, 0.05)
    for f1 in grid:
        for f2 in grid:
            w = walls(f1, f2)
            d = (w[0] - 44) ** 2 + (w[1] - 12) ** 2 + w[2] ** 2
            if best is None or d < best[0]:
                best = (d, f1, f2, w)
    d, f1, f2, w = best
    print("closest two-constant tie to 44, 12, 0: f1=%.2f f2=%.2f walls %s, miss %.1f; per-cube releases %s" % (
        f1, f2, [round(x, 1) for x in w], d ** 0.5, [round(64 * x, 1) for x in (a - f1, a + f1 - f2, a + f2)]))


def follow_walls(kinds_by_spec, specs, lam, kappa):
    f = {1: 2 * kappa, 2: kappa}
    return [min(-16 * k[0] + 4 * lam * k[1] + sum(f.get(d, 0) * c for d, c in k[4]) for k in kinds_by_spec[sp])
            for sp in specs]


def more():
    specs = SPECS + ["4,4,6", "4,6,6"]
    kinds = {}
    for sp in specs:
        (adj, part), _ = tie.parse(sp)
        kinds[sp] = [k for k in tie.kinds(adj, part) if not tie.is_null(k)]
    # (i) 44, none, 12: bisection on the two linear conditions, then exact pricing
    best = None
    for lam in np.arange(1.10, 1.30, 0.0005):
        for kap in np.arange(1.0, 2.5, 0.005):
            w = follow_walls(kinds, SPECS, lam, kap)
            d = (w[0] - 44) ** 2 + (w[2] - 12) ** 2 + (max(w[1], 0)) ** 2
            if best is None or d < best[0]:
                best = (d, lam, kap, w)
    d, lam, kap, w = best
    a = 4 * (lam - 1)
    print("44, none, 12 under the follow tie: lambda=%.4f kappa=%.3f walls %s; per-cube releases %s (untied %.1f each)" % (
        lam, kap, [round(x, 1) for x in w], [round(64 * x, 1) for x in (a - 2 * kap, a + kap, a + kap)], 64 * a))
    # (ii) short tori
    for lam2, kap2 in ((1 + 66 / 256, 0.0), (1 + 66 / 256, 2.03)):
        print("short tori at lambda=%.4f kappa=%.2f: 4,4,6 wall %.1f; 4,6,6 wall %.1f" % (
            lam2, kap2, *follow_walls(kinds, ["4,4,6", "4,6,6"], lam2, kap2)))
    best = None
    for f1 in np.arange(-2, 8.01, 0.05):
        for f2 in np.arange(-2, 8.01, 0.05):
            f = {1: f1, 2: f2}
            w = [min(-16 * k[0] + 4 * LAM * k[1] + sum(f.get(d, 0) * c for d, c in k[4]) for k in kinds[sp])
                 for sp in ("4,4,4x8", "4,4,6", "4,6,6")]
            dd = (w[0] - 44) ** 2 + (w[1] - 12) ** 2 + w[2] ** 2
            if best is None or dd < best[0]:
                best = (dd, f1, f2, w)
    print("short tori, closest two-constant tie to 44, 12, 0: f1=%.2f f2=%.2f walls %s, miss %.1f" % (
        best[1], best[2], [round(x, 1) for x in best[3]], best[0] ** 0.5))


if __name__ == "__main__":
    main()
    more()
