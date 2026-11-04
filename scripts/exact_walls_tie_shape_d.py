"""EXACT, exploratory: walls under a direction tie of any shape (VISION Update 30). Usage:

    python scripts/exact_walls_tie_shape_d.py --lam=1.25 --f=0,0.85,1.04,0 4,4,4x8 4,4,18 4,12,12 6,6,8

f lists the tie energy of a point with d = 0, 1, ..., D open directions (per point; a broken point, d > D, counts 0). The
follow form of Update 30 is f(d) = kappa (D - d) for 1 <= d <= D. Written 2026-09-25 to check whether the shape the
owner's triad needs (the first opening netting the least, later ones more) keeps X stuck and flat space stable. Each switch
is priced exactly from `exact_walls_tie_d.kinds` (the change in the count of points at each d).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exact_walls_tie_d as tie    # noqa: E402


def main(argv):
    lam, table, specs = 1.25, None, []
    for a in argv:
        if a.startswith("--lam="):
            lam = float(a[6:])
        elif a.startswith("--f="):
            table = [float(x) for x in a[4:].split(",")]
        else:
            specs.append(a)

    def f(d):
        return table[d] if d < len(table) else 0.0

    for spec in specs:
        (adj, part), _ = tie.parse(spec)
        n, dim, _, _, _, hist = tie.rung(adj, lam)
        assert len(table) == dim + 1, "f needs D + 1 values"
        t_rung = sum(f(k) * c for k, c in hist.items())
        ks = tie.kinds(adj, part)

        def cost(k):
            return -16.0 * k[0] + 4.0 * lam * k[1] + sum(f(d) * c for d, c in k[4])

        best = min((k for k in ks if not tie.is_null(k)), key=lambda k: (cost(k), k))
        h = float(tie.hamiltonian(adj, lam))
        print("%-10s N=%-5d rung above flat %.1f (%.3f per point) | wall %.2f (dS %d, dX %d)"
              % (spec, n, h + t_rung, (h + t_rung) / n, cost(best), best[0], best[1]), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
