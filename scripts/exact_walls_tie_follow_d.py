"""EXACT, exploratory: walls under the "follow" form of the direction tie (the owner's triad, VISION Update 30). Usage:

    python scripts/exact_walls_tie_follow_d.py --lam=1.10,1.25 --kappa=0,1,2,4,8 4,4,4x8 4,4,18 4,12,12 6,6,8

The first form of Update 30, kappa d (D - d), was found by exact counting to reward broken points (where the count of
open pairs exceeds D) and, even with those counted as zero, to make the middle of the ladder the hardest step, so that no
cascade can run from the curled side (docs/design/direction_tie_first_look.md). The owner's triad picture is asymmetric:
the push must open one direction ("red"), and once one is open the directions still curled are driven to follow. The
simplest form with that shape, written before this calculation:

    f(d) = kappa (D - d) for 1 <= d <= D;  f(0) = 0;  f(d) = 0 for d > D (a broken point has no directions to tie).

A fully curled point costs nothing, a point with one open direction costs kappa for each still curled, a flat point
nothing. Each switch is priced from `exact_walls_tie_d.kinds`, which records how many points change at each d, so the tie
change of any form is exact: sum over d of f(d) times the change in the count at d.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exact_walls_tie_d as tie    # noqa: E402


def f(d, dim):
    return (dim - d) if 1 <= d <= dim else 0


def tie_change(kind, dim):
    return sum(f(k, dim) * c for k, c in kind[4])


def cost(kind, lam, kappa, dim):
    ds, dx = kind[0], kind[1]
    return -16.0 * ds + 4.0 * lam * dx + kappa * tie_change(kind, dim)


def main(argv):
    lams, kappas, specs = [1.25], [0.0], []
    for a in argv:
        if a.startswith("--lam="):
            lams = [float(x) for x in a[6:].split(",")]
        elif a.startswith("--kappa="):
            kappas = [float(x) for x in a[8:].split(",")]
        else:
            specs.append(a)
    for spec in specs:
        (adj, part), _ = tie.parse(spec)
        n, dim, _, _, _, hist = tie.rung(adj, lams[0])
        t_rung = sum(f(k, dim) * c for k, c in hist.items())
        ks = tie.kinds(adj, part)
        for lam in lams:
            h = float(tie.hamiltonian(adj, lam))
            for kappa in kappas:
                best = min((k for k in ks if not tie.is_null(k)), key=lambda k: (cost(k, lam, kappa, dim), k))
                print("%-12s N=%-5d D=%d lam=%.2f kappa=%-4g | rung above flat: H %.1f + tie %.1f = %.1f (%.3f per point) | "
                      "wall %.2f (dS %d, dX %d, tie change %d)"
                      % (spec, n, dim, lam, kappa, h, kappa * t_rung, h + kappa * t_rung, (h + kappa * t_rung) / n,
                         cost(best, lam, kappa, dim), best[0], best[1], tie_change(best, dim)), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
