"""EXACT, exploratory: the owner's order of 26 September with time as a fourth opening (VISION Update 36). Usage:

    python scripts/exact_budget_fit_4d.py [--cache=path.pkl]

Her order: dark energy opens first and releases almost nothing at the burp; dark matter opens second, free, and its
release pays the third opening, ordinary matter; ordinary matter pays to open time, which is why it started lower.

The model (eight links, four curled directions). Each direction's curling costs a = 4(lambda - 1) per point; a tie f(d)
per point with d open directions (f(0) = f(4) = 0; f(1), f(2), f(3) its constants) shifts the four releases per point,
in the order of opening, to r1 = a - f1, r2 = a + f1 - f2, r3 = a + f2 - f3, r4 = a + f3. They always add to 4a.
"Ordinary pays for time" is read as: what is observed as ordinary matter is what is left after time's opening, r3 + r4.
The measured inputs are those of O74 (dark matter / ordinary = 5.36; dark energy ~ 0 at birth). So r1 = 0,
r2 = 4a 5.36/6.36, r3 + r4 = 4a/6.36, which fixes f1 = a, f2 = 2a - r2 = -1.371a, and leaves one constant free:
time's own release r4 = rho a (rho < 0: time's opening takes energy, paid by ordinary's; rho = 0: it takes none).
f3 = (rho - 1) a.

The walls are priced exactly (exact_walls_tie_d.kinds, every switch from one u1, whole-graph recomputation) on the
2,304-point ladder of O50 at the rung sizes of this script's SPECS, and each (lambda, rho) is checked against: X stuck
(its wall > 0), flat space stable (its wall > 0), dark matter's opening free (the second wall <= 0), and whether the
three-open state lies below flat space (then flat four-direction space is not the lowest state and time's opening is
uphill: the rung energy per point of three-open-one-curled above flat is r4).

Ours; exact on the built states; the assignment of kinds to openings is the owner's hypothesis, not the model's.
"""
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exact_walls_tie_d as tie    # noqa: E402

RATIO = 5.36
SPECS = {"X (gas of 8-cubes)": "4,4,4,4x4", "three curled": "4,4,4,12", "two curled": "4,4,8,8",
         "one curled": "4,8,8,8", "flat": "6,6,6,6"}
ORDER = ["X (gas of 8-cubes)", "three curled", "two curled", "one curled", "flat"]


def tie_for(rho):
    """(f1, f2, f3) in units of a for time's release rho a, the other shares fixed by the measured inputs."""
    r2 = 4.0 * RATIO / (1.0 + RATIO)
    return 1.0, 2.0 - r2, rho - 1.0


def releases(rho):
    f1, f2, f3 = tie_for(rho)
    return 1.0 - f1, 1.0 + f1 - f2, 1.0 + f2 - f3, 1.0 + f3


def walls(kinds, lam, rho):
    a = 4.0 * (lam - 1.0)
    f = dict(zip((1, 2, 3), (x * a for x in tie_for(rho))))
    return {n: min(-16 * k[0] + 4 * lam * k[1] + sum(f.get(d, 0.0) * c for d, c in k[4]) for k in kinds[n])
            for n in ORDER}


def load(cache):
    if cache and Path(cache).exists():
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    kinds = {}
    for name in ORDER:
        (adj, part), _ = tie.parse(SPECS[name])
        kinds[name] = [k for k in tie.kinds(adj, part) if not tie.is_null(k)]
    if cache:
        with open(cache, "wb") as fh:
            pickle.dump(kinds, fh)
    return kinds


def main(argv):
    cache = next((x.split("=", 1)[1] for x in argv if x.startswith("--cache=")), None)
    kinds = load(cache)
    f1, f2, _ = tie_for(0.0)
    print("fixed by the measured inputs: f1 = %.3f a, f2 = %.3f a; f3 = (rho - 1) a" % (f1, f2))
    for rho in (0.2, 0.1, 0.0, -0.1, -0.2, -0.4):
        r = releases(rho)
        print("\nrho = %+.1f: releases per point (DE, DM, ordinary, time) = %s a; ordinary net %.3f a"
              % (rho, [round(x, 3) for x in r], r[2] + r[3]))
        good = []
        for lam in np.arange(1.02, 1.81, 0.02):
            w = walls(kinds, lam, rho)
            stuck, flat, free2 = w[ORDER[0]] > 0, w["flat"] > 0, w[ORDER[1]] <= 0
            if stuck and flat:
                good.append((lam, free2, [round(w[n], 1) for n in ORDER]))
        if not good:
            print("  no lambda with X stuck and flat stable")
        for lam, free2, ws in good[:: max(1, len(good) // 5)]:
            print("  lambda %.2f  walls X, 1-open, 2-open, 3-open, flat = %s  DM free: %s" % (lam, ws, free2))
        free = [round(g[0], 2) for g in good if g[1]]
        print("  lambda with X stuck, flat stable and the dark-matter opening free: %s"
              % ("%s to %s" % (free[0], free[-1]) if free else "none"))
        if rho < 0:
            print("  three-open-one-curled lies %.3f a per point BELOW flat: flat four-direction space is not the lowest"
                  " state" % -rho)


if __name__ == "__main__":
    main(sys.argv[1:])
