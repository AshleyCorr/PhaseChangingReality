"""EXACT, exploratory: fitting the owner's triad to what is measured (VISION Update 35). Usage:

    python scripts/exact_budget_fit.py

The measured inputs (Planck 2018 central values as given by a search summary, to verify; docs/reading/notes/
2026-09-26_dark_charge.md): dark matter / ordinary matter = 5.36, fixed since both were made, because both thin as space
grows; dark energy's share today 68.6 %, but at the first atoms about one part in a billion, and smaller still at any
earlier birth, because its density stays constant while matter's falls. So at the burp the dark-energy opening's release
must be essentially zero next to the matter openings' (a share of about 2.2 / (1 + z_birth)^3). Those are the two
birth numbers: DM : ordinary = 5.36 and DE ≈ 0.

The model (six links, three curled directions). Each direction's curling costs a = 4(lambda - 1) per point; a tie f(d)
per point with d open directions (f(0) = f(3) = 0; f(1), f(2) the two constants; VISION Update 30) shifts the three
releases per point to r1 = a - f(1), r2 = a + f(1) - f(2), r3 = a + f(2), in the order of opening. For each of the six
ways of assigning ordinary matter, dark matter and dark energy to the three openings, the two birth numbers fix f(1) and
f(2) as multiples of a. The walls are then priced exactly (exact_walls_tie_d.kinds) for X (the gas of 6-cubes), the
two-curled and one-curled rungs, and flat space, across lambda, and each order is checked against the owner's
requirements: X stuck (its wall > 0), flat space stable (its wall > 0), and the change running once started (the later
walls <= 0), with "the first push the largest" reported. The follow form (f(1) = 2 kappa, f(2) = kappa) is one line of
this family and is reported for the order ordinary, dark matter, dark energy.
"""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exact_walls_tie_d as tie    # noqa: E402

RATIO = 5.36
SPECS = {"X (gas of 6-cubes)": "4,4,4x8", "two curled": "4,4,18", "one curled": "4,12,12", "flat": "6,6,8"}


def shares():
    ordinary = 1.0 / (1.0 + RATIO)
    return {"ordinary": ordinary, "dark matter": RATIO * ordinary, "dark energy": 0.0}


def main():
    kinds = {}
    for name, sp in SPECS.items():
        (adj, part), _ = tie.parse(sp)
        kinds[name] = [k for k in tie.kinds(adj, part) if not tie.is_null(k)]

    def walls(lam, f1, f2):
        f = {1: f1, 2: f2}
        return {n: min(-16 * k[0] + 4 * lam * k[1] + sum(f.get(d, 0) * c for d, c in k[4]) for k in kinds[n])
                for n in SPECS}

    sh = shares()
    print("birth shares used: ordinary %.4f, dark matter %.4f, dark energy %.1f (of the three releases)"
          % (sh["ordinary"], sh["dark matter"], sh["dark energy"]))
    for order in itertools.permutations(["ordinary", "dark matter", "dark energy"]):
        r = [3 * sh[k] for k in order]            # releases per point in units of a
        f1, f2 = 1 - r[0], r[2] - 1                # in units of a
        ok = []
        for lam in np.arange(1.01, 1.61, 0.01):
            a = 4 * (lam - 1)
            w = walls(lam, f1 * a, f2 * a)
            stuck = w["X (gas of 6-cubes)"] > 0
            flat = w["flat"] > 0
            runs = w["two curled"] <= 0 and w["one curled"] <= 0
            if stuck and flat:
                ok.append((round(lam, 2), runs, round(w["X (gas of 6-cubes)"], 1), round(w["two curled"], 1),
                           round(w["one curled"], 1), round(w["flat"], 1)))
        runs_at = [o[0] for o in ok if o[1]]
        print("\norder %s: releases per point r = %s a; tie f(1) = %.3f a, f(2) = %.3f a"
              % (" -> ".join(order), [round(x, 3) for x in r], f1, f2))
        print("  lambda with X stuck and flat stable: %s" % ([o[0] for o in ok][:1] + ["..."] + [o[0] for o in ok][-1:]
                                                           if ok else "none"))
        print("  of those, the change runs after the first push (later walls <= 0): %s" % (runs_at or "none"))
        for o in ok[:: max(1, len(ok) // 4)]:
            print("    lambda %.2f: walls X %.1f, two-curled %.1f, one-curled %.1f, flat %.1f" % (o[0], *o[2:]))
    # the follow form, one constant, for the owner's order
    kap = RATIO * 1 / (1 + 2 * RATIO + 1) if False else None
    print("\nfollow form (f1 = 2k, f2 = k), order ordinary -> dark matter -> dark energy: r = (a - 2k, a + k, a + k);"
          " DM/ordinary = 5.36 needs k = %.3f a, and then dark energy = dark matter at birth, not ~0." % (4.36 / 11.72))


if __name__ == "__main__":
    main()
