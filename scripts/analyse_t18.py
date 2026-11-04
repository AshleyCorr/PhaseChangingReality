"""The pre-registered T18 reading: how much room does the new space need, as lambda changes? Usage:

    python scripts/analyse_t18.py

PREREGISTRATION.md T18. Each run is classified by T9's rules exactly (analyse_t9.classify: sheet, melted, tube,
stalled, other). C*(lambda) is the smallest C at which a majority of the twenty runs end as a sheet; the reading
is R = C*(1.40) / C*(1.10): PROPORTIONAL if R >= 2.5, WEAKER THAN PROPORTIONAL if 1.5 < R < 2.5, FLAT if
R <= 1.5. Gates: energy exact in every run; a C* exists at each lambda.
"""
import csv
import glob
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse_t9 import classify                                         # noqa: E402


def c_star(by_c):
    """Smallest C with a majority of sheets, or None. by_c: {C: [kind, ...]}."""
    for c in sorted(by_c):
        kinds = by_c[c]
        if kinds.count("sheet") > len(kinds) / 2:
            return c
    return None


def reading(r):
    if r is None:
        return "NOT READ (a C* is missing)"
    if r >= 2.5:
        return "PROPORTIONAL"
    if r > 1.5:
        return "WEAKER THAN PROPORTIONAL"
    return "FLAT"


def main():
    stars = {}
    exact = True
    for f in sorted(glob.glob("results/t18_room_lam*.csv")):
        rows = list(csv.DictReader(open(f, newline="")))
        lam = float(rows[0]["lam"])
        # "to the last unit": energy steps are multiples of 4 lambda (4.4, 5.6 are not exact in binary), so
        # floating-point rounding of order 1e-13 is not a failure; 1e-6 is far below the smallest step
        exact = exact and all(float(r["drift"]) < 1e-6 for r in rows)
        by_c = defaultdict(list)
        for r in rows:
            by_c[int(r["C"])].append(classify(r))      # T9's rules: N, d0..d6 and f_final
        stars[lam] = c_star(by_c)
        print("lambda = %.2f  C* = %s" % (lam, stars[lam]))
        for c in sorted(by_c):
            k = by_c[c]
            print("   C=%-4d sheet %2d melted %2d tube %2d stalled %2d other %2d"
                  % (c, k.count("sheet"), k.count("melted"), k.count("tube"), k.count("stalled"), k.count("other")))
    print("energy exact in every run:", exact)
    r = (stars[1.4] / stars[1.1]) if stars.get(1.4) and stars.get(1.1) else None
    print("R = C*(1.40)/C*(1.10) =", r)
    print("PRE-REGISTERED READING:", reading(r) if exact else "NOT READ (energy gate failed)")


if __name__ == "__main__":
    main()
