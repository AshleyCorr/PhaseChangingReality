"""The pre-registered T21 reading: does a sealed sheet fold rather than melt when points are interchangeable? Usage:

    python scripts/analyse_t21.py

PREREGISTRATION.md T21. At each (size, protocol, budget) the folded share is the mean, over replicas with any
damage, of folded / (folded + melted), for named and for interchangeable points. Per (size, protocol):
FOLDS WITH INTERCHANGEABLE POINTS if at some budget interchangeable >= 0.5 while named < 0.5; MELTS EITHER WAY if
at every budget with damage both are below 0.5; FOLDS EITHER WAY if at some budget both are >= 0.5; otherwise
INCONCLUSIVE. The overall reading is the verdict at N = 64 under the bath protocol. Gate: energy exact.
"""
import csv
import glob
from collections import defaultdict

import numpy as np


def folded_share(rows):
    shares = [int(r["folded"]) / (int(r["folded"]) + int(r["melted"])) for r in rows
              if int(r["folded"]) + int(r["melted"]) > 0]
    return (float(np.mean(shares)), len(shares)) if shares else (None, 0)


def verdict(pairs):
    """pairs: {budget: (named_share or None, interchangeable_share or None)}."""
    damaged = {b: p for b, p in pairs.items() if p[0] is not None and p[1] is not None}
    if not damaged:
        return "INCONCLUSIVE (no damage)"
    if any(i >= 0.5 and n < 0.5 for n, i in damaged.values()):
        return "FOLDS WITH INTERCHANGEABLE POINTS"
    if any(i >= 0.5 and n >= 0.5 for n, i in damaged.values()):
        return "FOLDS EITHER WAY"
    if all(i < 0.5 and n < 0.5 for n, i in damaged.values()):
        return "MELTS EITHER WAY"
    return "INCONCLUSIVE"


def main():
    rows = []
    for f in sorted(glob.glob("results/t21_refold_n*.csv")):
        rows += list(csv.DictReader(open(f, newline="")))
    exact = all(float(r["drift"]) < 1e-6 for r in rows)
    cells = defaultdict(list)
    for r in rows:
        cells[(int(r["N"]), r["protocol"], float(r["budget_per_point"]), r["points"])].append(r)
    overall = None
    for n in sorted({k[0] for k in cells}):
        for proto in ("single", "bath"):
            budgets = sorted({k[2] for k in cells if k[0] == n and k[1] == proto})
            pairs = {}
            print("N=%d, %s:" % (n, proto))
            for b in budgets:
                nm, nn = folded_share(cells.get((n, proto, b, "named"), []))
                it, ni = folded_share(cells.get((n, proto, b, "interchangeable"), []))
                pairs[b] = (nm, it)
                sym = [int(r["symmetries_end"]) for r in cells.get((n, proto, b, "interchangeable"), [])]
                print("   budget %5.1f  folded share: named %s (%d damaged), interchangeable %s (%d damaged); "
                      "interchangeable end symmetries %s" % (b, "%.2f" % nm if nm is not None else "-", nn,
                                                             "%.2f" % it if it is not None else "-", ni, sym))
            v = verdict(pairs)
            print("   -> %s" % v)
            if n == 64 and proto == "bath":
                overall = v
    print("energy exact in every run:", exact)
    print("PRE-REGISTERED OVERALL READING (N = 64, bath):", overall if exact else "NOT READ (energy gate)")


if __name__ == "__main__":
    main()
