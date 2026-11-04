"""EXPLORATORY: how steep is each protocol's curve, against [T22] Fig. 3? Usage:

    python scripts/explore_gate_c_protocols.py results/explore_c_*.csv results/gatec_t22_fig3_p1s_a.csv

Not a gate and not pre-registered (ASSUMPTIONS O43 follow-up). For each run and each leg: the ln g where squares per
vertex crosses 9 and 2, and the width between them. The published curve has 9 at ln g 0.666 and 2 at 1.311, a width
of 0.646. The shift needed to line a leg up with the published curve at the 9 crossing is reported too, as the
factor our coupling would have to be divided by.
"""
import csv
import math
import sys
from pathlib import Path

import numpy as np

PUBLISHED = (0.666, 1.311)


def crossing(x, y, level):
    """ln g where the curve (ordered by ln g) first passes through level."""
    o = np.argsort(x)
    x, y = np.asarray(x)[o], np.asarray(y)[o]
    for i in range(len(x) - 1):
        if (y[i] - level) * (y[i + 1] - level) <= 0 and y[i] != y[i + 1]:
            return float(x[i] + (level - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))
    return math.nan


def main(paths):
    print("published [T22] Fig. 3: 9 at %.3f, 2 at %.3f, width %.3f" % (*PUBLISHED, PUBLISHED[1] - PUBLISHED[0]))
    for path in paths:
        rows = list(csv.DictReader(open(path, newline="")))
        for leg in ("cool", "heat"):
            r = [q for q in rows if q["leg"] == leg]
            x = [float(q["ln_g"]) for q in r]
            y = [float(q["squares_per_vertex"]) for q in r]
            a, b = crossing(x, y, 9.0), crossing(x, y, 2.0)
            top = max(y)
            print("%-32s %-4s 9 at %6.3f, 2 at %6.3f, width %6.3f; highest %5.2f; divide our g by %5.2f to line up at 9"
                  % (Path(path).stem, leg, a, b, b - a, top, math.exp(a - PUBLISHED[0]) if not math.isnan(a) else math.nan))


if __name__ == "__main__":
    main(sys.argv[1:])
