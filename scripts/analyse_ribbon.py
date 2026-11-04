"""Does the leftover of the shattered phase form one closed piece? Usage:

    python scripts/analyse_ribbon.py [results/cqg_lam0_ribbon.csv]

Prints to the screen and writes nothing.

THE PREDICTION BEING TESTED, which is not ours. [GV21] reports that the cold phase at the
penalty-off setting is about N/16 knots PLUS one closed bipartite "ribbon" holding whatever is
left over. Our earlier runs could not see it: their sizes were all multiples of 16, so the
graph shatters into a whole number of 4-cubes with no remainder at all. The sizes here are not.

THE TWO PICTURES, and they are far apart. In the shattered phase, if the leftover is a single
closed piece then it IS the largest piece whenever it is bigger than one knot, so

    one ribbon :  largest_frac  =  1 - baby_frac        and   pieces ~ baby_frac*N/16 + 1
    scattered  :  largest_frac  ~  16/N                 and   pieces much larger

The columns are averages over the measurement sweeps, so a row is a blur over whatever the run
visited; this can rule a picture out but confirming one properly needs a snapshot.
"""
import csv
import sys
from collections import defaultdict

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "results/cqg_lam0_ribbon.csv"
with open(path, newline="") as fh:
    rows = [r for r in csv.DictReader(fh)]

f = lambda r, k: float(r[k])
shattered = [r for r in rows if f(r, "baby_frac") > 0.5]
partial = [r for r in rows if 0.15 < f(r, "baby_frac") <= 0.95]

print("Ribbon test: %d rows, %d in the shattered phase (baby_frac > 0.5)\n" % (len(rows), len(shattered)))
print("%-6s %-6s %-7s %-8s %-8s %-9s %-11s %-11s %s"
      % ("N", "leg", "g", "phi", "pieces", "baby", "largest", "if ribbon", "if scattered"))

by = defaultdict(list)
for r in shattered:
    by[(int(r["N"]), r["g"])].append(r)

votes = {"ribbon": 0, "scattered": 0, "neither": 0}
for key in sorted(by, key=lambda k: (k[0], float(k[1]))):
    rs = by[key]
    n = int(rs[0]["N"])
    baby = np.mean([f(r, "baby_frac") for r in rs])
    largest = np.mean([f(r, "largest_frac") for r in rs])
    pieces = np.mean([f(r, "pieces") for r in rs])
    phi = np.mean([f(r, "phi") for r in rs])
    ribbon_pred, scatter_pred = 1.0 - baby, 16.0 / n
    d_r, d_s = abs(largest - ribbon_pred), abs(largest - scatter_pred)
    call = "ribbon" if d_r < d_s and d_r < 0.03 else "scattered" if d_s < 0.03 else "neither"
    votes[call] += 1
    print("%-6d %-6s %-7s %-8.3f %-8.2f %-9.4f %-11.4f %-11.4f %-11.4f  %s"
          % (n, rs[0]["leg"], rs[0]["g"], phi, pieces, baby, largest,
             ribbon_pred, scatter_pred, call))

print("\nRows favouring one ribbon: %d   scattered: %d   neither: %d"
      % (votes["ribbon"], votes["scattered"], votes["neither"]))

if partial:
    print("\nAnd the rows where shattering is INCOMPLETE, which is the half-converted state"
          "\nthat task T11 is about. If the unconverted remainder is one connected piece, the"
          "\nsame equality should hold there too:\n")
    print("%-6s %-7s %-8s %-8s %-9s %-11s %-11s %s"
          % ("N", "g", "phi", "pieces", "baby", "largest", "1 - baby", "agree?"))
    pby = defaultdict(list)
    for r in partial:
        pby[(int(r["N"]), r["g"])].append(r)
    agree = total = 0
    for key in sorted(pby, key=lambda k: (k[0], float(k[1]))):
        rs = pby[key]
        n = int(rs[0]["N"])
        baby = np.mean([f(r, "baby_frac") for r in rs])
        largest = np.mean([f(r, "largest_frac") for r in rs])
        ok = abs(largest - (1 - baby)) < 0.03
        agree += ok
        total += 1
        print("%-6d %-7s %-8.3f %-8.2f %-9.4f %-11.4f %-11.4f %s"
              % (n, rs[0]["g"], np.mean([f(r, "phi") for r in rs]),
                 np.mean([f(r, "pieces") for r in rs]), baby, largest, 1 - baby,
                 "yes" if ok else "no"))
    print("\n  %d of %d half-converted rows have the leftover in a single piece." % (agree, total))
