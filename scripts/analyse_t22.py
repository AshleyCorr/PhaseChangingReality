"""The pre-registered T22 reading: near lambda = 1, do exits from the curled torus fall back? Usage:

    python scripts/analyse_t22.py

PREREGISTRATION.md T22. At each (lambda, N): E, the mean exits per decay that went through, with its standard
error; T1, the mean sweeps to the first exit, with its standard error; tau from Eq. (2). F (fall-backs present):
E - 1 > 2 SE. L (first exit late): T1 - tau > 2 SE. Verdict read at lambda = 1.05 over both sizes: FALL-BACKS
(F, not L), LATE FIRST EXIT (L, not F), BOTH, NEITHER, or MIXED if the sizes disagree.

The historical score conditions both quantities on successful conversion. Its output
is retained explicitly as conditional; a separate first-exit summary includes every
observed exit, including replicas that never reached the conversion criterion.
The reciprocal mean exit count is a diagnostic, not a measured transmission probability.
"""
import csv
import glob
import math
from collections import defaultdict

import numpy as np


def tau(lam, g=1.5):
    return 1.0 / (3 * math.exp(-(32 - 16 * lam) / g) + 2 * math.exp(-(64 - 40 * lam) / g))


def flags(exits, first, t):
    exits, first = np.asarray(exits, float), np.asarray(first, float)
    se_e = exits.std(ddof=1) / math.sqrt(len(exits))
    se_t = first.std(ddof=1) / math.sqrt(len(first))
    f = exits.mean() - 1 > 2 * se_e
    late = first.mean() - t > 2 * se_t
    return f, late, exits.mean(), se_e, first.mean(), se_t


def verdict(pairs):
    """pairs: [(F, L), ...] for the sizes at lambda = 1.05."""
    names = {(True, False): "FALL-BACKS", (False, True): "LATE FIRST EXIT", (True, True): "BOTH",
             (False, False): "NEITHER"}
    kinds = {names[p] for p in pairs}
    return kinds.pop() if len(kinds) == 1 else "MIXED"


def observed_first_exits(rows):
    """All recorded first exits; conversion success is not a selection criterion."""
    return np.array([float(r["first_exit_sweeps"]) for r in rows
                     if r["first_exit_sweeps"] not in ("", None)
                     and float(r["first_exit_sweeps"]) >= 0], dtype=float)


def main():
    rows = []
    for f in sorted(glob.glob("results/t22_exits_*.csv")):
        rows += list(csv.DictReader(open(f, newline="")))
    cells = defaultdict(list)
    for r in rows:
        cells[(float(r["lam"]), int(r["N"]))].append(r)
    at_105 = []
    for (lam, n) in sorted(cells):
        rs = [r for r in cells[(lam, n)] if r["went_through"] == "True"]
        e = [int(r["exits"]) for r in rs]
        t1 = [float(r["first_exit_sweeps"]) for r in rs]
        f, late, em, se, tm, st = flags(e, t1, tau(lam))
        print("lambda=%.2f N=%-3d through %d of %d | exits per successful decay %.2f +- %.2f (reciprocal mean %.2f) %s | "
              "conditional first exit %.0f +- %.0f sweeps vs two-channel approximation %.0f %s | mean end %.0f sweeps"
              % (lam, n, len(rs), len(cells[(lam, n)]), em, se, 1 / em, "F" if f else "-", tm, st, tau(lam),
                 "L" if late else "-", np.mean([float(r["end_sweeps"]) for r in rs])))
        observed = observed_first_exits(cells[(lam, n)])
        if observed.size:
            print("  All observed first exits: %d/%d, mean %.3f sweeps; not selected by conversion."
                  % (observed.size, len(cells[(lam, n)]), observed.mean()))
        if abs(lam - 1.05) < 1e-9:
            at_105.append((f, late))
    print("PRE-REGISTERED VERDICT (lambda = 1.05; historical conditional scoring):",
          verdict(at_105) if at_105 else "no data")


if __name__ == "__main__":
    main()
