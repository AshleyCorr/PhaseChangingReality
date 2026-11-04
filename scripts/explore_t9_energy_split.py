"""EXPLORATORY, not pre-registered: where the energy went in the T9 sealed boxes. Usage:

    python scripts/explore_t9_energy_split.py

Reads the T9 end states (results/t9_n*_C*.csv, rows with final == 1), classifies each with the
pre-registered classifier of scripts/analyse_t9.py, and splits the conserved total (the tube's
N * 4(lambda - 1) plus the 12-unit spark) into what the network still holds above the flat sheet,
H = N * (16 (1 - phi) + 4 lambda * surplus), and what sits in the bath. Written 2026-09-22 to
answer the author's question whether a dark-matter-like leftover could be "a quality of
spacetime" -- energy kept spread through the geometry -- rather than a few scraps per seed.
A first look at existing data; nothing here is a finding (CLAUDE.md rule 7).
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyse_t9 as T9       # noqa: E402  (the pre-registered end-state classifier)

LAM = 1.25
SPARK = 12.0


def split(row):
    """(held in the network above the sheet, in the bath, conservation residual) for one end state."""
    n = int(row["N"])
    held = n * (16.0 * (1.0 - float(row["phi"])) + 4.0 * LAM * float(row["surplus"]))
    bath = float(row["bath_total"])
    return held, bath, held + bath - (n * 4.0 * (LAM - 1.0) + SPARK)


def main(out_dir="results"):
    groups, totals = defaultdict(list), defaultdict(int)
    for f in Path(out_dir).glob("t9_n*_C*.csv"):
        for r in csv.DictReader(open(f, newline="")):
            if r["final"] != "1":
                continue
            kind = T9.classify(r)
            totals[kind] += 1
            groups[(int(r["N"]), int(r["C"]), kind)].append(split(r))
    print("EXPLORATORY. End states over all T9 runs: %s (total %d)\n" % (dict(totals), sum(totals.values())))
    print("%-5s %-5s %-7s %-5s %-10s %-9s %-14s %s" % ("N", "C", "kind", "runs", "held", "bath", "share held", "max residual"))
    for key in sorted(groups):
        a = np.array(groups[key])
        print("%-5d %-5d %-7s %-5d %-10.1f %-9.1f %-14.3f %.1e"
              % (key + (len(a), a[:, 0].mean(), a[:, 1].mean(), (a[:, 0] / (a[:, 0] + a[:, 1])).mean(),
                        np.abs(a[:, 2]).max())))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
