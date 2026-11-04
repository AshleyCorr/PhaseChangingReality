"""The pre-registered T17 reading: does each seed leave its own leftover? Usage:

    python scripts/analyse_t17.py

PREREGISTRATION.md T17. Gate 1: energy drift 0 in every run. Gate 2: at every k, at least 18 of 20 runs
convert (f_final >= 0.9). The leftover count is the number of pieces at d = 1 (T10's observable); the
count of pieces of exactly four is reported beside it. The slope of the count against k, by least squares
over runs passing gate 2, with its standard error, decides the verdict:
ONE PER SEED 0.75 to 1.25; ONE PER TUBE within +-0.25 of 0; BETWEEN 0.25 to 0.75; otherwise INCONCLUSIVE.
The pre-registration also says "mean near 1 at every k" for ONE PER TUBE without a number; the verdict is
read on the slope, as its verdict table states, and the means are reported beside it so either reading
can be checked.
"""
import csv
import glob
import math

import numpy as np


def slope(ks, counts):
    ks, counts = np.asarray(ks, float), np.asarray(counts, float)
    kbar = ks.mean()
    sxx = ((ks - kbar) ** 2).sum()
    b = ((ks - kbar) * (counts - counts.mean())).sum() / sxx
    resid = counts - (counts.mean() + b * (ks - kbar))
    se = math.sqrt((resid ** 2).sum() / (len(ks) - 2) / sxx) if len(ks) > 2 else math.nan
    return b, se


def verdict(b, gates_ok):
    if not gates_ok:
        return "INCONCLUSIVE (a gate failed)"
    if 0.75 <= b <= 1.25:
        return "ONE PER SEED"
    if abs(b) <= 0.25:
        return "ONE PER TUBE"
    if 0.25 < b < 0.75:
        return "BETWEEN"
    return "INCONCLUSIVE (slope outside every band)"


def main():
    rows = []
    for f in sorted(glob.glob("results/t17_seeds_k*.csv")):
        rows += list(csv.DictReader(open(f, newline="")))
    by_k = {}
    for r in rows:
        by_k.setdefault(int(r["k"]), []).append(r)
    gate1 = all(float(r["drift"]) == 0.0 for r in rows)
    gates_ok = gate1
    ks, counts = [], []
    print("T17, PREREGISTRATION.md. Gate 1 (energy exact in every run): %s" % ("pass" if gate1 else "FAIL"))
    for k in sorted(by_k):
        rs = by_k[k]
        conv = [r for r in rs if float(r["f_final"]) >= 0.9]
        g2 = len(conv) >= 18
        gates_ok = gates_ok and g2
        c = [int(r["pieces_d1"]) for r in conv]
        fours = [sum(1 for s in r["sizes_d1"].split() if s == "4") for r in conv]
        ks += [k] * len(c)
        counts += c
        print("  k=%d: %d of %d converted (gate 2 %s); leftovers per tube mean %.2f (distribution %s); "
              "pieces of exactly four %.2f" % (k, len(conv), len(rs), "pass" if g2 else "FAIL", np.mean(c),
                                               {v: c.count(v) for v in sorted(set(c))}, np.mean(fours)))
    b, se = slope(ks, counts)
    print("  slope %.3f +- %.3f leftovers per extra seed" % (b, se))
    print("PRE-REGISTERED VERDICT:", verdict(b, gates_ok))


if __name__ == "__main__":
    main()
