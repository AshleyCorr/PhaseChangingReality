"""EXPLORATORY: which reading of [T22] Fig. 3's axis best maps our six-link curve onto it? Usage:

    python scripts/explore_gate_c_readings.py results/explore_c_heat500.csv results/gatec_t22_fig3_p1a.csv ...

Not a gate and not pre-registered (ASSUMPTIONS O43, O45 follow-up; written 2026-09-24, night, at the owner's
request to make guesses about the failed gate and test them). Each reading is a map from our coupling g to the
published axis, x = a ln g + b, and each is a named hypothesis about how the published run and ours differ:

  (i)   x = ln g               ħg is our g (the paper's Eqs. (1) to (3) read with one factor of g in the weight)
  (ii)  x = ln g − ln N^(1/3)  the published axis divided by N^(1 − 2/D) (O43's second reading)
  (iii) x = ln g − ln 2        a factor of 2 in the action's normalization
  (iv)  x = ½ ln g             the weight goes as 1/g² (Eq. (3) read literally: the action of Eq. (1) already
                               carries 1/g and the exponent divides by ħg again)
  (v)   x = ln g − ln 5.5      O45's empirical factor for a fast heating leg
  (fit) x = a ln g + b, a and b free: what the data would choose with two free numbers, for reference only

For each result file and leg: the rms difference from the published points over the overlap, the ln-axis
positions of the crossings of 9 and 2 squares per vertex against the published 0.666 and 1.311, and the
width between them (published 0.645). A reading that puts the crossings and the width right with no free
number is the one to pre-register as Gate C′; the free fit says how much better two free numbers could do.
"""
import csv
import math
import sys
from pathlib import Path

import numpy as np

TARGET = Path(__file__).resolve().parents[1] / "docs" / "published" / "T22_fig3_digitised.csv"
PUB_9, PUB_2 = 0.666, 1.311
READINGS = {
    "(i)   x = ln g": (1.0, 0.0),
    "(ii)  x = ln g - ln N^(1/3)": (1.0, -math.log(500 ** (1 / 3))),
    "(iii) x = ln g - ln 2": (1.0, -math.log(2)),
    "(iv)  x = (1/2) ln g": (0.5, 0.0),
    "(v)   x = ln g - ln 5.5": (1.0, -math.log(5.5)),
}


def crossing(x, y, level):
    o = np.argsort(x)
    x, y = np.asarray(x)[o], np.asarray(y)[o]
    for i in range(len(x) - 1):
        if (y[i] - level) * (y[i + 1] - level) <= 0 and y[i] != y[i + 1]:
            return float(x[i] + (level - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))
    return math.nan


def rms(x, y, tx, ty):
    """rms of ours minus published, over the published points that lie inside our mapped range."""
    o = np.argsort(x)
    x, y = np.asarray(x)[o], np.asarray(y)[o]
    inside = (tx >= x.min()) & (tx <= x.max())
    if inside.sum() < 5:
        return math.nan, int(inside.sum())
    return float(np.sqrt(np.mean((np.interp(tx[inside], x, y) - ty[inside]) ** 2))), int(inside.sum())


def report(name, g, y, tx, ty):
    lng = np.log(np.asarray(g, dtype=float))
    print("  %s (%d couplings, g %.2f to %.2f)" % (name, len(g), min(g), max(g)))
    for label, (a, b) in READINGS.items():
        x = a * lng + b
        r, n = rms(x, y, tx, ty)
        c9, c2 = crossing(x, y, 9.0), crossing(x, y, 2.0)
        print("     %-30s rms %.2f over %2d pts | 9 at %+.2f (pub %+.2f) | 2 at %+.2f (pub %+.2f) | width %.2f (pub %.3f)"
              % (label, r, n, c9, PUB_9, c2, PUB_2, c2 - c9, PUB_2 - PUB_9))
    best = None
    for a in np.arange(0.3, 1.21, 0.01):
        for b in np.arange(-3.0, 1.01, 0.02):
            r, n = rms(a * lng + b, y, tx, ty)
            if not math.isnan(r) and n >= 30 and (best is None or r < best[0]):
                best = (r, a, b)
    if best:
        x = best[1] * lng + best[2]
        print("     %-30s rms %.2f | a = %.2f, b = %+.2f (a factor of %.2f in g if a were 1) | width %.2f"
              % ("(fit) free a, b", best[0], best[1], best[2], math.exp(-best[2]), crossing(x, y, 2.0) - crossing(x, y, 9.0)))


def main(paths):
    tgt = list(csv.DictReader(open(TARGET, newline="")))
    tx = np.array([float(r["ln_hbar_g"]) for r in tgt])
    ty = np.array([float(r["squares_per_vertex"]) for r in tgt])
    for p in paths:
        rows = list(csv.DictReader(open(p, newline="")))
        print("== %s" % Path(p).name)
        for leg in ("cool", "heat"):
            sel = [r for r in rows if r["leg"] == leg]
            if not sel:
                continue
            by_g = {}
            for r in sel:
                by_g.setdefault(float(r["g"]), []).append(float(r["squares_per_vertex"]))
            g = sorted(by_g)
            y = [np.mean(by_g[k]) for k in g]
            report(leg, g, y, tx, ty)


if __name__ == "__main__":
    main(sys.argv[1:])
