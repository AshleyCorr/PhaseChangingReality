"""The pre-registered Gate C' reading: which reading of [T22] Fig. 3's axis is ours, and does it hold in 2D? Usage:

    python scripts/analyse_gatec2.py

Implements PREREGISTRATION.md section Gate C' (written 2026-09-24, night, before the runs) on
results/gatec2_3d_n500_{a,b}.csv (run A) and results/gatec2_2d_n2000_{a,b}.csv (run B).

Run A is scored under reading (iv), x = (1/2) ln g, on each leg separately: (A1) the crossing of 9 squares per
vertex within 0.10 of the published 0.666; (A2) the crossing of 6 within 0.10 of 0.996; (A3) the width between
them within 30 % of the published 0.330. READING (iv) HOLDS if A1 to A3 pass on at least one leg of at least
one replica; FAILS otherwise. Reported, not scored: the crossing of 2 (published 1.311), the cold plateau
(published 10.07), and A1 to A3 under readings (i) x = ln g, (iii) x = ln g - ln 2 and (v) x = ln g - ln 5.5.

Run B is scored on the width alone: W = ln g at which 4S/N falls through 0.17 minus ln g at which it falls
through 3.1, per leg. ONE POWER if W < 3 on both legs of every replica read; TWO POWERS if W > 3 on all;
MIXED otherwise. Pure functions of parsed rows, tested in tests/test_gatec2.py.
"""
import csv
import math
import sys
from pathlib import Path

import numpy as np

PUB_9, PUB_6, PUB_2, PUB_PLATEAU = 0.666, 0.996, 1.311, 10.07
TOL_CROSS, TOL_WIDTH = 0.10, 0.30
READINGS = {"(i)": (1.0, 0.0), "(iii)": (1.0, -math.log(2)), "(iv)": (0.5, 0.0), "(v)": (1.0, -math.log(5.5))}
B_LEVELS, B_SPLIT = (3.1, 0.17), 3.0


def crossing(x, y, level):
    """x where y first falls through level going from cold to hot (x ascending); nan if never."""
    o = np.argsort(x)
    x, y = np.asarray(x, dtype=float)[o], np.asarray(y, dtype=float)[o]
    for i in range(len(x) - 1):
        if (y[i] - level) * (y[i + 1] - level) <= 0 and y[i] != y[i + 1]:
            return float(x[i] + (level - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))
    return math.nan


def legs(rows):
    """{leg: (g ascending, squares per vertex)}"""
    out = {}
    for leg in ("cool", "heat"):
        sel = {}
        for r in rows:
            if r["leg"] == leg:
                sel.setdefault(float(r["g"]), []).append(float(r["squares_per_vertex"]))
        if sel:
            g = sorted(sel)
            out[leg] = (np.array(g), np.array([np.mean(sel[k]) for k in g]))
    return out


def score_a(g, y, reading="(iv)"):
    a, b = READINGS[reading]
    x = a * np.log(g) + b
    c9, c6, c2 = crossing(x, y, 9.0), crossing(x, y, 6.0), crossing(x, y, 2.0)
    width = c6 - c9
    a1 = abs(c9 - PUB_9) <= TOL_CROSS
    a2 = abs(c6 - PUB_6) <= TOL_CROSS
    a3 = abs(width - (PUB_6 - PUB_9)) <= TOL_WIDTH * (PUB_6 - PUB_9)
    return dict(c9=c9, c6=c6, c2=c2, width=width, a1=bool(a1), a2=bool(a2), a3=bool(a3),
                passes=bool(a1 and a2 and a3), plateau=float(y[np.argmin(g)]))


def verdict_a(scores):
    """scores: list of per-(replica, leg) dicts under reading (iv)."""
    if not scores:
        return "NOT READ"
    return "READING (iv) HOLDS" if any(s["passes"] for s in scores) else "FAILS"


def width_b(g, y):
    lng = np.log(g)
    return crossing(lng, y, B_LEVELS[1]) - crossing(lng, y, B_LEVELS[0])


def verdict_b(widths):
    ws = [w for w in widths if not math.isnan(w)]
    if not ws:
        return "NOT READ"
    if all(w < B_SPLIT for w in ws):
        return "ONE POWER"
    if all(w > B_SPLIT for w in ws):
        return "TWO POWERS"
    return "MIXED"


def load(name, out_dir):
    p = Path(out_dir) / (name + ".csv")
    return list(csv.DictReader(open(p, newline=""))) if p.exists() else None


def main(out_dir="results"):
    print("== Run A (D = 3, N = 500), scored under reading (iv)")
    scores = []
    for tag in ("a", "b"):
        rows = load("gatec2_3d_n500_" + tag, out_dir)
        if rows is None:
            print("  replica %s: not yet on disk" % tag)
            continue
        for leg, (g, y) in legs(rows).items():
            s = score_a(g, y)
            scores.append(s)
            print("  replica %s %-4s (iv): 9 at %.3f (pub %.3f) %s | 6 at %.3f (pub %.3f) %s | width %.3f (pub 0.330) %s "
                  "| 2 at %.3f (pub %.3f) | plateau %.2f (pub %.2f) -> %s"
                  % (tag, leg, s["c9"], PUB_9, "ok" if s["a1"] else "no", s["c6"], PUB_6, "ok" if s["a2"] else "no",
                     s["width"], "ok" if s["a3"] else "no", s["c2"], PUB_2, s["plateau"], PUB_PLATEAU,
                     "PASS" if s["passes"] else "fail"))
            for rd in ("(i)", "(iii)", "(v)"):
                t = score_a(g, y, rd)
                print("      %-5s 9 at %.3f | 6 at %.3f | width %.3f | 2 at %.3f (reported, not scored)"
                      % (rd, t["c9"], t["c6"], t["width"], t["c2"]))
    print("VERDICT A:", verdict_a(scores))
    print("\n== Run B (D = 2, N = 2000), the width between 4S/N = 3.1 and 0.17 in ln g (published 2.0; split at 3)")
    widths = []
    for tag in ("a", "b"):
        rows = load("gatec2_2d_n2000_" + tag, out_dir)
        if rows is None:
            print("  replica %s: not yet on disk" % tag)
            continue
        for leg, (g, y) in legs(rows).items():
            w = width_b(g, y)
            widths.append(w)
            lng = np.log(g)
            at = {k: float(np.interp(k, lng, y)) for k in (-2, -1, 0, 1)}
            at2 = {k: float(np.interp(2 * k, lng, y)) for k in (-2, -1, 0, 1)}
            print("  replica %s %-4s: W = %.2f | at ln g = -2,-1,0,1: %s | at 2x: %s (published 3.5, 3.1, 1.17, 0.17)"
                  % (tag, leg, w, ["%.2f" % at[k] for k in (-2, -1, 0, 1)], ["%.2f" % at2[k] for k in (-2, -1, 0, 1)]))
    print("VERDICT B:", verdict_b(widths))


if __name__ == "__main__":
    main(*sys.argv[1:])
