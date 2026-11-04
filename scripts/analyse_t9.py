"""The T9 verdict table. Usage:

    python scripts/analyse_t9.py

Reads every results/t9_n*_C*.csv, classifies each replica's end state, finds the crossover in C
at each size, and applies PREREGISTRATION.md section T9 as written.

END STATES are legacy count-based labels, not independent geometric certificates.
They use d=6-q and f=mean(d)-1. A connected surface requires separate wiring checks.
No observed "stalled" end state does not exclude equilibrium coexistence.
Raw bath_T is mean store energy, not temperature. The original N/3.5 prediction
is retained only to reproduce the registration; see O108 for its correction.

END STATES, from the final square-count histogram and the conversion statistic f:
  sheet      f >= 0.9 and at least 90 % of vertices at d = 2 (leftover rings allowed: d = 1 in
             small clusters), and less than 5 % at d >= 3 or d = 0
  melted     at least 15 % of vertices at d >= 3 or d = 0 -- the order, old or new, is gone
  tube       f < 0.1 and at least 90 % at d = 1: never converted
  stalled    0.1 <= f < 0.9, at least 85 % of vertices at d in {1, 2}, less than 5 % at d >= 3
             or d = 0 -- both orders present, neither melting: the supercooled-water outcome
  other      anything else
"""
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

G_MELT = 3.5      # coupling at which the lambda = 1.25 sheet's phi has fallen to 0.9 (pre-registered)


def classify(r):
    n = int(r["N"])
    d = np.array([int(r["d%d" % i]) for i in range(7)], float) / n
    f = float(r["f_final"])
    disorder = d[0] + d[3:].sum()
    if disorder >= 0.15:
        return "melted"
    if f >= 0.9 and d[2] >= 0.9 and disorder < 0.05:
        return "sheet"
    if f < 0.1 and d[1] >= 0.9:
        return "tube"
    if 0.1 <= f < 0.9 and d[1] + d[2] >= 0.85 and disorder < 0.05:
        return "stalled"
    return "other"


def main(out_dir="results"):
    runs = defaultdict(dict)         # (N, C) -> replica -> final row
    for f in Path(out_dir).glob("t9_n*_C*.csv"):
        for r in csv.DictReader(open(f, newline="")):
            if r["final"] == "1":
                runs[(int(r["N"]), int(r["C"]))][int(r["replica"])] = r
    if not runs:
        print("no T9 results yet"); return
    print("T9, sealed tube, lambda = 1.25. PREREGISTRATION.md section T9.\n")
    print("%-5s %-5s %-6s %-6s %-7s %-6s %-7s %-6s %-9s %-9s  %s"
          % ("N", "C", "reps", "sheet", "melted", "tube", "stalled", "other", "mean E", "drift", "d=1 pieces"))
    per_n = defaultdict(list)
    for (n, c) in sorted(runs):
        rows = list(runs[(n, c)].values())
        kinds = [classify(r) for r in rows]
        cnt = {k: kinds.count(k) for k in ("sheet", "melted", "tube", "stalled", "other")}
        bath = np.mean([float(r["bath_T"]) for r in rows])
        drift = max(float(r["drift"]) for r in rows)
        sheets = [r for r, k in zip(rows, kinds) if k == "sheet"]
        rings = np.mean([int(r["pieces_d1"]) > 0 for r in sheets]) if sheets else float("nan")
        print("%-5d %-5d %-6d %-6d %-7d %-6d %-7d %-6d %-9.2f %-9.1e  %s"
              % (n, c, len(rows), cnt["sheet"], cnt["melted"], cnt["tube"], cnt["stalled"], cnt["other"],
                 bath, drift, ("%.0f%% of sheets" % (100 * rings)) if sheets else "-"))
        per_n[n].append(dict(C=c, reps=len(rows), **cnt, bath=bath))

    print("\nCrossover per size: the smallest C at which a majority of replicas end as a sheet, and the")
    print("largest C at which a majority end melted. Historical prediction (a), using an uncalibrated thermometer: C* ~ N * 4(lambda-1) / g_melt = N / %.1f."
          % G_MELT)
    cross = {}
    for n in sorted(per_n):
        rows = sorted(per_n[n], key=lambda d: d["C"])
        sheet_c = [d["C"] for d in rows if d["sheet"] > d["reps"] / 2]
        melt_c = [d["C"] for d in rows if d["melted"] > d["reps"] / 2]
        stall_c = [d["C"] for d in rows if d["stalled"] > d["reps"] / 2]
        cross[n] = (min(sheet_c) if sheet_c else None, max(melt_c) if melt_c else None, stall_c)
        print("   N=%-4d legacy C* = %5.1f   first majority-sheet C = %s   last majority-melted C = %s   majority-stalled at C = %s"
              % (n, n / G_MELT, cross[n][0], cross[n][1], stall_c or "none"))

    # the verdict, as written
    a = all(v[0] is not None and v[1] is not None and v[1] < v[0] for v in cross.values())
    ratios = [v[0] / n for n, v in cross.items() if v[0]]
    scales = len(ratios) >= 2 and (max(ratios) / min(ratios) < 2.5)
    b_fails = any(v[2] for v in cross.values())
    stall_everywhere = all(v[2] for v in cross.values())
    print("\n  (a) bonfire above a C*, boil-off below, at every size:  %s" % ("yes" if a else "no"))
    print("      C*/N at each size: %s  -> scales with N: %s" % (["%.3f" % r for r in ratios], "yes" if scales else "no"))
    print("  (b) no supercooled-water stall:  %s" % ("holds" if not b_fails else "FAILS (stalled majorities at some C)"))
    if a and scales and not b_fails:
        verdict = "BONFIRE WITH A THRESHOLD"
    elif stall_everywhere:
        verdict = "SLUSH"
    else:
        verdict = "INCONCLUSIVE"
    print("\n  PRE-REGISTERED VERDICT, T9:  %s" % verdict)
    print("Scope: count-based end-state verdict; no equilibrium coexistence conclusion.")
    print("Conditional delta=1 store model: u(3.5)=%.6f, so neglecting seed and graph energy gives C*~N/%.6f."
          % (1.0 / np.expm1(1.0 / G_MELT), 1.0 / np.expm1(1.0 / G_MELT)))
    print("Full estimate: C*~[epsilon*N+seed-U_graph(g_m)]/u(g_m); spectrum and mixing unverified.")


if __name__ == "__main__":
    main()
