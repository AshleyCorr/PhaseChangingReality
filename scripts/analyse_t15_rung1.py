"""The pre-registered T15 rung 1 reading: does the time spent in each arrangement equal the count? Usage:

    python scripts/analyse_t15_rung1.py

PREREGISTRATION.md T15 rung 1. For each (N, lambda, g) and route (A: the interchangeable chain; B: the named
chain reweighted by symmetry), the replica-mean fraction per class, its standard error over replicas, and
TV = 1/2 sum |mean - p| against the exact interchangeable probabilities. The control compares the named chain's
raw fractions with the named probabilities. AGREES: both routes at every setting have TV < 0.03 and every class
with p >= 0.02 within 4 standard errors + 0.005 of p. DISAGREES: a route fails where its control passes.
INCONCLUSIVE: a control fails, or anything else.
"""
import csv
import glob
from collections import defaultdict

import numpy as np


def check(means, ses, p):
    tv = 0.5 * float(np.abs(means - p).sum())
    worst = [k for k in range(len(p)) if p[k] >= 0.02 and abs(means[k] - p[k]) > 4 * ses[k] + 0.005]
    return tv, worst, (tv < 0.03 and not worst)


def main():
    rows = []
    for f in sorted(glob.glob("results/t15_rung1_n*.csv")):
        rows += list(csv.DictReader(open(f, newline="")))
    cells = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        cells[(int(r["N"]), float(r["lam"]), float(r["g"]))][int(r["replica"])][int(r["class_id"])] = r
    all_routes, all_controls = True, True
    for key in sorted(cells):
        reps = cells[key]
        classes = sorted(next(iter(reps.values())))
        p_int = np.array([float(reps[min(reps)][k]["p_interchangeable"]) for k in classes])
        p_nam = np.array([float(reps[min(reps)][k]["p_named"]) for k in classes])
        out = {}
        for col, p in (("f_interchangeable_chain", p_int), ("f_named_reweighted", p_int), ("f_named_chain", p_nam)):
            m = np.array([[float(reps[r][k][col]) for k in classes] for r in sorted(reps)])
            out[col] = check(m.mean(axis=0), m.std(axis=0, ddof=1) / np.sqrt(len(m)), p)
        a, b, c = out["f_interchangeable_chain"], out["f_named_reweighted"], out["f_named_chain"]
        all_routes = all_routes and a[2] and b[2]
        all_controls = all_controls and c[2]
        print("N=%d lambda=%.1f g=%.1f: route A TV %.4f %s | route B TV %.4f %s | control TV %.4f %s"
              % (*key, a[0], "ok" if a[2] else "FAILS %s" % a[1], b[0], "ok" if b[2] else "FAILS %s" % b[1],
                 c[0], "ok" if c[2] else "FAILS %s" % c[1]))
    if not all_controls:
        print("PRE-REGISTERED VERDICT: INCONCLUSIVE (a control failed)")
    else:
        print("PRE-REGISTERED VERDICT:", "AGREES" if all_routes else "DISAGREES")


if __name__ == "__main__":
    main()
