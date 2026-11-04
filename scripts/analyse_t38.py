"""T38: the rare long wait, read by the rules written before the runs. Usage:

    python scripts/analyse_t38.py

PREREGISTRATION T38 (written 2026-09-25, before any run). Reads `results/t38_lam<l>_n<N>_<j>.csv`, the jobs of one
seeded set pooled per cell (lambda, N). The wait w is the runner's `waiting` (the sweep at which phi first leaves the
tube's resting band), and w' = w - 200, since nothing is read before sweep 200. Per cell:

- tau_hat = median(w') / ln 2, the exponential scale read robustly;
- k10, the number of waits with w' > 10 tau_hat; one exponential expects n e^-10 of them (0.18 in 4,000 decays);
- the two-population fit: maximum likelihood of p Exp(tau_1) + (1 - p) Exp(tau_2) against one exponential, reported
  with 2 ln(likelihood ratio);
- per cell TAIL if k10 >= 3, NO TAIL if k10 <= 1, UNCLEAR otherwise.

Verdict: TWO POPULATIONS if at least two cells read TAIL; ONE POPULATION if every cell reads NO TAIL; UNCLEAR otherwise.
Only recorded detector crossings enter this historical score. Missing crossings are not
ordinary right-censored survivors: O108 finds all 190 missing crossings reach f>=0.75.
The mixture fit describes selected recorded waits; it does not diagnose the mechanism.
Reported, not scored: the saved graphs of tubes still waiting at 5,000, 10,000 and 20,000 sweeps, read exactly (the
energy above the start, the local-dimension histogram, whether the wiring is still the perfect tube).
"""
import csv
import glob
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

NAME = re.compile(r"t38_lam(\d+)_n(\d+)_\d+\.csv$")
REST = 200


def waits_by_cell(results="results"):
    cells = defaultdict(list)
    for path in sorted(glob.glob(str(Path(results) / "t38_*.csv"))):
        m = NAME.search(path.replace("\\", "/"))
        if not m:
            continue
        key = (int(m.group(1)) / 100.0, int(m.group(2)))
        for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
            if r["waiting"]:
                cells[key].append(float(r["waiting"]) - REST)
    return cells


def tail_count(w):
    w = np.asarray(w, dtype=float)
    tau = float(np.median(w)) / math.log(2)
    return tau, int((w > 10 * tau).sum())


def fit_mixture(w, iters=500):
    """EM for p Exp(t1) + (1 - p) Exp(t2), t1 < t2. Returns (p, t1, t2, 2 ln LR against one exponential)."""
    w = np.maximum(np.asarray(w, dtype=float), 1e-9)
    t_one = w.mean()
    ll_one = float(np.sum(-np.log(t_one) - w / t_one))
    p, t1, t2 = 0.95, float(np.median(w)) / math.log(2), float(np.percentile(w, 99))
    for _ in range(iters):
        a = p / t1 * np.exp(-w / t1)
        b = (1 - p) / t2 * np.exp(-w / t2)
        r = a / (a + b + 1e-300)
        p = float(np.clip(r.mean(), 1e-6, 1 - 1e-6))
        t1 = float((r * w).sum() / r.sum())
        t2 = float(((1 - r) * w).sum() / max((1 - r).sum(), 1e-12))
    ll_two = float(np.sum(np.log(p / t1 * np.exp(-w / t1) + (1 - p) / t2 * np.exp(-w / t2) + 1e-300)))
    if t1 > t2:
        p, t1, t2 = 1 - p, t2, t1
    return p, t1, t2, 2 * (ll_two - ll_one)


def cell_reading(k10):
    if k10 >= 3:
        return "TAIL"
    if k10 <= 1:
        return "NO TAIL"
    return "UNCLEAR"


def verdict(readings):
    if sum(1 for r in readings if r == "TAIL") >= 2:
        return "TWO POPULATIONS"
    if readings and all(r == "NO TAIL" for r in readings):
        return "ONE POPULATION"
    return "UNCLEAR"


def main(results="results"):
    cells = waits_by_cell(results)
    if not cells:
        print("no T38 results")
        return
    readings = []
    for (lam, n), w in sorted(cells.items()):
        tau, k10 = tail_count(w)
        p, t1, t2, lr = fit_mixture(w)
        word = cell_reading(k10)
        readings.append(word)
        print("lambda=%.2f N=%-4d waits %d | tau_hat %.0f | k10 %d (one exponential expects %.2f) -> %s | mixture: "
              "short share %.4f tau %.0f, long tau %.0f, 2lnLR %.1f | longest %.0f (%.1f tau)"
              % (lam, n, len(w), tau, k10, len(w) * math.exp(-10), word, p, t1, t2, lr, max(w), max(w) / tau))
    print("VERDICT T38: %s" % verdict(readings))
    print("Scope: recorded detector crossings only; missing crossings are not ordinary censored survivors.")
    print("See O108 and analyse_curled_revision.py for full denominators and snapshot availability.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
