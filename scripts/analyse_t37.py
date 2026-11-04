"""T37: many natural seeds in long tubes, read by the rules written before the runs. Usage:

    python scripts/analyse_t37.py

PREREGISTRATION T37 (written 2026-09-25, before any run). Reads `results/t37_*.csv` (scripts/run_tube_decay.py with
`count_patches_every` set) and each decay's saved final graph (`results/t37_*_adj/`). Per replica: the number of
separate seeds k, the first seed's sweep t1, the converted count every `count_patches_every` sweeps, and from the final
graph the leftovers: connected pieces of points not at d = 2, a "column" being a piece of exactly four points all at
d = 1 (paper 2's relic), anything else "other".

Per cell (g, L): mean k, mean columns, mean other pieces, with standard errors; the Avrami exponent n from each replica
with k >= 4 (a straight-line fit of ln(-ln(1 - X)) against ln t over 0.1 <= X <= 0.9, X the converted count over its
final value, t the sweep); the KJMA prediction of k from quantities measured in the same cell and nothing fitted to k:
I = 1 / (L * mean t1) seeds per column per sweep, K the median over replicas of the Avrami coefficient with n fixed at 2
(X = 1 - exp(-K t^2)), and k_pred = (I L / 2) sqrt(pi / K), which is the integral of I L (1 - X) over time.

Predictions and verdict rules are in PREREGISTRATION T37; `verdict_*` below implement them.
"""
import csv
import glob
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphity.cqg import surplus, total_squares                 # noqa: E402
from graphity.dimension import local_dimension, piece_labels   # noqa: E402

NAME = re.compile(r"t37_lam125_g(\d+)_L(\d+)(?:_[a-z])?\.csv$")
X_LO, X_HI, MIN_K_FOR_AVRAMI = 0.10, 0.90, 4
CLEAN = 0.90          # an end state is CLEAN if at least this share of points is at d = 2, DEFECTED otherwise


def end_state(adj, lam):
    """(energy left above the flat torus, exact; share of points at d = 2; CLEAN or DEFECTED)."""
    n = adj.shape[0]
    h = 16.0 * (n - total_squares(adj)) + 4.0 * lam * surplus(adj)
    share = float((local_dimension(adj) == 2).mean())
    return float(h), share, ("CLEAN" if share >= CLEAN else "DEFECTED")


def leftovers(adj):
    """(columns, other pieces) among the connected pieces of points not at d = 2."""
    d = local_dimension(adj)
    labels = piece_labels(adj, d != 2)
    if labels.max() < 0:
        return 0, 0
    columns = other = 0
    for k in range(labels.max() + 1):
        members = np.flatnonzero(labels == k)
        if len(members) == 4 and (d[members] == 1).all():
            columns += 1
        else:
            other += 1
    return columns, other


def avrami(series, every):
    """Fit ln(-ln(1 - X)) = ln K + n ln t over 0.1 <= X <= 0.9; returns (n, K) or (nan, nan)."""
    c = np.asarray(series, dtype=float)
    if len(c) < 4 or c[-1] <= 0:
        return math.nan, math.nan
    x = c / c[-1]
    t = every * np.arange(1, len(c) + 1)
    m = (x >= X_LO) & (x <= X_HI)
    if m.sum() < 3:
        return math.nan, math.nan
    y = np.log(-np.log(1.0 - x[m]))
    n, lnk = np.polyfit(np.log(t[m]), y, 1)
    return float(n), float(math.exp(lnk))


def avrami_k_at_two(series, every):
    """K with the exponent fixed at 2: the mean of -ln(1 - X) / t^2 over 0.1 <= X <= 0.9."""
    c = np.asarray(series, dtype=float)
    if len(c) < 2 or c[-1] <= 0:
        return math.nan
    x = c / c[-1]
    t = every * np.arange(1, len(c) + 1)
    m = (x >= X_LO) & (x <= X_HI)
    if not m.any():
        return math.nan
    return float(np.exp(np.mean(np.log(-np.log(1.0 - x[m]) / t[m] ** 2))))


def kjma_prediction(t1_mean, length, k_two):
    """k_pred = (I L / 2) sqrt(pi / K), I = 1 / (L t1_mean)."""
    if not (t1_mean > 0 and k_two > 0):
        return math.nan
    i = 1.0 / (length * t1_mean)
    return 0.5 * i * length * math.sqrt(math.pi / k_two)


def mean_se(v):
    v = np.asarray([x for x in v if not math.isnan(x)], dtype=float)
    if len(v) == 0:
        return math.nan, math.nan
    return float(v.mean()), (float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else math.nan)


def read_cells(results="results"):
    cells = defaultdict(list)
    for path in sorted(glob.glob(str(Path(results) / "t37_*.csv"))):
        m = NAME.search(path.replace("\\", "/"))
        if not m:
            continue
        g, length = int(m.group(1)) / 100.0, int(m.group(2))
        adj_dir = Path(path[:-4] + "_adj")
        for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
            rec = dict(k=int(r["seeds"]), every=None)
            sw = r["seed_sweeps"].split()
            rec["t1"] = float(sw[0].split(":")[0]) if sw else math.nan
            rec["series"] = [int(v) for v in r["converted_series"].split()]
            f = adj_dir / ("N%s_rep%s.npz" % (r["N"], r["replica"]))
            if f.exists():
                adj = np.load(f)["adj"]
                rec["columns"], rec["other"] = leftovers(adj)
                rec["e_left"], rec["flat_share"], rec["state"] = end_state(adj, float(r["lam"]))
            else:
                rec["columns"], rec["other"] = math.nan, math.nan
                rec["e_left"], rec["flat_share"], rec["state"] = math.nan, math.nan, "MISSING"
            cells[(g, length)].append(rec)
    return cells


def cell_summary(recs, length, every):
    k = mean_se([r["k"] for r in recs])
    cols = mean_se([r["columns"] for r in recs])
    oth = mean_se([r["other"] for r in recs])
    t1 = mean_se([r["t1"] for r in recs])
    ns = [avrami(r["series"], every)[0] for r in recs if r["k"] >= MIN_K_FOR_AVRAMI]
    ns = [n for n in ns if not math.isnan(n)]
    k2 = [avrami_k_at_two(r["series"], every) for r in recs]
    k2 = [v for v in k2 if not math.isnan(v)]
    kpred = kjma_prediction(t1[0], length, float(np.median(k2))) if k2 else math.nan
    e_left = mean_se([r["e_left"] for r in recs])
    defected = sum(1 for r in recs if r["state"] == "DEFECTED")
    return dict(n=len(recs), k=k, columns=cols, other=oth, t1=t1, e_left=e_left, defected=defected,
                avrami_n=(float(np.median(ns)) if ns else math.nan), avrami_count=len(ns), k_pred=kpred)


def verdict_scaling(ks_by_length):
    """P1: the exponent of mean k against L over L >= 256, within [0.7, 1.1]. Returns (alpha, holds)."""
    pts = [(L, k) for L, k in sorted(ks_by_length.items()) if L >= 256 and k > 0]
    if len(pts) < 2:
        return math.nan, None
    a, _ = np.polyfit(np.log([p[0] for p in pts]), np.log([p[1] for p in pts]), 1)
    return float(a), bool(0.7 <= a <= 1.1)


def verdict_scraps(left_by_length):
    """The owner's question: the mean energy left in the final graph per tube (the scrap, in energy) at the largest L
    against the smallest."""
    ls = sorted(left_by_length)
    lo, hi = left_by_length[ls[0]], left_by_length[ls[-1]]
    if not (lo > 0):
        return "MANY SEEDS, MANY SCRAPS" if hi > 0 else "ONE SCRAP HOWEVER LARGE"
    r = hi / lo
    if r >= 3.0:
        return "MANY SEEDS, MANY SCRAPS"
    if r < 1.5:
        return "ONE SCRAP HOWEVER LARGE"
    return "BETWEEN"


def main(results="results"):
    cells = read_cells(results)
    if not cells:
        print("no T37 results")
        return
    every = 100
    by_g = defaultdict(dict)
    for (g, length), recs in sorted(cells.items()):
        s = cell_summary(recs, length, every)
        by_g[g][length] = s
        print("g=%.2f L=%-5d n=%-3d seeds %.2f +- %.2f | left %.1f +- %.1f (%.3f per point) | DEFECTED %d | columns %.2f "
              "+- %.2f | other %.2f | t1 %.0f | Avrami n %.2f (%d) | k_pred %.2f"
              % (g, length, s["n"], s["k"][0], s["k"][1], s["e_left"][0], s["e_left"][1], s["e_left"][0] / (4 * length),
                 s["defected"], s["columns"][0], s["columns"][1], s["other"][0], s["t1"][0], s["avrami_n"],
                 s["avrami_count"], s["k_pred"]))
    for g, row in sorted(by_g.items()):
        alpha, p1 = verdict_scaling({L: s["k"][0] for L, s in row.items()})
        ns = [s["avrami_n"] for L, s in row.items() if not math.isnan(s["avrami_n"])]
        p2 = (bool(1.6 <= float(np.median(ns)) <= 2.4) if ns else None)
        p3 = [(L, s["k_pred"] / s["k"][0]) for L, s in row.items() if L >= 512 and s["k"][0] > 0]
        p3_holds = (all(1 / 1.5 <= r <= 1.5 for _, r in p3) if p3 else None)
        scraps = verdict_scraps({L: s["e_left"][0] for L, s in row.items()}) if len(row) > 1 else "NOT READ"
        print("g=%.2f  P1 exponent %.2f -> %s | P2 Avrami n %s -> %s | P3 k_pred/k %s -> %s | scraps: %s"
              % (g, alpha, p1, ("%.2f" % float(np.median(ns))) if ns else "nan", p2,
                 ", ".join("L%d %.2f" % (L, r) for L, r in p3), p3_holds, scraps))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
