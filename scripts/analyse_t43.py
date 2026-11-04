"""The pre-registered T43 reading: how long does a planted allotrope last at λ = 1? Usage:

    python scripts/analyse_t43.py [results_dir]

Implements PREREGISTRATION.md section T43 (docs/design/planted_allotrope.md, section 6) on results/t43_*.csv, written by
scripts/run_planted_allotrope.py. The rules, fixed before any run:

  * The planted region is alive while e = f_R - f_B >= 1/2. Its lifetime tau_R is the sweep of the first snapshot at
    which e is below 1/2 and stays below at the next snapshot too (two snapshots in a row). A replica with no such pair
    is censored at n_sweeps.
  * The background's order lifetime tau_B is defined the same way on q_B < 1/2.
  * Per cell (object, g), the medians over the replicas (censored replicas count at n_sweeps): LASTS if the median
    tau_R >= 1,000 sweeps, DISSOLVES otherwise; tagged FIRST if median tau_R < median tau_B / 2, WITH ITS BACKGROUND if
    not. The two controls have no planted set and are read for tau_B only (CONTROL).
  * Verdict, read on the fold: ALLOTROPE LASTS if it LASTS at some g, DISSOLVES if at none. The handle and the flat
    handle are reported beside, not scored.

Reported beside: T36's persistence excess for R at lags of 50, 200 and 2,000 sweeps (L_t = the points of R touching at
most background - 1 squares at snapshot t; E = mean over t with L_t nonempty of |L_t ∩ L_{t+lag}| / |L_t|, minus
rho = the mean share of the whole graph touching at most background - 1 squares, as in analyse_t36.py); q_R and H at the
start, at 2,000 sweeps and at the end. Pure functions of parsed rows, tested in tests/test_t43.py.
"""
import csv
import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ALIVE, LASTS_AT, SCORED = 0.5, 1000, "hyperbolic_fold"
LAGS = (50, 200, 2000)


def lifetime(sweeps, values, n_sweeps, threshold=ALIVE):
    """(tau, censored): the first snapshot at which values < threshold there and at the next snapshot."""
    for i in range(len(values) - 1):
        if values[i] < threshold and values[i + 1] < threshold:
            return sweeps[i], False
    return n_sweeps, True


def replicas(rows):
    """{(object, g, replica): rows sorted by sweep}."""
    out = defaultdict(list)
    for r in rows:
        out[(r["object"], float(r["g"]), int(r["replica"]))].append(r)
    for k in out:
        out[k].sort(key=lambda r: int(r["sweep"]))
    return out


def replica_times(blocks, n_sweeps):
    """tau_R, censored_R, tau_B, censored_B for one replica; tau_R is None when there is no planted set."""
    sweeps = [int(b["sweep"]) for b in blocks]
    q_b = [float(b["q_B"]) for b in blocks]
    tau_b, cens_b = lifetime(sweeps, q_b, n_sweeps)
    if int(blocks[0]["n_R"]) == 0:
        return dict(tau_R=None, cens_R=None, tau_B=tau_b, cens_B=cens_b)
    e = [float(b["f_R"]) - float(b["f_B"]) for b in blocks]
    tau_r, cens_r = lifetime(sweeps, e, n_sweeps)
    return dict(tau_R=tau_r, cens_R=cens_r, tau_B=tau_b, cens_B=cens_b)


def cell_reading(times):
    """times: the replica_times of one (object, g). Returns the cell's medians, reading and tag."""
    tau_b = float(np.median([t["tau_B"] for t in times]))
    out = dict(n=len(times), tau_B=tau_b, cens_B=sum(t["cens_B"] for t in times))
    if times[0]["tau_R"] is None:
        return dict(out, tau_R=None, cens_R=None, reading="CONTROL", tag="")
    tau_r = float(np.median([t["tau_R"] for t in times]))
    return dict(out, tau_R=tau_r, cens_R=sum(t["cens_R"] for t in times),
                reading="LASTS" if tau_r >= LASTS_AT else "DISSOLVES",
                tag="FIRST" if tau_r < tau_b / 2 else "WITH ITS BACKGROUND")


def verdict(cells):
    """cells: {(object, g): reading}. Read on the fold only."""
    fold = [r["reading"] for (obj, _), r in cells.items() if obj == SCORED]
    if not fold:
        return "NOT READ"
    return "ALLOTROPE LASTS" if "LASTS" in fold else "DISSOLVES"


def excess(blocks, lag):
    """T36's persistence excess for R at a lag in sweeps, over the snapshot pairs that lag apart; nan if none."""
    by_sweep = {int(b["sweep"]): set(b["low_R"].split()) for b in blocks}
    rho = float(np.mean([int(b["n_low"]) / int(b["N"]) for b in blocks]))
    vals = [len(s & by_sweep[t + lag]) / len(s) for t, s in by_sweep.items() if s and (t + lag) in by_sweep]
    return float(np.mean(vals)) - rho if vals else math.nan


def mean_se(values):
    v = np.array([x for x in values if not math.isnan(x)])
    if len(v) == 0:
        return math.nan, math.nan
    return float(v.mean()), (float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else math.nan)


def at_sweep(reps, sweep, key):
    vals = [float(b[key]) for blocks in reps for b in blocks if int(b["sweep"]) == sweep and b[key] != ""]
    return float(np.mean(vals)) if vals else math.nan


def main(out_dir="results"):
    rows, n_sweeps = [], None
    for f in sorted(glob.glob(str(Path(out_dir) / "t43_*.csv"))):
        n_sweeps = int(json.loads(Path(f).with_suffix(".meta.json").read_text())["config"]["n_sweeps"])
        rows += list(csv.DictReader(open(f, newline="")))
    if not rows:
        print("no T43 results yet")
        return
    reps = replicas(rows)
    by_cell = defaultdict(list)
    for (obj, g, _), blocks in reps.items():
        by_cell[(obj, g)].append(blocks)
    cells = {}
    print("object                 g      reps  median tau_R (censored)  median tau_B (censored)  reading    tag"
          "                  excess at 50 / 200 / 2000       q_R start/2000/end   H start/2000/end")
    for (obj, g) in sorted(by_cell):
        blocks = by_cell[(obj, g)]
        r = cell_reading([replica_times(b, n_sweeps) for b in blocks])
        cells[(obj, g)] = r
        ex = [mean_se([excess(b, lag) for b in blocks]) for lag in LAGS] if r["tau_R"] is not None else []
        tr = "%7.0f (%2d)" % (r["tau_R"], r["cens_R"]) if r["tau_R"] is not None else "      -     "
        exs = " / ".join("%+.3f+/-%.3f" % e for e in ex) if ex else "-"
        qs = "/".join("%.2f" % at_sweep(blocks, s, "q_R") for s in (0, 2000, n_sweeps)) if ex else "-"
        hs = "/".join("%.0f" % at_sweep(blocks, s, "H") for s in (0, 2000, n_sweeps))
        print("%-22s %-6.3f %4d  %s              %7.0f (%2d)             %-10s %-20s %-31s %-20s %s"
              % (obj, g, r["n"], tr, r["tau_B"], r["cens_B"], r["reading"], r["tag"], exs, qs, hs))
    print("VERDICT T43 (the fold): %s" % verdict(cells))


if __name__ == "__main__":
    main(*sys.argv[1:])
