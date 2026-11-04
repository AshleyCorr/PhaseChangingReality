"""The pre-registered T36 reading: are there allotropes in an equilibrated λ = 1 graph? Usage:

    python scripts/analyse_t36.py

Implements PREREGISTRATION.md section T36 (written 2026-09-25, before the runs) on results/t36_*.csv and their _adj
directories, written by scripts/run_allotrope_search.py. L_t is the set of points with c(v) <= 2 (they touch at most two
squares) at snapshot t; rho = mean |L_t| / N. Persistence at a lag of k snapshots: P_k = mean over t of
|L_t ∩ L_{t+k}| / |L_t|, and its excess over chance E_k = P_k - rho. Per replica, E at the scoring lag (LAG_SWEEPS);
over the replicas of a cell (N, g), its mean and standard error. PERSISTENT if the mean exceeds 2 standard errors.
The persistent set of a replica is the intersection of L_t over the last LAG_SWEEPS of the run; its pieces are read in
the saved final graph; a REGION is a piece of at least MIN_REGION points. Per cell: ALLOTROPES if PERSISTENT and a
majority of replicas hold a region; SCATTERED if PERSISTENT without; TRANSIENT if not PERSISTENT. The verdict of T36
is read at the scored size (SCORED_N): ALLOTROPES if any of its couplings reads so, TRANSIENT if all do, otherwise
SCATTERED. Other sizes are reported beside. Pure functions of parsed rows, tested in tests/test_t36.py.
"""
import csv
import glob
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphity.dimension import pieces_of          # noqa: E402

LAG_SWEEPS, MIN_REGION, SCORED_N = 2000, 4, 196


def replicas(rows):
    out = defaultdict(list)
    for r in rows:
        out[(int(r["N"]), float(r["g"]), r["start"], int(r["replica"]))].append(r)
    for k in out:
        out[k].sort(key=lambda r: int(r["snap"]))
    return out


def sets_of(blocks):
    return [set(int(x) for x in b["low"].split()) if b["low"] else set() for b in blocks]


def excess(sets, n, k):
    """E_k = mean |L_t ∩ L_{t+k}| / |L_t| - rho, over t with L_t nonempty."""
    rho = float(np.mean([len(s) for s in sets])) / n
    vals = [len(sets[t] & sets[t + k]) / len(sets[t]) for t in range(len(sets) - k) if sets[t]]
    return (float(np.mean(vals)) - rho) if vals else math.nan, rho


def persistent_set(sets, k):
    """Points in L_t at every one of the last k + 1 snapshots."""
    tail = sets[-(k + 1):]
    out = set(tail[0])
    for s in tail[1:]:
        out &= s
    return out


def region_size(adj, members):
    if not members:
        return 0
    mask = np.zeros(adj.shape[0], dtype=bool)
    mask[sorted(members)] = True
    sizes = pieces_of(adj, mask)
    return sizes[0] if sizes else 0


def cell_reading(reps, snap_every, adj_for=None):
    """reps: {key: blocks} for one (N, g). Returns the reading dict."""
    k = max(1, int(round(LAG_SWEEPS / snap_every)))
    es, rhos, regions = [], [], []
    for key, blocks in reps.items():
        n = key[0]
        sets = sets_of(blocks)
        e, rho = excess(sets, n, k)
        es.append(e); rhos.append(rho)
        if adj_for is not None:
            adj = adj_for(key, int(blocks[-1]["snap"]))
            regions.append(region_size(adj, persistent_set(sets, k)) if adj is not None else 0)
    es = np.array([e for e in es if not math.isnan(e)])
    mean = float(es.mean()) if len(es) else math.nan
    se = float(es.std(ddof=1) / math.sqrt(len(es))) if len(es) > 1 else math.nan
    persistent = (not math.isnan(se)) and mean > 2 * se and mean > 0
    with_region = sum(1 for r in regions if r >= MIN_REGION)
    if not persistent:
        word = "TRANSIENT"
    elif with_region > len(reps) / 2:
        word = "ALLOTROPES"
    else:
        word = "SCATTERED"
    return dict(excess=mean, se=se, rho=float(np.mean(rhos)), persistent=persistent, regions=regions,
                with_region=with_region, n=len(reps), reading=word, lag=k)


def verdict(readings_at_scored):
    words = [r["reading"] for r in readings_at_scored]
    if not words:
        return "NOT READ"
    if any(w == "ALLOTROPES" for w in words):
        return "ALLOTROPES"
    if all(w == "TRANSIENT" for w in words):
        return "TRANSIENT"
    return "SCATTERED"


def main(out_dir="results"):
    rows, snap_every, adj_dirs = [], {}, {}
    for f in sorted(glob.glob(str(Path(out_dir) / "t36_*.csv"))):
        import json
        cfg = json.loads(Path(f).with_suffix(".meta.json").read_text())["config"]
        these = list(csv.DictReader(open(f, newline="")))
        for r in these:
            snap_every[int(r["N"])] = int(cfg["snap_every"])
            adj_dirs[int(r["N"])] = Path(f).with_suffix("").as_posix() + "_adj"
        rows += these
    if not rows:
        print("no T36 results yet")
        return
    reps = replicas(rows)
    cells = defaultdict(dict)
    for key, blocks in reps.items():
        cells[(key[0], key[1])][key] = blocks

    def adj_for(key, snap):
        n, g, start, rep = key
        f = Path(adj_dirs[n]) / ("g%d_%s_rep%d_snap%d.npz" % (round(g * 1000), start, rep, snap))
        return np.load(f)["adj"] if f.exists() else None

    scored = []
    for (n, g) in sorted(cells):
        r = cell_reading(cells[(n, g)], snap_every[n], adj_for)
        if n == SCORED_N:
            scored.append(r)
        print("N=%-4d g=%.3f replicas %2d | rho %.3f | excess at %d sweeps %+.4f ± %.4f -> %s | persistent regions >= %d: %d of %d (sizes %s)"
              % (n, g, r["n"], r["rho"], LAG_SWEEPS, r["excess"], r["se"], r["reading"], MIN_REGION, r["with_region"], r["n"],
                 sorted(r["regions"], reverse=True)[:8]))
    print("VERDICT T36 (N = %d): %s" % (SCORED_N, verdict(scored)))


if __name__ == "__main__":
    main(*sys.argv[1:])
