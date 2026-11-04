"""T49: does one fully curled piece, a single hypercube, open all its directions, and in what order? Read by the rules
written before the runs. Usage:

    python scripts/analyse_t49.py

PREREGISTRATION T49 (written 2026-09-27, before any run). Reads `results/t49_*.csv` (scripts/run_sealed_curled_d.py
from a "gas" of one piece: the 6-cube at six links, the 8-cube at eight; named points, or interchangeable ones through
graphity.interchangeable_d). D is read from the piece's sides.

Per replica, the end from its last reading: DAMAGED if at least a quarter of the points are above D; else ALL OPEN if at
least 90 % are at d = D; else STUCK if at least half are still at d = 0; else PART OPEN. The path, for ALL OPEN and PART
OPEN replicas, from every reading: for each rung k from 1 to D - 1, the first reading at which at least half of the points
sit at d = k. IN ORDER if every such rung held that majority at some reading and the first times increase with k;
TOGETHER if no rung from 1 to D - 1 ever held a majority; PARTLY IN ORDER otherwise. Per cell (points, sides, lambda,
treatment of points) the majority end and, among the replicas that moved, the majority path, else MIXED.
"""
import csv
import glob
import sys
from collections import Counter, defaultdict
from pathlib import Path


def dim_of(row):
    return len(str(row["dims"]).split("(")[-1].rstrip(")").split())


def rungs(row, dim):
    d = [int(row["d%d" % k]) for k in range(8)]
    return d[:dim + 1], sum(d[dim + 1:])


def end_of(last, dim):
    n = int(last["N"])
    r, damaged = rungs(last, dim)
    if damaged >= n / 4:
        return "DAMAGED"
    if r[dim] >= 0.9 * n:
        return "ALL OPEN"
    if r[0] >= n / 2:
        return "STUCK"
    return "PART OPEN"


def path_of(blocks, dim):
    n = int(blocks[0]["N"])
    first = {}
    for b in blocks:
        r, _ = rungs(b, dim)
        for k in range(1, dim):
            if k not in first and r[k] >= n / 2:
                first[k] = int(b["sweep"])
    if not first:
        return "TOGETHER"
    ks = list(range(1, dim))
    if all(k in first for k in ks) and all(first[a] <= first[b] for a, b in zip(ks, ks[1:])):
        return "IN ORDER"
    return "PARTLY IN ORDER"


def treatment(row):
    if str(row.get("interchangeable", "")) in ("True", "true", "1") and str(row.get("weighted", "True")) in ("True", "true", "1"):
        return "interchangeable"
    return "named"


def summarize(rows):
    runs = defaultdict(list)
    for r in rows:
        runs[(int(r["N"]), float(r["lam"]), treatment(r), float(r["spark"]), int(r["replica"]))].append(r)
    ends, paths = defaultdict(list), defaultdict(list)
    for (n, lam, how, spark, rep), blocks in runs.items():
        blocks.sort(key=lambda r: int(r["sweep"]))
        dim = dim_of(blocks[0])
        e = end_of(blocks[-1], dim)
        ends[(n, lam, how, spark)].append(e)
        if e in ("ALL OPEN", "PART OPEN"):
            paths[(n, lam, how, spark)].append(path_of(blocks, dim))

    def majority(v):
        if not v:
            return "-"
        top, votes = Counter(v).most_common(1)[0]
        return top if votes > len(v) / 2 else "MIXED"
    return {k: dict(ends=dict(Counter(v)), end=majority(v), paths=dict(Counter(paths[k])), path=majority(paths[k]))
            for k, v in ends.items()}


def main(out_dir="results"):
    rows = []
    for path in sorted(glob.glob(str(Path(out_dir) / "t49_*.csv"))):
        rows += list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    if not rows:
        print("no T49 results")
        return
    for k, v in sorted(summarize(rows).items()):
        print("N=%-4d lambda=%.2f %-15s E=%-6g ends %-50s -> %-10s | paths %-40s -> %s"
              % (*k, v["ends"], v["end"], v["paths"], v["path"]))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
