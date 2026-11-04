"""T40: the push that starts the change in four directions, and the pattern it starts. Usage:

    python scripts/analyse_t40.py

PREREGISTRATION T40 (written 2026-09-25, before any run). Reads `results/t40_*.csv` (scripts/run_sealed_curled_d.py).
T33's pattern rules, with one repair written before any T40 run: the runner's `melted` column counts points with more
than three open directions, which at eight links includes flat points, so melted is read here from the histogram as the
points with more than four (d5 to d7). T33 was not affected (nothing in it opened).

Per replica (start rung s: 1 for the three-curled torus, 0 for the gas): LEAVES if some block has a majority rung above s;
the pattern by T33's rules (FOUR TOGETHER, SINGLETON PLUS THREE, THREE TOGETHER, ONE AT A TIME, STALLS, MELTED, OTHER).
Per spark: the share that LEAVES and the majority pattern. **E\\*** per start: the smallest spark at which a majority
LEAVES. **Verdict per start:** the pattern that is the majority at E\\* and above, if one pattern holds that majority at
every such spark; MIXED if not; NEVER STARTS if no spark reaches E\\*.
"""
import csv
import glob
import sys
from collections import Counter, defaultdict
from pathlib import Path

REST, MELT_FRAC, FLAT = 5000, 0.25, 4


def majority_rung(row, n):
    for d in range(8):
        if int(row["d%d" % d]) > n / 2:
            return d
    return None


def melted(row):
    return sum(int(row["d%d" % d]) for d in range(FLAT + 1, 8))


def read_replica(blocks, n):
    start = majority_rung(blocks[0], n)
    seq, rests = [], set()
    run_rung = run_start = None
    for b in blocks:
        m = majority_rung(b, n)
        sw = int(b["sweep"])
        if m is not None and m not in seq:
            seq.append(m)
        if m is not None and m == run_rung:
            if sw - run_start >= REST:
                rests.add(m)
        else:
            run_rung, run_start = m, sw
    leaves = any(r is not None and r > start for r in seq)
    if melted(blocks[-1]) >= MELT_FRAC * n:
        return leaves, "MELTED"
    between = set(range(start + 1, FLAT))
    inner = rests & between
    if FLAT in seq:
        if not inner:
            return leaves, ("FOUR TOGETHER" if start == 0 else "THREE TOGETHER")
        if start == 0 and inner == {1}:
            return leaves, "SINGLETON PLUS THREE"
        if inner == between:
            return leaves, "ONE AT A TIME"
        return leaves, "OTHER"
    reached = [r for r in seq if r > start]
    if reached and inner and inner == set(range(start + 1, max(reached) + 1)):
        return leaves, "ONE AT A TIME"
    return leaves, "STALLS"


def summarize(rows):
    reps = defaultdict(list)
    for r in rows:
        reps[(r["dims"], float(r["lam"]), int(r["N"]), float(r["spark"]), int(r["replica"]))].append(r)
    by_spark = defaultdict(list)
    for (dims, lam, n, spark, rep), blocks in reps.items():
        blocks.sort(key=lambda r: int(r["sweep"]))
        by_spark[(dims, lam, n, spark)].append(read_replica(blocks, n))
    out = {}
    for key, res in sorted(by_spark.items()):
        pats = [p for _, p in res]
        top, votes = Counter(pats).most_common(1)[0]
        out[key] = dict(n=len(res), leave=sum(1 for lv, _ in res if lv), counts=dict(Counter(pats)),
                        majority=(top if votes > len(res) / 2 else "MIXED"))
    return out


def verdicts(summary):
    starts = defaultdict(dict)
    for (dims, lam, n, spark), s in summary.items():
        starts[(dims, lam, n)][spark] = s
    result = {}
    for key, sparks in starts.items():
        e_star = next((e for e in sorted(sparks) if sparks[e]["leave"] > sparks[e]["n"] / 2), None)
        if e_star is None:
            result[key] = (None, "NEVER STARTS")
            continue
        majorities = {sparks[e]["majority"] for e in sorted(sparks) if e >= e_star}
        result[key] = (e_star, majorities.pop() if len(majorities) == 1 else "MIXED")
    return result


def main(out_dir="results"):
    rows = []
    for path in sorted(glob.glob(str(Path(out_dir) / "t40_*.csv"))):
        rows += list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    if not rows:
        print("no T40 results")
        return
    s = summarize(rows)
    for (dims, lam, n, spark), c in s.items():
        print("start %-18s lambda=%.2f N=%-5d E=%-6.1f leave %d/%d %-45s -> %s" % (dims, lam, n, spark, c["leave"], c["n"], c["counts"], c["majority"]))
    for (dims, lam, n), (e_star, word) in verdicts(s).items():
        print("VERDICT T40 start %s lambda=%.2f: E* = %s, pattern %s" % (dims, lam, e_star, word))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
