"""The pre-registered T33 reading: in what pattern do the curled directions of a four-direction torus open? Usage:

    python scripts/analyse_t33.py

Implements PREREGISTRATION.md section T33 (written 2026-09-25, before the runs) on results/t33_*.csv, written by
scripts/run_sealed_curled_d.py at eight links. The local dimension d is the number of open directions at a point,
0 to 4. Per replica, every block gives the **majority rung**: the d held by more than half the points, or None.
The **rung sequence** is the list of distinct majority rungs in the order first reached. A rung is **rested on**
if it stays the majority for at least REST consecutive sweeps. A replica is MELTED if at least MELT_FRAC of the
points have d above the flat value (4) in the final block.

Patterns, read from the sequence of rests among the rungs strictly between the start rung and the flat rung 4:
  FOUR TOGETHER        the start is the gas (rung 0) and rung 4 is reached with no rest on 1, 2 or 3;
  SINGLETON PLUS THREE from rung 0: a rest on rung 1, then rung 4 with no rest on 2 or 3; from rung 1 (the
                       4 x 4 x 4 x L start, one direction already open): rung 4 with no rest on 2 or 3;
  ONE AT A TIME        a rest on every intermediate rung on the way to 4, or on the way as far as it got;
  STALLS               rung 4 not reached and no rest pattern above fits (including never leaving the start);
  MELTED               as above.
Per cell (start, lambda, N, C): the majority pattern, else MIXED. The verdict names the majority pattern of each
start's cells: for the gas, FOUR TOGETHER / SINGLETON PLUS THREE / ONE AT A TIME / NO CASCADE (stalls or melts);
for the three-curled start, THREE TOGETHER (the sequence 1 -> 4 with no rest between), ONE AT A TIME, NO CASCADE.
Pure functions of parsed rows, tested in tests/test_t33.py.
"""
import csv
import glob
import sys
from collections import Counter, defaultdict
from pathlib import Path

REST, MELT_FRAC, FLAT = 5000, 0.25, 4


def by_replica(rows):
    out = defaultdict(list)
    for r in rows:
        out[(r["dims"], float(r["lam"]), int(r["N"]), int(r["C"]), float(r["spark"]), int(r["replica"]))].append(r)
    for k in out:
        out[k].sort(key=lambda r: int(r["sweep"]))
    return out


def majority_rung(row, n):
    for d in range(8):
        if int(row["d%d" % d]) > n / 2:
            return d
    return None


def rests_and_sequence(blocks, n):
    """(rungs rested on, set; rungs reached in order, list; rung 4 reached, bool)"""
    seq, rests = [], set()
    run_rung, run_start = None, None
    for b in blocks:
        m = majority_rung(b, n)
        sw = int(b["sweep"])
        if m is not None and (not seq or seq[-1] != m) and m not in seq:
            seq.append(m)
        if m == run_rung and m is not None:
            if sw - run_start >= REST:
                rests.add(m)
        else:
            run_rung, run_start = m, sw
    return rests, seq, FLAT in seq


def pattern(blocks, n):
    last = blocks[-1]
    if int(last["melted"]) >= MELT_FRAC * n:
        return "MELTED"
    start = majority_rung(blocks[0], n)
    rests, seq, flat = rests_and_sequence(blocks, n)
    between = set(range(start + 1, FLAT))          # rungs strictly between the start and flat
    inner_rests = rests & between
    if flat:
        if not inner_rests:
            return "FOUR TOGETHER" if start == 0 else "THREE TOGETHER"
        if start == 0 and inner_rests == {1}:
            return "SINGLETON PLUS THREE"
        if inner_rests == between:
            return "ONE AT A TIME"
        return "OTHER"
    reached = [r for r in seq if r > start]
    if reached and inner_rests and inner_rests == set(range(start + 1, max(reached) + 1)):
        return "ONE AT A TIME"
    return "STALLS"


def cells(rows):
    reps = by_replica(rows)
    grouped = defaultdict(list)
    for (dims, lam, n, c, spark, rep), blocks in reps.items():
        grouped[(dims, lam, n, c)].append(pattern(blocks, n))
    out = {}
    for k, pats in grouped.items():
        top, votes = Counter(pats).most_common(1)[0]
        out[k] = dict(n=len(pats), counts=dict(Counter(pats)), majority=(top if votes > len(pats) / 2 else "MIXED"))
    return out


def verdict(cell_map, start_is_gas):
    chosen = [c["majority"] for k, c in cell_map.items() if ("x(" in k[0]) == start_is_gas]
    if not chosen:
        return "NOT READ"
    for name in (("FOUR TOGETHER", "SINGLETON PLUS THREE", "ONE AT A TIME") if start_is_gas else ("THREE TOGETHER", "ONE AT A TIME")):
        if any(v == name for v in chosen):
            return name
    return "NO CASCADE"


def main(out_dir="results"):
    rows = []
    for f in sorted(glob.glob(str(Path(out_dir) / "t33_*.csv"))):
        rows += list(csv.DictReader(open(f, newline="")))
    if not rows:
        print("no T33 results yet")
        return
    cm = cells(rows)
    for k in sorted(cm):
        print("  start %-12s lambda=%.2f N=%-5d C=%-5d n=%2d %s -> %s" % (k[0], k[1], k[2], k[3], cm[k]["n"], cm[k]["counts"], cm[k]["majority"]))
    print("VERDICT (gas, all four curled):", verdict(cm, True))
    print("VERDICT (three curled, one open):", verdict(cm, False))


if __name__ == "__main__":
    main(*sys.argv[1:])
