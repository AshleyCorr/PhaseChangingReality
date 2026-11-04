"""T48: does a connected curled space open into one space when the energy it releases stays where it is released? Read by
the rules written before the runs. Usage:

    python scripts/analyse_t48.py

PREREGISTRATION T48 (written 2026-09-27, before any run). Reads `results/t48_*.csv` (scripts/run_sealed_curled_d.py with
"local_heat": the push in one point's own store, one store per point). D, the number of directions, is read from the
torus's sides and the starting rung r0 from how many sides are open (longer than 4).

Per replica, from its last reading: DAMAGED if at least a quarter of the points have more open directions than D; else
OPENS if at least half the points are at d = D and the largest connected piece at d = D holds at least half of all
points; else ADVANCES if the rung holding the most points (d from 0 to D) is above r0; else STAYS. Per cell (torus,
lambda, push) the majority, else MIXED. Per setting (torus, lambda): **ONE SPACE** if the cell with the push equal to the
wall has an OPENS majority; **ONE SPACE WITH A BIGGER PUSH** if only the larger push's cell does; **ADVANCES ONLY** if no
cell opens and some cell has an ADVANCES majority; **DAMAGED** if the best cell is DAMAGED; **STAYS** otherwise.
Reported beside, not scored: per replica the first reading at which the largest open piece holds a tenth and a half of the
points (the front's arrival), and the rung each replica rests on.
"""
import csv
import glob
import sys
from collections import Counter, defaultdict
from pathlib import Path

ORDER = ["STAYS", "DAMAGED", "ADVANCES", "OPENS"]


def dims_of(row):
    return [int(x) for x in str(row["dims"]).split()]


def hist(row, dim):
    """Points at d = 0 .. D, and the points above D (read from d0..d7, d7 meaning 7 or more)."""
    d = [int(row["d%d" % k]) for k in range(8)]
    return d[:dim + 1], sum(d[dim + 1:])


def read_replica(blocks):
    """(outcome, rung it rests on, first sweep with the largest open piece >= N/10, first sweep >= N/2)."""
    last = blocks[-1]
    sides = dims_of(last)
    dim, n = len(sides), int(last["N"])
    r0 = sum(1 for s in sides if s > 4)
    rungs, damaged = hist(last, dim)
    rests = max(range(dim + 1), key=lambda k: rungs[k])
    largest = int(last["largest_open_d"])
    if damaged >= n / 4:
        out = "DAMAGED"
    elif rungs[dim] >= n / 2 and largest >= n / 2:
        out = "OPENS"
    elif rests > r0:
        out = "ADVANCES"
    else:
        out = "STAYS"
    tenth = next((int(b["sweep"]) for b in blocks if int(b["largest_open_d"]) >= n / 10), None)
    half = next((int(b["sweep"]) for b in blocks if int(b["largest_open_d"]) >= n / 2), None)
    return out, rests, tenth, half


def replicas(rows):
    runs = defaultdict(list)
    for r in rows:
        runs[(str(r["dims"]), float(r["lam"]), float(r["spark"]), int(r["replica"]))].append(r)
    for v in runs.values():
        v.sort(key=lambda r: int(r["sweep"]))
    return runs


def summarize(rows):
    cells, notes = defaultdict(list), defaultdict(list)
    for (dims, lam, spark, rep), blocks in replicas(rows).items():
        out, rests, tenth, half = read_replica(blocks)
        cells[(dims, lam, spark)].append(out)
        notes[(dims, lam, spark)].append((rests, tenth, half))
    majority = {}
    for k, outs in cells.items():
        top, votes = Counter(outs).most_common(1)[0]
        majority[k] = top if votes > len(outs) / 2 else "MIXED"
    sparks = defaultdict(set)
    for (dims, lam, spark) in majority:
        sparks[(dims, lam)].add(spark)
    verdict = {}
    for key, ss in sparks.items():
        small = min(ss)
        mine = {sp: m for (d, la, sp), m in majority.items() if (d, la) == key}
        if mine[small] == "OPENS":
            verdict[key] = "ONE SPACE"
        elif "OPENS" in mine.values():
            verdict[key] = "ONE SPACE WITH A BIGGER PUSH"
        elif "ADVANCES" in mine.values():
            verdict[key] = "ADVANCES ONLY"
        elif "DAMAGED" in mine.values():
            verdict[key] = "DAMAGED"
        else:
            verdict[key] = "STAYS"
    return dict(cells={k: dict(Counter(v)) for k, v in cells.items()}, majority=majority, verdict=verdict, notes=notes)


def main(out_dir="results"):
    rows = []
    for path in sorted(glob.glob(str(Path(out_dir) / "t48_*.csv"))):
        rows += list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    if not rows:
        print("no T48 results")
        return
    s = summarize(rows)
    for k in sorted(s["cells"]):
        print("%-12s lambda=%.2f E=%-8g %-44s -> %s" % (*k, s["cells"][k], s["majority"][k]))
        print("    rests at / front at N/10 / at N/2: %s" % s["notes"][k])
    for key, v in sorted(s["verdict"].items()):
        print("VERDICT T48 %s lambda=%.2f: %s" % (key[0], key[1], v))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
