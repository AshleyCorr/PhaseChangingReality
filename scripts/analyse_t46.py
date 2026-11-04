"""T46: the owner's order of 26 September in four directions (dark energy, dark matter, ordinary matter, time). Read by
the rules written before the runs. Usage:

    python scripts/analyse_t46.py

PREREGISTRATION T46 (written 2026-09-26, before any run). Reads `results/t46_*.csv` (scripts/run_sealed_curled_d.py with
"ftable_per_a", eight links). Per replica the outcome (MELTED, FLAT, STUCK, PARTLY OPEN) and the bath's gain at each
stage are T44's (analyse_t44.read_replica) with four directions: FLAT means at least 90 % of points at d = 4, MELTED a
quarter or more above 4. Per cell (tie, lambda, C, spark) the majority, else MIXED. Per setting (tie, lambda): **ONE PUSH
OPENS ALL** if a cell with the smaller spark (the first wall alone) has a FLAT majority; **PUSHED THROUGH** if only cells
with the larger spark (the sum of the walls) do; **NOT ALL** otherwise. Reported beside, not scored: where each replica
rests (the rung holding the most points at the end) and the bath gain per point at each stage against the fitted releases.
"""
import csv
import glob
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyse_t44 as t44    # noqa: E402
import analyse_t45 as t45    # noqa: E402

DIM = 4


def summarize(rows):
    cells, ledgers = defaultdict(list), defaultdict(list)
    for (label, lam, n, c, spark, rep), blocks in t45.replicas(rows).items():
        out, pattern, stages, end = t44.read_replica(blocks, lam, n, dim=DIM)
        d = t44.hist(blocks[-1])
        rests = max(range(DIM + 1), key=lambda k: d[k])
        cells[(label, lam, c, spark)].append(out)
        ledgers[(label, lam, c, spark)].append((stages, rests, pattern))
    majority = {}
    for k, outs in cells.items():
        top, votes = Counter(outs).most_common(1)[0]
        majority[k] = top if votes > len(outs) / 2 else "MIXED"
    sparks = defaultdict(set)
    for (label, lam, c, spark) in majority:
        sparks[(label, lam)].add(spark)
    verdict = {}
    for key, ss in sparks.items():
        small = min(ss)
        flat_small = any(m == "FLAT" for (l, la, c, sp), m in majority.items() if (l, la) == key and sp == small)
        flat_any = any(m == "FLAT" for (l, la, c, sp), m in majority.items() if (l, la) == key)
        verdict[key] = "ONE PUSH OPENS ALL" if flat_small else ("PUSHED THROUGH" if flat_any else "NOT ALL")
    return dict(cells={k: dict(Counter(v)) for k, v in cells.items()}, majority=majority, verdict=verdict,
                ledgers=ledgers)


def main(out_dir="results"):
    rows = []
    for path in sorted(glob.glob(str(Path(out_dir) / "t46_*.csv"))):
        rows += list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    if not rows:
        print("no T46 results")
        return
    s = summarize(rows)
    for k in sorted(s["cells"]):
        print("%-16s lambda=%.2f C=%-5d E=%-7g %-40s -> %s" % (*k, s["cells"][k], s["majority"][k]))
        for stages, rests, pattern in s["ledgers"][k]:
            print("    gain per point at d>=1..4: %s | rests at d = %d %s"
                  % (" / ".join("%.3f" % stages[j] if j in stages else "-" for j in range(1, DIM + 1)), rests, pattern))
    for key, v in sorted(s["verdict"].items()):
        print("VERDICT T46 %s lambda=%.2f: %s" % (key[0], key[1], v))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
