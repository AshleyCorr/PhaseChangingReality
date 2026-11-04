"""T45: the budget-fitted triad, run. Read by the rules written before the runs. Usage:

    python scripts/analyse_t45.py

PREREGISTRATION T45 (written 2026-09-26, before any run). Reads `results/t45_*.csv` (scripts/run_sealed_curled_d.py with
"ftable_per_a"). Per replica the outcome (MELTED, FLAT, STUCK, PARTLY OPEN), the pattern of FLAT replicas and the bath's
gain at each stage are T44's (analyse_t44.read_replica), unchanged. Per cell (tie, lambda, C, spark) the majority, else
MIXED. Per setting (tie, lambda): **ONE PUSH OPENS ALL** if a cell with the smaller spark (the first wall alone) has a
FLAT majority; **PUSHED THROUGH** if only cells with the larger spark (the sum of the walls) do; **NOT ALL** otherwise.
Reported beside, not scored: the measured bath gain per point at each stage against the fitted releases.
"""
import csv
import glob
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyse_t44 as t44    # noqa: E402


def replicas(rows):
    out = defaultdict(list)
    for r in rows:
        out[(r["tie_label"], float(r["lam"]), int(r["N"]), int(r["C"]), float(r["spark"]), int(r["replica"]))].append(r)
    for k in out:
        out[k].sort(key=lambda r: int(r["sweep"]))
    return out


def summarize(rows):
    cells, ledgers = defaultdict(list), defaultdict(list)
    for (label, lam, n, c, spark, rep), blocks in replicas(rows).items():
        out, pattern, stages, end = t44.read_replica(blocks, lam, n)
        cells[(label, lam, c, spark)].append(out)
        ledgers[(label, lam, c, spark)].append((stages, end, pattern))
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
    for path in sorted(glob.glob(str(Path(out_dir) / "t45_*.csv"))):
        rows += list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    if not rows:
        print("no T45 results")
        return
    s = summarize(rows)
    for k in sorted(s["cells"]):
        print("%-12s lambda=%.2f C=%-5d E=%-6g %-45s -> %s" % (*k, s["cells"][k], s["majority"][k]))
        for stages, end, pattern in s["ledgers"][k]:
            print("    gain per point at d>=1 / d>=2 / d=3: %s | end %s %s"
                  % (" / ".join("%.3f" % stages[j] if j in stages else "-" for j in (1, 2, 3)), end, pattern))
    for key, v in sorted(s["verdict"].items()):
        print("VERDICT T45 %s lambda=%.2f: %s" % (key[0], key[1], v))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
