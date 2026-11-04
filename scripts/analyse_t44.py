"""T44: does one push open all three directions of the fully curled X under the follow tie? Read by the rules written
before the runs. Usage:

    python scripts/analyse_t44.py

PREREGISTRATION T44 (written 2026-09-26, before any run). Reads `results/t44_*.csv` (scripts/run_sealed_curled_d.py with
"kappa"). Six links, a gas of eight 6-cubes (every point at d = 0), lambda = 1.25; a = 4(lambda - 1) per point per
direction.

Per replica, from the blocks: the flat state is reached at the first block with H/N <= 0.10 a (H without the tie, which
is zero on flat space anyway) and at least 90 % of points at d = 3. Outcome from the last block: MELTED (a quarter or
more of points above 3), FLAT (the flat state reached and 90 % at d = 3 at the end), STUCK (more than half still at
d = 0), PARTLY OPEN otherwise. For FLAT replicas the pattern: STEPWISE if at some block more than half the points sat at
d = 1 or more than half at d = 2 (a middle rung held the majority), TOGETHER otherwise.

Per cell (kappa, C): the majority outcome, else MIXED. Per row (kappa): ONE PUSH OPENS ALL if some cell has a FLAT
majority, NOT ALL otherwise.

Reported, not scored (the owner's ledger, VISION Update 33): for each replica, the energy the bath had gained by the
first block at which at least half the points had one or more directions open (d >= 1), two or more (d >= 2), and all
three (d = 3), per point, and the tie's share of the energy at those blocks.
"""
import csv
import glob
import sys
from collections import Counter, defaultdict
from pathlib import Path

TOL, CLEAN, MELT_FRAC = 0.10, 0.90, 0.25
DIM = 3


def replicas(rows):
    out = defaultdict(list)
    for r in rows:
        out[(float(r["kappa"]), int(r["N"]), int(r["C"]), float(r["spark"]), int(r["replica"]))].append(r)
    for k in out:
        out[k].sort(key=lambda r: int(r["sweep"]))
    return out


def hist(b):
    return [int(b["d%d" % k]) for k in range(8)]


def read_replica(blocks, lam, n, dim=DIM):
    """dim: the number of directions (3 for T44 and T45; 4 for T46, added 2026-09-26 with the default unchanged)."""
    DIM = dim  # noqa: N806
    a = 4.0 * (lam - 1.0)
    t_flat = None
    middle = False
    stages = {}
    e_bath0 = float(blocks[0]["total"])
    for b in blocks:
        h, d = float(b["h_per_vertex"]), hist(b)
        if t_flat is None and h <= a * TOL and d[DIM] >= CLEAN * n:
            t_flat = int(b["sweep"])
        if any(d[j] > n / 2 for j in range(1, DIM)):
            middle = True
        for k in range(1, DIM + 1):
            if k not in stages and sum(d[k:DIM + 1]) >= n / 2:
                stages[k] = (float(b["total"]) - e_bath0) / n
    d = hist(blocks[-1])
    if sum(d[DIM + 1:]) >= MELT_FRAC * n:
        out = "MELTED"
    elif t_flat is not None and d[DIM] >= CLEAN * n:
        out = "FLAT"
    elif d[0] > n / 2:
        out = "STUCK"
    else:
        out = "PARTLY OPEN"
    pattern = ("STEPWISE" if middle else "TOGETHER") if out == "FLAT" else ""
    return out, pattern, stages, [round(x / n, 3) for x in d[:5]] + [round(sum(d[DIM + 1:]) / n, 3)]


def summarize(rows):
    cells, patterns, ledgers, ends = defaultdict(list), defaultdict(Counter), defaultdict(list), defaultdict(list)
    for (kappa, n, c, spark, rep), blocks in replicas(rows).items():
        lam = float(blocks[0]["lam"])
        out, pattern, stages, end = read_replica(blocks, lam, n)
        cells[(kappa, n, c, spark)].append(out)
        ends[(kappa, n, c, spark)].append(end)
        if pattern:
            patterns[kappa][pattern] += 1
        ledgers[(kappa, n, c, spark)].append(stages)
    majority = {}
    for k, outs in cells.items():
        top, votes = Counter(outs).most_common(1)[0]
        majority[k] = top if votes > len(outs) / 2 else "MIXED"
    rows_verdict = {}
    for (kappa, n, c, spark), m in majority.items():
        rows_verdict[kappa] = "ONE PUSH OPENS ALL" if (m == "FLAT" or rows_verdict.get(kappa) == "ONE PUSH OPENS ALL") \
            else "NOT ALL"
    return dict(cells={k: dict(Counter(v)) for k, v in cells.items()}, majority=majority, rows=rows_verdict,
                patterns={k: dict(v) for k, v in patterns.items()}, ledgers=ledgers, ends=ends)


def main(out_dir="results"):
    rows = []
    for path in sorted(glob.glob(str(Path(out_dir) / "t44_*.csv"))):
        rows += list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    if not rows:
        print("no T44 results")
        return
    s = summarize(rows)
    for k in sorted(s["cells"]):
        kappa, n, c, spark = k
        print("kappa=%.2f N=%-4d C=%-5d E=%-5g %-50s -> %s" % (kappa, n, c, spark, s["cells"][k], s["majority"][k]))
        for stages, end in zip(s["ledgers"][k], s["ends"][k]):
            print("    bath gain per point at d>=1 / d>=2 / d=3: %s | end fractions d0..d4, >3: %s"
                  % (" / ".join("%.3f" % stages[j] if j in stages else "-" for j in (1, 2, 3)), end))
    for kappa, v in sorted(s["rows"].items()):
        print("VERDICT T44 kappa=%.2f: %s  pattern %s" % (kappa, v, s["patterns"].get(kappa, {})))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
