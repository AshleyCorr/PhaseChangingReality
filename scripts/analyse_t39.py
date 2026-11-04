"""T39: the cascade window, read by the rules written before the runs. Usage:

    python scripts/analyse_t39.py

PREREGISTRATION T39 (written 2026-09-25, before any run). Reads `results/t39_*.csv` (scripts/run_sealed_curled_d.py).
Every definition is T30's, written for any number of directions D (the runner's own `melted` and `pieces_flat`
columns assume six links, so this reads the local-dimension histogram directly): flat points are at d = D, the
one-curled rung at d = D - 1, the start (two curled) at d = D - 2, melted points above D.

Per replica, from the blocks: the middle rung is reached at the first block with H/N <= 4(lambda - 1)(1 + 0.10) and at
least half the points at d >= D - 1; the flat state at the first block with H/N <= 0.10 * 4(lambda - 1) and at least
90 % of points at d = D; it rests on the middle rung if H/N stays within 10 % of 4(lambda - 1) for 5,000 consecutive
sweeps. Outcome from the last block: MELTED (a quarter or more of points above D), FLAT (the flat state reached and 90 %
at d = D at the end), MIDDLE, STUCK (more than half still at d = D - 2), OTHER. Reported beside it, the census reading
of the T30 correction: FIRST OPEN (at most 7 % still at d = D - 2), STALLED (36 % or more), PARTWAY, PAST (a quarter or
more at d = D).

Per cell (D, lambda, C): the majority outcome, else MIXED. Per row (D, lambda): WINDOW if some cell has a FLAT majority,
NO WINDOW otherwise; the mechanism over every flat-reaching replica, CASCADE (no rest on the middle rung) or STEPWISE.
"""
import csv
import glob
import sys
from collections import Counter, defaultdict
from pathlib import Path

TOL, CLEAN, REST, MELT_FRAC = 0.10, 0.90, 5000, 0.25


def replicas(rows):
    out = defaultdict(list)
    for r in rows:
        out[(len(r["dims"].split()), float(r["lam"]), int(r["N"]), int(r["C"]), int(r["replica"]))].append(r)
    for k in out:
        out[k].sort(key=lambda r: int(r["sweep"]))
    return out


def hist(b):
    return [int(b["d%d" % k]) for k in range(8)]


def read_replica(blocks, dim, lam, n):
    one = 4.0 * (lam - 1.0)
    t_mid = t_flat = rest_start = None
    rested = False
    for b in blocks:
        h, d = float(b["h_per_vertex"]), hist(b)
        sw = int(b["sweep"])
        if t_mid is None and h <= one * (1 + TOL) and sum(d[dim - 1:]) - sum(d[dim + 1:]) >= n / 2:
            t_mid = sw
        if t_flat is None and h <= one * TOL and d[dim] >= CLEAN * n:
            t_flat = sw
        if t_flat is None and abs(h - one) <= TOL * one:
            if rest_start is None:
                rest_start = sw
            elif sw - rest_start >= REST:
                rested = True
        else:
            rest_start = None
    d = hist(blocks[-1])
    above = sum(d[dim + 1:])
    if above >= MELT_FRAC * n:
        out = "MELTED"
    elif t_flat is not None and d[dim] >= CLEAN * n:
        out = "FLAT"
    elif t_mid is not None:
        out = "MIDDLE"
    elif d[dim - 2] > n / 2:
        out = "STUCK"
    else:
        out = "OTHER"
    start = d[dim - 2] / n
    if d[dim] >= 0.25 * n:
        census = "PAST"
    elif start <= 0.07:
        census = "FIRST OPEN"
    elif start >= 0.36:
        census = "STALLED"
    else:
        census = "PARTWAY"
    return out, census, rested


def summarize(rows):
    cells, census, mech = defaultdict(list), defaultdict(list), defaultdict(Counter)
    for (dim, lam, n, c, rep), blocks in replicas(rows).items():
        out, cen, rested = read_replica(blocks, dim, lam, n)
        cells[(dim, lam, n, c)].append(out)
        census[(dim, lam, n, c)].append(cen)
        if out == "FLAT":
            mech[(dim, lam)]["STEPWISE" if rested else "CASCADE"] += 1
    majority = {}
    for k, outs in cells.items():
        top, votes = Counter(outs).most_common(1)[0]
        majority[k] = top if votes > len(outs) / 2 else "MIXED"
    rows_verdict = {}
    for (dim, lam, n, c), m in majority.items():
        key = (dim, lam)
        rows_verdict[key] = "WINDOW" if (m == "FLAT" or rows_verdict.get(key) == "WINDOW") else "NO WINDOW"
    return dict(cells={k: dict(Counter(v)) for k, v in cells.items()},
                census={k: dict(Counter(v)) for k, v in census.items()},
                majority=majority, rows=rows_verdict, mechanism={k: dict(v) for k, v in mech.items()})


def main(out_dir="results"):
    rows = []
    for path in sorted(glob.glob(str(Path(out_dir) / "t39_*.csv"))):
        rows += list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    if not rows:
        print("no T39 results")
        return
    s = summarize(rows)
    for k in sorted(s["cells"]):
        dim, lam, n, c = k
        print("D=%d lambda=%.2f N=%-5d C=%-5d %-40s -> %-7s | census %s" % (dim, lam, n, c, s["cells"][k], s["majority"][k], s["census"][k]))
    for (dim, lam), v in sorted(s["rows"].items()):
        print("VERDICT T39 D=%d lambda=%.2f: %s  mechanism %s" % (dim, lam, v, s["mechanism"].get((dim, lam), {})))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
