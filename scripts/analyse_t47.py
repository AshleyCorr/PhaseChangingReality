"""T47: is there a speed limit? Read by the rules written before the runs. Usage:

    python scripts/analyse_t47.py

PREREGISTRATION T47 (written 2026-09-26, before any run).

Part A (`results/t47_front_*.csv`, scripts/run_front_speed.py). Per replica, t(p) is the first recorded sweep with
converted fraction f >= p. Each replica that reaches f = 0.5 gives the pace ratio R = [t(0.5) - t(0.3)] / [t(0.3) - t(0.1)]
(1 for a front at fixed speed, 2 for one that spreads like diffusion from the seed, below 1 for one that speeds up) and
the per-front speed v = 0.4 L / 2 / [t(0.5) - t(0.1)], in columns per sweep. Per (bath, L): the median R and median v,
and the share of replicas reaching f = 0.5. Per bath: NO FRONT if at any L fewer than half the replicas reach f = 0.5;
else FIXED SPEED if the median R lies in [0.7, 1.4] at every L and the median v at each L is within 25 % of their mean;
DIFFUSIVE if the median R >= 1.6 at every L; ACCELERATING if the median R < 0.7 at every L; MIXED otherwise.

Part B (`results/t47_recurl_*.csv`, scripts/run_sealed_curled_d.py from flat eight-link space). Per replica, from the
last block: MELTED if a quarter or more of the points sit above d = 4; FRONT if a quarter or more sit at d = 3 (one
direction curled); HEALS if fewer than 2 % sit anywhere but d = 4; STALLED otherwise. Per setting the majority, else
MIXED. For FRONT replicas the growth of the d = 3 count is reported, not scored.
"""
import csv
import glob
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

LO, MID, HI = 0.1, 0.3, 0.5


def first_at(rows, p):
    for r in rows:
        if float(r["f"]) >= p:
            return int(r["sweep"])
    return None


def read_front(rows):
    """(R, v) for one replica's rows sorted by sweep, or None if it never reaches f = 0.5."""
    t = [first_at(rows, p) for p in (LO, MID, HI)]
    if None in t or t[1] == t[0] or t[2] == t[0]:
        return None
    L = int(rows[0]["L"])
    return (t[2] - t[1]) / (t[1] - t[0]), (HI - LO) * L / 2.0 / (t[2] - t[0])


def summarize_a(rows):
    reps = defaultdict(list)
    for r in rows:
        reps[(r["bath"], int(r["L"]), int(r["replica"]))].append(r)
    cells = defaultdict(list)
    for (bath, L, rep), rr in reps.items():
        rr.sort(key=lambda r: int(r["sweep"]))
        cells[(bath, L)].append(read_front(rr))
    table, verdict = {}, {}
    for (bath, L), res in cells.items():
        ok = [x for x in res if x is not None]
        table[(bath, L)] = dict(reached=len(ok) / len(res),
                                R=statistics.median([x[0] for x in ok]) if ok else None,
                                v=statistics.median([x[1] for x in ok]) if ok else None)
    for bath in {b for b, _ in table}:
        t = [table[k] for k in sorted(table) if k[0] == bath]
        if any(c["reached"] < 0.5 for c in t):
            verdict[bath] = "NO FRONT"
            continue
        rs, vs = [c["R"] for c in t], [c["v"] for c in t]
        mean_v = sum(vs) / len(vs)
        if all(0.7 <= r <= 1.4 for r in rs) and all(abs(v - mean_v) <= 0.25 * mean_v for v in vs):
            verdict[bath] = "FIXED SPEED"
        elif all(r >= 1.6 for r in rs):
            verdict[bath] = "DIFFUSIVE"
        elif all(r < 0.7 for r in rs):
            verdict[bath] = "ACCELERATING"
        else:
            verdict[bath] = "MIXED"
    return table, verdict


def read_recurl(last, n):
    d = [int(last["d%d" % k]) for k in range(8)]
    if sum(d[5:]) >= 0.25 * n:
        return "MELTED"
    if d[3] >= 0.25 * n:
        return "FRONT"
    if n - d[4] < 0.02 * n:
        return "HEALS"
    return "STALLED"


def summarize_b(rows):
    reps = defaultdict(list)
    for r in rows:
        reps[(r["tie_label"], int(r["replica"]))].append(r)
    cells = defaultdict(list)
    for (label, rep), rr in reps.items():
        rr.sort(key=lambda r: int(r["sweep"]))
        cells[label].append(read_recurl(rr[-1], int(rr[-1]["N"])))
    out = {}
    for label, outs in cells.items():
        top, votes = Counter(outs).most_common(1)[0]
        out[label] = (dict(Counter(outs)), top if votes > len(outs) / 2 else "MIXED")
    return out


def _rows(pattern, out_dir):
    rows = []
    for path in sorted(glob.glob(str(Path(out_dir) / pattern))):
        rows += list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    return rows


def main(out_dir="results"):
    a = _rows("t47_front_*.csv", out_dir)
    if a:
        table, verdict = summarize_a(a)
        for k in sorted(table):
            c = table[k]
            print("A %-5s L=%-4d reached %.2f  R %s  v %s" % (*k, c["reached"], c["R"], c["v"]))
        for b, v in sorted(verdict.items()):
            print("VERDICT T47 A %s: %s" % (b, v))
    b = _rows("t47_recurl_*.csv", out_dir)
    for label, (counts, v) in sorted(summarize_b(b).items()) if b else []:
        print("VERDICT T47 B %s: %s %s" % (label, counts, v))
    if not a and not b:
        print("no T47 results")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
