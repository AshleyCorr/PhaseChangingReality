"""The pre-registered T34 reading: does concentrated energy fold flat six-link space? Usage:

    python scripts/analyse_t34.py

Implements PREREGISTRATION.md section T34 (written 2026-09-25, midday, before the runs) on results/t34_*.csv, written
by scripts/run_sealed_curled_d.py from a flat 3-torus. Per replica, from the final block: folded = d < 3, melted = d > 3
(the runner's `melted` column), damage = their sum. FOLDED: damage >= 4 and folded >= melted; MELTED: damage >= 4 and
folded < melted; HEALED: damage < 4. Per cell (protocol, N, E): the majority, else MIXED, with at least MIN_REPLICAS.
RE-CURLS if some cell has a FOLDED majority; MELTS if none has and some has a MELTED majority; HEALS if every read
cell is HEALED; MIXED otherwise. Reported, not scored: ONE DIRECTION replicas (folded reaches N/3 at some block),
CASCADE replicas (2N/3), the largest folded piece, melt-then-fold. Pure functions, tested in tests/test_t34.py.
"""
import csv
import glob
import sys
from collections import Counter, defaultdict
from pathlib import Path

MIN_DAMAGE, MIN_REPLICAS = 4, 6


def folded_of(row):
    return sum(int(row["d%d" % k]) for k in range(3))


def by_replica(rows):
    out = defaultdict(list)
    for r in rows:
        proto = "packed" if r.get("local_heat", "False") == "True" else "spread"
        out[(proto, int(r["N"]), float(r["spark"]), int(r["replica"]))].append(r)
    for k in out:
        out[k].sort(key=lambda r: int(r["sweep"]))
    return out


def outcome(blocks):
    last = blocks[-1]
    folded, melted = folded_of(last), int(last["melted"])
    if folded + melted < MIN_DAMAGE:
        return "HEALED"
    return "FOLDED" if folded >= melted else "MELTED"


def reach(blocks, n):
    """(one direction reached, cascade reached)"""
    top = max(folded_of(b) for b in blocks)
    return top >= n / 3, top >= 2 * n / 3


def melt_then_fold(blocks):
    seen = False
    for b in blocks:
        f, m = folded_of(b), int(b["melted"])
        if m >= MIN_DAMAGE:
            seen = True
        elif seen and f >= MIN_DAMAGE and f > m:
            return True
    return False


def cells(rows):
    reps = by_replica(rows)
    grouped = defaultdict(list)
    for (proto, n, e, rep), blocks in reps.items():
        grouped[(proto, n, e)].append(blocks)
    out = {}
    for k, runs in grouped.items():
        counts = Counter(outcome(b) for b in runs)
        m = len(runs)
        top, votes = counts.most_common(1)[0]
        n = k[1]
        out[k] = dict(n=m, counts=dict(counts), majority=(top if (m >= MIN_REPLICAS and votes > m / 2) else ("NOT READ" if m < MIN_REPLICAS else "MIXED")),
                      one_direction=sum(reach(b, n)[0] for b in runs), cascade=sum(reach(b, n)[1] for b in runs),
                      melt_then_fold=sum(melt_then_fold(b) for b in runs), largest_folded=max(folded_of(b[-1]) for b in runs))
    return out


def verdict(cell_map):
    read = [c["majority"] for c in cell_map.values() if c["majority"] != "NOT READ"]
    if not read:
        return "NOT READ"
    if any(v == "FOLDED" for v in read):
        return "RE-CURLS"
    if any(v == "MELTED" for v in read):
        return "MELTS"
    if all(v == "HEALED" for v in read):
        return "HEALS"
    return "MIXED"


def main(out_dir="results"):
    rows = []
    for f in sorted(glob.glob(str(Path(out_dir) / "t34_*.csv"))):
        rows += list(csv.DictReader(open(f, newline="")))
    if not rows:
        print("no T34 results yet")
        return
    cm = cells(rows)
    for k in sorted(cm):
        c = cm[k]
        print("  %-6s N=%-4d E=%-6g n=%d %s -> %s | one direction %d, cascade %d, melt then fold %d, largest folded %d"
              % (k[0], k[1], k[2], c["n"], c["counts"], c["majority"], c["one_direction"], c["cascade"], c["melt_then_fold"], c["largest_folded"]))
    print("VERDICT T34:", verdict(cm))


if __name__ == "__main__":
    main(*sys.argv[1:])
