"""The pre-registered T26 and T27 readings: energy packed into one place, sealed (T26) or leaking (T27). Usage:

    python scripts/analyse_t26.py            # both, from results/t26_*.csv and results/t27_*.csv

Implements PREREGISTRATION.md sections T26 and T27 (written 2026-09-24, before the runs). Per replica, from
its final block: folded = vertices at d < 2, melted = vertices at d > 2, damage = folded + melted. Outcomes:

  FOLDED   damage >= MIN_DAMAGE and folded >= melted        (a re-curled region: the author's black hole)
  MELTED   damage >= MIN_DAMAGE and folded < melted         (a random bubble: [T25]'s)
  HEALED   damage < MIN_DAMAGE                              (the sheet is flat again, to within one column)

MIN_DAMAGE is 4, the smallest curled object (one column). A cell is one (protocol, points, E, radius, leak,
N); its outcome is the majority over replicas, else MIXED; at least MIN_REPLICAS replicas to be read.

T26 verdict over its cells (leak = 0):
  RE-CURLS   at least one cell's majority is FOLDED;
  MELTS      no FOLDED majority, at least one MELTED majority;
  HEALS      every read cell's majority is HEALED;
  MIXED      otherwise.
T26's verdict reads the one-place protocols only ("stores" and "patch", leak 0). T27's verdict reads the
"bath" protocol with leak > 0: the same words, read as FOLDS BEFORE IT FLATTENS / STAYS MELTED / FLATTENS /
MIXED; the leak-0 bath cells are its sealed control, reported and not scored.

Reported, not scored: the share of replicas that melt first and fold later ("melt then fold": some block with
melted >= MIN_DAMAGE followed by a later block with folded >= MIN_DAMAGE and folded > melted), the largest
folded piece at the end, and with interchangeable points the symmetry count at the end. Pure functions of
parsed rows, tested in tests/test_t26.py.
"""
import csv
import glob
import sys
from collections import Counter, defaultdict
from pathlib import Path

MIN_DAMAGE = 4
MIN_REPLICAS = 6


def key(r):
    return (r["protocol"], r["points"], float(r["E"]), r["radius"], float(r["leak"]), int(r["N"]),
            r.get("local_heat", "False"))


def by_replica(rows):
    out = defaultdict(list)
    for r in rows:
        out[(key(r), int(r["replica"]))].append(r)
    for k in out:
        out[k].sort(key=lambda r: int(r["block"]))
    return out


def outcome(blocks):
    last = blocks[-1]
    folded, melted = int(last["folded"]), int(last["melted"])
    if folded + melted < MIN_DAMAGE:
        return "HEALED"
    return "FOLDED" if folded >= melted else "MELTED"


def melt_then_fold(blocks):
    seen_melt = False
    for b in blocks:
        folded, melted = int(b["folded"]), int(b["melted"])
        if melted >= MIN_DAMAGE:
            seen_melt = True
        elif seen_melt and folded >= MIN_DAMAGE and folded > melted:
            return True
    return False


def cells(rows):
    """{cell key: dict(n, counts, majority, melt_then_fold, largest_folded)}"""
    reps = by_replica(rows)
    grouped = defaultdict(list)
    for (k, _), blocks in reps.items():
        grouped[k].append(blocks)
    out = {}
    for k, runs in grouped.items():
        counts = Counter(outcome(b) for b in runs)
        n = len(runs)
        top, votes = counts.most_common(1)[0]
        majority = top if (n >= MIN_REPLICAS and votes > n / 2) else ("NOT READ" if n < MIN_REPLICAS else "MIXED")
        out[k] = dict(n=n, counts=dict(counts), majority=majority,
                      melt_then_fold=sum(melt_then_fold(b) for b in runs) / n,
                      largest_folded=max(int(b[-1]["largest_folded"]) for b in runs))
    return out


def verdict(cell_map, leaking):
    if leaking:
        chosen = {k: c for k, c in cell_map.items() if k[0] == "bath" and k[4] > 0}
    else:
        chosen = {k: c for k, c in cell_map.items() if k[0] in ("stores", "patch") and k[4] == 0}
    read = {k: c["majority"] for k, c in chosen.items() if c["majority"] != "NOT READ"}
    if not read:
        return "NOT READ"
    words = (("FOLDS BEFORE IT FLATTENS", "STAYS MELTED", "FLATTENS") if leaking else ("RE-CURLS", "MELTS", "HEALS"))
    if any(v == "FOLDED" for v in read.values()):
        return words[0]
    if any(v == "MELTED" for v in read.values()):
        return words[1]
    if all(v == "HEALED" for v in read.values()):
        return words[2]
    return "MIXED"


def main(out_dir="results"):
    for section, pattern, leaking in (("T26", "t26_*.csv", False), ("T27", "t27_*.csv", True)):
        rows = []
        for f in sorted(glob.glob(str(Path(out_dir) / pattern))):
            rows += list(csv.DictReader(open(f, newline="")))
        print("== %s" % section)
        if not rows:
            print("no results yet")
            continue
        cm = cells(rows)
        for k in sorted(cm, key=lambda k: (k[0], k[1], k[5], k[2])):
            c = cm[k]
            print("%-6s %-15s N=%-4d E=%-5g radius=%-3s leak=%-5g local_heat=%-5s n=%2d %s -> %s | melt then fold %.2f "
                  "| largest folded %d"
                  % (k[0], k[1], k[5], k[2], k[3] or "all", k[4], k[6], c["n"], c["counts"], c["majority"],
                     c["melt_then_fold"], c["largest_folded"]))
        print("VERDICT %s: %s\n" % (section, verdict(cm, leaking)))


if __name__ == "__main__":
    main(*sys.argv[1:])
