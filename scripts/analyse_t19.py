"""The pre-registered T19 reading: does a leftover move, stay, or anneal away? Usage:

    python scripts/analyse_t19.py

PREREGISTRATION.md T19. Per followed replica: MOVED if its displacement reaches 2 or more at any block while a
leftover exists; ANNEALED if, never having moved, it has no vertex at d = 1 at the end; STAYED if, never having
moved, it still exists at the end. Per coupling: the outcome of more than half the followed replicas (MOVES,
ANNEALS, STAYS), otherwise MIXED; fewer than six followed replicas: NOT READ.
"""
import csv
from collections import defaultdict


def outcome(blocks):
    """blocks: the rows of one followed replica, in time order."""
    moved = any(r["displacement"] not in ("", None) and int(r["displacement"]) >= 2 and int(r["n_d1"]) > 0
                for r in blocks)
    if moved:
        return "moved"
    return "stayed" if int(blocks[-1]["n_d1"]) > 0 else "annealed"


def verdict(outcomes):
    if len(outcomes) < 6:
        return "NOT READ"
    for kind, name in (("moved", "MOVES"), ("annealed", "ANNEALS"), ("stayed", "STAYS")):
        if outcomes.count(kind) > len(outcomes) / 2:
            return name
    return "MIXED"


def main():
    rows = list(csv.DictReader(open("results/t19_mobility.csv", newline="")))
    reps = defaultdict(list)
    for r in rows:
        reps[(float(r["g"]), int(r["replica"]))].append(r)
    for g in sorted({k[0] for k in reps}):
        outs, not_followed, lifetimes = [], 0, []
        for (gg, rep), rs in sorted(reps.items()):
            if gg != g:
                continue
            if rs[0]["followed"] != "True":
                not_followed += 1
                continue
            rs = sorted(rs, key=lambda r: int(r["sweep"]))
            outs.append(outcome(rs))
            gone = [int(r["sweep"]) for r in rs if int(r["n_d1"]) == 0]
            lifetimes.append(gone[0] if gone else None)
        print("g = %.2f: followed %d, not followed %d; moved %d, annealed %d, stayed %d; first sweep without a "
              "leftover: %s -> %s" % (g, len(outs), not_followed, outs.count("moved"), outs.count("annealed"),
                                      outs.count("stayed"), lifetimes, verdict(outs)))


if __name__ == "__main__":
    main()
