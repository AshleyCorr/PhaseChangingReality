"""The pre-registered T25 reading: does the scrap freeze in before it heals? Usage:

    python scripts/analyse_t25.py

Implements PREREGISTRATION.md section T25 (written 2026-09-24, before the runs) on results/t25_race_*.csv.
Per cooling time t_cool: the followed replicas, and of those the share whose leftover **survives**, meaning
at least one vertex at d = 1 in the final block (after the hold at g_cold). At least MIN_FOLLOWED followed
replicas are needed to read a cooling time. Verdicts:

  FREEZES IN     some cooling time has a survival share above one half, and the fastest cooling time read
                 has the highest share;
  ALWAYS HEALS   no cooling time read has a survival share above one half;
  MIXED          a share above one half somewhere, but the fastest cooling read is not the best.

Reported, not scored: the freeze-out time t* (the longest cooling time with survival above one half), and
for each healed replica the sweep and the coupling at which the leftover was last seen. Pure functions of
parsed rows, tested in tests/test_t25.py.
"""
import csv
import glob
import math
import sys
from collections import defaultdict
from pathlib import Path

MIN_FOLLOWED = 8
HALF = 0.5


def by_replica(rows):
    """{(t_cool, replica): [rows in block order]} over followed replicas only."""
    out = defaultdict(list)
    for r in rows:
        if r["followed"] == "True":
            out[(int(r["t_cool"]), int(r["replica"]))].append(r)
    for k in out:
        out[k].sort(key=lambda r: int(r["block"]))
    return out


def survived(blocks):
    return int(blocks[-1]["n_d1"]) > 0


def last_seen(blocks):
    """(sweep, g) of the last block in which the leftover was present; None if it survived."""
    if survived(blocks):
        return None
    present = [b for b in blocks if int(b["n_d1"]) > 0]
    b = present[-1] if present else blocks[0]
    return int(b["sweep"]), float(b["g"])


def shares(rows):
    """{t_cool: (survivors, followed)}"""
    reps = by_replica(rows)
    out = defaultdict(lambda: [0, 0])
    for (t, _), blocks in reps.items():
        out[t][1] += 1
        out[t][0] += 1 if survived(blocks) else 0
    return {t: (s, f) for t, (s, f) in out.items()}


def verdict(share_by_t):
    """share_by_t: {t_cool: (survivors, followed)}."""
    read = {t: s / f for t, (s, f) in share_by_t.items() if f >= MIN_FOLLOWED}
    if not read:
        return "NOT READ", None
    above = [t for t, p in read.items() if p > HALF]
    t_star = max(above) if above else None
    if not above:
        return "ALWAYS HEALS", None
    fastest = min(read)
    if read[fastest] >= max(read.values()):
        return "FREEZES IN", t_star
    return "MIXED", t_star


def main(out_dir="results"):
    rows = []
    for f in sorted(glob.glob(str(Path(out_dir) / "t25_race_*.csv"))):
        rows += list(csv.DictReader(open(f, newline="")))
    if not rows:
        print("no T25 results yet")
        return
    sh = shares(rows)
    reps = by_replica(rows)
    for t in sorted(sh):
        s, f = sh[t]
        healed = [last_seen(b) for (tt, _), b in reps.items() if tt == t and not survived(b)]
        sweeps = sorted(x[0] for x in healed)
        print("t_cool=%-7d followed %2d survived %2d (%.2f)%s | healed at sweeps %s"
              % (t, f, s, s / f if f else math.nan, "" if f >= MIN_FOLLOWED else " NOT READ",
                 sweeps[:12] if sweeps else "-"))
    v, t_star = verdict(sh)
    print("\nVERDICT: %s%s" % (v, "" if t_star is None else " (freeze-out time t* = %d sweeps)" % t_star))


if __name__ == "__main__":
    main(*sys.argv[1:])
