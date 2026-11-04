"""T71 (draft): read the slab-under-the-layer-form runs by the rules of PREREGISTRATION.md section T71. Usage:

    python scripts/analyse_t71.py [results/t71_*.csv ...]

Per replica: the sweep of the first split, of the first paid clean switch move and of the first contraction; whether
a clean move came before any split (SPENT FIRST); the release at the first split (the fall in F between the row
before it and the row at it; exact only when rows are logged every sweep) against a N_slab; the number of refolds
(contractions after the first split); the open share (the fraction of the logged sweeps after the first split spent
at width >= 6); the end width and the end energy above flat. Words per replica where the release goes to the stores:
OPENS AND STAYS (a split and open share >= 0.9), OSCILLATES (a split, at least one refold, open share < 0.9), NEVER
SPLITS. Where the release leaves into a daughter ledger (release_to "daughter", VISION Update 66): CYCLES (two or more
daughters and at least one refold between them: the slab reforms and burps again), ONE BURP (one daughter and no
refold), NEVER SPLITS; and beside the word the number of daughters, the median energy per daughter against a N_slab, and
the median period between daughters in sweeps. Per cell (lam, L, rate, gate, leak, move_set, release_to,
feed_stops_at_split): the majority word, the count of each word, the median first split with its range, the median
release against a N_slab, the median end energy above flat, how many replicas were SPENT FIRST, and under "daughter"
the median daughters, energy per daughter and period. Nothing here scores the held-out size; that is done against the dated amendment once it exists.
"""
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

OPEN_SHARE = 0.9


def width_of(dims):
    return int(str(dims).split("x")[0])


def read_replica(rows):
    """rows: the logged rows of one replica, as dicts with the runner's columns, in sweep order."""
    rows = sorted(rows, key=lambda r: int(r["sweep"]))
    a_n = 4.0 * (float(rows[0]["lam"]) - 1.0) * int(rows[0]["N_slab"])
    first_split = rows[0]["first_split"]
    first_split = int(first_split) if str(first_split) not in ("", "None") else None
    first_clean = rows[0]["first_clean"]
    first_clean = int(first_clean) if str(first_clean) not in ("", "None") else None
    daughter_mode = rows[0].get("release_to", "stores") == "daughter"
    n_dau = int(rows[-1].get("daughters", 0) or 0)
    out = dict(first_split=first_split, first_clean=first_clean,
               spent_first=bool(first_clean is not None and (first_split is None or first_clean < first_split)),
               end_width=width_of(rows[-1]["dims"]), end_F=float(rows[-1]["F"]), a_n=a_n, release=None,
               refolds=None, open_share=None, slab_share=None, daughters=n_dau, per_daughter=None, period=None)
    if daughter_mode and n_dau:
        out["per_daughter"] = float(rows[-1]["daughter_energy"]) / n_dau
        births = [int(r["sweep"]) for prev, r in zip(rows[:-1], rows[1:])
                  if int(r["daughters"]) > int(prev["daughters"])]
        if int(rows[0]["daughters"]) > 0:
            births = [int(rows[0]["sweep"])] + births
        if len(births) >= 2:
            out["period"] = float(np.median(np.diff(births)))
    if first_split is None:
        out["word"] = "NEVER SPLITS"
        return out
    by_sweep = {int(r["sweep"]): r for r in rows}
    before = [r for r in rows if int(r["sweep"]) < first_split]
    if first_split in by_sweep:
        # F changes only in a sweep with a move, and every such sweep is logged (layer_feed, log_events), so the last
        # row before the split holds F as it stood the sweep before; with no row before it, F was the slab's a N_slab.
        f_before = float(before[-1]["F"]) if before else a_n
        out["release"] = f_before - float(by_sweep[first_split]["F"])
    after = [r for r in rows if int(r["sweep"]) >= first_split]
    n_contract_at = int(by_sweep[first_split]["n_contract"]) if first_split in by_sweep else 0
    out["refolds"] = int(rows[-1]["n_contract"]) - n_contract_at
    # the share of sweeps after the split spent at width >= 6, weighted by how long each logged width stood (the width
    # changes only in logged sweeps); the last row counts for one sweep
    if after:
        sweeps = [int(r["sweep"]) for r in after]
        durations = [b - a for a, b in zip(sweeps[:-1], sweeps[1:])] + [1]
        wide = [width_of(r["dims"]) >= 6 for r in after]
        out["open_share"] = float(sum(d for d, w in zip(durations, wide) if w) / sum(durations))
    else:
        out["open_share"] = 0.0
    out["slab_share"] = 1.0 - out["open_share"]
    if daughter_mode:
        out["word"] = "CYCLES" if (n_dau >= 2 and out["refolds"] >= 1) else "ONE BURP"
        return out
    # the share decides the word; the refolds are reported beside it (a refold undone again still counts as open)
    out["word"] = "OPENS AND STAYS" if out["open_share"] >= OPEN_SHARE else "OSCILLATES"
    return out


def read_cells(rows):
    """rows: every row of one or more result files. Returns {cell key: dict} with the per-cell reading."""
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (float(r["lam"]), int(r["L"]), float(r["rate"]), float(r["gate"]), float(r["leak"]), r["move_set"],
               r.get("release_to", "stores"), int(r["feed_stops_at_split"]))
        groups[key][int(r["replica"])].append(r)
    cells = {}
    for key, reps in sorted(groups.items()):
        reads = [read_replica(rs) for _, rs in sorted(reps.items())]
        words = Counter(x["word"] for x in reads)
        splits = [x["first_split"] for x in reads if x["first_split"] is not None]
        releases = [x["release"] for x in reads if x["release"] is not None]
        cells[key] = dict(
            n=len(reads), words=dict(words), majority=words.most_common(1)[0][0],
            median_first_split=float(np.median(splits)) if splits else None,
            range_first_split=(min(splits), max(splits)) if splits else None,
            median_release=float(np.median(releases)) if releases else None, a_n=reads[0]["a_n"],
            median_end_F=float(np.median([x["end_F"] for x in reads])),
            spent_first=sum(x["spent_first"] for x in reads), replicas=reads,
            median_daughters=float(np.median([x["daughters"] for x in reads])),
            median_per_daughter=(float(np.median([x["per_daughter"] for x in reads if x["per_daughter"] is not None]))
                                 if any(x["per_daughter"] is not None for x in reads) else None),
            median_period=(float(np.median([x["period"] for x in reads if x["period"] is not None]))
                           if any(x["period"] is not None for x in reads) else None),
            median_slab_share=(float(np.median([x["slab_share"] for x in reads if x["slab_share"] is not None]))
                               if any(x["slab_share"] is not None for x in reads) else None))
    return cells


def main(paths):
    rows = []
    for p in paths:
        rows.extend(csv.DictReader(open(p, newline="")))
    cells = read_cells(rows)
    for key, c in cells.items():
        lam, L, rate, gate, leak, move_set, release_to, stops = key
        print("lam=%.2f L=%2d r=%-7g gate=%-7.1f leak=%-7g %-8s %-8s feed stops=%d | %-15s %s | first split median %s range %s | release median %s vs a N_slab %.1f | end F median %.1f | spent first %d/%d%s"
              % (lam, L, rate, gate, leak, move_set, release_to, stops, c["majority"], c["words"], c["median_first_split"],
                 c["range_first_split"], c["median_release"], c["a_n"], c["median_end_F"], c["spent_first"], c["n"],
                 (" | daughters median %s, per daughter %s, period %s, slab share %s" % (c["median_daughters"], c["median_per_daughter"], c["median_period"], c["median_slab_share"]))
                 if release_to == "daughter" else ""))
    return cells


if __name__ == "__main__":
    paths = sys.argv[1:] or sorted(str(p) for p in Path("results").glob("t71_*.csv"))
    main(paths)
