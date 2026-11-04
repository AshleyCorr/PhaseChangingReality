"""Score T71's held-out size L = 16 against the numbers written before it ran (PREREGISTRATION.md T71, Amendment 4,
section 6; CLAUDE.md rule 14). Usage:

    PYTHONPATH="src;scripts" python scripts/t71_heldout_score.py [configs/t71_heldout_prediction.csv]

Reads the nine L = 16 files `results/t71_{model,layer,layerref}_lam{110,125,140}_l16.csv` by name, each replica with
`t71_heldout_numbers.per_replica` (the reading the numbers were written from: `analyse_t71.read_replica`, with releases
and energies in units of a N_slab), and checks every row of the prediction table:

* a word holds if the cell's majority word equals it;
* a number holds if it lies in [low, high] (a blank bound is open), with 1e-9 of slack for the exact rules;
* medians are over the replicas that have the quantity, as at stage 1; "earliest" and "latest" are the minimum and
  maximum first split; SPENT FIRST and clean moves are totals over the 16 replicas.

The verdict is AS PREDICTED if every row marked scored holds, and otherwise not AS PREDICTED with each miss named with
its rule. Rows marked not scored are printed beside, as the amendment reports them.
"""
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import t71_heldout_numbers as H   # noqa: E402

FILES = ["t71_%s_lam%d_l16" % (arm, lam) for arm in ("model", "layer", "layerref") for lam in (110, 125, 140)]
SLACK = 1e-9
MEDIANS = {"first split, median": "first_split", "release at the first split, median": "release",
           "end energy above flat, median": "end_F", "daughters, median": "daughters",
           "energy per daughter, median": "per_daughter", "period, median": "period",
           "slab share, median": "slab_share", "open share, median": "open_share", "refolds, median": "refolds"}


def cells_from_rows(rows):
    """{cell name as in the prediction table: [per-replica dicts]} from the rows of the L = 16 files."""
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["move_set"], r["release_to"], float(r["lam"]), float(r["rate"]))
        groups[key][int(r["replica"])].append(r)
    return {H.cell_name(k): [H.per_replica(rs) for _, rs in sorted(v.items())] for k, v in groups.items()}


def value(stat, reps):
    if stat == "majority word":
        return Counter(x["word"] for x in reps).most_common(1)[0][0]
    if stat in MEDIANS:
        vals = [x[MEDIANS[stat]] for x in reps if x[MEDIANS[stat]] is not None]
        return float(np.median(vals)) if vals else None
    splits = [x["first_split"] for x in reps if x["first_split"] is not None]
    if stat == "first split, earliest replica":
        return min(splits) if splits else None
    if stat == "first split, latest replica":
        return max(splits) if splits else None
    if stat == "SPENT FIRST count":
        return sum(x["spent_first"] for x in reps)
    if stat == "clean moves in all":
        return sum(x["clean_total"] for x in reps)
    if stat == "replicas with a clean move":
        return sum(x["clean_total"] > 0 for x in reps)
    raise KeyError(stat)


def holds(row, v):
    if v is None:
        return None
    if row["rule"] == "word":
        return v == row["predicted"]
    try:
        lo = float(row["low"]) if row["low"] != "" else -np.inf
        hi = float(row["high"]) if row["high"] != "" else np.inf
    except ValueError:          # a range in words ("a majority"): reported only
        return None
    return lo - SLACK <= float(v) <= hi + SLACK


def score(pred_rows, cells):
    out = []
    for row in pred_rows:
        reps = cells.get(row["cell"])
        v = value(row["statistic"], reps) if reps else None
        out.append(dict(row, value=v, holds=holds(row, v) if reps else None, n=len(reps) if reps else 0))
    scored = [r for r in out if r["scored"] == "yes"]
    missing = [r for r in scored if r["holds"] is None]
    misses = [r for r in scored if r["holds"] is False]
    verdict = "NOT READ" if missing else ("AS PREDICTED" if not misses else "not AS PREDICTED")
    return verdict, out, misses, missing


def main(path=ROOT / "configs" / "t71_heldout_prediction.csv"):
    pred = list(csv.DictReader(open(path, newline="")))
    rows = []
    for name in FILES:
        rows.extend(csv.DictReader(open(ROOT / "results" / (name + ".csv"), newline="")))
    cells = cells_from_rows(rows)
    verdict, out, misses, missing = score(pred, cells)
    for r in out:
        v = r["value"]
        vs = "-" if v is None else (v if isinstance(v, str) else "%.6g" % v)
        mark = {True: "holds", False: "MISSES", None: "not read" if r["scored"] == "yes" else "reported"}[r["holds"]]
        if r["scored"] != "yes" and r["holds"] is not None:
            mark = "reported (%s)" % ("inside" if r["holds"] else "outside")
        print("%-30s %-36s %-8s | predicted %s (%s to %s) | L = 16 %s | %s"
              % (r["cell"], r["statistic"], r["rule"], r["predicted"], r["low"] or "-", r["high"] or "-", vs, mark))
    print("\nverdict at L = 16:", verdict)
    for r in misses:
        print("  miss, rule %s: %s, %s: %s against %s to %s" % (r["rule"], r["cell"], r["statistic"], r["value"],
                                                               r["low"] or "-", r["high"] or "-"))
    for r in missing:
        print("  not read: %s, %s" % (r["cell"], r["statistic"]))
    return verdict, out


if __name__ == "__main__":
    main(*sys.argv[1:2])
