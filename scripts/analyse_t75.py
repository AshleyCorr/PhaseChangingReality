"""T75's verdicts, read from the runner's rows (scripts/run_t75.py). Usage, naming every run read:

    python scripts/analyse_t75.py results/t75_n512_beside_g1p5.csv results/t75_n512_far_g1p5.csv ...

Implements the verdict words of PREREGISTRATION.md T75 as amended (Amendment 2, item 4; Amendment 7 for the cells and
the cross word), with the reader's choices
below fixed as written by Amendment 3, item 4. Per replica, records in order of fair time, D the diameter of the
replica's starting graph:

  healed    the damage is gone at the last record;
  confined  damage remains at the last record and R is never above 3;
  reached   R reaches D/4 (R >= D/4) at some record; t_reach is the first such record;
  travels   reached, with at most 8 points damaged at every record up to and including t_reach;
  jumped    a record marked `jump` at or before t_reach (at any record if R never reaches D/4);
  speed     for a replica that reached: the least-squares slope of R against fair time over the records from the
            first with R >= 3 to t_reach, both included (NaN if that stretch has fewer than two records).

Per cell (placement, coupling, size), 32 replicas, the first word that holds, in this order: HEALS (at least 24
healed), CONFINED (24 confined), TRAVELS (24 travels), SPREADS (24 reached), JUMPS (8 jumped), else MIXED. TRAVELS is
read before SPREADS because every travelling replica also reaches D/4. Whether at least 8 jumped is reported beside
the word whatever the word is, so "SPREADS, with jumps" can be read. The cell's speed is the mean of its reached
replicas' speeds, with its standard error. The flaw's cost (Amendment 3, item 1, and Amendment 4: the cheapest
valid switch at its point, beside a knot the cheapest touching no knot point, always above zero by Amendment 5; it varies from place to place) is reported beside every word: its smallest, mean and largest over the
cell's replicas.

Per (coupling, size), across the two kinds of disturbance (Amendment 7: the single flaw far from knots, placement
"far", against the pair of flaws far from knots, placement "pair"; the single flaw beside a knot is reported beside them
as a placement and is not in this word): ONE SPEED if both cells SPREAD or TRAVEL and their speeds differ by at most
20 % of the larger; SEVERAL SPEEDS if both do and differ by more; NO SIGNAL if neither does.
When exactly one does, the word is UNSCORED, said so (Amendment 3, item 4).

Prints the table. Nothing random; pure functions tested in tests/test_t75.py.
"""
import csv
import sys
from collections import defaultdict

import numpy as np

QUORUM, JUMP_QUORUM = 24, 8
CONFINED_R, SPREAD_FROM, TRAVEL_MAX = 3, 3, 8
SAME_SPEED = 0.20
SIGNAL = ("SPREADS", "TRAVELS")


def replica_reading(records, d):
    """The per-replica flags of the docstring; `records` is a list of dicts with fair_sweep, damaged, R, jump."""
    rec = sorted(records, key=lambda r: r["fair_sweep"])
    t = np.array([r["fair_sweep"] for r in rec], dtype=float)
    dam = np.array([r["damaged"] for r in rec])
    rad = np.array([r["R"] for r in rec])
    jump = np.array([bool(r["jump"]) for r in rec])
    healed = dam[-1] == 0
    confined = dam[-1] > 0 and rad.max() <= CONFINED_R
    hit = np.flatnonzero(rad >= d / 4)
    reached = hit.size > 0
    end = int(hit[0]) if reached else len(rec) - 1
    travels = reached and bool((dam[:end + 1] <= TRAVEL_MAX).all())
    jumped = bool(jump[:end + 1].any())
    speed = float("nan")
    if reached:
        start = int(np.flatnonzero(rad >= SPREAD_FROM)[0])
        if end - start >= 1:
            speed = float(np.polyfit(t[start:end + 1], rad[start:end + 1], 1)[0])
    return dict(healed=bool(healed), confined=bool(confined), reached=bool(reached), travels=bool(travels),
                jumped=jumped, speed=speed, t_reach=(float(t[end]) if reached else None),
                cost=float(rec[0].get("flaw_cost", float("nan"))))


def cell_word(readings):
    """The cell's word, its counts, whether at least JUMP_QUORUM jumped, and its speed (mean, standard error, n)."""
    count = {k: sum(r[k] for r in readings) for k in ("healed", "confined", "travels", "reached", "jumped")}
    if count["healed"] >= QUORUM:
        word = "HEALS"
    elif count["confined"] >= QUORUM:
        word = "CONFINED"
    elif count["travels"] >= QUORUM:
        word = "TRAVELS"
    elif count["reached"] >= QUORUM:
        word = "SPREADS"
    elif count["jumped"] >= JUMP_QUORUM:
        word = "JUMPS"
    else:
        word = "MIXED"
    speeds = np.array([r["speed"] for r in readings if r["reached"] and np.isfinite(r["speed"])])
    mean = float(speeds.mean()) if speeds.size else float("nan")
    err = float(speeds.std(ddof=1) / np.sqrt(speeds.size)) if speeds.size > 1 else float("nan")
    costs = np.array([r.get("cost", float("nan")) for r in readings])
    return dict(word=word, replicas=len(readings), jumps=count["jumped"] >= JUMP_QUORUM, speed=mean, speed_err=err,
                n_speed=int(speeds.size), cost_min=float(np.min(costs)), cost_mean=float(np.mean(costs)),
                cost_max=float(np.max(costs)), **count)


def signal_word(cell_a, cell_b):
    """ONE SPEED / SEVERAL SPEEDS / NO SIGNAL across the two placements (see the docstring)."""
    a, b = cell_a["word"] in SIGNAL, cell_b["word"] in SIGNAL
    if not a and not b:
        return "NO SIGNAL"
    if a != b:
        return "UNSCORED (exactly one placement spreads)"
    va, vb = cell_a["speed"], cell_b["speed"]
    if not (np.isfinite(va) and np.isfinite(vb)):
        return "UNSCORED: a speed could not be fitted"
    return "ONE SPEED" if abs(va - vb) <= SAME_SPEED * max(abs(va), abs(vb)) else "SEVERAL SPEEDS"


def read_rows(paths):
    rows = []
    for p in paths:
        with open(p, newline="") as fh:
            for r in csv.DictReader(fh):
                rows.append(dict(N=int(r["N"]), g=float(r["g"]), placement=r["placement"], replica=int(r["replica"]),
                                 D=int(r["D"]), fair_sweep=float(r["fair_sweep"]), damaged=int(r["damaged"]),
                                 R=int(r["R"]), jump=r["jump"] == "True", flaw_cost=float(r["flaw_cost"])))
    return rows


def verdicts(rows):
    """{(N, g, placement): cell} and {(N, g): signal word}."""
    by_rep = defaultdict(list)
    for r in rows:
        by_rep[(r["N"], r["g"], r["placement"], r["replica"])].append(r)
    by_cell = defaultdict(list)
    for (n, g, place, _), recs in by_rep.items():
        by_cell[(n, g, place)].append(replica_reading(recs, recs[0]["D"]))
    cells = {k: cell_word(v) for k, v in by_cell.items()}
    signal = {}
    for (n, g) in sorted({(k[0], k[1]) for k in cells}):
        if (n, g, "far") in cells and (n, g, "pair") in cells:
            signal[(n, g)] = signal_word(cells[(n, g, "far")], cells[(n, g, "pair")])
    return cells, signal


def main(paths):
    cells, signal = verdicts(read_rows(paths))
    for (n, g, place), c in sorted(cells.items()):
        print("N %5d  g %-6g %-6s  %-8s%s  healed %2d confined %2d travels %2d reached %2d jumped %2d of %d;"
              "  speed %.4g +- %.2g (%d);  flaw cost %g / %.4g / %g (min / mean / max)"
              % (n, g, place, c["word"], " with jumps" if c["jumps"] else "", c["healed"], c["confined"], c["travels"],
                 c["reached"], c["jumped"], c["replicas"], c["speed"], c["speed_err"], c["n_speed"], c["cost_min"],
                 c["cost_mean"], c["cost_max"]))
    for (n, g), w in sorted(signal.items()):
        print("N %5d  g %-6g  single far flaw against the pair: %s" % (n, g, w))
    return cells, signal


if __name__ == "__main__":
    main(sys.argv[1:])
