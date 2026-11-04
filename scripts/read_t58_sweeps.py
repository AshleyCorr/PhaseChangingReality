"""Read the sweep-by-sweep replay of T58's nine decays (exploratory; 10 October 2026). Usage:

    python scripts/read_t58_sweeps.py

The owner's question after T58 was read: could a second curl have come and gone between two looks of five sweeps,
and so have stretched the two long waits without being seen? `configs/t58_exploratory_sweeps_lam130_n64.json`
replays the same nine decays from their seeds with every sweep written and with the number of moves the chain
accepted in each block of five sweeps. This script first checks that the replay is the same history as T58 stage A
(every column of every row, and every block of every trace), then says, for each decay, what happened between the
start and the sweep its detector fired:

    quiet blocks   blocks of five sweeps in which the chain accepted no move at all. The graph did not change in
                   them, so nothing of any kind happened: no exit, no curl, nothing between looks.
    moves          moves accepted in the other blocks, and the stretches of sweeps they fall in
    off the tube   sweeps that ended with squares or surplus squares different from the tube's
    most squares   the highest square count after any sweep (a second curl is more squares than the tube's)
    cubes, babies  the most 4-cubes, and points in baby universes, after any sweep (a pinched-off knot is both)
    pieces         the most connected pieces after any sweep

Exploratory: a closer look at histories already read, not a pre-registered test, and no verdict.
"""
import csv
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
OLD, NEW = "t58_targets_lam130_n64", "t58_exploratory_sweeps_lam130_n64"
TARGETS = (553, 2748, 3072)
BLOCK = 5


def rows_of(path):
    return list(csv.DictReader(open(path, newline="")))


def same_history(results=RESULTS):
    """True if the replay's rows and block traces are, column for column, those of T58 stage A."""
    old, new = rows_of(results / (OLD + ".csv")), rows_of(results / (NEW + ".csv"))
    if old != new:
        return False
    for row in new:
        name = "N%s_rep%s.csv" % (row["N"], row["replica"])
        if (results / (OLD + "_trace") / name).read_bytes() != (results / (NEW + "_trace") / name).read_bytes():
            return False
    return True


def stretches(sweeps):
    """[5, 10, 15, 40] (block ends) -> [(1, 15), (36, 40)]: runs of neighbouring blocks, as first and last sweep."""
    runs = []
    for end in sweeps:
        if runs and end - runs[-1][1] == BLOCK:
            runs[-1][1] = end
        else:
            runs.append([end - BLOCK + 1, end])
    return [tuple(r) for r in runs]


def summarize(trace, until, s_tube, x_tube):
    """What the sweep-by-sweep trace shows up to and including sweep `until` (the sweep the detector fired)."""
    rows = [t for t in trace if int(t["sweep"]) <= until]
    ends = [t for t in rows if int(t["sweep"]) % BLOCK == 0]
    busy = [int(t["sweep"]) for t in ends if int(t["accepted_in_block"]) > 0]
    off = [int(t["sweep"]) for t in rows if (int(t["S"]), int(t["X"])) != (s_tube, x_tube)]
    return dict(blocks=len(ends), quiet=len(ends) - len(busy),
                moves=sum(int(t["accepted_in_block"]) for t in ends), busy=stretches(busy),
                off=len(off), first_off=(off[0] if off else None),
                s_max=max(int(t["S"]) for t in rows), cubes=max(int(t["cubes"]) for t in rows),
                baby=max(int(t["baby"]) for t in rows), pieces=max(int(t["pieces"]) for t in rows))


def main(results=RESULTS):
    if not same_history(results):
        sys.exit("the replay is not the history T58 stage A read: nothing below would mean anything")
    print("Same history as T58 stage A: every row and every block of every trace is unchanged.\n")
    print("EXPLORATORY. Up to the sweep each detector fired:\n")
    for row in rows_of(results / (NEW + ".csv")):
        n, rep, wait = int(row["N"]), int(row["replica"]), int(row["waiting"])
        trace = rows_of(results / (NEW + "_trace") / ("N%d_rep%d_sweeps.csv" % (n, rep)))
        s_tube, x_tube = (5 * n) // 4, n
        d = summarize(trace, wait, s_tube, x_tube)
        print("decay %-5d%s detector fired at sweep %d" % (rep, " (long wait)" if rep in TARGETS else "", wait))
        print("   quiet blocks %d of %d; %d moves accepted, in sweeps %s"
              % (d["quiet"], d["blocks"], d["moves"], ", ".join("%d to %d" % r for r in d["busy"]) or "none"))
        print("   off the tube after %d sweeps (first at %s); most squares %d (tube %d); most 4-cubes %d; most "
              "points in baby universes %d; most pieces %d\n"
              % (d["off"], d["first_off"], d["s_max"], s_tube, d["cubes"], d["baby"], d["pieces"]))


if __name__ == "__main__":
    main()
