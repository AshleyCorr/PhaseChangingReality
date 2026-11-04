"""EXPLORATORY, read-only (11 October 2026): is a replica that reads OPENS by the count wide flat space, or the
starting shape glued together differently? Usage:

    python scripts/read_open_shells.py [results_dir]

Why. T48's rule, which T53 and T56 use, calls a replica OPENS when at least half its points are at d = D and the
largest connected piece of such points holds at least half of all points. d is a count at one point (the pairs of
its links that close no square), and a connected piece of such points says nothing about how wide the space is.
Reading T56 on 11 October, one rod (4 x 4 x 18, lambda = 1.20, g = 2.5) ended with every point at d = 3, none
damaged, and 1.2 per point above flat space. So the count had to be checked against the shape.

What is read. For every replica of T48, T53 and T56 whose last reading is OPENS by that rule, the saved final
graph, and from it the shells: the number of points at each distance from a point. A rod's shells level off at
twice its cross-section and its ends are far apart; a slab's grow in proportion to the distance; flat space's grow
as the distance squared. Printed beside each replica: the same numbers for the torus it started from and, where a
flat torus with the same number of points and every side at least 6 exists, for that torus.

Nothing here is a registered rule and no verdict changes. It reads results/ and writes nothing.
"""
import csv
import glob
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyse_t48                                                # noqa: E402

# The flat torus with the same number of points and every side at least 6 and even, where there is one.
FLAT = {(4, 48): (12, 16), (4, 4, 18): (6, 6, 8), (4, 4, 36): (6, 8, 12), (4, 4, 72): (8, 12, 12),
        (4, 8, 12): (6, 8, 8), (4, 12, 12): (6, 8, 12)}
SAMPLE = 12                           # points per graph the shells are counted from


def torus(dims):
    """Neighbour array of the product torus with these sides."""
    index = np.arange(int(np.prod(dims))).reshape(dims)
    adj = np.empty((index.size, 2 * len(dims)), dtype=np.int64)
    for point in np.ndindex(*dims):
        k = 0
        for axis in range(len(dims)):
            for step in (-1, 1):
                other = list(point)
                other[axis] = (other[axis] + step) % dims[axis]
                adj[index[point], k] = index[tuple(other)]
                k += 1
    return adj


def shells(adj, start):
    """The number of points at distance 0, 1, 2, ... from `start`, within its connected piece."""
    seen = np.zeros(adj.shape[0], dtype=bool)
    seen[start] = True
    front, out = np.array([start]), [1]
    while True:
        near = np.unique(adj[front].ravel())
        front = near[~seen[near]]
        if front.size == 0:
            return out
        seen[front] = True
        out.append(int(front.size))


def profile(adj, sample=SAMPLE):
    """(median largest shell, median distance to the farthest point, median points within four steps) over
    `sample` evenly spaced starting points."""
    starts = np.linspace(0, adj.shape[0] - 1, sample).astype(int)
    all_shells = [shells(adj, int(s)) for s in starts]
    return (statistics.median(max(s) for s in all_shells), statistics.median(len(s) - 1 for s in all_shells),
            statistics.median(sum(s[:5]) for s in all_shells))


def saved_graph(csv_path, row):
    """The saved final graph of this replica, or None. T48 names the file by bath and spark; T53 and T56 by replica."""
    folder = Path(str(csv_path)[:-4] + "_adj")
    names = ["rep%d.npz" % int(row["replica"])]
    if "spark" in row and "C" in row:
        names.insert(0, "C%d_E%d_rep%d.npz" % (int(row["C"]), int(round(float(row["spark"]))), int(row["replica"])))
    for name in names:
        if (folder / name).is_file():
            return folder / name
    return None


def open_replicas(csv_path):
    """[(last row, blocks)] for the replicas of one result file that read OPENS at their last reading."""
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    runs = defaultdict(list)
    for r in rows:
        runs[(r.get("C", ""), r.get("spark", ""), int(r["replica"]))].append(r)
    out = []
    for blocks in runs.values():
        blocks.sort(key=lambda r: int(r["sweep"]))
        if analyse_t48.read_replica(blocks)[0] == "OPENS":
            out.append(blocks[-1])
    return out


def main(results="results"):
    reference = {}
    print("Replicas that read OPENS by the count, and their shape from the saved wiring (exploratory, read only).")
    print("largest shell / farthest point / points within four steps; then the same for the start and for flat\n")
    tally = defaultdict(lambda: [0, 0, 0])
    for pattern in ("t48_*.csv", "t53_*.csv", "t56_*.csv"):
        for path in sorted(glob.glob(str(Path(results) / pattern))):
            for last in open_replicas(path):
                dims = tuple(analyse_t48.dims_of(last))
                graph = saved_graph(path, last)
                if graph is None:
                    print("%-34s rep %-2s no saved graph" % (Path(path).stem, last["replica"]))
                    continue
                for d in (dims, FLAT.get(dims)):
                    if d is not None and d not in reference:
                        reference[d] = profile(torus(d))
                got = profile(np.load(graph)["adj"])
                start, flat = reference[dims], reference.get(FLAT.get(dims))
                if flat is None:
                    like = "no flat torus of this size"
                else:
                    # nearer the start or nearer flat, by the largest shell, on a log scale
                    like = ("like the start" if abs(np.log(got[0] / start[0])) < abs(np.log(got[0] / flat[0]))
                            else "like flat")
                n = int(last["N"])
                key = (Path(path).stem.split("_")[0], "x".join(map(str, dims)), float(last["lam"]))
                tally[key][0] += 1
                tally[key][1] += like == "like flat"
                tally[key][2] += like == "like the start"
                print("%-34s rep %-2s %-10s lam %.2f | at d=D %.2f, damaged %.2f | %5g / %3g / %5g | start %g / %g / %g"
                      " | flat %s | %s"
                      % (Path(path).stem, last["replica"], "x".join(map(str, dims)), float(last["lam"]),
                         int(last["d%d" % len(dims)]) / n, int(last["damaged_d"]) / n, *got, *start,
                         "%g / %g / %g" % flat if flat else "none", like))
    print("\nBy run, start and lambda: replicas reading OPENS, of which like flat, like the start")
    for key in sorted(tally):
        print("  %-4s %-10s lam %.2f: %d, %d, %d" % (*key, *tally[key]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "results"))
