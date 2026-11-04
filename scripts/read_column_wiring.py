"""EXPLORATORY, read-only (14 October 2026): the wiring of the four-point leftover, the column. Usage:

    python scripts/read_column_wiring.py [results_dir]

Why. The papers' plain-language editions are to draw the common leftover, and a drawing must show what the wiring
fixes and nothing more. O16 corrected "one ring of the tube" to "a four-point remnant lying along the tube", and O107
calls it the column: four points at d = 1 with 4 units at their points and 14 within one step in most. Two things
were not read: which links at its points carry a third square, and whether its loop of four goes around the new
tube (as a ring of the old tube went around the old one).

What is read. Every saved final graph of paper 1's tube maps (T8, T23, T24, T60) and of paper 2's large test (T51).
A column is a connected piece of points not at d = 2 that has four points, all at d = 1 (`analyse_t55.is_column`).
For each:
    pattern   the squares on its four inside links, sorted; the squares on its eight links out, sorted; and whether
              the inside links on three squares are two opposite sides of the loop
    groups    remove the four points; how many groups their eight outside neighbours fall into when each may reach
              the others only within REACH steps. A loop that went around the tube, as a ring of the old tube did,
              would cut it locally into two groups; a loop that does not would leave one.
Exact for each graph. Which graphs were read and how they are grouped is ours and was not registered. Nothing here
is a verdict; it reads results/ and writes nothing.
"""
import glob
import sys
from collections import Counter, deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyse_t55 import is_column, pieces_of                       # noqa: E402
from graphity.cqg_d import squares_on_edge                          # noqa: E402

RUNS = ("t8_lam*_adj", "t23_lam*_adj", "t24_lam*_adj", "t60_lam*_adj", "t51_*_adj")
REACH = 10


def pattern(adj, col):
    """(squares on the four inside links, sorted; on the eight links out, sorted; the inside links on three squares
    are two opposite sides)."""
    inside, out, three = [], [], []
    for v in col:
        for w in adj[v]:
            w, s = int(w), int(squares_on_edge(adj, v, int(w)))
            if w in col:
                if v < w:
                    inside.append(s)
                    if s == 3:
                        three.append({v, w})
            else:
                out.append(s)
    opposite = len(three) == 2 and not (three[0] & three[1])
    return tuple(sorted(inside)), tuple(sorted(out)), opposite


def groups(adj, col, reach=REACH):
    """How many groups the outside neighbours of `col` fall into, with `col` removed and walks limited to `reach`."""
    gone = set(col)
    outside = sorted({int(w) for v in col for w in adj[v]} - gone)
    group = {}
    for s in outside:
        if s in group:
            continue
        dist, queue = {s: 0}, deque([s])
        while queue:
            v = queue.popleft()
            if dist[v] == reach:
                continue
            for w in adj[v]:
                w = int(w)
                if w not in gone and w not in dist:
                    dist[w] = dist[v] + 1
                    queue.append(w)
        for t in outside:
            if t in dist:
                group.setdefault(t, s)
    return len(set(group.values()))


def columns(adj):
    pieces, d = pieces_of(adj)
    return [p for p in pieces if is_column(p, d)]


def read_all(results="results", runs=RUNS):
    """Counter of (pattern, groups) over every column in the named runs; and the number of graphs read."""
    tally, graphs = Counter(), 0
    for run in runs:
        for path in sorted(glob.glob(str(Path(results) / run / "*.npz"))):
            adj = np.load(path)["adj"]
            graphs += 1
            for col in columns(adj):
                tally[(pattern(adj, col), groups(adj, col))] += 1
    return tally, graphs


def main(results="results"):
    tally, graphs = read_all(results)
    total = sum(tally.values())
    print("Columns in %d saved graphs of %s: %d (exploratory; exact per graph).\n" % (graphs, ", ".join(RUNS), total))
    print("count  inside        out                        opposite  groups")
    for ((inside, out, opposite), g), n in tally.most_common():
        print("%5d  %-12s  %-25s  %-8s  %d" % (n, inside, out, opposite, g))
    cut = sum(n for (_, g), n in tally.items() if g > 1)
    print("\nColumns whose removal cuts the tube within %d steps: %d of %d." % (REACH, cut, total))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
