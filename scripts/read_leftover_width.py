"""EXPLORATORY, read-only (12 October 2026): the tube runs' end states that are not exactly flat. Usage:

    python scripts/read_leftover_width.py [results_dir]

Why. `read_endpoint_width.py` and `read_flat_kind.py` read the end states with H = 0, and found them six around.
About one tube run in six ends with energy left instead, resting on a leftover, and those were not read. T63's
control (12 October) hinted that they are not six around.

What is read. Every saved final graph of the tube runs T8, T23, T24 and T60 (the bath runs of paper 1's maps; the
sealed runs of T61 are another question). Each is put in one class:
    FLAT           every link on exactly two squares (H = 0 at every lambda)
    STILL A TUBE   more than a quarter of a surplus square per point left (the curled tube has one per point)
    LEFTOVER       the rest: converted, with energy left
and for each, `sees3`: the share of its points with exactly 25 points within three steps, which is what a point
of a flat sheet has. By the thresholds T61 to T64 use: WIDE if sees3 >= 0.5, TUBE if sees3 < 0.05, PATCHWORK
between. A point of a flat torus six around has fewer than 25, so an exactly flat tube six around has sees3 = 0.

A flaw changes the count of every point near it, so a count of 25 beside a flaw says nothing about width. So a
second number is read from positions: the points with **no flawed link within three steps** (a flawed link is one
not on exactly two squares). Round such a point everything it can see in three steps is locally flat, and then
the count is a statement about the tube there: 25 if no loop of six runs round it, 22 on a tube six around, 20
on the curled tube. These are the "clear" points. Last, where an end state has clear points of both kinds: is
any clear point seeing 22 linked to a clear point seeing 25? If never, the two widths meet only across a flaw.

What this is not. sees3 certifies no geometry and names no shape; a count is not a position. Nothing here is a
registered rule and no verdict changes. It reads results/ and writes nothing.
"""
import glob
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from numba import njit                                            # noqa: E402

from graphity.cqg import surplus, total_squares                   # noqa: E402
from graphity.squares import squares_on_edge                      # noqa: E402
from run_sixloop_widen import balls                               # noqa: E402

RUNS = ("t8_lam*_adj", "t23_lam*_adj", "t24_lam*_adj", "t60_lam*_adj")
FLAT, STILL, LEFTOVER = "FLAT", "STILL A TUBE", "LEFTOVER"
WIDE_AT, TUBE_BELOW = 0.5, 0.05


REACH = 3


@njit(cache=True)
def flaw_distance(adj):
    """Steps from each point to the nearest point on a flawed link (a link not on exactly two squares); -1 where
    there is none to reach."""
    n = adj.shape[0]
    dist = np.full(n, -1, dtype=np.int64)
    queue = np.empty(n, dtype=np.int64)
    tail = 0
    for u in range(n):
        for k in range(4):
            if squares_on_edge(adj, u, adj[u, k]) != 2 and dist[u] < 0:
                dist[u] = 0
                queue[tail] = u
                tail += 1
    head = 0
    while head < tail:
        v = queue[head]
        head += 1
        for k in range(4):
            w = adj[v, k]
            if dist[w] < 0:
                dist[w] = dist[v] + 1
                queue[tail] = w
                tail += 1
    return dist


def clear_counts(adj):
    """The three-step counts of the clear points: those with no flawed link within three steps."""
    dist = flaw_distance(adj)
    return balls(adj)[:, 0][(dist < 0) | (dist > REACH)]


def joint(adj):
    """(both, touch). both: clear points seeing 22 and clear points seeing 25 both occur. touch: some clear point
    seeing 22 is linked to a clear point seeing 25, so the two widths meet with no flaw between them."""
    dist = flaw_distance(adj)
    clear = (dist < 0) | (dist > REACH)
    count = balls(adj)[:, 0]
    six, wide = clear & (count == 22), clear & (count == 25)
    return bool(six.any() and wide.any()), bool(np.any(six[:, None] & wide[adj]))


def sees3(adj):
    return float(np.mean(balls(adj)[:, 0] == 25))


def word(share):
    return "WIDE" if share >= WIDE_AT else ("TUBE" if share < TUBE_BELOW else "PATCHWORK")


def read(adj):
    """(class, sees3, clear points, clear points seeing 25, clear points seeing 22, both widths present, the two
    widths touching) of one end state."""
    n = adj.shape[0]
    s, x = int(total_squares(adj)), int(surplus(adj))
    kind = FLAT if (s == n and x == 0) else (STILL if x > n / 4 else LEFTOVER)
    clear = clear_counts(adj)
    return (kind, sees3(adj), len(clear), int(np.sum(clear == 25)), int(np.sum(clear == 22))) + joint(adj)


def table(results="results", runs=RUNS):
    """{(run, lambda tag, N): list of read(adj)}"""
    out = defaultdict(list)
    for pattern in runs:
        for folder in sorted(glob.glob(str(Path(results) / pattern))):
            m = re.match(r"(t\d+)_lam(\d+)", Path(folder).name)
            for path in sorted(Path(folder).glob("N*_rep*.npz")):
                adj = np.load(path)["adj"]
                out[(m.group(1), m.group(2), adj.shape[0])].append(read(adj))
    return out


def summary(rows):
    kinds = Counter(r[0] for r in rows)
    left = [r for r in rows if r[0] == LEFTOVER]
    shares = [r[1] for r in left]
    return dict(states=len(rows), flat=kinds[FLAT], still=kinds[STILL], leftover=len(left),
                flat_tube=sum(word(r[1]) == "TUBE" for r in rows if r[0] == FLAT),
                words=Counter(word(s) for s in shares),
                median=(statistics.median(shares) if shares else float("nan")),
                clear=sum(r[2] for r in left), clear25=sum(r[3] for r in left), clear22=sum(r[4] for r in left),
                some_wide=sum(r[3] > 0 for r in left),          # a clear point with no loop of six round it
                mostly_wide=sum(r[2] > 0 and 2 * r[3] > r[2] for r in left),
                all_six=sum(r[2] > 0 and r[4] == r[2] for r in left),
                all_wide=sum(r[2] > 0 and r[3] == r[2] for r in left),      # no loop of six away from the flaws
                both=sum(r[3] > 0 and r[4] > 0 and r[3] + r[4] == r[2] for r in left),
                two_widths=sum(r[5] for r in left), touching=sum(r[6] for r in left),
                no_clear=sum(r[2] == 0 for r in left))


def main(results="results"):
    cells = table(results)
    if not cells:
        print("no saved end states of %s in %s" % (", ".join(RUNS), results))
        return 1
    print("End states of the tube runs that are not exactly flat (exploratory; sees3 is a count, not a shape).\n")
    by_size = defaultdict(list)
    for (run, lam, n), rows in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        s = summary(rows)
        by_size[n] += rows
        print("%-4s lam %s N = %-4d states %4d | flat %4d (six around by sees3: %4d) | still a tube %3d | leftover %3d: "
              "TUBE %d, PATCHWORK %d, WIDE %d%s" % (
                  run, lam, n, s["states"], s["flat"], s["flat_tube"], s["still"], s["leftover"], s["words"]["TUBE"],
                  s["words"]["PATCHWORK"], s["words"]["WIDE"],
                  "" if not s["leftover"] else " | median sees3 %.2f" % s["median"]))
    print("\nAll four runs and every lambda together, by size:")
    for n, rows in sorted(by_size.items()):
        s = summary(rows)
        print("N = %-4d states %5d | flat %5d (six around by sees3: %5d) | still a tube %4d | leftover %4d: TUBE %d, "
              "PATCHWORK %d, WIDE %d | median sees3 %.2f, about %.0f points" % (
                  n, s["states"], s["flat"], s["flat_tube"], s["still"], s["leftover"], s["words"]["TUBE"],
                  s["words"]["PATCHWORK"], s["words"]["WIDE"], s["median"], s["median"] * n))
    print()
    print("The same end states with a leftover, read from their clear points (no flawed link within three steps):")
    for n, rows in sorted(by_size.items()):
        s = summary(rows)
        print("N = %-4d leftover %4d | no clear point %4d | every clear point six around %4d | a clear point with no loop "
              "of six round it %4d (most of the clear points so %4d; every one %4d; some 22 and some 25 and nothing else "
              "%4d) | clear points %6d: seeing 22 %6d, seeing 25 %6d | both widths among the clear points %4d, of those "
              "with the two widths linked directly %d" % (
                  n, s["leftover"], s["no_clear"], s["all_six"], s["some_wide"], s["mostly_wide"], s["all_wide"],
                  s["both"], s["clear"], s["clear22"], s["clear25"], s["two_widths"], s["touching"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
