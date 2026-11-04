"""EXPLORATORY, read-only (11 October 2026): when a tube opens and ends flat, how wide is what it opened into? Usage:

    python scripts/read_endpoint_width.py [results_dir]

Why. A tube is a torus four steps around. Its end state counts as flat when every edge carries exactly two squares
(H = 0). That is a count at each edge, and a torus only six steps around already satisfies it. Paper 1 says so and
says that growth of both periods has not been measured. Reading T56 on 11 October (`scripts/read_open_shells.py`)
showed end states that are flat by every count and still long and thin. This script measures the width.

What is read. Every saved final graph of the tube runs T8, T23, T24 and T60 (sizes 64 to 192, and T60's held-out
size 288, which was left out until it had been read by its own rule on 11 October). For each graph with H = 0
exactly: its
**width**, the length of the shortest closed path that cannot be shrunk to a point. It is read from the balls: in
an endless flat sheet the number of points within r steps of a point is 2r^2 + 2r + 1, and on a flat torus the
count first falls short of that at r = width / 2 (exact for a flat torus: two paths of r steps close a loop of
2r). The tube's width is 4; the least a flat torus can have is 6; a torus as wide as it is long has about the
square root of 2N.

Nothing here is a registered rule and no verdict changes. It reads results/ and writes nothing.
"""
import glob
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graphity.cqg import surplus, total_squares                   # noqa: E402
from read_open_shells import shells                               # noqa: E402

RUNS = ("t8_lam*_adj", "t23_lam*_adj", "t24_lam*_adj", "t60_lam*_adj")
SIZES = (64, 96, 144, 192, 288)       # 288 added 11 October, after T60's held-out size was read by its rule


def twisted_tube(length, shift=2):
    """The 4 x `length` tube with its seam shifted by `shift` steps along the tube: point (x, y) has x in 0..3 round
    the tube and y in 0..length-1 along it, and stepping round past x = 3 arrives at (0, y - shift). With shift 0 it
    is the tube itself; with shift 2 no four steps round it close, the shortest way round is six steps, and every
    edge carries exactly two squares."""
    def index(x, y):
        return x * length + (y % length)
    adj = np.empty((4 * length, 4), dtype=np.int64)
    for x in range(4):
        for y in range(length):
            ahead = index(0, y - shift) if x == 3 else index(x + 1, y)
            back = index(3, y + shift) if x == 0 else index(x - 1, y)
            adj[index(x, y)] = (ahead, back, index(x, y + 1), index(x, y - 1))
    return adj


def width(adj, start=0):
    """Twice the first radius at which the ball round `start` is smaller than the flat sheet's; None if the whole
    graph is reached first. For a flat torus this is the length of its shortest loop that cannot be shrunk."""
    reached = 0
    for r, shell in enumerate(shells(adj, start)):
        reached += shell
        if reached < 2 * r * r + 2 * r + 1:
            return 2 * r
    return None


def is_flat(adj):
    """Every edge on exactly two squares: as many squares as points and no surplus. Then H = 0 at every lambda."""
    return int(total_squares(adj)) == adj.shape[0] and int(surplus(adj)) == 0


def main(results="results"):
    table = defaultdict(lambda: [0, 0, Counter(), 0])       # saved, flat, widths of the flat ones, uneven
    for pattern in RUNS:
        for folder in sorted(glob.glob(str(Path(results) / pattern))):
            m = re.search(r"(t\d+)_lam(\d+)", Path(folder).name)
            run, lam = m.group(1), int(m.group(2)) / 100.0
            for path in sorted(Path(folder).glob("N*_rep*.npz")):
                n = int(re.match(r"N(\d+)_", path.name).group(1))
                if n not in SIZES:
                    continue
                adj = np.load(path)["adj"]
                cell = table[(run, lam, n)]
                cell[0] += 1
                if not is_flat(adj):
                    continue
                cell[1] += 1
                seen = {width(adj, s) for s in (0, n // 3, 2 * n // 3)}
                cell[3] += len(seen) > 1                     # a flat torus looks the same from every point
                cell[2][min(seen)] += 1
    print("Saved end states of the tube runs: how many are exactly flat, and how wide those are (exploratory).")
    print("The tube is 4 around; a flat torus is at least 6; a square one of N points is about sqrt(2N).\n")
    print("%-4s %-5s %-4s %6s %5s  %s" % ("run", "lam", "N", "saved", "flat", "width: count, among the flat"))
    pooled = defaultdict(Counter)
    for (run, lam, n), (saved, flat, widths, uneven) in sorted(table.items()):
        print("%-4s %-5.2f %-4d %6d %5d  %s%s" % (run, lam, n, saved, flat,
              ", ".join("%d: %d" % kv for kv in sorted(widths.items())), "  (uneven: %d)" % uneven if uneven else ""))
        pooled[n].update(widths)
    print("\nAll runs and lambda together, by size (width: count):")
    for n in sorted(pooled):
        total = sum(pooled[n].values())
        print("  N = %-4d flat end states %5d | %s | at the least width, 6: %.0f %%"
              % (n, total, ", ".join("%d: %d" % kv for kv in sorted(pooled[n].items())),
                 100.0 * pooled[n][6] / total if total else float("nan")))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "results"))
