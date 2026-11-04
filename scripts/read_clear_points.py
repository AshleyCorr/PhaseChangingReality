"""EXPLORATORY, read-only (12 October 2026): the end states of T62 and T64 read from their clear points. Usage:

    python scripts/read_clear_points.py [results_dir]

Why. T62 and T64 call an end state WIDE when at least half its points have exactly 25 points within three steps
(sees3). A flaw changes that count for every point near it (scripts/read_leftover_width.py), and these end states
hold flaws. So the same end states are read here only from their **clear points**: those with no flawed link
within three steps, round which everything in view is locally flat. A clear point sees 25 if no loop of six runs
round the tube there, and 22 on a tube six around.

What is printed, per cell: how many of its points are clear at all; of the clear points how many see 25 and how
many 22; how many end states have every clear point seeing 25, or every one seeing 22; and whether the two widths
ever touch with no flaw between them.

Nothing here is a registered rule and no verdict changes. It reads a named set of runs in results/ and writes
nothing.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import read_leftover_width as w                                   # noqa: E402

SIZES = (96, 192, 384)
T62 = ["t62_mu%03d_g%03d_n%d" % (m, g, n) for n in SIZES for m in (0, 10, 20, 25, 30) for g in (150, 200)]
T64 = ["t64_mu%04d_n%d" % (m, n) for n in SIZES for m in (0, 25, 50, 100, 200)]


def cell(results, name):
    """dict for one run's saved end states, or None if the run has none saved."""
    files = sorted((Path(results) / (name + "_adj")).glob("N*_rep*.npz"))
    if not files:
        return None
    rows = [w.read(np.load(f)["adj"]) for f in files]
    points = sum(np.load(f)["adj"].shape[0] for f in files)
    clear = sum(r[2] for r in rows)
    return dict(states=len(rows), points=points, clear=clear, see25=sum(r[3] for r in rows),
                see22=sum(r[4] for r in rows), with_clear=sum(r[2] > 0 for r in rows),
                all25=sum(r[2] > 0 and r[3] == r[2] for r in rows), all22=sum(r[2] > 0 and r[4] == r[2] for r in rows),
                two_widths=sum(r[5] for r in rows), touching=sum(r[6] for r in rows))


def main(results="results"):
    print("End states of T62 and T64 read from their clear points (exploratory; no verdict changes).\n")
    seen = 0
    for name in T62 + T64:
        c = cell(results, name)
        if c is None:
            continue
        seen += 1
        print("%-20s states %2d | clear points %5d of %5d (%.2f) | of the clear: seeing 25 %5d, seeing 22 %5d | end "
              "states with every clear point seeing 25: %2d, seeing 22: %2d, with no clear point: %2d | both widths %2d, "
              "of those touching %d" % (name, c["states"], c["clear"], c["points"], c["clear"] / c["points"], c["see25"],
                                        c["see22"], c["all25"], c["all22"], c["states"] - c["with_clear"],
                                        c["two_widths"], c["touching"]))
    if not seen:
        print("no saved end states of T62 or T64 in %s" % results)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
