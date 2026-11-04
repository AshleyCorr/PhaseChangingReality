"""Reported, not scored (PREREGISTRATION T65, "Definitions": the leftover pieces). Usage:

    python scripts/read_t65_pieces.py

For every saved end graph of T65 (results/t65_*_adj/*.npz): the census (open d = 3, curled d < 3, damaged d > 3),
the number and sizes of separate open regions, of separate curled pieces and of separate damaged pieces, read from
the wiring with `graphity.dimension.pieces_of`. Curled pieces are listed only for replicas in which the open points
are the majority, so that they are leftovers inside an opened space and not the unopened slab. Writes nothing.
"""
import glob
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphity.dimension import local_dimension_d, pieces_of  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def read_graph(path):
    z = np.load(path)
    adj = z["adj"]
    n = adj.shape[0]
    d = local_dimension_d(adj)
    op, cu, dm = d == 3, d < 3, d > 3
    out = dict(n=n, open=int(op.sum()), curled=int(cu.sum()), damaged=int(dm.sum()))
    out["open_regions"] = sorted(pieces_of(adj, op), reverse=True) if op.any() else []
    out["damaged_pieces"] = sorted(pieces_of(adj, dm), reverse=True) if dm.any() else []
    out["curled_pieces"] = sorted(pieces_of(adj, cu), reverse=True) if cu.any() else []
    return out


def main():
    dirs = sorted(glob.glob(str(ROOT / "results" / "t65_*_adj")))
    for dd in dirs:
        name = Path(dd).name.replace("_adj", "")
        cells = defaultdict(list)
        for f in sorted(glob.glob(dd + "/*.npz")):
            m = re.match(r"C(\d+)_E(\d+)_rep(\d+)", Path(f).name)
            cells[(int(m.group(1)), int(m.group(2)))].append(read_graph(f))
        print("==", name)
        for (c, e), reps in sorted(cells.items()):
            n = reps[0]["n"]
            opened = [r for r in reps if r["open"] > n // 2]
            curled_sizes = Counter(s for r in opened for s in r["curled_pieces"])
            dam_sizes = Counter(s for r in reps for s in r["damaged_pieces"])
            largest_open = [r["open_regions"][0] if r["open_regions"] else 0 for r in reps]
            n_open_regions = [len(r["open_regions"]) for r in reps]
            print("  C=%d E=%d: open median %d/%d, open regions median %d (largest median %d); damaged pieces by size %s; "
                  "replicas with open majority %d, their curled pieces by size %s"
                  % (c, e, int(np.median([r["open"] for r in reps])), n, int(np.median(n_open_regions)),
                     int(np.median(largest_open)), dict(sorted(dam_sizes.items())), len(opened),
                     dict(sorted(curled_sizes.items()))))


if __name__ == "__main__":
    main()
