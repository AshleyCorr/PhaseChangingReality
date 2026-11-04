"""What the "damaged" points of the opening runs are, read from the saved final wiring. Usage:

    python scripts/read_opening_damage.py [glob of *_adj directories ...]

WHY. O79 reported that where the openings of T44 to T46 (and T39 untied) released energy, a quarter or more of the points
ended "damaged": more open directions than flat space allows (d > D). That was a count. The owner's question of 27 Sep
(what does "melted" mean?) showed that T42's "melted" was a flickering scar, not a melt (O82, correction), so this reads
the openings' damage from the positions before anything is built on it (the project's rule: read positions before naming
a geometry). Nothing here changes a verdict.

WHAT IS READ, per saved final graph (exact; no dynamics): the census of curled (d < D), open (d = D) and damaged (d > D)
points; for the damaged points, the share that touch a curled point, an open point, both, or only other damaged points
(a damaged point touching both sides sits where an opened region meets a curled one: a seam; one touching only damaged
points is inside a damaged blob); the pieces of the damaged set and of the open set (several open pieces separated by
damage is a mosaic of patches; one open piece with damage beside it is not). For a run started from a gas of separate
pieces (T44 to T46), also: the largest fully open region, whether any open region contains points of two starting pieces,
and how many starting pieces opened completely. Reads results/ only; writes nothing.
"""
import glob
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphity.dimension import local_dimension_d, piece_labels  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ["results/t44_*_adj", "results/t45_*_adj", "results/t46_*_adj", "results/t39_*_adj"]


def sizes(labels):
    return sorted(np.bincount(labels[labels >= 0]).tolist(), reverse=True) if labels.max() >= 0 else []


def read(adj):
    """The census and the neighbourhood classes of the damaged points of one graph."""
    n, deg = adj.shape
    dim = deg // 2
    d = np.asarray(local_dimension_d(adj))
    curled, open_, bad = d < dim, d == dim, d > dim
    touch_c = np.array([curled[adj[v]].any() for v in range(n)])
    touch_o = np.array([open_[adj[v]].any() for v in range(n)])
    nb = int(bad.sum())
    seam = int((bad & touch_c & touch_o).sum())
    only_c = int((bad & touch_c & ~touch_o).sum())
    only_o = int((bad & touch_o & ~touch_c).sum())
    inner = int((bad & ~touch_c & ~touch_o).sum())
    return dict(n=n, curled=int(curled.sum()), open=int(open_.sum()), damaged=nb, seam=seam, only_curled=only_c,
                only_open=only_o, inner=inner, damaged_pieces=sizes(piece_labels(adj, bad)),
                open_pieces=sizes(piece_labels(adj, open_)), curled_pieces=sizes(piece_labels(adj, curled)))


def by_piece(adj, piece):
    """For a run started from a gas of separate pieces of `piece` points each (numbered one after another, as
    run_sealed_curled_d.gas builds them): the largest fully open region, how many open regions contain points of two
    or more starting pieces, how many starting pieces are fully open, and the connected components of the whole graph."""
    n, deg = adj.shape
    d = np.asarray(local_dimension_d(adj))
    origin = np.arange(n) // piece
    lab = piece_labels(adj, d == deg // 2)
    largest, spanning = 0, 0
    if lab.max() >= 0:
        largest = int(np.bincount(lab[lab >= 0]).max())
        spanning = sum(1 for k in range(lab.max() + 1) if len(set(origin[lab == k])) > 1)
    full = sum(1 for c in range(n // piece) if (d[origin == c] == deg // 2).all())
    return dict(largest_open=largest, spanning=spanning, fully_open_pieces=full,
                components=int(piece_labels(adj, np.ones(n, bool)).max()) + 1)


def gas_piece(folder):
    """Points per starting piece if the run's config started from a gas, else None."""
    cfg = ROOT / "configs" / (os.path.basename(folder)[:-4] + ".json")
    if not cfg.exists():
        return None
    import json
    g = json.loads(cfg.read_text()).get("gas")
    return int(np.prod([int(x) for x in g["dims"]])) if g else None


def main(patterns):
    for pat in patterns:
        for folder in sorted(glob.glob(str(ROOT / pat))):
            piece = gas_piece(folder)
            if piece:
                per = [by_piece(np.load(f)["adj"], piece) for f in sorted(glob.glob(os.path.join(folder, "*.npz")))]
                print("%-34s gas of %d-point pieces | largest open region per graph %s | open regions spanning two "
                      "pieces %d | fully open pieces %s | components %s"
                      % (os.path.basename(folder)[:-4], piece, [p["largest_open"] for p in per],
                         sum(p["spanning"] for p in per), [p["fully_open_pieces"] for p in per],
                         [p["components"] for p in per]), flush=True)
            rows = [read(np.load(f)["adj"]) for f in sorted(glob.glob(os.path.join(folder, "*.npz")))]
            rows = [r for r in rows if r["damaged"] > 0]
            if not rows:
                print("%-34s no damaged final graph" % os.path.basename(folder)[:-4])
                continue
            tot = {k: sum(r[k] for r in rows) for k in ("n", "curled", "open", "damaged", "seam", "only_curled",
                                                         "only_open", "inner")}
            nd = tot["damaged"]
            print("%-34s %2d graphs | curled %4.1f%% open %4.1f%% damaged %4.1f%% | damaged touching: both %4.1f%%, "
                  "curled only %4.1f%%, open only %4.1f%%, neither %4.1f%% | damaged pieces (largest, count) %s | "
                  "open pieces (largest two, count) %s"
                  % (os.path.basename(folder)[:-4], len(rows), 100 * tot["curled"] / tot["n"], 100 * tot["open"] / tot["n"],
                     100 * nd / tot["n"], 100 * tot["seam"] / nd, 100 * tot["only_curled"] / nd,
                     100 * tot["only_open"] / nd, 100 * tot["inner"] / nd,
                     [(r["damaged_pieces"][0], len(r["damaged_pieces"])) for r in rows[:4]],
                     [(r["open_pieces"][:2], len(r["open_pieces"])) for r in rows[:3]]), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT)
