"""T55: what are the pieces T51 leaves that no cooling removes? Read from the wiring, exact. Usage:

    python scripts/analyse_t55.py configs/t55_pieces.json [out_dir]

Implements PREREGISTRATION.md section T55 (written 2026-10-09, before any computation; the owner's prediction SEAM
recorded before this ran). For every saved final graph of T51 (`results/t51_*_adj/*.npz`) and every connected piece
of points not at d = 2 (`graphity.dimension.piece_labels` on `local_dimension != 2`, as T37 and O104 read them):

  1. the census: size, the d of each point, the squares on each link at its points, and the energy held at its points,
     e(v) = 16 - 2 sum_e S_e + 2 lambda sum_e (S_e - 2)_+ over the links e at v, which sums to H over the graph
     (ours: H = 16 (N - S) + 4 lambda X with S = sum_v sum_e S_e / 8 and X = sum_v sum_e (S_e - 2)_+ / 2);
  2. one move from flat?  Every single switch with at least one of its four points in the piece is tried, the partner
     point drawn from within graph distance NEAR of the piece (a switch whose two links share no square loses every
     square on both and cannot lower the energy or heal anything, so this is exact for the questions asked; the same
     argument as scripts/exact_walls_d.walls). The piece is ONE MOVE FROM FLAT if some valid switch lowers the energy
     and leaves every point of the piece at d = 2 with the number of points off d = 2 down by exactly the piece's size;
     LOWERABLE if some valid switch lowers the energy without that; otherwise A DIP. The cheapest switch is reported;
  3. far links: the way round of every link at a point of the piece (scripts/explore_far_links.way_round; 3 on a
     flat sheet); FAR if any is 7 or more;
  4. where it sits: the graph distance to the nearest other piece not at d = 2, and to the nearest column.

The verdict per size (2 and 8), over all instances at the verdict's length (L = 256), is the majority class:
SCAR (ONE MOVE FROM FLAT), SEAM (FAR), KNOT (A DIP and not FAR), else MIXED; SEAM is read before KNOT when both apply.
Other sizes and the other lengths are reported beside. One row per piece is written to <out_dir>/<name>.csv with the
config and versions in its .meta.json (rule 5). Nothing random. Pure functions, tested in tests/test_t55.py.
"""
import json
import platform
import sys
from collections import Counter, defaultdict
from glob import glob
from pathlib import Path

import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graphity import __version__                                             # noqa: E402
from graphity.cqg_d import _new_edge_ok, _switch, hamiltonian, squares_on_edge   # noqa: E402
from graphity.dimension import local_dimension, piece_labels                 # noqa: E402
from graphity.results import ResultWriter                                    # noqa: E402
from exact_hidden_relics import ball                                         # noqa: E402
from explore_far_links import way_round                                      # noqa: E402

NEAR, FAR, VERDICT_L = 4, 7, 256
EPS = 1e-9


def pieces_of(adj):
    """[(sorted points of each connected piece of points not at d = 2)], with d; columns are size 4, all at d = 1."""
    d = local_dimension(adj)
    labels = piece_labels(adj, d != 2)
    out = [tuple(int(v) for v in np.flatnonzero(labels == k)) for k in range(int(labels.max()) + 1)]
    return out, d


def is_column(points, d):
    return len(points) == 4 and all(d[v] == 1 for v in points)


def point_energy(adj, v, lam):
    """e(v): the per-point split of H (see the module docstring)."""
    s = [squares_on_edge(adj, v, int(w)) for w in adj[v] if w >= 0]
    return 16.0 - 2.0 * sum(s) + 2.0 * lam * sum(max(x - 2, 0) for x in s)


def census(adj, points, lam, d):
    return dict(size=len(points), ds=" ".join(str(int(d[v])) for v in points),
                squares=" ".join(str(squares_on_edge(adj, v, int(w))) for v in points for w in adj[v] if w >= 0),
                energy=float(sum(point_energy(adj, v, lam) for v in points)))


def switches_touching(adj, part, points, near=NEAR):
    """Every switch (u1, v1, u2, v2), u on side 0 and v on side 1, with some point of `points` among its four, the
    other link's points within graph distance `near` of the piece. Each unordered switch once."""
    close = ball(adj, points, near)
    seen = set()
    for p in points:
        for q in adj[p]:
            q = int(q)
            u1, v1 = (p, q) if part[p] == 0 else (q, p)
            nbr1 = {int(x) for x in adj[u1]}
            for u2 in close:
                if part[u2] != 0 or u2 == u1:
                    continue
                nbr2 = {int(x) for x in adj[u2]}
                if v1 in nbr2:
                    continue
                for v2 in adj[u2]:
                    v2 = int(v2)
                    if v2 == v1 or v2 in nbr1:
                        continue
                    key = (min((u1, v1), (u2, v2)), max((u1, v1), (u2, v2)))
                    if key in seen:
                        continue
                    seen.add(key)
                    yield u1, v1, u2, v2


def one_move(adj, part, points, lam, d, near=NEAR):
    """(class, cheapest dH, the switch that gives it): ONE MOVE FROM FLAT, LOWERABLE or A DIP."""
    h0 = hamiltonian(adj, lam)
    off0 = int((d != 2).sum())
    best, best_move, heals = None, None, False
    for u1, v1, u2, v2 in switches_touching(adj, part, points, near):
        t = adj.copy()
        _switch(t, u1, v1, u2, v2)
        if not (_new_edge_ok(t, u1, v2) and _new_edge_ok(t, u2, v1)):     # the kernel's own hard-core check
            continue
        dh = float(hamiltonian(t, lam) - h0)
        if best is None or dh < best - EPS:
            best, best_move = dh, (u1, v1, u2, v2)
        if dh < -EPS and not heals:
            dt = local_dimension(t)
            if all(dt[v] == 2 for v in points) and int((dt != 2).sum()) == off0 - len(points):
                heals = True
    if heals:
        return "ONE MOVE FROM FLAT", best, best_move
    if best is not None and best < -EPS:
        return "LOWERABLE", best, best_move
    return "A DIP", best, best_move


def far_links(adj, points):
    """(the largest way round of a link at a point of the piece, how many such links are FAR)."""
    ways = [way_round(adj, v, int(w)) for v in points for w in adj[v] if w >= 0]
    return max(ways), sum(1 for w in ways if w >= FAR)


def distance_to(adj, points, targets):
    """Graph distance from the piece to the nearest point in `targets` (outside the piece); -1 if none."""
    targets = set(targets) - set(points)
    if not targets:
        return -1
    seen, edge, steps = set(points), list(points), 0
    while edge:
        steps += 1
        nxt = []
        for v in edge:
            for w in adj[v]:
                w = int(w)
                if w in seen:
                    continue
                if w in targets:
                    return steps
                seen.add(w)
                nxt.append(w)
        edge = nxt
    return -1


def classify(row):
    """SCAR, SEAM or KNOT for one piece's row (SEAM before KNOT)."""
    if row["move"] == "ONE MOVE FROM FLAT":
        return "SCAR"
    if int(row["far"]) > 0:
        return "SEAM"
    if row["move"] == "A DIP":
        return "KNOT"
    return "LOWERABLE"


def verdict(classes):
    """The majority class over a list of SCAR / SEAM / KNOT / LOWERABLE, else MIXED; empty gives NONE."""
    if not classes:
        return "NONE"
    top, k = Counter(classes).most_common(1)[0]
    return top if k > len(classes) / 2 else "MIXED"


def read_graph(adj, part, lam):
    """Every piece of one graph as a row (without the file columns)."""
    pieces, d = pieces_of(adj)
    columns = [v for p in pieces if is_column(p, d) for v in p]
    others = [v for p in pieces for v in p]
    rows = []
    for points in pieces:
        c = census(adj, points, lam, d)
        move, dh, sw = one_move(adj, part, points, lam, d)
        far_max, far_n = far_links(adj, points)
        rows.append(dict(points=" ".join(str(v) for v in points), column=is_column(points, d), **c, move=move,
                         cheapest=("" if dh is None else round(dh, 6)),
                         switch=("" if sw is None else " ".join(str(v) for v in sw)),
                         way_round_max=far_max, far=far_n,
                         to_piece=distance_to(adj, points, others), to_column=distance_to(adj, points, columns)))
    return rows


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    files = sorted(glob(cfg["states"]))
    if not files:
        raise FileNotFoundError("no saved graph matches %s" % cfg["states"])
    meta = dict(config=cfg, config_path=str(path), package=__version__, python=platform.python_version(),
                numpy=np.__version__, numba=numba.__version__, graphs=len(files),
                preregistration="PREREGISTRATION.md section T55, written 2026-10-09")
    by_size = defaultdict(list)
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for f in files:
            z = np.load(f)
            adj, part, lam = z["adj"], z["part"], float(z["lam"])
            length = adj.shape[0] // 4
            for r in read_graph(adj, part, lam):
                r = dict(state=Path(f).as_posix(), L=length, t_cool=int(z["t_cool"]), replica=int(z["replica"]), **r)
                r["class"] = classify(r)
                out.write(r)
                by_size[(length, r["size"])].append(r["class"])
            print("%s: %d pieces" % (Path(f).as_posix(), len(by_size)), flush=True)
    print("\nT55, by the rules of PREREGISTRATION T55 (written 2026-10-09 before any computation).")
    for (length, size), classes in sorted(by_size.items()):
        tag = "VERDICT" if length == VERDICT_L and size in (2, 8) else "reported"
        print("  %-8s L=%-4d size %2d: %-4d pieces  %s  -> %s"
              % (tag, length, size, len(classes), dict(Counter(classes)), verdict(classes)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results"))
