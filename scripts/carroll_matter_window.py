"""T94: a window with stored matter in it, read through Carroll's reader. Exact.

Usage:
    python scripts/carroll_matter_window.py <stage>        stage is "stage1" (R = 3, 4) or "heldout" (R = 6)

THE RULE READ (the owner's, VISION Update 155): matter, energy and twist are stored between points of the same kind, which
are never linked; photons and gravitational waves travel between points of different kinds. T91 (scripts/carroll_reader.py)
put a share of information on the links only, which under that rule is empty space. Here a share is also put on STORED
PAIRS: two points at graph distance two with at least two common neighbours, which on the flat sheet are exactly the two
diagonals of each square. A share is given as the pair's reader distance in link units, alpha = (-ln x_m)/(-ln x_0):
alpha = 1 shares as much as a link, alpha = 0.5 more, alpha = 2 the same as the path through links (so alpha = 2 must
reproduce EMPTY; tests/test_carroll_matter_window.py checks it).

THE WINDOW is the same point set in every case, the unweighted ball of graph radius R round the center, so only the
distances change; the distances are shortest weighted paths in the whole graph. Then classical multidimensional scaling
and the registered count D_MDS, as in T91 (the functions are imported from carroll_reader.py).

CASES: EMPTY; UNIFORM (every stored pair of the torus carries a share), alpha = 1 and 0.5; PATCH (the eight stored pairs
of the four squares meeting at the center), alpha = 1 and 0.5. Pre-registered as PREREGISTRATION.md section T94.
"""
import csv
import json
import platform
import sys
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graphity import __version__                     # noqa: E402
from graphity.cqg import NO_CAP, torus                 # noqa: E402
import carroll_reader as cr                           # noqa: E402

STAGES = {"stage1": (3, 4), "heldout": (6,), "matter": (3, 4, 6)}
ALPHAS = (1.0, 0.5)


def stored_pairs(g):
    """Pairs at graph distance two with at least two common neighbours (the diagonals of squares), as sorted tuples."""
    out = set()
    for u in g:
        for a, b in combinations(sorted(g[u]), 2):
            if not g.has_edge(a, b) and len(set(g[a]) & set(g[b])) >= 2:
                out.add((a, b))
    return sorted(out)


def pairs_at_center(pairs, g, center):
    """The stored pairs of the squares that meet at `center`: both points are neighbours of the center, or one is the
    center itself (the pair across a square through the center)."""
    nb = set(g[center])
    return [(a, b) for a, b in pairs if (a in nb and b in nb) or center in (a, b)]


def weighted_graph(g, pairs, alpha):
    """Links of weight 1 plus the given stored pairs as extra edges of weight alpha."""
    w = nx.Graph()
    w.add_edges_from(g.edges, weight=1.0)
    for a, b in pairs:
        w.add_edge(a, b, weight=float(alpha))
    return w


def window_weighted(g, w, center, radius):
    """The unweighted ball (same points as T91's window) with shortest weighted distances between its points."""
    pts = sorted(nx.single_source_shortest_path_length(g, center, cutoff=radius))
    dist = []
    for p in pts:
        lengths = nx.single_source_dijkstra_path_length(w, p, weight="weight")
        dist.append([lengths[q] for q in pts])
    return pts, dist


def cases(g, center, part=None, matter_only=False):
    pairs = stored_pairs(g)
    patch = pairs_at_center(pairs, g, center)
    if matter_only:
        # T98: the owner's matter sits on the black diagonals alone (Update 156); part == 0 is black
        black = [(a, b) for a, b in pairs if part[a] == 0 and part[b] == 0]
        black_patch = [(a, b) for a, b in patch if part[a] == 0 and part[b] == 0]
        yield "EMPTY", 0.0, []
        yield "UNIFORM", 1.0, pairs
        yield "UNIFORM_MATTER", 1.0, black
        yield "PATCH", 1.0, patch
        yield "PATCH_MATTER", 1.0, black_patch
        return
    yield "EMPTY", 0.0, []
    for alpha in ALPHAS:
        yield "UNIFORM", alpha, pairs
    for alpha in ALPHAS:
        yield "PATCH", alpha, patch


def read(adj, center, radii, part=None, matter_only=False):
    g = cr.graph_of(adj)
    rows = []
    for name, alpha, pairs in cases(g, center, part, matter_only):
        w = weighted_graph(g, pairs, alpha) if pairs else weighted_graph(g, [], 1.0)
        for r in radii:
            pts, dist = window_weighted(g, w, center, r)
            ev = cr.mds_spectrum(dist)
            rows.append(dict(case=name, alpha=alpha, stored_pairs=len(pairs), R=r, points=len(pts), d_mds=cr.d_mds(ev),
                             l1=float(ev[0]), r2=float(ev[1] / ev[0]), r3=float(ev[2] / ev[0]), r4=float(ev[3] / ev[0]),
                             r5=float(ev[4] / ev[0])))
    return rows


def main(stage, out_dir="results"):
    radii = STAGES[stage]
    name = ("t98_matter_only" if stage == "matter" else "t94_matter_window_" + stage)
    path = Path(out_dir) / (name + ".csv")
    if path.exists():
        raise SystemExit("refusing to overwrite " + str(path))
    adj, part = torus(32, 32, NO_CAP)
    rows = read(adj, 0, radii, part, matter_only=(stage == "matter"))
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    meta = dict(script="scripts/carroll_matter_window.py", stage=stage, radii=radii, alphas=ALPHAS, state="flat 32x32 torus",
                threshold=cr.THRESHOLD, package=__version__, python=platform.python_version(), numpy=np.__version__,
                networkx=nx.__version__, preregistration="PREREGISTRATION.md section " + ("T98" if stage == "matter" else "T94"))
    (Path(out_dir) / (name + ".meta.json")).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main(sys.argv[1])
