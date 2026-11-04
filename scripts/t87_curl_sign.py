"""T87: does a curled loop of four in the toy have a sign of its own, read from its surroundings? Exact, no chain. Usage:

    python scripts/t87_curl_sign.py <N> <npz> [<npz> ...]        (writes results/t87_curl_sign_N<N>.csv and .meta.json)

Implements PREREGISTRATION.md section T87 (written 2026-10-30 before any registered computation; the owner's correction
of the same day: "Why can't a size be negative? This is the 2d equation set not a plane"). A loop of four has two ways
round. Its curl HAS A SIGN at radius r if no renaming of the points within graph distance r of the loop, preserving every
link among them, every point's distance from the loop and each point's side (the bipartition), maps the loop onto itself
the other way round. If such a renaming exists, the two ways round are the same arrangement seen in a mirror, and the
curl has no sign at that radius.

For every column (a piece of four points off the sheet, all at local dimension 1, the curled loop of four; T55's reading)
in every saved graph, and for r = 2, 3, 4 and 6: the ball of radius r round the loop; a reflection is either of the two
side-preserving reversals of the 4-cycle (a, b, c, d) -> (a, d, c, b) or (c, b, a, d); the question is whether either
extends to an isomorphism of the ball onto itself that fixes every point's distance from the loop (networkx VF2 with
node attributes (distance, side, anchor)). Reported per column and radius: SIGNED (no such renaming) or MIRROR (one exists).

Words, per size, over all columns at that size: SIGNED if the majority are SIGNED at r = 3; MIRROR if the majority are
MIRROR at r = 3; else MIXED. The other radii are reported beside. Verdict read at the held-out size only (rule 14).
Pure functions, tested in tests/test_t86_t87.py.
"""
import platform
import sys
from pathlib import Path

import networkx as nx
import numba
import numpy as np
from networkx.algorithms import isomorphism

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graphity import __version__                                   # noqa: E402
from graphity.dimension import local_dimension, piece_labels       # noqa: E402
from graphity.results import ResultWriter, provenance              # noqa: E402

SHEET_D, COLUMN_D = 2, 1
RADII = (2, 3, 4, 6)
VERDICT_R = 3


def graph_of(adj):
    g = nx.Graph()
    g.add_nodes_from(range(adj.shape[0]))
    for v in range(adj.shape[0]):
        for w in adj[v]:
            if w >= 0 and v < w:
                g.add_edge(int(v), int(w))
    return g


def columns_of(adj):
    d = local_dimension(adj)
    labels = piece_labels(adj, d != SHEET_D)
    out = []
    if labels.max() < 0:
        return out
    for k in range(int(labels.max()) + 1):
        pts = [int(v) for v in np.flatnonzero(labels == k)]
        if len(pts) == 4 and all(d[v] == COLUMN_D for v in pts):
            out.append(pts)
    return out


def cycle_order(g, pts):
    """The four points of a 4-cycle in cyclic order, or None if they are not a 4-cycle."""
    sub = g.subgraph(pts)
    if sub.number_of_edges() != 4 or any(sub.degree(v) != 2 for v in pts):
        return None
    a = pts[0]
    b = next(iter(sub.neighbors(a)))
    c = next(x for x in sub.neighbors(b) if x != a)
    d = next(x for x in sub.neighbors(c) if x != b)
    return [a, b, c, d]


def has_mirror(g, side, loop, radius):
    """True if some renaming of the ball of `radius` round `loop`, preserving links, distance from the loop and side,
    sends the loop (a, b, c, d) to (a, d, c, b) or to (c, b, a, d)."""
    dist = nx.multi_source_dijkstra_path_length(g, set(loop), cutoff=radius)
    ball = g.subgraph(dist.keys()).copy()
    anchors = {loop[0]: 0, loop[1]: 1, loop[2]: 2, loop[3]: 3}
    for target in ([loop[0], loop[3], loop[2], loop[1]], [loop[2], loop[1], loop[0], loop[3]]):
        g1, g2 = ball.copy(), ball.copy()
        for v in ball.nodes:
            g1.nodes[v]["lab"] = (dist[v], int(side[v]), anchors.get(v, -1))
            g2.nodes[v]["lab"] = (dist[v], int(side[v]), -1)
        for i, v in enumerate(target):           # the image of loop point i must be target[i]
            g2.nodes[v]["lab"] = (dist[v], int(side[v]), i)
        gm = isomorphism.GraphMatcher(g1, g2, node_match=isomorphism.categorical_node_match("lab", None))
        if gm.is_isomorphic():
            return True
    return False


def read_graph(adj, side, source):
    g = graph_of(adj)
    rows = []
    for pts in columns_of(adj):
        loop = cycle_order(g, pts)
        if loop is None:
            continue
        row = dict(source=source, points=" ".join(str(v) for v in loop))
        for r in RADII:
            row["r%d" % r] = "MIRROR" if has_mirror(g, side, loop, r) else "SIGNED"
        rows.append(row)
    return rows


def word(rows, r=VERDICT_R):
    if not rows:
        return "NONE", 0
    key = "r%d" % r
    signed = sum(1 for row in rows if row[key] == "SIGNED")
    n = len(rows)
    if signed > n / 2:
        return "SIGNED", n
    if n - signed > n / 2:
        return "MIRROR", n
    return "MIXED", n


def main(n_size, paths, out_dir="results"):
    name = "t87_curl_sign_N%d" % n_size
    meta = dict(task="T87", N=n_size, files=[str(p) for p in paths], radii=list(RADII), verdict_radius=VERDICT_R,
                graphity=__version__, python=platform.python_version(), numpy=np.__version__, numba=numba.__version__,
                networkx=nx.__version__, **provenance())
    all_rows = []
    with ResultWriter(name, meta, out_dir) as out:
        for p in paths:
            d = np.load(p)
            adj, side = d["adj"], d["part"]
            assert adj.shape[0] == n_size, (p, adj.shape)
            rows = read_graph(adj, side, str(p))
            for row in rows:
                out.write(row)
            all_rows.extend(rows)
    for r in RADII:
        wd, n = word(all_rows, r)
        print("N = %d  r = %d: %-7s over %d columns (%d SIGNED)" % (n_size, r, wd, n, sum(1 for row in all_rows if row["r%d" % r] == "SIGNED")))
    return all_rows


if __name__ == "__main__":
    main(int(sys.argv[1]), [Path(p) for p in sys.argv[2:]])
