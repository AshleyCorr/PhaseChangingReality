"""The general-D move pricer (scripts/exact_exits_d.py) against numbers already on the record, and against networkx."""
import importlib.util
from pathlib import Path

import networkx as nx
import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "exact_exits_d", Path(__file__).resolve().parents[1] / "scripts" / "exact_exits_d.py")
e = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(e)


def brute_squares(adj):
    """Squares (4-cycles) by networkx: each edge's count of 4-cycles through it, and the total."""
    g = nx.Graph()
    for u in range(adj.shape[0]):
        for v in adj[u]:
            g.add_edge(u, int(v))
    per_edge = {}
    for u, v in g.edges():
        c = 0
        for a in g[u]:
            if a == v:
                continue
            for b in g[v]:
                if b != u and g.has_edge(a, b):
                    c += 1
        per_edge[(u, v)] = c
    return per_edge, sum(per_edge.values()) // 4


def test_square_counts_agree_with_networkx_in_2d_and_3d():
    for dims, dim in [((6, 6), 2), ((16, 4), 2), ((4, 4), 2), ((6, 6, 6), 3), ((4, 6, 6), 3), ((4, 4, 4), 3)]:
        adj, _ = e.torus(dims)
        per_edge, total = brute_squares(adj)
        s, x = e.s_and_x(adj, 2 * dim - 2)
        assert s == total
        assert x == sum(max(c - (2 * dim - 2), 0) for c in per_edge.values())


def test_the_ladder_energies():
    # H/N = 4(lam - 1) per curled direction, in D = 2 and D = 3; the 6-cube has 5 squares on every edge.
    for dims, dim, curled in [((6, 6), 2, 0), ((16, 4), 2, 1), ((4, 4), 2, 2),
                              ((6, 6, 6), 3, 0), ((4, 6, 6), 3, 1), ((4, 4, 6), 3, 2), ((4, 4, 4), 3, 3)]:
        adj, _ = e.torus(dims)
        c0, c1 = e.energy_per_vertex(adj, dim)
        assert abs(c0 + 4 * curled) < 1e-12 and abs(c1 - 4 * curled) < 1e-12
    s, _ = e.s_and_x(e.torus((4, 4, 4))[0], 4)
    assert s == 240


def test_paper_1_moves_a_and_b_and_the_offer_rate():
    adj, side = e.torus((16, 4))
    counts, rate = e.census(adj, side, 2)
    n = adj.shape[0]
    assert counts[(-2, -4)] == 3 * n and counts[(-4, -10)] == 2 * n
    assert abs(counts[(-2, -4)] * rate - 3.0) < 1e-12 and abs(counts[(-4, -10)] * rate - 2.0) < 1e-12
    assert e.stuck_window(counts) == (1.0, 1.6)


def test_o40_the_2d_knot_gas_is_never_stuck_above_one():
    adj, side = e.disjoint([e.torus((4, 4)), e.torus((4, 4))])
    counts, _ = e.census(adj, side, 2)
    assert counts[(-2, -8)] == 96 and counts[(-6, -20)] == 1024
    assert e.stuck_window(counts) is None


def test_every_counted_move_is_valid_and_bipartite():
    adj, side = e.torus((4, 4, 6))
    assert e._locally_valid(adj, np.arange(adj.shape[0], dtype=np.int64))
    assert all(side[u] != side[v] for u in range(adj.shape[0]) for v in adj[u])
