"""src/graphity/sixloop.py: a price on loops of six (VISION Update 56; ASSUMPTIONS O124). The checks CLAUDE.md
rule 6 asks of a new kernel: an independent brute-force count (networkx), exactness of the incremental update
against full recomputation, constraint preservation and same-seed reproducibility; and that at zero price it is
cqg.run_chain draw for draw."""
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from graphity.cqg import (NO_CAP, _new_edge_ok, _switch, is_valid, run_chain, surplus, torus,          # noqa: E402
                          total_squares)
from graphity.sixloop import FLAT, excess6, hamiltonian6, loop_counts, loops6, run_chain6              # noqa: E402
from graphity.squares import has_edge                                                               # noqa: E402
from read_endpoint_width import twisted_tube                                                        # noqa: E402


def graph(adj):
    return nx.Graph((i, int(j)) for i in range(adj.shape[0]) for j in adj[i])


def side_of_tube(length):
    return np.array([x * length + y for x in range(4) for y in range(length) if (x + y) % 2 == 0], dtype=np.int64)


def side_of(adj):
    colour = nx.bipartite.color(graph(adj))
    if colour[0] == 1:
        colour = {v: 1 - c for v, c in colour.items()}
    return np.array(sorted(v for v, c in colour.items() if c == 0), dtype=np.int64)


def melted(n_side=8, g=6.0, sweeps=60, seed=5):
    adj, part = torus(n_side, n_side, NO_CAP)
    side = np.flatnonzero(part == 0).astype(np.int64)
    run_chain(adj, side, 1.0 / g, sweeps, 1, seed, 1.25, NO_CAP)
    return adj, side


def brute_loops6(g, u, v):
    """Loops of six through link (u, v): simple paths of exactly five links from v back to u."""
    return sum(1 for path in nx.all_simple_paths(g, v, u, cutoff=5) if len(path) == 6)


def test_loops_of_six_against_networkx():
    cases = [torus(8, 8, NO_CAP)[0], torus(6, 10, NO_CAP)[0], twisted_tube(12, 2), twisted_tube(12, 0), melted()[0]]
    for adj in cases:
        g = graph(adj)
        for u in range(0, adj.shape[0], 3):
            for v in adj[u]:
                assert loops6(adj, u, int(v)) == brute_loops6(g, u, int(v))
                assert loops6(adj, int(v), u) == loops6(adj, u, int(v))            # either end gives the same


def test_a_wide_flat_sheet_pays_nothing_and_narrow_ones_pay_a_fixed_amount_per_point():
    for dims in ((8, 8), (8, 12), (10, 12)):
        adj, part = torus(*dims, NO_CAP)
        side = np.flatnonzero(part == 0).astype(np.int64)
        assert set(loop_counts(adj, side)[side].ravel()) == {FLAT} and excess6(adj, side) == 0
    for length in (12, 24):
        side, n = side_of_tube(length), 4 * length
        assert excess6(twisted_tube(length, 0), side) == 18 * n                    # the tube, four around
        assert excess6(twisted_tube(length, 2), side) == 15 * n                    # re-glued, six around
        assert excess6(twisted_tube(length, 4), side) == 0                         # eight around: nothing
        assert hamiltonian6(twisted_tube(length, 2), side, 1.25, 0.1) == pytest.approx(1.5 * n)
        assert hamiltonian6(twisted_tube(length, 0), side, 1.25, 0.1) == pytest.approx(n + 1.8 * n)
    adj, part = torus(6, 16, NO_CAP)                                               # a plain torus six around pays too
    assert excess6(adj, np.flatnonzero(part == 0).astype(np.int64)) > 0


@njit
def reference_chain(adj, side_u, inv_g, n_sweeps, seed, lam, mu):
    """The same proposals and the same Metropolis rule with nothing incremental: the whole energy is recomputed
    for every candidate. Slow, and obviously right."""
    np.random.seed(seed)
    n = adj.shape[0]
    nu = side_u.shape[0]
    h = hamiltonian6(adj, side_u, lam, mu)
    out = np.zeros((n_sweeps, 3), dtype=np.int64)
    for sweep in range(n_sweeps):
        for _ in range(2 * n):
            u1 = side_u[np.random.randint(0, nu)]
            v1 = adj[u1, np.random.randint(0, 4)]
            u2 = side_u[np.random.randint(0, nu)]
            v2 = adj[u2, np.random.randint(0, 4)]
            if u1 == u2 or v1 == v2 or has_edge(adj, u1, v2) or has_edge(adj, u2, v1):
                continue
            _switch(adj, u1, v1, u2, v2)
            if not (_new_edge_ok(adj, u1, v2, NO_CAP) and _new_edge_ok(adj, u2, v1, NO_CAP)):
                _switch(adj, u1, v2, u2, v1)
                continue
            h_new = hamiltonian6(adj, side_u, lam, mu)
            d_h = h_new - h
            if d_h > 0.0 and np.random.random() >= np.exp(-inv_g * d_h):
                _switch(adj, u1, v2, u2, v1)
                continue
            h = h_new
        out[sweep, 0] = total_squares(adj)
        out[sweep, 1] = surplus(adj)
        out[sweep, 2] = excess6(adj, side_u)
    return out


@pytest.mark.parametrize("mu, g, start", [(0.25, 4.0, "six"), (0.5, 1.0, "six"), (0.125, 4.0, "six"), (0.25, 2.0, "tube"),
                                          (0.5, 2.5, "melt")])
def test_the_incremental_chain_is_the_full_recomputation_chain(mu, g, start):
    """lam = 1.25 and these prices make every energy a multiple of 1/8, so both chains do exact arithmetic and
    must agree move for move, not only on average."""
    if start == "melt":
        adj, side = melted()
    else:
        adj, side = twisted_tube(12, 2 if start == "six" else 0), side_of_tube(12)
    one, two = adj.copy(), adj.copy()
    sweeps = 120
    s, x, e, acc = run_chain6(one, side, 1.0 / g, 0, sweeps, 20261012, 1.25, mu)
    ref = reference_chain(two, side, 1.0 / g, sweeps, 20261012, 1.25, mu)
    assert np.array_equal(one, two)
    assert np.array_equal(s, ref[:, 0]) and np.array_equal(x, ref[:, 1]) and np.array_equal(e, ref[:, 2])
    assert (s[-1], x[-1], e[-1]) == (total_squares(one), surplus(one), excess6(one, side))
    assert acc > 0 and len(set(e)) > 1                                            # moves were made, and E6 moved
    assert is_valid(one, NO_CAP)


def test_long_walks_keep_the_books_exact_and_the_graph_valid():
    for mu, g in ((0.3, 1.2), (0.1, 0.8), (1.0, 4.0)):
        adj, side = twisted_tube(24, 2), side_of_tube(24)
        for block in range(4):                                                    # carried on without re-seeding
            s, x, e, _acc = run_chain6(adj, side, 1.0 / g, 0, 150, 99 if block == 0 else -1, 1.25, mu)
            assert (s[-1], x[-1], e[-1]) == (total_squares(adj), surplus(adj), excess6(adj, side))
            g_nx = graph(adj)
            assert is_valid(adj, NO_CAP) and nx.is_bipartite(g_nx) and set(dict(g_nx.degree()).values()) == {4}
            assert g_nx.number_of_edges() == 2 * adj.shape[0]                     # no doubled link


def test_at_zero_price_it_is_the_toys_own_chain_draw_for_draw():
    for lam, g, shift in ((1.25, 1.5, 0), (1.25, 3.0, 2), (1.0, 2.0, 0)):
        one, two, side = twisted_tube(16, shift), twisted_tube(16, shift), side_of_tube(16)
        s1, x1, acc1 = run_chain(one, side, 1.0 / g, 0, 400, 7, lam, NO_CAP)
        s2, x2, e2, acc2 = run_chain6(two, side, 1.0 / g, 0, 400, 7, lam, 0.0)
        assert np.array_equal(one, two) and np.array_equal(s1, s2) and np.array_equal(x1, x2) and acc1 == acc2
        assert e2[-1] == excess6(two, side)                                       # E6 is still read, though it costs nothing


def test_same_seed_same_chain_and_other_seed_another():
    def go(seed):
        adj, side = twisted_tube(12, 2), side_of_tube(12)
        out = run_chain6(adj, side, 1.0 / 4.0, 10, 200, seed, 1.25, 0.3)       # warm enough that moves are made
        return adj, out
    a1, o1 = go(11)
    a2, o2 = go(11)
    a3, _o3 = go(12)
    assert np.array_equal(a1, a2) and all(np.array_equal(p, q) for p, q in zip(o1[:3], o2[:3])) and o1[3] == o2[3]
    assert not np.array_equal(a1, a3)


def test_side_helper_matches_the_tubes_own_sides():
    adj = twisted_tube(12, 2)
    assert np.array_equal(side_of(adj), side_of_tube(12))
