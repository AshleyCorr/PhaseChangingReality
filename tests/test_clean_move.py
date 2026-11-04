"""The clean two-switch move (graphity.clean_move): rule-6 checks.

dH exact against the full Hamiltonian for every accepted move; the clean guarantee (no point d > 3 from a clean start);
constraint preservation by brute force (networkx: simple, 6-regular, bipartite; and the hard-core rule); conservation
of H + stores - fed to 1e-9; same-seed reproducibility; and the control (allow_break) does break.
"""
import networkx as nx
import numpy as np

from graphity import clean_move as M
from graphity.cqg_d import hamiltonian, is_valid, torus
from graphity.dimension import local_dimension_d


def brute_ok(adj, part):
    g = nx.Graph()
    for u in range(adj.shape[0]):
        for v in adj[u]:
            g.add_edge(u, int(v))
    assert g.number_of_nodes() == adj.shape[0]
    assert all(d == 6 for _, d in g.degree())
    assert nx.is_bipartite(g)
    colour = nx.bipartite.color(g)
    n = adj.shape[0]
    assert all(colour[u] == part[u] for u in range(n)) or all(colour[u] != part[u] for u in range(n))
    assert is_valid(adj)


def _accepted_moves_are_exact(allow_break, want, cap):
    adj, part = torus((4, 6, 6))
    lam = 1.25
    n = adj.shape[0]
    side0 = np.flatnonzero(part == 0).astype(np.int64)
    stores = np.full(n, 50.0)                      # enough that cost never refuses a move
    work = M.work_arrays(6)
    np.random.seed(5)
    M.seed_numba(5)
    attempts = 0
    for _ in range(want):
        before = hamiltonian(adj, lam)
        a, dh, made = M.attempt_until_accepted(adj, part, stores, side0, lam, allow_break, cap - attempts, *work)
        attempts += made
        assert a == 1, (attempts, cap)
        after = hamiltonian(adj, lam)
        assert abs((after - before) - dh) < 1e-9, (after - before, dh)
        if not allow_break:
            assert local_dimension_d(adj).max() <= 3
    brute_ok(adj, part)
    return adj, attempts


def test_dh_is_exact_for_every_accepted_move_of_the_control():
    _accepted_moves_are_exact(True, want=200, cap=400000)


def test_clean_moves_out_of_the_slab_are_exact_and_leave_no_point_broken():
    # clean moves out of the slab are rare (the first clean state is two switches away, T69), which is the point of
    # the move set; a refused attempt leaves the graph as it was, which the exactness check covers by construction
    adj, attempts = _accepted_moves_are_exact(False, want=5, cap=20_000_000)
    assert local_dimension_d(adj).max() <= 3
    assert attempts > 1000                           # if this ever fails, the slab has stopped being a dip for the move


def test_the_driven_protocol_conserves_energy_repeats_with_the_seed_and_opens_cleanly():
    adj_a, part_a, stores_a, rows_a = M.run_clean_feed((4, 6, 6), 1.25, rate=8.0, seed=2, n_sweeps=300, every=50)
    adj_b, part_b, stores_b, rows_b = M.run_clean_feed((4, 6, 6), 1.25, rate=8.0, seed=2, n_sweeps=300, every=50)
    assert rows_a == rows_b
    assert np.array_equal(np.sort(adj_a, axis=1), np.sort(adj_b, axis=1)) and np.array_equal(stores_a, stores_b)
    for row in rows_a:
        assert abs(row["residual"]) < 1e-9 * max(1.0, row["sweep"] * 8.0)
        assert row["broken"] == 0 and row["low"] == 0          # the clean guarantee, every block
    assert rows_a[-1]["open"] > 0                                # the slab did start to open at this feed
    brute_ok(adj_a, part_a)


def test_the_control_breaks_points():
    _, _, _, rows = M.run_clean_feed((4, 6, 6), 1.25, rate=8.0, seed=2, n_sweeps=300, every=50, allow_break=True)
    assert any(row["broken"] > 0 for row in rows)
    for row in rows:
        assert abs(row["residual"]) < 1e-9 * max(1.0, row["sweep"] * 8.0)


def test_nothing_moves_with_nothing_in_the_stores():
    adj0, _ = torus((4, 6, 6))
    adj, part, stores, rows = M.run_clean_feed((4, 6, 6), 1.25, rate=0.0, seed=1, n_sweeps=20, every=10)
    assert np.array_equal(np.sort(adj, axis=1), np.sort(adj0, axis=1))
    assert all(row["acc"] == 0 for row in rows)
