"""The sealed bath with the follow tie of VISION Update 30 (graphity.sealed_tie_d; PREREGISTRATION T44).

Rule 6: at kappa = 0 it is sealed_d.run_sealed_bath_d draw for draw; the incremental tie equals full recomputation
after every block of a hot run; the tie total agrees with a networkx brute force; H + kappa T + stores is conserved;
the constraints hold; the same seed gives the same run.
"""
import itertools

import networkx as nx
import numpy as np

from graphity import cqg_d
from graphity.dimension import local_dimension_d
from graphity.sealed_d import energy_d, run_sealed_bath_d
from graphity.sealed_tie_d import (energy_table_d, energy_tie_d, follow, run_sealed_bath_table_d, run_sealed_bath_tie_d,
                                  table_total, tie_total)


def _gas(dims, copies):
    a, p = cqg_d.torus(dims)
    n = a.shape[0]
    return np.concatenate([a + k * n for k in range(copies)]), np.concatenate([p] * copies)


def test_kappa_zero_is_run_sealed_bath_d_draw_for_draw():
    lam = 1.10
    for adj0, part in (cqg_d.torus([4, 4, 6]), _gas([4, 4, 4], 3)):
        a1, a2 = adj0.copy(), adj0.copy()
        side_u = np.flatnonzero(part == 0)
        d1 = np.zeros(2 * adj0.shape[0]); d1[0] = 60.0
        d2 = d1.copy()
        s1, x1, m1, t1, acc1 = run_sealed_bath_d(a1, side_u, d1, 150, 42, lam)
        s2, x2, tt, m2, t2, acc2 = run_sealed_bath_tie_d(a2, side_u, d2, 150, 42, lam, 0.0)
        assert (s1 == s2).all() and (x1 == x2).all() and (a1 == a2).all()
        assert np.array_equal(d1, d2) and acc1 == acc2
        assert tt[-1] == tie_total(a2)


def test_the_tie_reads_the_ladder():
    for dims, per_point in (([6, 6, 6], 0), ([4, 6, 6], 1), ([4, 4, 6], 2)):
        adj, _ = cqg_d.torus(dims)
        assert tie_total(adj) == per_point * adj.shape[0], dims
    adj, _ = _gas([4, 4, 4], 2)
    assert tie_total(adj) == 0
    assert [follow(d, 3) for d in range(8)] == [0, 2, 1, 0, 0, 0, 0, 0]


def _brute_tie(adj):
    g = nx.Graph()
    for v in range(adj.shape[0]):
        for w in adj[v]:
            g.add_edge(v, int(w))
    dim = adj.shape[1] // 2
    t = 0
    for v in g.nodes:
        open_pairs = 0
        for a, b in itertools.combinations(sorted(g[v]), 2):
            common = (set(g[a]) & set(g[b])) - {v}
            if not common:
                open_pairs += 1
        t += (dim - open_pairs) if 1 <= open_pairs <= dim else 0
    return t


def test_incremental_tie_is_exact_and_agrees_with_networkx_on_a_damaged_run():
    lam, kappa = 1.25, 1.5
    adj, part = cqg_d.torus([4, 4, 6])
    side_u = np.flatnonzero(part == 0)
    stores = np.zeros(2 * adj.shape[0]); stores[:] = 3.0           # hot: the run damages the torus
    e0 = energy_tie_d(adj, lam, kappa) + stores.sum()
    changed = False
    for block in range(6):
        s, x, t, _, tot, _ = run_sealed_bath_tie_d(adj, side_u, stores, 20, 9 if block == 0 else -1, lam, kappa)
        assert t[-1] == tie_total(adj) == _brute_tie(adj)
        assert s[-1] == cqg_d.total_squares(adj) and x[-1] == cqg_d.surplus(adj)
        assert abs(energy_tie_d(adj, lam, kappa) + stores.sum() - e0) < 1e-8
        h = 16.0 * (3 * adj.shape[0] - s) + 4.0 * lam * x + kappa * t
        assert np.allclose(h + tot, e0)
        changed = changed or len(set(local_dimension_d(adj).tolist())) > 1
    assert changed                                                    # the check was not vacuous
    assert cqg_d.is_valid(adj) and (stores >= -1e-12).all()
    assert all(len(set(adj[v].tolist())) == 6 for v in range(adj.shape[0]))
    sides = np.zeros(adj.shape[0], dtype=bool); sides[side_u] = True
    assert all(sides[v] != sides[w] for v in range(adj.shape[0]) for w in adj[v])


def test_gas_with_the_tie_conserves_per_vertex_and_same_seed_repeats():
    lam, kappa = 1.25, 2.0
    runs = []
    for _ in range(2):
        adj, part = _gas([4, 4, 4], 2)
        side_u = np.flatnonzero(part == 0)
        stores = np.zeros(adj.shape[0]); stores[int(side_u[0])] = 200.0
        e0 = energy_tie_d(adj, lam, kappa) + stores.sum()
        s, x, t, _, tot, _ = run_sealed_bath_tie_d(adj, side_u, stores, 60, 5, lam, kappa, by_vertex=True)
        assert abs(energy_tie_d(adj, lam, kappa) + stores.sum() - e0) < 1e-8
        assert t[-1] == tie_total(adj)
        runs.append((s, x, t, adj.copy(), stores.copy()))
    assert all(np.array_equal(a, b) for a, b in zip(runs[0], runs[1]))


def test_energy_tie_d_is_energy_d_plus_kappa_t():
    adj, _ = cqg_d.torus([4, 4, 6])
    assert abs(energy_tie_d(adj, 1.25, 2.0) - (energy_d(adj, 1.25) + 2.0 * 2 * adj.shape[0])) < 1e-9


def test_table_kernel_is_the_follow_kernel_draw_for_draw_and_exact_for_any_shape():
    lam, kappa = 1.25, 2.0
    adj0, part = cqg_d.torus([4, 4, 6])
    side_u = np.flatnonzero(part == 0)
    a1, a2 = adj0.copy(), adj0.copy()
    d1 = np.full(2 * adj0.shape[0], 2.0); d2 = d1.copy()
    s1, x1, t1, _, _, acc1 = run_sealed_bath_tie_d(a1, side_u, d1, 60, 13, lam, kappa)
    ftab = kappa * np.array([0.0, 2.0, 1.0, 0.0])
    s2, x2, t2, _, _, acc2 = run_sealed_bath_table_d(a2, side_u, d2, 60, 13, lam, ftab)
    assert (s1 == s2).all() and (x1 == x2).all() and (a1 == a2).all() and acc1 == acc2
    assert np.allclose(kappa * t1, t2)
    # any shape, including negative entries: incremental total exact, energy conserved
    ftab = np.array([0.0, 1.6, -0.84, 0.0])
    adj, part = cqg_d.torus([4, 4, 6])
    stores = np.full(2 * adj.shape[0], 3.0)
    e0 = energy_table_d(adj, lam, ftab) + stores.sum()
    for block in range(4):
        _, _, t, _, _, _ = run_sealed_bath_table_d(adj, side_u, stores, 20, 21 if block == 0 else -1, lam, ftab)
        assert abs(t[-1] - table_total(adj, ftab)) < 1e-9
        assert abs(energy_table_d(adj, lam, ftab) + stores.sum() - e0) < 1e-8
    assert cqg_d.is_valid(adj)


def test_table_kernel_at_eight_links_is_exact_and_conserves():
    """T46 runs the table tie with eight links (four directions); checked here on a hot, damaged 4 x 4 x 4 x 6 torus."""
    lam = 1.30
    a = 4.0 * (lam - 1.0)
    ftab = np.array([0.0, 1.0, -1.3711, -1.2, 0.0]) * a
    adj, part = cqg_d.torus([4, 4, 4, 6])
    side_u = np.flatnonzero(part == 0)
    stores = np.full(2 * adj.shape[0], 4.0)
    e0 = energy_table_d(adj, lam, ftab) + stores.sum()
    changed = False
    for block in range(3):
        _, _, t, _, _, _ = run_sealed_bath_table_d(adj, side_u, stores, 10, 31 if block == 0 else -1, lam, ftab)
        assert abs(t[-1] - table_total(adj, ftab)) < 1e-9
        assert abs(energy_table_d(adj, lam, ftab) + stores.sum() - e0) < 1e-8
        changed = changed or len(set(local_dimension_d(adj).tolist())) > 1
    assert changed and cqg_d.is_valid(adj) and (stores >= -1e-12).all()


def test_thermal_chain_with_a_zero_table_is_run_chain_draw_for_draw():
    """run_chain_table_d (9 October, for T56): with ftab all zero it must be cqg_d.run_chain, draw for draw."""
    from graphity.sealed_tie_d import run_chain_table_d
    lam, inv_g = 1.25, 1.0 / 2.5
    adj0, part = cqg_d.torus([4, 4, 6])
    side_u = np.flatnonzero(part == 0)
    a1, a2 = adj0.copy(), adj0.copy()
    s1, x1, acc1 = cqg_d.run_chain(a1, side_u, inv_g, 0, 40, 17, lam)
    s2, x2, t2, acc2 = run_chain_table_d(a2, side_u, inv_g, 40, 17, lam, np.zeros(4))
    assert (s1 == s2).all() and (x1 == x2).all() and (a1 == a2).all() and acc1 == acc2 and (t2 == 0).all()


def test_thermal_chain_with_a_table_is_exact_valid_and_repeats():
    """The incremental tie agrees with full recomputation on a warm, damaged run of the owner's 'all at the last'
    shape; the graph stays valid; the same seed gives the same chain; a carried-on stream equals one call."""
    from graphity.sealed_tie_d import run_chain_table_d
    lam = 1.20
    a = 4.0 * (lam - 1.0)
    ftab = np.array([0.0, a, 2.0 * a, 0.0])
    adj, part = cqg_d.torus([4, 8, 8])
    side_u = np.flatnonzero(part == 0)
    one = adj.copy()
    s, x, t, _ = run_chain_table_d(one, side_u, 1.0 / 3.0, 80, 23, lam, ftab)
    assert abs(t[-1] - table_total(one, ftab)) < 1e-9
    assert s[-1] == cqg_d.total_squares(one) and x[-1] == cqg_d.surplus(one)
    assert abs(energy_table_d(one, lam, ftab) - (16.0 * (3 * one.shape[0] - s[-1]) + 4.0 * lam * x[-1] + t[-1])) < 1e-9
    assert cqg_d.is_valid(one)
    assert (local_dimension_d(one) != 3).any(), "the warm run should have moved the slab"
    again = adj.copy()
    s2, x2, t2, _ = run_chain_table_d(again, side_u, 1.0 / 3.0, 80, 23, lam, ftab)
    assert (again == one).all() and (s2 == s).all() and np.allclose(t2, t)
    blocks = adj.copy()
    out = []
    for k in range(4):
        sb, _, _, _ = run_chain_table_d(blocks, side_u, 1.0 / 3.0, 20, 23 if k == 0 else -1, lam, ftab)
        out += list(sb)
    assert (blocks == one).all() and out == list(s)


def test_thermal_chain_tie_matches_networkx_energy_on_a_damaged_graph():
    """Full recomputation by an independent route: the tie total from local_dimension_d, H from networkx cycles."""
    from graphity.sealed_tie_d import run_chain_table_d
    lam = 1.15
    a = 4.0 * (lam - 1.0)
    ftab = np.array([0.0, a, 2.0 * a, 0.0])
    adj, part = cqg_d.torus([4, 4, 6])
    side_u = np.flatnonzero(part == 0)
    s, x, t, _ = run_chain_table_d(adj, side_u, 1.0 / 3.5, 30, 5, lam, ftab)
    d = local_dimension_d(adj)
    assert abs(t[-1] - sum(ftab[k] for k in d if k < 4)) < 1e-9
    g = nx.Graph()
    for u in range(adj.shape[0]):
        for v in adj[u]:
            g.add_edge(u, int(v))
    squares = sum(1 for c in nx.simple_cycles(g, length_bound=4) if len(c) == 4)
    assert squares == s[-1]
