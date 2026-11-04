"""Checks on the 2D combinatorial quantum gravity kernel."""
from collections import Counter

import networkx as nx
import numpy as np
import pytest

from graphity.analysis import block_bootstrap_mean_var
from graphity.cqg import (CAP, NO_CAP, hamiltonian, is_valid, run_chain, squares_on_edge,
                          surplus, torus, total_squares)


def to_nx(adj):
    g = nx.Graph()
    for u in range(adj.shape[0]):
        for v in adj[u]:
            g.add_edge(u, int(v))
    return g


def brute_squares(g):
    return sum(1 for c in nx.simple_cycles(g, length_bound=4) if len(c) == 4)


def brute_hamiltonian(g, lam):
    """H = 16 (N - S) + 4 lam X from networkx alone, sharing no code with the kernel."""
    on_edge, n_squares = Counter(), 0
    for c in nx.simple_cycles(g, length_bound=4):
        if len(c) == 4:
            n_squares += 1
            for i in range(4):
                on_edge[frozenset((c[i], c[(i + 1) % 4]))] += 1
    x = sum(max(0, k - 2) for k in on_edge.values())
    return 16 * (g.number_of_nodes() - n_squares) + 4 * lam * x


def test_torus_is_the_ground_state():
    adj, part = torus(6)
    assert is_valid(adj)
    assert total_squares(adj) == 36                       # S = N, so H = 16 (N - S) = 0
    assert all(squares_on_edge(adj, u, int(v)) == 2 for u in range(36) for v in adj[u])


def test_four_cube_is_excluded():
    with pytest.raises(ValueError):
        torus(4)                                          # 4x4 torus = Q4, three squares per edge (Q3)


def test_rectangular_torus():
    """TASKS T1: the 16 x 10 torus, N = 160 as in [T25] Fig. 3."""
    adj, part = torus(16, 10)
    g = to_nx(adj)
    assert adj.shape == (160, 4)
    assert all(d == 4 for _, d in g.degree())             # 4-regular
    assert nx.is_bipartite(g)
    assert all(part[u] != part[v] for u, v in g.edges())  # and `part` is a valid two-colouring
    assert part.sum() == 80                               # equal halves
    assert total_squares(adj) == brute_squares(g) == 160  # S = N, checked independently
    assert all(squares_on_edge(adj, u, int(v)) == 2 for u in range(160) for v in adj[u])
    assert is_valid(adj)


def test_square_torus_is_unchanged():
    """torus(L) must stay exactly what it was, slot order included, or old seeds stop reproducing."""
    adj, part = torus(10)
    adj2, part2 = torus(10, 10)
    assert np.array_equal(adj, adj2) and np.array_equal(part, part2)
    assert list(adj[0]) == [10, 90, 1, 9]                 # +x, -x, +y, -y


def test_short_or_odd_sides_are_refused():
    for lx, ly in [(16, 4), (4, 16), (7, 10), (10, 7)]:   # a side of 4 wraps into extra squares
        with pytest.raises(ValueError):
            torus(lx, ly)
    with pytest.raises(ValueError):
        torus(2, 8, cap=NO_CAP)                           # a side of 2 would double the edges


def test_chain_runs_on_a_rectangle_and_melts_to_the_published_floor():
    """Hot-phase square density at N = 160 against the published 0.126 [T25 Fig. 3].

    This is the floor only. It depends on which graphs are allowed, not on the
    energy, so it is NOT Gate B. Our value is 0.120, about 5 % BELOW the published
    one, which is a large-N formula; the hard-core rule (Q2) causes the deficit,
    the cap does not (ASSUMPTIONS O5).
    """
    adj, part = torus(16, 10)
    s, _, _ = run_chain(adj, np.flatnonzero(part == 0), 0.0, 300, 700, 5)
    assert is_valid(adj)
    assert s[-1] == total_squares(adj) == brute_squares(to_nx(adj))
    floor = s.mean() / 160
    assert abs(floor - 0.126) < 0.01                      # Gate B's tolerance, met with little room
    assert abs(floor - 0.120) < 0.004                     # our own value: guards the rules and the sampler


def test_chain_stays_in_configuration_space():
    adj, part = torus(8)
    side_u = np.flatnonzero(part == 0)
    for inv_g, seed in [(0.0, 1), (0.3, 2)]:
        s, x, acc = run_chain(adj, side_u, inv_g, 20, 20, seed)
        g = to_nx(adj)
        assert is_valid(adj)
        assert all(d == 4 for _, d in g.degree())
        assert nx.is_bipartite(g)
        assert all(part[u] != part[v] for u, v in g.edges())   # bipartition preserved
        assert s[-1] == total_squares(adj) == brute_squares(g)  # incremental count is exact
        assert not x.any() and surplus(adj) == 0                # under the cap X is always zero
        assert 0 < acc < 1


def test_melts_at_infinite_temperature():
    adj, part = torus(10)
    s, _, _ = run_chain(adj, np.flatnonzero(part == 0), 0.0, 300, 50, 3)
    assert s.mean() / 100 < 0.25                           # far from the ordered value S/N = 1


def test_same_seed_same_result():
    runs = []
    for _ in range(2):
        adj, part = torus(6)
        runs.append(run_chain(adj, np.flatnonzero(part == 0), 0.2, 5, 20, 42)[0])
    assert np.array_equal(runs[0], runs[1])


# ---- TASKS T2: the full Hamiltonian with the lambda knob, cap optional ----------------------

GOLDEN_S = [35, 36, 36, 38, 34, 36, 38, 37, 37, 39, 39, 34, 33, 38, 40, 40, 37, 34, 37, 37, 36, 36, 36, 37]
GOLDEN_ACC = 0.12729779411764705
GOLDEN_ADJ = 20242


def test_capped_kernel_is_bit_for_bit_what_it_was():
    """T2 regression. The numbers were recorded from the kernel as it stood before T2
    (commit f383c01). Under the cap lam must make no difference at all."""
    for lam in (1.0, 0.0, 0.37):
        adj, part = torus(8)
        side_u = np.flatnonzero(part == 0)
        run_chain(adj, side_u, 0.0, 30, 1, 11)
        s, x, acc = run_chain(adj, side_u, 0.15, 10, 24, 12, lam, CAP, False)
        assert list(s) == GOLDEN_S and acc == GOLDEN_ACC and not x.any()
        assert int((adj * np.arange(1, 5)).sum()) == GOLDEN_ADJ     # same graph, same slot order


def test_four_cube_energies():
    """T2: the 4x4 torus is the 4-cube; H = 0 at lam = 1 and -128 at lam = 0 (Q1)."""
    adj, _ = torus(4, cap=NO_CAP)
    g = to_nx(adj)
    assert nx.is_isomorphic(g, nx.hypercube_graph(4))
    assert is_valid(adj, NO_CAP) and not is_valid(adj, CAP)
    assert total_squares(adj) == brute_squares(g) == 24
    assert all(squares_on_edge(adj, u, int(v)) == 3 for u in range(16) for v in adj[u])
    assert surplus(adj) == 32                                       # 32 edges, one surplus square each
    assert hamiltonian(adj, 1.0) == brute_hamiltonian(g, 1.0) == 0  # degenerate with the flat torus
    assert hamiltonian(adj, 0.0) == brute_hamiltonian(g, 0.0) == -128
    assert hamiltonian(adj, 0.5) == brute_hamiltonian(g, 0.5) == -64


def test_flat_torus_has_zero_energy_for_every_lambda():
    for shape in [(6, 6), (16, 10)]:
        adj, _ = torus(*shape)
        assert all(hamiltonian(adj, lam) == 0 for lam in (0.0, 0.5, 1.0, 7.0))


@pytest.mark.parametrize("lam", [0.0, 0.5, 1.0, 3.0])
def test_energy_edge_by_edge(lam):
    """VISION Update 5: H = 4 * sum over edges of [(2 - S_e) + lam * (S_e - 2)+], and no edge carries
    more than 3 squares. So each edge is cheapest at S_e = 3 below lam = 1 and at S_e = 2 above it."""
    adj, part = torus(8, cap=NO_CAP)
    run_chain(adj, np.flatnonzero(part == 0), 0.12, 40, 1, 5, 0.0, NO_CAP, False)   # lam = 0: over-full edges appear
    s_e = [squares_on_edge(adj, u, int(v)) for u in range(len(adj)) for v in adj[u] if u < v]
    assert len(s_e) == 2 * len(adj) and {2, 3} <= set(s_e) and max(s_e) == 3
    assert 4 * sum((2 - k) + lam * max(k - 2, 0) for k in s_e) == hamiltonian(adj, lam)


def test_melted_graph_has_energy_near_16n():
    adj, part = torus(30, cap=NO_CAP)
    run_chain(adj, np.flatnonzero(part == 0), 0.0, 300, 1, 8, 1.0, NO_CAP, False)
    assert 0.95 < hamiltonian(adj, 1.0) / (16 * 900) <= 1.0          # about 20 squares among 900 vertices


@pytest.mark.parametrize("lam", [0.0, 0.5, 1.0])
def test_incremental_energy_is_exact(lam):
    """T2: running totals against full recomputation and against networkx, after about 10^4
    moves in one go and after every one of 100 further sweeps. Starts from the 4 x 8 torus,
    where half the edges carry three squares, so the local term is at work from the first move."""
    adj, part = torus(4, 8, cap=NO_CAP)
    side_u = np.flatnonzero(part == 0)
    assert surplus(adj) > 0
    s, x, acc = run_chain(adj, side_u, 0.10, 60, 100, 21, lam, NO_CAP, False)   # 160 sweeps x 64 moves
    assert acc > 0.05 and len(set(x)) >= 8                          # g = 10: many moves accepted, X busy
    for sweep in range(100):
        if sweep:
            s, x, _ = run_chain(adj, side_u, 0.10, 0, 1, 100 + sweep, lam, NO_CAP, False)
        assert s[-1] == total_squares(adj) and x[-1] == surplus(adj)
        assert 16 * (32 - s[-1]) + 4 * lam * x[-1] == hamiltonian(adj, lam)
    assert hamiltonian(adj, lam) == brute_hamiltonian(to_nx(adj), lam)


def test_uncapped_chain_stays_in_configuration_space():
    adj, part = torus(8, cap=NO_CAP)
    side_u = np.flatnonzero(part == 0)
    for inv_g, lam, glauber, seed in [(0.0, 1.0, False, 1), (0.3, 0.0, False, 2), (0.3, 1.0, True, 3)]:
        s, x, acc = run_chain(adj, side_u, inv_g, 20, 20, seed, lam, NO_CAP, glauber)
        g = to_nx(adj)
        assert is_valid(adj, NO_CAP)
        assert all(d == 4 for _, d in g.degree()) and nx.is_bipartite(g)
        assert all(part[u] != part[v] for u, v in g.edges())
        assert max(squares_on_edge(adj, u, int(v)) for u in range(64) for v in adj[u]) <= 3   # Q2
        assert hamiltonian(adj, lam) == brute_hamiltonian(g, lam)
        assert 0 < acc < 1


def test_same_seed_same_result_uncapped():
    runs = []
    for _ in range(2):
        adj, part = torus(6, cap=NO_CAP)
        runs.append(run_chain(adj, np.flatnonzero(part == 0), 0.2, 5, 20, 42, 0.5, NO_CAP, True))
    assert np.array_equal(runs[0][0], runs[1][0]) and np.array_equal(runs[0][1], runs[1][1])


def test_glauber_and_metropolis_sample_the_same_distribution():
    """Both rules obey detailed balance (Q4), so their averages must agree within errors."""
    means, errs = [], []
    for glauber in (False, True):
        adj, part = torus(6, cap=NO_CAP)
        side_u = np.flatnonzero(part == 0)
        run_chain(adj, side_u, 0.0, 100, 1, 30, 1.0, NO_CAP, glauber)
        s, _, _ = run_chain(adj, side_u, 1 / 8.0, 300, 6000, 31, 1.0, NO_CAP, glauber)
        m, e, _, _ = block_bootstrap_mean_var(s / 36, seed=32)
        means.append(m)
        errs.append(e)
    assert abs(means[0] - means[1]) < 4 * np.hypot(*errs)
    assert 0.3 < means[0] < 0.9                                     # in the crossover, not at an end


@pytest.mark.parametrize("cap, lam, inv_g", [(CAP, 1.0, 0.12), (NO_CAP, 0.0, 0.12), (NO_CAP, 1.0, 0.0)])
def test_quick_validity_check_agrees_with_the_full_one(cap, lam, inv_g):
    """T4: the kernel decides whether a switch stays inside the configuration space by looking only near the
    two new edges. Tried here on EVERY possible switch from a few states, against is_valid on the whole graph.
    A quick check that refused valid moves would bias the sampling; one that let invalid ones through would leave
    the space."""
    from graphity.cqg import _new_edge_ok, _switch
    adj, part = torus(6, 6, cap) if cap == CAP else torus(6, 4, cap)
    side_u = np.flatnonzero(part == 0)
    tried = allowed = 0
    for stage in range(3):
        run_chain(adj, side_u, inv_g, 40, 1, 60 + stage, lam, cap, False)
        assert is_valid(adj, cap)
        for u1 in side_u:
            for u2 in side_u:
                for v1 in adj[u1].copy():
                    for v2 in adj[u2].copy():
                        if u1 == u2 or v1 == v2 or v2 in adj[u1] or v1 in adj[u2]:
                            continue
                        _switch(adj, u1, v1, u2, v2)
                        quick = _new_edge_ok(adj, u1, v2, cap) and _new_edge_ok(adj, u2, v1, cap)
                        assert quick == is_valid(adj, cap)
                        _switch(adj, u1, v2, u2, v1)
                        tried += 1
                        allowed += quick
    assert tried > 1000 and 0 < allowed < tried
