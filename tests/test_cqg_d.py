"""The any-D kernel (cqg_d.py): the same chain as cqg.py at D = 2, and the rule-6 checks at D = 3."""
import networkx as nx
import numpy as np
import pytest

from graphity import cqg, cqg_d

GOLOMB6 = [0, 1, 4, 10, 12, 17]          # pairwise differences all distinct: a start with no squares at D = 3


@pytest.mark.parametrize("lx,ly", [(6, 6), (16, 4), (8, 10), (4, 4)])
def test_the_2d_torus_is_cqgs_torus(lx, ly):
    a, p = cqg.torus(lx, ly, cqg.NO_CAP)
    b, q = cqg_d.torus((lx, ly))
    assert np.array_equal(a, b) and np.array_equal(p, q)


@pytest.mark.parametrize("lam,glauber", [(1.25, False), (0.5, False), (1.0, True)])
def test_at_four_links_the_chain_is_cqgs_chain_draw_for_draw(lam, glauber):
    for start in [(16, 4), (8, 8)]:
        a, part = cqg.torus(*start, cqg.NO_CAP)
        b = a.copy()
        side = np.flatnonzero(part == 0)
        cqg.run_chain(a, side, 0.0, 30, 0, 11, 1.0, cqg.NO_CAP)     # melt a little, same stream both ways
        cqg_d.run_chain(b, side, 0.0, 30, 0, 11)
        assert np.array_equal(a, b)
        r1 = cqg.run_chain(a, side, 1 / 1.5, 40, 60, 5, lam, cqg.NO_CAP, glauber)
        r2 = cqg_d.run_chain(b, side, 1 / 1.5, 40, 60, 5, lam, glauber)
        assert np.array_equal(r1[0], r2[0]) and np.array_equal(r1[1], r2[1]) and r1[2] == r2[2]
        assert np.array_equal(a, b)


def brute(adj):
    g = nx.Graph()
    for u in range(adj.shape[0]):
        for v in adj[u]:
            g.add_edge(u, int(v))
    sat = adj.shape[1] - 2
    per = []
    for u, v in g.edges():
        per.append(sum(1 for a in g[u] if a != v for b in g[v] if b != u and g.has_edge(a, b)))
    return sum(per) // 4, sum(max(c - sat, 0) for c in per)


@pytest.mark.parametrize("dims", [(6, 6, 6), (4, 6, 6), (4, 4, 6), (4, 4, 4)])
def test_squares_and_surplus_agree_with_networkx_in_3d(dims):
    adj, _ = cqg_d.torus(dims)
    assert (cqg_d.total_squares(adj), cqg_d.surplus(adj)) == brute(adj)


def test_the_3d_ladder_energies():
    for dims, curled in [((6, 6, 6), 0), ((4, 6, 6), 1), ((4, 4, 6), 2), ((4, 4, 4), 3)]:
        adj, _ = cqg_d.torus(dims)
        n = adj.shape[0]
        for lam in (0.0, 1.0, 1.1, 1.25):
            assert abs(cqg_d.hamiltonian(adj, lam) - 4 * curled * (lam - 1) * n) < 1e-9


def test_the_circulant_start_is_valid_and_has_no_squares():
    adj, part = cqg_d.circulant(250, GOLOMB6)
    assert adj.shape == (500, 6) and cqg_d.is_valid(adj)
    assert cqg_d.total_squares(adj) == 0
    assert all(part[u] != part[v] for u in range(500) for v in adj[u])


# Hot enough that moves are accepted (at g = 1.5 the curled 3-torus has walls near 43 and nothing moves),
# plus one cold case whose only moves are downhill.
@pytest.mark.parametrize("start,lam,g", [("torus", 1.1, 20.0), ("torus", 0.5, 30.0), ("circulant", 1.0, 2.0),
                                         ("cube", 1.25, 1.5)])
def test_3d_incremental_energy_is_exact_and_constraints_hold(start, lam, g):
    if start == "torus":
        adj, part = cqg_d.torus((4, 6, 6))
    elif start == "cube":
        adj, part = cqg_d.torus((4, 4, 4))
    else:
        adj, part = cqg_d.circulant(60, GOLOMB6)
    side = np.flatnonzero(part == 0)
    s, x, acc = cqg_d.run_chain(adj, side, 1.0 / g, 20, 30, 3, lam)
    assert s[-1] == cqg_d.total_squares(adj) and x[-1] == cqg_d.surplus(adj)
    assert (s[-1], x[-1]) == brute(adj)
    assert cqg_d.is_valid(adj) and adj.shape[1] == 6
    assert all(part[u] != part[v] for u in range(adj.shape[0]) for v in adj[u])
    assert 0.0 < acc < 1.0


def test_3d_same_seed_same_chain():
    out = []
    for _ in range(2):
        adj, part = cqg_d.torus((4, 6, 6))
        out.append((cqg_d.run_chain(adj, np.flatnonzero(part == 0), 1 / 1.5, 10, 20, 9, 1.1), adj))
    assert np.array_equal(out[0][0][0], out[1][0][0]) and np.array_equal(out[0][1], out[1][1])
