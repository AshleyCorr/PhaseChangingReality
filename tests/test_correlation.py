"""T16: the correlation length of [KTB19] Fig. 9. Checked against a brute-force networkx computation."""
import networkx as nx
import numpy as np

from graphity.correlation import correlation_by_distance, correlation_curve, edge_squares, xi_literal
from graphity.cqg import NO_CAP, is_valid, run_chain, torus
from graphity.graphs import to_networkx


def melted(lx, ly, seed, sweeps=200):
    adj, part = torus(lx, ly, NO_CAP)
    run_chain(adj, np.flatnonzero(part == 0), 0.0, sweeps, 1, seed, 1.0, NO_CAP, False)
    assert is_valid(adj, NO_CAP)
    return adj


def brute_force(adj):
    """The same quantities written the slow, obvious way with networkx."""
    g = to_networkx(adj)
    s_e = {}
    for u, v in g.edges():
        cycles = 0
        for a in g[v]:
            for b in g[u]:
                if a != u and b != v and a != b and g.has_edge(a, b):
                    cycles += 1
        s_e[frozenset((u, v))] = cycles / 2.0                   # phi_sq = S_e / (d - 2)
    phi = np.mean(list(s_e.values()))
    edge_var = np.mean([(x - phi) ** 2 for x in s_e.values()])
    f = {u: np.mean([s_e[frozenset((u, v))] for v in g[u]]) - phi for u in g}
    by_r = {}
    for u, dists in nx.all_pairs_shortest_path_length(g):
        for v, r in dists.items():
            if r > 0:
                by_r.setdefault(r, []).append(f[u] * f[v])
    return phi, edge_var, {r: np.mean(x) / edge_var for r, x in by_r.items()}, max(by_r)


def test_matches_networkx_on_melted_graphs():
    for seed in (1, 2, 3):
        adj = melted(8, 6, seed)
        num, count, edge_var, phi, diam = correlation_by_distance(adj, 20)
        c = correlation_curve(num, count, edge_var)
        phi_b, var_b, c_b, diam_b = brute_force(adj)
        assert abs(phi - phi_b) < 1e-12 and abs(edge_var - var_b) < 1e-12 and diam == diam_b
        assert abs(phi - edge_squares(adj).sum() / 8 / adj.shape[0]) < 1e-12      # = S / N
        for r, value in c_b.items():
            assert abs(c[r] - value) < 1e-10, (r, c[r], value)
        assert np.all(np.isnan(c[diam + 1:]))


def test_perfect_lattice_has_nothing_to_correlate():
    adj, _ = torus(10, 10, NO_CAP)
    num, count, edge_var, phi, diam = correlation_by_distance(adj, 20)
    assert phi == 1.0 and edge_var == 0.0 and diam == 10
    assert np.all(np.isnan(correlation_curve(num, count, edge_var)))


def test_xi_literal_by_hand():
    c = np.array([np.nan, np.exp(-1.0), np.exp(-1.0), -0.1, 1.2, np.nan])
    xi, used, skipped = xi_literal(c, diameter=4)
    assert used == 2 and skipped == 2
    assert abs(xi - (1.0 + 2.0) / 2) < 1e-12                  # -r / log C at r = 1 and 2 is 1 and 2
