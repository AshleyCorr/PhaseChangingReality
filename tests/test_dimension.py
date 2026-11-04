"""T7 gate 1: the per-vertex local dimension must read 2 / 1 / 0 on the exact sheet, tube and
4-cube, at every vertex, before it is used on anything else."""
from math import comb

import networkx as nx
import numpy as np
import pytest

from graphity.cqg import NO_CAP, torus, total_squares
from graphity.dimension import (heat_kernel_dimension, local_dimension, pieces_of, spectral_dimension,
                                volume_growth)


def test_sheet_tube_cube_at_every_vertex():
    for shape, want in (((6, 6), 2), ((8, 8), 2), ((16, 4), 1), ((24, 4), 1), ((4, 4), 0)):
        adj, _ = torus(shape[0], shape[1], NO_CAP)
        d = local_dimension(adj)
        assert (d == want).all(), (shape, np.bincount(d))


def test_it_agrees_with_the_square_count_ladder():
    """Design brief: curling a side adds a quarter of a square per vertex and removes one large
    direction. So mean d + squares per vertex is constant across the ladder (2 + 1 = 1 + 1.25 =
    0 + 1.5? no -- but 2 - d equals 4 * (squares per vertex - 1)). Check that relation."""
    from graphity.cqg import total_squares
    for shape in ((6, 6), (16, 4), (4, 4)):
        adj, _ = torus(shape[0], shape[1], NO_CAP)
        n = adj.shape[0]
        d = local_dimension(adj).mean()
        sq = total_squares(adj) / n
        assert abs((2 - d) - 4 * (sq - 1)) < 1e-12, (shape, d, sq)


def test_pieces_of_a_subset():
    adj, _ = torus(8, 8, NO_CAP)
    n = adj.shape[0]
    # one connected 2x2 block of the lattice, plus one isolated vertex far away
    members = np.zeros(n, dtype=bool)
    members[[0, 1, 8, 9]] = True
    members[36] = True
    assert pieces_of(adj, members) == [4, 1]
    assert pieces_of(adj, np.ones(n, dtype=bool)) == [n]
    assert pieces_of(adj, np.zeros(n, dtype=bool)) == []


def test_piece_labels_agree_with_pieces_of():
    """T37: the labels number the same pieces pieces_of sizes."""
    from graphity.dimension import piece_labels
    adj, _ = torus(12, 10, NO_CAP)
    rng = np.random.default_rng(3)
    for _ in range(20):
        members = rng.random(adj.shape[0]) < 0.45
        labels = piece_labels(adj, members)
        assert (labels[~members] == -1).all() and (labels[members] >= 0).all()
        sizes = sorted(np.bincount(labels[labels >= 0]).tolist(), reverse=True) if members.any() else []
        assert sizes == pieces_of(adj, members)


def test_new_seeds_counts_only_pieces_touching_nothing_counted():
    """T37: a new piece is a seed; a piece that grew, split or merged is not; small pieces are ignored."""
    from graphity.dimension import new_seeds
    counted = np.zeros(10, dtype=bool)
    labels = np.array([0, 0, 0, -1, 1, 1, 1, -1, 2, -1])          # two pieces of 3, one of 1
    k, counted = new_seeds(labels, counted, 3)
    assert k == 2 and counted.tolist() == [1, 1, 1, 0, 1, 1, 1, 0, 0, 0]
    labels = np.array([0, 0, 0, 0, 0, 0, 0, -1, -1, -1])          # the two merged: no new seed
    k, counted = new_seeds(labels, counted, 3)
    assert k == 0
    labels = np.array([0, -1, 1, 1, -1, -1, -1, 2, 2, 2])         # a split piece (0, 1) and a new one (2)
    k, counted = new_seeds(labels, counted, 1)
    assert k == 1 and counted[7:].all()


# ---------------------------------------------------------------------------------------------------------------
# T80's estimator (draft): spectral_dimension, volume_growth and heat_kernel_dimension, on built graphs only. No saved
# state is read. Built perfect objects are measured here, as exact references: a ring; square-lattice tori; the tube
# (16 x 4) and the 4-cube as built, at the first step (D_s between t = 2 and 4, D_v between r = 1 and 2, both
# restatements of the square count); and the 10 x 10 x 10 cubic torus for t < 10, whose P(t) fixes the perfect
# six-link lattice's shortest-scale D_s: 2.53, 2.74 and 2.80 at t_mid = 2.8, 4.9 and 6.9. These are exact arithmetic
# on perfect objects, known before any run, so none of them can count as a test of either prediction in T80; that
# includes the six-link flat torus's short-scale value, the one number on which T80's draft separates the owner's
# inferred prediction (2) from ours (3). The toy's saved states are read only after her predictions are written.
# ---------------------------------------------------------------------------------------------------------------

def _ring(m):
    """The ring of m points as an (m, 2) neighbour array."""
    i = np.arange(m)
    return np.stack([(i + 1) % m, (i - 1) % m], axis=1).astype(np.int64)


def _from_networkx(g):
    """(N, k) neighbour array of a k-regular networkx graph whose points are numbered 0 .. N-1."""
    return np.array([sorted(g.neighbors(v)) for v in range(g.number_of_nodes())], dtype=np.int64)


def _line(t):
    """The chance a walker on the infinite line is back at its start after t steps (t even): C(t, t/2) / 2^t."""
    return comb(t, t // 2) / 2 ** t


def test_ring_reads_one():
    """The ring of 400 points. Below 400 steps a walker cannot go round, so P(t) is the infinite line's exactly
    (checked to 1e-12; measured 6e-14). D_s is within 0.025 of 1 for t_mid from 20 to 600 (it reads 0.83 at t_mid = 2.8: the lattice correction). The ball
    is 2r + 1 points while r < 200, so D_v rises toward 1."""
    m = 400
    res = spectral_dimension(_ring(m), 600)
    short = res["t"] < m
    want = np.array([_line(int(t)) for t in res["t"][short]])
    np.testing.assert_allclose(res["P"][short], want, rtol=1e-12)
    assert res["floor"] == 2 / m
    window = (res["t_mid"] >= 20) & (res["t_mid"] <= 600)
    assert np.abs(res["D_s"][window] - 1).max() < 0.025
    vg = volume_growth(_ring(m), 150)
    assert (vg["V"] == 2 * vg["r"] + 1).all()
    assert (np.diff(vg["D_v"]) > 0).all() and 0.99 < vg["D_v"][-1] < 1


def test_square_torus_return_probability_is_the_plane_s_and_reads_two():
    """The 32 x 32 lattice torus. Exact: for t < L a walker cannot go round, so P(t) is the infinite square lattice's,
    [C(t, t/2) / 2^t]^2 (checked to 1e-12; measured 1e-14). Window and tolerance: D_s within 0.05 of 2 for t_mid from 20 to L^2/8 = 128. The floor is
    2/N (one connected bipartite piece), 6.3 % of P(t) at t = 20 and 39.4 % at t = 128; past t of order L^2 the
    plain D_s falls to 0 and the reading above the floor rises without bound, both of them the graph's size (the
    docstring's statement, checked)."""
    L = 32
    adj, _ = torus(L, L, NO_CAP)
    res = spectral_dimension(adj, 4 * L * L)
    short = res["t"] < L
    want = np.array([_line(int(t)) ** 2 for t in res["t"][short]])
    np.testing.assert_allclose(res["P"][short], want, rtol=1e-12)
    assert res["floor"] == 2 / L ** 2
    window = (res["t_mid"] >= 20) & (res["t_mid"] <= L * L / 8)
    assert np.abs(res["D_s"][window] - 2).max() < 0.05
    early = (res["t_mid"] >= 20) & (res["t_mid"] <= 64)
    assert (res["D_s_above_floor"][early] > 2.05).all()          # subtracting the floor too early reads high
    assert abs(res["D_s"][0] - 1.66) < 0.005                      # the lattice correction at the shortest scale
    share = res["floor"] / res["P"]                                # the floor's share of P(t) at the window's ends
    assert abs(share[res["t"] == 20][0] - 0.063) < 5e-4 and abs(share[res["t"] == 128][0] - 0.394) < 5e-4
    late = res["t_mid"] >= 2 * L * L
    assert (res["D_s"][late] < 0.01).all() and (res["D_s_above_floor"][late] > 30).all()


def test_square_torus_ball_is_the_diamond_until_it_wraps():
    """Exact: on the 64 x 64 torus the ball of radius r is the lattice's diamond, 2r^2 + 2r + 1 points, for every
    r <= L/2 - 1 = 31; at r = L/2 the points (+-L/2, 0) and (0, +-L/2) meet their opposites and it holds 2 fewer.
    D_v is between 1.85 and 2 for r_mid from 8 to L/4 = 16 (about 2 - 1/r: the diamond's own correction)."""
    L = 64
    adj, _ = torus(L, L, NO_CAP)
    vg = volume_growth(adj, L // 2)
    r = vg["r"]
    diamond = 2 * r ** 2 + 2 * r + 1
    assert (vg["V"][:L // 2] == diamond[:L // 2]).all()
    assert vg["V"][L // 2] == diamond[L // 2] - 2
    window = (vg["r_mid"] >= 8) & (vg["r_mid"] <= L / 4)
    assert ((vg["D_v"][window] > 1.85) & (vg["D_v"][window] < 2)).all()
    assert (np.diff(vg["D_v"][window]) > 0).all()


def test_four_cube_return_probability_matches_the_hypercube_formula():
    """The 4 x 4 torus is the 4-cube (C4 x C4 = Q4; checked against networkx). The walk's eigenvalues on the d-cube
    are 1 - 2k/d, C(d, k) times each, so P(t) = 2^-d sum_k C(d, k) (1 - 2k/d)^t, d = 4: to machine precision."""
    adj, _ = torus(4, 4, NO_CAP)
    edges = {(u, int(v)) for u in range(16) for v in adj[u]}
    assert nx.is_isomorphic(nx.Graph(list(edges)), nx.hypercube_graph(4))
    res = spectral_dimension(adj, 40)
    d = 4
    want = [sum(comb(d, k) * (1 - 2 * k / d) ** int(t) for k in range(d + 1)) / 2 ** d for t in res["t"]]
    np.testing.assert_allclose(res["P"], want, rtol=0, atol=1e-14)
    assert res["floor"] == 2 / 16


def test_random_four_regular_graph_reads_high_and_its_balls_grow_like_a_tree_s():
    """A random 4-regular graph (networkx, seed 2026, N = 1000; connected and not bipartite) is locally a tree. Its
    floor is 1/N (the eigenvalue +1 only). D_s is already 2.38 between t = 2 and 4 (the tree's own value: P(2) = 1/4,
    P(4) = 28/256) and rises at every step, past 4 by t_mid = 9. Its balls are never larger than the 4-regular tree's,
    2 * 3^r - 1 points, and within 3 % of them up to r = 3 (16.9 of 17 at r = 2, 51.8 of 53 at r = 3: a few short
    cycles); D_v rises while the larger ball holds under half the
    points, and V(r) = N from the diameter (networkx) on."""
    n = 1000
    g = nx.random_regular_graph(4, n, seed=2026)
    assert nx.is_connected(g) and not nx.is_bipartite(g)
    adj = _from_networkx(g)
    res = spectral_dimension(adj, 16)
    assert res["floor"] == 1 / n
    assert res["D_s"][0] > 2.3 and (np.diff(res["D_s"]) > 0).all()
    assert (res["D_s"][res["t_mid"] >= 8.9] > 4).all()
    vg = volume_growth(adj, 15)
    tree = 2 * 3.0 ** vg["r"] - 1
    assert (vg["V"] <= tree + 1e-9).all()
    assert (vg["V"][:4] > 0.97 * tree[:4]).all()
    growing = vg["V"][2:] < n / 2                                  # D_v[j] runs from r = j + 1 to r = j + 2
    assert growing.sum() >= 3 and (np.diff(vg["D_v"][growing]) > 0).all()
    diameter = nx.diameter(g)
    assert (vg["V"][diameter:] == n).all() and (vg["V"][:diameter] < n).all()


def test_return_probability_agrees_with_matrix_powers():
    """Independent check: P(t) = Tr(W^t) / N with W built by networkx and raised to the power t by numpy, on a random
    4-regular graph (not bipartite) and on networkx's own periodic 6 x 6 grid (bipartite); and the slope arithmetic.
    Both to 1e-12 (measured: 1e-14 for P, 1.4e-13 for the slope)."""
    graphs = (nx.random_regular_graph(4, 40, seed=7),
              nx.convert_node_labels_to_integers(nx.grid_2d_graph(6, 6, periodic=True)))
    for g in graphs:
        n = g.number_of_nodes()
        res = spectral_dimension(_from_networkx(g), 24)
        w = nx.to_numpy_array(g, nodelist=range(n)) / 4
        want = np.array([np.trace(np.linalg.matrix_power(w, int(t))) / n for t in res["t"]])
        np.testing.assert_allclose(res["P"], want, rtol=1e-12)
        slope = -2 * np.diff(np.log(want)) / np.diff(np.log(res["t"]))
        np.testing.assert_allclose(res["D_s"], slope, rtol=1e-12)


def test_volume_growth_agrees_with_networkx_distances():
    """Independent check of the breadth-first search: V(r) from networkx's shortest-path lengths."""
    g = nx.random_regular_graph(4, 200, seed=11)
    vg = volume_growth(_from_networkx(g), 10)
    lengths = [list(nx.single_source_shortest_path_length(g, s).values()) for s in range(200)]
    want = [np.mean([sum(1 for x in row if x <= r) for row in lengths]) for r in vg["r"]]
    np.testing.assert_allclose(vg["V"], want, rtol=0, atol=1e-12)


def test_floor_counts_pieces_and_bipartite_pieces():
    """Two separate 6 x 6 tori: +1 and -1 each twice, so the floor is 4/N."""
    adj, _ = torus(6, 6, NO_CAP)
    two = np.vstack([adj, adj + 36])
    assert spectral_dimension(two, 10)["floor"] == 4 / 72


def test_same_input_same_output_and_the_input_is_untouched():
    adj, _ = torus(8, 6, NO_CAP)
    before = adj.copy()
    for run in (lambda: spectral_dimension(adj, 30), lambda: volume_growth(adj, 6),
                lambda: heat_kernel_dimension(adj, [0.5, 5.0, 50.0])):
        a, b = run(), run()
        assert a.keys() == b.keys()
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
    assert (adj == before).all()


def test_refuses_what_it_cannot_measure():
    """All three refuse the same arrays that are not simple graphs (one shared check), each with a ValueError: before
    23 October volume_growth returned a wrong V for the lopsided and out-of-range ones without a word, and
    spectral_dimension raised an IndexError for the out-of-range one; until the second check of 23 October a self-loop,
    a double link listed from both ends and a fractional float (cast to a whole number) all passed. Each also refuses
    a scale it cannot read a slope at."""
    adj, _ = torus(6, 6, NO_CAP)
    broken = adj.copy()
    broken[0, 0] = -1                                              # a slot emptied mid-move
    lopsided = adj.copy()
    lopsided[0, 0] = lopsided[0, 1]                                # 0 lists one neighbour twice; it lists 0 once
    self_loop = adj.copy()                                         # 0 and a list themselves, not each other
    a = self_loop[0, 0]
    self_loop[0, 0] = 0
    self_loop[a, list(self_loop[a]).index(0)] = a
    double = adj.copy()                                            # a switch that doubles the link 0 - a
    a, b = double[0, 0], double[0, 1]
    x = next(int(w) for w in double[a] if w != 0 and w not in double[b])  # a neighbour of a, not of b
    double[0, 1] = a                                               # 0 lists a twice, not b
    double[a, list(double[a]).index(x)] = 0                        # a lists 0 twice, not x
    double[b, list(double[b]).index(0)] = x                        # b and x take the link the other two gave up
    double[x, list(double[x]).index(a)] = b
    fractional = adj.astype(float)
    fractional[0, 0] += 0.7                                        # a cast would truncate it back to a neighbour
    for past_the_end in (36, 10 ** 6):                             # a point that does not exist
        outside = adj.copy()
        outside[0, 0] = past_the_end
        for bad in (broken, lopsided, outside, self_loop, double, fractional):
            with pytest.raises(ValueError):
                spectral_dimension(bad, 10)
            with pytest.raises(ValueError):
                volume_growth(bad, 4)
            with pytest.raises(ValueError):
                heat_kernel_dimension(bad, [1.0])
    with pytest.raises(ValueError):
        spectral_dimension(adj, 3)
    with pytest.raises(ValueError):
        volume_growth(adj, 1)
    for t in ([], [0.0], [1.0, -1.0], [np.nan], [np.inf]):
        with pytest.raises(ValueError):
            heat_kernel_dimension(adj, t)


def test_the_reading_above_the_floor_is_nan_once_it_underflows():
    """On the 4-cube, P_above_floor underflows to exactly 0 from t = 1,074. Every pair that touches a zero reads nan
    (the first used to read +inf); the plain D_s stays finite throughout."""
    adj, _ = torus(4, 4, NO_CAP)
    res = spectral_dimension(adj, 2400)
    zero = res["P_above_floor"] == 0
    assert res["t"][np.flatnonzero(zero)[0]] == 1074 and zero[np.flatnonzero(zero)[0]:].all()
    touches_zero = zero[:-1] | zero[1:]
    assert np.isnan(res["D_s_above_floor"][touches_zero]).all()
    assert np.isfinite(res["D_s_above_floor"][~touches_zero]).all()
    assert np.isfinite(res["D_s"]).all()


def _z3_returns(t):
    """Closed walks of t steps on the infinite cubic lattice Z^3 (t even): choose which steps go along x, y and z, and
    in each direction half go forward."""
    n = t // 2
    return sum(comb(t, 2 * a) * comb(2 * a, a) * comb(t - 2 * a, 2 * b) * comb(2 * b, b)
               * comb(t - 2 * a - 2 * b, n - a - b) for a in range(n + 1) for b in range(n + 1 - a))


def test_cubic_torus_return_probability_is_the_cubic_lattice_s():
    """Six links per point. The 10 x 10 x 10 lattice torus (networkx's periodic grid): for t < 10 a walker cannot go
    round, so P(t) is the infinite cubic lattice's, 6, 90, 1860, 44730 closed walks over 6^t at t = 2, 4, 6, 8
    (counted by the formula, and the formula checked against those four numbers); to 1e-14 (measured 9e-16)."""
    assert [_z3_returns(t) for t in (2, 4, 6, 8)] == [6, 90, 1860, 44730]
    g = nx.convert_node_labels_to_integers(nx.grid_graph(dim=(10, 10, 10), periodic=True))
    adj = _from_networkx(g)
    assert adj.shape == (1000, 6)
    res = spectral_dimension(adj, 8)
    want = np.array([_z3_returns(int(t)) / 6 ** int(t) for t in res["t"]])
    np.testing.assert_allclose(res["P"], want, rtol=1e-14)
    assert res["floor"] == 2 / 1000


def _random_bipartite(n, k, seed):
    """A random simple k-regular bipartite graph on two sides of n points (0 .. n-1 and n .. 2n-1): the union of k
    perfect matchings, each a random permutation drawn again until it repeats no link already chosen."""
    rng = np.random.default_rng(seed)
    side = np.empty((n, 0), dtype=np.int64)
    while side.shape[1] < k:
        match = n + rng.permutation(n)
        if not (side == match[:, None]).any():
            side = np.hstack([side, match[:, None]])
    other = np.empty((n, k), dtype=np.int64)
    fill = np.zeros(n, dtype=np.int64)
    for u in range(n):
        for v in side[u]:
            other[v - n, fill[v - n]] = u
            fill[v - n] += 1
    return np.vstack([side, other])


def _squares_per_point(adj):
    """q = 4S/N with S the number of 4-cycles, counted independently of the toy's square counter: each 4-cycle has
    two diagonal pairs of points, and a pair with c common neighbours closes c(c - 1)/2 of them."""
    n, k = adj.shape
    a = np.zeros((n, n), dtype=np.int64)
    np.add.at(a, (np.repeat(np.arange(n), k), adj.ravel()), 1)
    c = a @ a
    np.fill_diagonal(c, 0)
    return 4 * ((c * (c - 1) // 2).sum() // 4) / n                 # each diagonal pair is seen from both ends


def test_random_bipartite_four_regular_graph():
    """A random bipartite 4-regular graph (seed 2026, N = 400; simple, connected), like the toy's melts in being
    bipartite but not built to obey the hard-core rule. Its floor is 2/N; P(t) equals Tr(A^t) / (N 4^t) counted in
    whole numbers to 1e-12; V(r) equals networkx's distances; and the first step reads the identity's value."""
    adj = _random_bipartite(200, 4, seed=2026)
    g = nx.Graph([(u, int(v)) for u in range(400) for v in adj[u]])
    assert g.number_of_edges() == 800 and nx.is_connected(g) and nx.is_bipartite(g)
    res = spectral_dimension(adj, 12)
    assert res["floor"] == 2 / 400
    a = nx.to_numpy_array(g, nodelist=range(400), dtype=np.int64)
    power, want = np.eye(400, dtype=np.int64), []
    for t in range(1, 13):
        power = power @ a
        if t % 2 == 0:
            want.append(np.trace(power) / (400 * 4 ** t))
    np.testing.assert_allclose(res["P"], want, rtol=1e-12)
    q = _squares_per_point(adj)
    assert abs(res["D_s"][0] - (-2 * np.log2((2 * 16 - 4 + 2 * q) / 64))) < 1e-12
    vg = volume_growth(adj, 8)
    lengths = [list(nx.single_source_shortest_path_length(g, s).values()) for s in range(400)]
    np.testing.assert_allclose(vg["V"], [np.mean([sum(1 for x in row if x <= r) for row in lengths])
                                         for r in vg["r"]], rtol=0, atol=1e-12)


def test_the_first_step_restates_the_square_count():
    """Exact: closed walks of four steps on a simple k-regular graph are 8 per square plus N(2k^2 - k) that retrace
    themselves, so P(2) = 1/k, P(4) = (2k^2 - k + 2q) / k^4 with q = 4S/N, and the first reading is
    D_s(sqrt 8) = -2 log2[(2k^2 - k + 2q) / k^3]. On the flat torus (q = 4), the tube (q = 5) and the 4-cube (q = 6),
    all as built, with q from the toy's square counter and from the independent count; to 1e-12."""
    for shape, q_want, d_want in (((16, 16), 4, 1.660), ((16, 4), 5, 1.504), ((4, 4), 6, 1.356)):
        adj, _ = torus(shape[0], shape[1], NO_CAP)
        n, k = adj.shape
        q = 4 * total_squares(adj) / n
        assert q == q_want == _squares_per_point(adj)
        res = spectral_dimension(adj, 4)
        assert res["P"][0] == pytest.approx(1 / k, rel=1e-14)
        identity = -2 * np.log2((2 * k * k - k + 2 * q) / k ** 3)
        assert abs(res["D_s"][0] - identity) < 1e-12 and abs(identity - d_want) < 5e-4


def test_the_first_volume_growth_step_restates_the_square_count():
    """Exact: on a simple k-regular graph with no triangles in which no two points share more than two neighbours
    (every valid state of the toy), V(1) = k + 1 and V(2) = k^2 + 1 - q with q = 4S/N, so the first reading is
    D_v(sqrt 2) = log2[(k^2 + 1 - q) / (k + 1)]. On the flat torus (q = 4), the tube (q = 5) and the 4-cube (q = 6),
    all as built, with both conditions checked and q from the toy's square counter and the independent count;
    to 1e-12."""
    for shape, q_want, d_want in (((16, 16), 4, 1.3785), ((16, 4), 5, 1.2630), ((4, 4), 6, 1.1375)):
        adj, _ = torus(shape[0], shape[1], NO_CAP)
        n, k = adj.shape
        g = nx.Graph([(u, int(v)) for u in range(n) for v in adj[u]])
        common = nx.to_numpy_array(g, nodelist=range(n), dtype=np.int64)
        common = common @ common
        np.fill_diagonal(common, 0)
        assert nx.is_bipartite(g) and common.max() <= 2              # no triangle; at most two shared neighbours
        q = 4 * total_squares(adj) / n
        assert q == q_want == _squares_per_point(adj)
        vg = volume_growth(adj, 2)
        assert vg["V"][1] == k + 1 and vg["V"][2] == k * k + 1 - q
        identity = np.log2((k * k + 1 - q) / (k + 1))
        assert vg["r_mid"][0] == pytest.approx(np.sqrt(2), rel=1e-15)
        assert abs(vg["D_v"][0] - identity) < 1e-12 and abs(identity - d_want) < 5e-5


# [T22] Eqs. (15) to (17): the published model's continuous-time spectral function.

def test_heat_kernel_on_the_four_cube():
    """Exact: on the 4-cube the eigenvalues of Delta are -j/2, C(4, j) times each, so K(t) = (1 + e^{-t/2})^4 and
    d_s(t) = 4t / (e^{t/2} + 1). Both to 1e-12."""
    adj, _ = torus(4, 4, NO_CAP)
    t = np.geomspace(1e-3, 200, 60)
    res = heat_kernel_dimension(adj, t)
    np.testing.assert_allclose(res["K"], (1 + np.exp(-t / 2)) ** 4, rtol=1e-12)
    np.testing.assert_allclose(res["d_s"], 4 * t / (np.exp(t / 2) + 1), rtol=1e-12, atol=1e-300)


def test_heat_kernel_on_the_square_torus():
    """Exact: on the L x L lattice torus W has eigenvalues (cos k1 + cos k2) / 2, k = 2 pi m / L, so
    K(t) = sum over m1, m2 of exp(-t [1 - (cos k1 + cos k2) / 2]). L = 12 and a rectangular 12 x 8; to 1e-12."""
    t = np.geomspace(1e-3, 1e4, 80)
    for lx, ly in ((12, 12), (12, 8)):
        adj, _ = torus(lx, ly, NO_CAP)
        k1 = 2 * np.pi * np.arange(lx) / lx
        k2 = 2 * np.pi * np.arange(ly) / ly
        rate = 1 - (np.cos(k1)[:, None] + np.cos(k2)[None, :]) / 2
        want = np.exp(-t[:, None, None] * rate[None]).sum(axis=(1, 2))
        np.testing.assert_allclose(heat_kernel_dimension(adj, t)["K"], want, rtol=1e-12)


def test_heat_kernel_on_the_torus_is_the_square_lattice_s_and_overshoots_two():
    """Exact: on the infinite square lattice each direction is a walk at rate 1/2, back at its start with chance
    e^{-t/2} I_0(t/2), so K/N = [e^{-t/2} I_0(t/2)]^2 and d_s(t) = 2t [1 - I_1(t/2) / I_0(t/2)]. On the 48 x 48 torus
    the copies of the start one way round add less than 1e-16 of K for t <= 50 (I_48(25) / I_0(25) = 1.4e-17), so
    both agree to 1e-12 (measured 2e-14). The docstring's numbers: d_s overshoots to between 2.43 and 2.44 near
    t = 3.4 and comes down to 2 from above as 2 + 1/t + about 2/t^2 (2.021 at t = 50)."""
    from scipy.special import ive                                  # I_n(x) e^{-x}
    adj, _ = torus(48, 48, NO_CAP)
    t = np.geomspace(1e-3, 50, 200)
    res = heat_kernel_dimension(adj, t)
    np.testing.assert_allclose(res["K"], 48 * 48 * ive(0, t / 2) ** 2, rtol=1e-12)
    np.testing.assert_allclose(res["d_s"], 2 * t * (1 - ive(1, t / 2) / ive(0, t / 2)), rtol=1e-12)
    top = np.argmax(res["d_s"])
    assert 2.43 < res["d_s"][top] < 2.44 and 3.2 < t[top] < 3.6
    late = t >= 20
    assert ((res["d_s"][late] > 2) & (np.abs(res["d_s"][late] - (2 + 1 / t[late])) < 3 / t[late] ** 2)).all()


def test_heat_kernel_slope_agrees_with_a_finite_difference_and_a_matrix_exponential():
    """Independent checks of d_s and K: d_s against a central difference of ln K in ln t (step 1e-4, so it agrees to
    about 1e-8; asserted to 1e-6), and K against the trace of scipy's matrix exponential of Delta = -I + A/4, built by
    networkx, on a random 4-regular graph (not bipartite) and the 10 x 10 torus."""
    from scipy.linalg import expm
    graphs = (nx.random_regular_graph(4, 60, seed=9),
              nx.convert_node_labels_to_integers(nx.grid_2d_graph(10, 10, periodic=True)))
    t = np.geomspace(1e-2, 1e3, 40)
    h = 1e-4
    for g in graphs:
        n = g.number_of_nodes()
        adj = _from_networkx(g)
        res = heat_kernel_dimension(adj, t)
        up, down = heat_kernel_dimension(adj, t * np.exp(h))["K"], heat_kernel_dimension(adj, t * np.exp(-h))["K"]
        np.testing.assert_allclose(res["d_s"], -(np.log(up) - np.log(down)) / h, rtol=0, atol=1e-6)
        delta = nx.to_numpy_array(g, nodelist=range(n)) / 4 - np.eye(n)
        for s in (0.1, 3.0, 40.0):
            assert heat_kernel_dimension(adj, [s])["K"][0] == pytest.approx(np.trace(expm(s * delta)), rel=1e-10)


def test_heat_kernel_starts_at_zero_and_its_floor_is_the_number_of_pieces():
    """The two differences from the discrete walk (the docstring's statements, checked). At short t, K(0) = N and
    K'(0) = -N, so d_s / t -> 2: at t = 1e-6 it is 2(1 - t/k) to first order. At long t, K(t) falls to the number of
    connected pieces: 1 for the 6 x 6 torus, so K/N falls to 1/N where the discrete walk's floor is 2/N on this
    bipartite graph, and 2 for two separate copies."""
    adj, _ = torus(6, 6, NO_CAP)
    short = heat_kernel_dimension(adj, [1e-6])
    assert abs(short["K"][0] - 36 * np.exp(-1e-6)) < 1e-9
    assert abs(short["d_s"][0] / 1e-6 - 2) < 1e-5
    long = heat_kernel_dimension(adj, [2000.0])
    assert abs(long["K"][0] - 1) < 1e-12 and abs(long["d_s"][0]) < 1e-12
    assert spectral_dimension(adj, 2000)["floor"] == 2 / 36
    two = np.vstack([adj, adj + 36])
    assert abs(heat_kernel_dimension(two, [2000.0])["K"][0] - 2) < 1e-12
