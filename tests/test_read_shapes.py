"""The two exploratory shape readers of 11 October 2026 (scripts/read_open_shells.py, scripts/read_endpoint_width.py):
the shells and the width on tori whose answers are known, and on one saved end state."""
import sys
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from read_endpoint_width import is_flat, twisted_tube, width      # noqa: E402
from read_open_shells import profile, shells, torus               # noqa: E402


def graph(adj):
    return nx.Graph((i, int(j)) for i in range(adj.shape[0]) for j in adj[i])


def squares_on_each_edge(adj):
    """Brute force with networkx: for every edge, the number of squares through it."""
    g = graph(adj)
    return {sum(1 for a in g[u] if a != v for b in g[v] if b not in (u, a) and g.has_edge(a, b)) for u, v in g.edges()}


def test_shells_of_known_tori():
    assert shells(torus((4, 4, 18)), 0) == [1, 6, 16, 26, 31, 32, 32, 32, 32, 31, 26, 16, 6, 1]
    assert shells(torus((6, 6, 8)), 0) == [1, 6, 18, 36, 53, 60, 53, 36, 18, 6, 1]
    for dims in ((4, 12), (6, 16), (4, 8, 8)):
        adj = torus(dims)
        counted = shells(adj, 5)
        assert sum(counted) == adj.shape[0]
        by_networkx = nx.single_source_shortest_path_length(graph(adj), 5)
        assert counted == [list(by_networkx.values()).count(r) for r in range(len(counted))]


def test_a_rod_and_flat_space_are_told_apart_by_their_largest_shell():
    rod, flat = profile(torus((4, 4, 18))), profile(torus((6, 6, 8)))
    assert rod[0] == 32 and flat[0] == 60                     # twice the rod's cross-section; flat space's widest shell
    assert rod[1] == 13 and flat[1] == 10                     # the rod's far end is farther away


def test_width_is_the_shortest_way_round():
    for dims, expected in (((4, 12), 4), ((6, 16), 6), ((8, 8), 8), ((8, 12), 8), ((12, 16), 12)):
        adj = torus(dims)
        assert width(adj) == expected
        assert is_flat(adj) == (min(dims) >= 6)


def test_the_count_from_inside_falls_short_at_half_the_way_round():
    """The table of paper 1's plain-language edition, Chapter 3: the points within r steps of a point, on a grid
    that wraps both ways, match the endless grid's 2r^2 + 2r + 1 until r is half the short way round."""
    endless = [2 * r * r + 2 * r + 1 for r in range(7)]
    assert [endless[r] for r in (1, 2, 3, 4, 6)] == [5, 13, 25, 41, 85]
    for round_, expected in ((4, [5, 12, 20, 28, 44]), (8, [5, 13, 25, 40, 72]), (12, [5, 13, 25, 41, 84])):
        adj = torus((round_, 48))
        within = np.cumsum(shells(adj, 0))
        assert [int(within[r]) for r in (1, 2, 3, 4, 6)] == expected
        by_networkx = nx.single_source_shortest_path_length(graph(adj), 0)
        assert [sum(1 for d in by_networkx.values() if d <= r) for r in (1, 2, 3, 4, 6)] == expected
        first_short = next(r for r in range(7) if within[r] < endless[r])
        assert first_short == round_ // 2 and width(adj) == round_
    # the saved end states six round have their seam shifted (the next test), and count 22 within three steps
    assert int(np.cumsum(shells(twisted_tube(48, 2), 0))[3]) == 22


def test_the_tube_with_its_seam_shifted_by_two_is_flat_and_six_round():
    for length in (16, 24, 48):
        tube, twisted = twisted_tube(length, 0), twisted_tube(length, 2)
        assert nx.is_isomorphic(graph(tube), graph(torus((4, length))))
        assert squares_on_each_edge(tube) == {2, 3} and not is_flat(tube) and width(tube) == 4
        assert squares_on_each_edge(twisted) == {2} and is_flat(twisted) and width(twisted) == 6
        assert nx.is_bipartite(graph(twisted)) and set(dict(graph(twisted).degree()).values()) == {4}
        assert shells(twisted, 0)[:6] == [1, 4, 8, 9, 8, 8]   # eight points at each distance: still a tube


def test_a_saved_flat_end_state_is_that_twisted_tube():
    """ASSUMPTIONS O116: one saved end state of T60 (lambda = 1.05, N = 64), read from its wiring."""
    saved = np.load(ROOT / "results" / "t60_lam105_n64_adj" / "N64_rep0.npz")["adj"]
    assert is_flat(saved) and width(saved) == 6
    assert nx.is_isomorphic(graph(saved), graph(twisted_tube(16, 2)))


def test_a_flat_piece_is_a_torus_or_a_klein_bottle():
    """scripts/read_flat_kind.py: two-sided and one-sided flat surfaces built for the purpose are told apart."""
    from read_flat_kind import KLEIN, TORUS, describe, kind, klein_bottle, pieces
    for adj in (torus((8, 8)), torus((6, 16)), twisted_tube(24, 2), twisted_tube(24, 4)):
        assert is_flat(adj) and kind(adj) == TORUS
        assert len({width(adj, s) for s in range(adj.shape[0])}) == 1          # a torus looks the same from every point
    for around, along in ((8, 8), (6, 16), (8, 12)):
        bottle = klein_bottle(around, along)
        assert squares_on_each_edge(bottle) == {2} and is_flat(bottle)          # zero energy at every lambda
        assert nx.is_bipartite(graph(bottle)) and set(dict(graph(bottle).degree()).values()) == {4}
        assert nx.is_connected(graph(bottle)) and kind(bottle) == KLEIN
    # independent of the gluing argument: a one-sided flat surface does not look the same from every point
    assert {width(klein_bottle(6, 16), s) for s in range(96)} == {6, 8, 10, 12}
    assert kind(twisted_tube(24, 0)).startswith("not flat")                     # the tube itself: links with three squares
    # two separate pieces in one graph
    a, b = torus((6, 6)), klein_bottle(6, 8)
    both = np.vstack([a, b + a.shape[0]])
    assert [p.shape[0] for p in pieces(both)] == [36, 48]
    assert describe(both) == ((36, TORUS, 6), (48, KLEIN, 6))


def test_a_saved_flat_end_state_is_a_torus_beside_a_klein_bottle():
    """One end state of T61's stage 1 (lambda = 1.40, N = 96, two stores per point, replica 6) has H = 0 and is two
    pieces: a torus six around and a one-sided piece. Checked against networkx for the pieces."""
    from read_flat_kind import KLEIN, TORUS, describe, pieces
    saved = np.load(ROOT / "results" / "t61_lam140_n96_adj" / "N96_C192_rep6.npz")["adj"]
    assert is_flat(saved)
    assert sorted(len(c) for c in nx.connected_components(graph(saved))) == [40, 56]
    parts = describe(saved)
    assert [(n, k) for n, k, _ in parts] == [(40, KLEIN), (56, TORUS)] and parts[1][2] == 6
    bottle = [p for p in pieces(saved) if p.shape[0] == 40][0]
    assert {width(bottle, s) for s in range(40)} == {6, 8}
    # a second way: the one-sided piece is none of the flat tori of 40 points, and the other piece is one of 56
    import exact_flat_torus_widths as flat_tori
    other = [p for p in pieces(saved) if p.shape[0] == 56][0]
    assert not any(nx.is_isomorphic(graph(bottle), graph(flat_tori.graph(a, b, d))) for a, b, d, _ in flat_tori.wrappings(40))
    assert any(nx.is_isomorphic(graph(other), graph(flat_tori.graph(a, b, d))) for a, b, d, _ in flat_tori.wrappings(56))
