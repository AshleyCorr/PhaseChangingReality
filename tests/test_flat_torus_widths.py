"""scripts/exact_flat_torus_widths.py: the count of flat tori by width, checked against built graphs."""
import sys
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import exact_flat_torus_widths as e                               # noqa: E402
from graphity.cqg import NO_CAP, hamiltonian, is_valid            # noqa: E402
from read_endpoint_width import is_flat, twisted_tube             # noqa: E402
from read_endpoint_width import width as graph_width              # noqa: E402


def as_graph(adj):
    return nx.Graph((i, int(j)) for i in range(adj.shape[0]) for j in adj[i])


def test_every_wrapping_of_64_points_is_a_valid_flat_graph_of_the_counted_width():
    seen = e.wrappings(64)
    assert len(seen) == 47 and e.count(64) == {6: 8, 8: 31, 10: 8}
    for a, b, d, w in seen:
        adj = e.graph(a, b, d)
        g = as_graph(adj)
        assert set(dict(g.degree()).values()) == {4} and nx.is_bipartite(g) and nx.is_connected(g)
        assert is_valid(adj, NO_CAP) and is_flat(adj) and hamiltonian(adj, 1.25) == 0
        assert graph_width(adj) == w                              # the lattice's shortest wrap is the graph's width


def test_the_tube_is_not_flat_and_its_seam_shifted_by_two_is():
    assert e.width(4, 0, 16) == 4 and not is_flat(e.graph(4, 0, 16))          # the tube: a wrap of four
    assert e.width(4, 2, 16) == 4                                             # a shift the other way keeps the wrap of four
    assert e.width(32, 4, 2) == 6 and is_flat(e.graph(32, 4, 2))              # wraps (4, 2) and (0, 16): the seam shifted by two
    twisted = as_graph(twisted_tube(16, 2))
    assert nx.is_isomorphic(as_graph(e.graph(32, 4, 2)), twisted)
    six = [as_graph(e.graph(a, b, d)) for a, b, d, w in e.wrappings(64) if w == 6]
    assert len(six) == 8 and sum(1 for g in six if nx.is_isomorphic(g, twisted)) >= 1


def test_the_recorded_counts():
    """ASSUMPTIONS O119."""
    assert e.count(96) == {6: 20, 8: 32, 10: 16, 12: 40}
    c192 = e.count(192)
    assert sum(c192.values()) == 236 and c192[6] == 20 and e.median_width(c192) == 12 and max(c192) == 18
    c288, c1024, c4096 = e.count(288), e.count(1024), e.count(4096)
    assert (sum(c288.values()), c288[6]) == (387, 20)
    assert (sum(c1024.values()), c1024[6], e.median_width(c1024)) == (1007, 8, 30)
    assert (sum(c4096.values()), c4096[6], e.median_width(c4096)) == (4079, 8, 58)
