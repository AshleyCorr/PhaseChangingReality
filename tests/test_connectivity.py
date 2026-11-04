"""T3: connected pieces, the largest piece, baby universes, and the 4-cubes among them."""
import networkx as nx
import numpy as np

from graphity.connectivity import component_labels, connectivity, is_four_cube
from graphity.cqg import NO_CAP, _switch, hamiltonian, has_edge, is_valid, run_chain, squares_on_edge, torus

FANO_LINES = [{0, 1, 3}, {1, 2, 4}, {2, 3, 5}, {3, 4, 6}, {4, 5, 0}, {5, 6, 1}, {6, 0, 2}]


def to_nx(adj):
    g = nx.Graph()
    g.add_nodes_from(range(len(adj)))
    g.add_edges_from((u, int(v)) for u in range(len(adj)) for v in adj[u])
    return g


def side_by_side(*parts):
    """Disjoint union of (adj, part) pairs: the later graphs' vertex numbers are shifted up."""
    adjs, sides, offset = [], [], 0
    for adj, part in parts:
        adjs.append(adj + offset)
        sides.append(part)
        offset += len(adj)
    return np.vstack(adjs), np.concatenate(sides)


def biplane():
    """Seven points, seven blocks of four (the complements of the Fano lines); a point is joined to its blocks.
    Any two points lie in exactly two blocks, which is what puts three squares on every edge."""
    blocks = [sorted(set(range(7)) - line) for line in FANO_LINES]
    adj = np.empty((14, 4), dtype=np.int64)
    for b, block in enumerate(blocks):
        adj[7 + b] = block
    for p in range(7):
        adj[p] = [7 + b for b, block in enumerate(blocks) if p in block]
    return adj, np.array([0] * 7 + [1] * 7)


def brute_connectivity(adj):
    """The same four numbers from networkx, with a true isomorphism test for the cubes."""
    g = to_nx(adj)
    pieces = [g.subgraph(c) for c in nx.connected_components(g)]
    babies = [p for p in pieces if all(squares_on_edge(adj, u, v) == 3 for u, v in p.edges)]
    cubes = sum(len(p) == 16 and nx.is_isomorphic(p, nx.hypercube_graph(4)) for p in babies)
    return len(pieces), max(len(p) for p in pieces), sum(len(p) for p in babies), cubes


def test_one_torus():
    adj, _ = torus(6, 8)
    assert connectivity(adj) == (1, 48, 0, 0)
    assert set(component_labels(adj)) == {0}


def test_two_disjoint_tori():
    adj, _ = side_by_side(torus(6), torus(8, 6))
    assert connectivity(adj) == (2, 48, 0, 0)
    assert list(component_labels(adj)) == [0] * 36 + [1] * 48


def test_torus_plus_a_four_cube():
    adj, _ = side_by_side(torus(6), torus(4, cap=NO_CAP))          # the 4 x 4 torus is the 4-cube (Q1)
    assert connectivity(adj) == brute_connectivity(adj) == (2, 36, 16, 1)


def test_three_four_cubes():
    cube = torus(4, cap=NO_CAP)
    adj, _ = side_by_side(cube, cube, cube)
    assert connectivity(adj) == (3, 16, 48, 3)


def test_the_four_cube_is_not_the_only_baby_universe():
    """Q8: the 14-vertex biplane graph is a valid state with three squares on every edge, so it has
    the 4-cube's energy per vertex at every lam. It counts as a baby universe and not as a cube."""
    adj, _ = biplane()
    g = to_nx(adj)
    assert nx.is_connected(g) and nx.is_bipartite(g) and all(d == 4 for _, d in g.degree())
    assert is_valid(adj, NO_CAP)
    assert {squares_on_edge(adj, u, v) for u, v in g.edges} == {3}
    cube, _ = torus(4, cap=NO_CAP)
    for lam in (0.0, 0.5, 1.0):
        assert hamiltonian(adj, lam) / 14 == hamiltonian(cube, lam) / 16 == -8 * (1 - lam)
    assert connectivity(adj) == brute_connectivity(adj) == (1, 14, 14, 0)
    both, _ = side_by_side(biplane(), torus(4, cap=NO_CAP), torus(6))
    assert connectivity(both) == (3, 36, 30, 1)


def test_sixteen_vertices_are_not_enough_to_be_a_cube():
    """One edge switch inside a 4-cube leaves 16 vertices, 4-regular, bipartite, connected: not Q4."""
    adj, _ = torus(4, cap=NO_CAP)
    assert is_four_cube(adj, 0) and is_four_cube(adj, 11)          # any vertex of the piece will do
    u1, u2 = 0, 5                                                  # same side: (0,0) and (1,1)
    v1 = next(int(v) for v in adj[u1] if not has_edge(adj, u2, v))
    v2 = next(int(v) for v in adj[u2] if not has_edge(adj, u1, v))
    _switch(adj, u1, v1, u2, v2)
    g = to_nx(adj)
    assert nx.is_connected(g) and nx.is_bipartite(g) and all(d == 4 for _, d in g.degree())
    assert not nx.is_isomorphic(g, nx.hypercube_graph(4))
    assert not is_four_cube(adj, 0)
    assert connectivity(adj) == (1, 16, 0, 0)


def test_agrees_with_networkx_on_shattered_graphs():
    """lam = 0 at low coupling is where pieces break off; compare every count with networkx."""
    shattered = cubes = 0
    for seed in range(4):
        adj, part = torus(8, cap=NO_CAP)
        side_u = np.flatnonzero(part == 0)
        run_chain(adj, side_u, 0.0, 30, 1, seed, 0.0, NO_CAP, False)               # melt
        for inv_g in (0.10, 0.14, 0.18, 0.25):                                       # then cool in steps
            run_chain(adj, side_u, inv_g, 150, 1, 100 + seed, 0.0, NO_CAP, False)
            assert connectivity(adj) == brute_connectivity(adj)
            shattered += connectivity(adj)[0] > 1
            cubes += connectivity(adj)[3]
    assert shattered > 0 and cubes > 0                        # the test did see pieces, and 4-cubes among them


def test_looking_does_not_change_the_chain():
    """The conn argument fills in the four numbers and leaves S, X and the graph exactly as they were."""
    runs = []
    for look in (False, True):
        adj, part = torus(8, cap=NO_CAP)
        side_u = np.flatnonzero(part == 0)
        run_chain(adj, side_u, 0.0, 30, 1, 3, 0.0, NO_CAP, False)
        conn = np.zeros((40, 4), dtype=np.int64) if look else None
        s, x, acc = run_chain(adj, side_u, 0.16, 100, 40, 4, 0.0, NO_CAP, False, conn)
        runs.append((list(s), list(x), acc, adj.copy()))
    assert runs[0][:3] == runs[1][:3] and (runs[0][3] == runs[1][3]).all()
    assert tuple(conn[-1]) == connectivity(adj)
    assert (conn[:, 0] >= 1).all() and (conn[:, 1] <= 64).all()
    assert (16 * conn[:, 3] <= conn[:, 2]).all() and (conn[:, 2] <= 64).all()
