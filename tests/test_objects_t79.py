"""graphity.objects for T79: the uncurled twin (count_ring2) and flat a x b grids (count_grid), checked against
networkx (grid monomorphisms with the no-three-squares condition applied afterwards) and against the tori.
"""
import networkx as nx
import numpy as np
import pytest
from networkx.algorithms import isomorphism

from graphity.cqg import NO_CAP, run_chain, torus, total_squares
from graphity.objects import count_grid, count_patches, count_ring2, count_rings
from graphity.squares import squares_on_edge

GRIDS = [(2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (3, 5), (4, 4), (4, 5)]


def to_nx(adj):
    g = nx.Graph()
    g.add_edges_from((int(u), int(v)) for u in range(adj.shape[0]) for v in adj[u])
    return g


def grids_by_networkx(adj, a, b, flat_only):
    g = to_nx(adj)
    grid = nx.grid_2d_graph(a, b)
    gm = isomorphism.GraphMatcher(g, grid)
    n = 0
    for m in gm.subgraph_monomorphisms_iter():            # m: graph node -> grid node
        if flat_only:
            inv = {v: k for k, v in m.items()}
            if any(squares_on_edge(adj, inv[p], inv[q]) > 2 for p, q in grid.edges()):
                continue
        n += 1
    sym = 8 if a == b else 4
    assert n % sym == 0
    return n // sym


def ring2_by_networkx(adj):
    g = to_nx(adj)
    return sum(1 for cyc in nx.simple_cycles(g, length_bound=4) if len(cyc) == 4
               and all(squares_on_edge(adj, cyc[i], cyc[(i + 1) % 4]) == 2 for i in range(4)))


def test_tori():
    adj, _ = torus(14, 14, NO_CAP)
    assert count_ring2(adj) == 196 and count_rings(adj) == 0
    for a, b in GRIDS:                                 # a rectangle lies along either lattice axis: two orientations
        expect = 196 if a == b else 392
        assert count_grid(adj, a, b, True) == expect
        assert count_grid(adj, a, b, False) == expect
    assert count_grid(adj, 4, 4, False) == count_patches(adj) == 196
    assert count_grid(adj, 2, 2, False) == total_squares(adj)
    adj, _ = torus(16, 4, NO_CAP)
    assert count_ring2(adj) == 0 and count_rings(adj) == 16
    assert count_grid(adj, 4, 4, False) == count_patches(adj) == 64
    assert count_grid(adj, 4, 4, True) == 0          # every grid there uses a three-square link


@pytest.mark.parametrize("g,seed", [(3.0, 11), (3.433, 12), (3.697, 13)])
def test_against_networkx_on_warm_graphs(g, seed):
    adj, part = torus(8, 8, NO_CAP)
    side_u = np.flatnonzero(part == 0)
    run_chain(adj, side_u, 1.0 / g, 300, 1, seed, 1.0, NO_CAP, False)
    assert count_ring2(adj) == ring2_by_networkx(adj)
    assert count_grid(adj, 2, 2, False) == total_squares(adj)
    for a, b in [(2, 2), (2, 3), (2, 4), (3, 3), (3, 4)]:
        for flat_only in (False, True):
            assert count_grid(adj, a, b, flat_only) == grids_by_networkx(adj, a, b, flat_only), (a, b, flat_only)
    assert count_grid(adj, 4, 4, False) == count_patches(adj)
