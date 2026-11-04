"""graphity.objects: the curled ring and the flat patch, counted exactly (PREREGISTRATION T78).

Checks: against networkx on small graphs the chain has disordered (rings as 4-cycles whose links all carry three
squares; patches as subgraph monomorphisms of the 4 x 4 grid, divided by 8); and against the known counts on the flat
14 x 14 torus (0 rings, 196 patches) and the curled 16 x 4 torus (16 rings; 64 grids, since a row of four round the
short side is four distinct points).
"""
import networkx as nx
import numpy as np
import pytest
from networkx.algorithms import isomorphism

from graphity.cqg import NO_CAP, run_chain, torus
from graphity.objects import count_patches, count_rings
from graphity.squares import squares_on_edge


def to_nx(adj):
    g = nx.Graph()
    g.add_edges_from((int(u), int(v)) for u in range(adj.shape[0]) for v in adj[u])
    return g


def rings_by_networkx(adj):
    g = to_nx(adj)
    count = 0
    for cyc in nx.simple_cycles(g, length_bound=4):
        if len(cyc) != 4:
            continue
        links = [(cyc[i], cyc[(i + 1) % 4]) for i in range(4)]
        if all(squares_on_edge(adj, u, v) == 3 for u, v in links):
            count += 1
    return count


def patches_by_networkx(adj):
    g = to_nx(adj)
    grid = nx.grid_2d_graph(4, 4)
    gm = isomorphism.GraphMatcher(g, grid)
    n = sum(1 for _ in gm.subgraph_monomorphisms_iter())
    assert n % 8 == 0
    return n // 8


def test_known_tori():
    adj, _ = torus(14, 14, NO_CAP)
    assert count_rings(adj) == 0
    assert count_patches(adj) == 196
    adj, _ = torus(16, 4, NO_CAP)
    assert count_rings(adj) == 16
    assert count_patches(adj) == 64


@pytest.mark.parametrize("g,seed", [(3.0, 1), (3.433, 2), (3.697, 3), (6.0, 4)])
def test_against_networkx_on_warm_graphs(g, seed):
    adj, part = torus(8, 8, NO_CAP)
    side_u = np.flatnonzero(part == 0)
    run_chain(adj, side_u, 1.0 / g, 300, 1, seed, 1.0, NO_CAP, False)
    assert count_rings(adj) == rings_by_networkx(adj)
    assert count_patches(adj) == patches_by_networkx(adj)


def test_curled_torus_against_networkx():
    adj, _ = torus(8, 4, NO_CAP)
    assert count_rings(adj) == rings_by_networkx(adj) == 8
    assert count_patches(adj) == patches_by_networkx(adj)
