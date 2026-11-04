"""The full energy with triangles and pentagons allowed (braces), on small graphs, exactly."""
import networkx as nx
import numpy as np
import pytest

from graphity import full_curvature as fc
from graphity.cqg import CAP, NO_CAP, hamiltonian, run_chain, torus


def cycle_census(g):
    cycles = fc.short_cycles(g)
    return tuple(sum(len(c) == k for c in cycles) for k in (3, 4, 5))


@pytest.mark.parametrize("cap, lam, inv_g, seed", [(CAP, 1.0, 0.0, 1), (NO_CAP, 1.0, 0.15, 2), (NO_CAP, 0.0, 0.2, 3)])
def test_general_formula_agrees_with_the_kernel_on_two_sided_graphs(cap, lam, inv_g, seed):
    """On a bipartite graph there are no triangles or pentagons, and the general energy must be the kernel's
    H at lam = 1, whatever state the chain has reached."""
    adj, part = torus(6, 6, cap)
    run_chain(adj, np.flatnonzero(part == 0), inv_g, 60, 1, seed, lam, cap, False)
    g = nx.Graph((u, int(v)) for u in range(len(adj)) for v in adj[u])
    assert fc.is_valid(g) and cycle_census(g)[0] == cycle_census(g)[2] == 0
    assert fc.energy(g) == hamiltonian(adj, 1.0)


def test_flat_sheet_stays_a_dip_when_braces_are_allowed():
    sheet = fc.square_sheet(6, 6)
    assert fc.is_valid(sheet) and cycle_census(sheet) == (0, 36, 0) and fc.energy(sheet) == 0
    cheapest, n_moves = fc.way_out(sheet)
    assert cheapest == 16 and n_moves > 500


def test_braced_sheet_is_higher_and_is_not_a_dip():
    """Kagome: every edge in one triangle, no squares, no pentagons. 4 per vertex above the flat sheet at the
    published prices, and one move (which puts a pentagon beside two triangles) already lowers it by 20."""
    braced = fc.kagome_sheet(3)
    assert fc.is_valid(braced) and cycle_census(braced) == (18, 0, 0)
    assert fc.energy(braced) == 4 * braced.number_of_nodes()
    cheapest, _ = fc.way_out(braced)
    assert cheapest == -20


def test_braced_closed_piece_ties_with_the_flat_sheet():
    """The 30-vertex piece with a triangle and a pentagon on every edge (the icosidodecahedron) has zero
    curvature on every edge, so H = 0 like the flat sheet, and it is a dip. Braces allowed, the flat sheet is
    therefore not the only lowest state of the published energy, and this one is a closed piece, not a space."""
    piece = nx.convert_node_labels_to_integers(nx.line_graph(nx.dodecahedral_graph()))
    assert fc.is_valid(piece) and cycle_census(piece) == (20, 0, 12)
    assert fc.energy(piece) == 0
    cheapest, _ = fc.way_out(piece)
    assert cheapest == 20
