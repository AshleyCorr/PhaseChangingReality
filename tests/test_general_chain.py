"""Checks on the chain that allows triangles and pentagons (ASSUMPTIONS Q10, Q18).

What cannot be checked here, and why it is said out loud: there is no exactly known
distribution to compare the chain with. The hard-core rule leaves no valid 4-regular graph
below 14 vertices (Q8), and enumerating the states of a 14- or 16-vertex general graph is out
of reach, so the sampling has no exact reference the way `cqg` has one at N = 16 and 18 (Q9).
What is checked: the energy and validity reader against the exact reference already tested in
`test_full_curvature`, that the walk never leaves the valid states, that a seed reproduces a
run, and that the two arrangements this module was written to compare both sit still when cold.
"""
import networkx as nx

from graphity import full_curvature as fc
from graphity import general_chain as gc


def census(g):
    counts = [0, 0, 0]
    for cycle in fc.short_cycles(g):
        counts[len(cycle) - 3] += 1
    return tuple(counts)


def test_read_agrees_with_the_exact_reference():
    """One listing of the short cycles gives what `is_valid`, `energy` and a census give separately."""
    starts = [fc.square_sheet(6, 5), gc.icosidodecahedron(), fc.kagome_sheet(3)]
    hot, _, _ = gc.melt(fc.square_sheet(6, 5), 10, seed=7)           # states a run actually reaches
    starts.append(hot)
    for g in starts:
        valid, h, t, s, p = gc.read(g)
        assert valid == fc.is_valid(g)
        assert h == fc.energy(g)
        assert (t, s, p) == census(g)


def test_the_chain_never_leaves_the_valid_states():
    """4-regular, simple, hard-core rule: every recorded state, hot enough to move a lot."""
    end, rows, acceptance = gc.run(fc.square_sheet(6, 5), 40.0, 20, seed=3)
    assert acceptance > 0.01 and len(rows) == 20
    assert fc.is_valid(end) and all(d == 4 for _, d in end.degree())
    assert end.number_of_nodes() == 30 and end.number_of_edges() == 60


def test_the_same_seed_gives_the_same_run():
    a = gc.run(fc.square_sheet(6, 5), 20.0, 8, seed=11)[1]
    b = gc.run(fc.square_sheet(6, 5), 20.0, 8, seed=11)[1]
    c = gc.run(fc.square_sheet(6, 5), 20.0, 8, seed=12)[1]
    assert a == b and a != c


def test_both_zero_energy_arrangements_sit_still_when_cold():
    """The flat sheet and the closed 30-point piece tie at H = 0 and are both dips (Q10), so a cold
    chain should stay in each: this is the control for the runs that ask whether space refolds."""
    for start, expect in [(fc.square_sheet(6, 5), (0, 30, 6)), (gc.icosidodecahedron(), (20, 0, 12))]:
        assert census(start) == expect and fc.energy(start) == 0
        end, rows, _ = gc.run(start, 1.0, 20, seed=5)
        assert all(r["H"] == 0.0 for r in rows)
        assert census(end) == expect                    # nothing moved, in either arrangement
        assert nx.is_connected(end)


def test_a_sheet_that_wraps_in_an_odd_number_of_steps_carries_free_pentagons():
    """The 6 x 5 sheet has a pentagon down each of its 6 columns and they cost nothing: once an edge
    carries two squares both brackets of the curvature are already at zero, so further loops on it are
    free. Worth pinning, because it is why a 30-point sheet can be compared with the 30-point piece."""
    sheet = fc.square_sheet(6, 5)
    assert census(sheet) == (0, 30, 6) and fc.energy(sheet) == 0 == fc.energy(fc.square_sheet(6, 6))
    # still a dip, but the free pentagons halve the wall: 8 to get out, against 16 on an even sheet
    assert fc.way_out(sheet)[0] == 8 and fc.way_out(fc.square_sheet(6, 6))[0] == 16


def shares_at_most_one_edge(g):
    """The hard-core rule as `full_curvature.is_valid` applies it, without the 4-regular part,
    so that it can be asked of a small hand-built piece."""
    cycles = list(fc.short_cycles(g))
    return all(len(a & b) <= 1 for a, b in gc._pairs(cycles))


def test_a_triangle_may_not_touch_a_square_or_another_triangle():
    """Two exclusions that follow from the hard-core rule rather than from the prices, and that
    decide the floor below. A triangle and a square sharing an edge leave a pentagon that shares
    two edges with each; two triangles on one edge leave a square that does the same."""
    triangle_and_square = nx.Graph([("u", "v"), ("v", "w"), ("w", "u"),      # the triangle
                                    ("v", "b"), ("b", "a"), ("a", "u")])    # the square, on edge u-v
    assert not shares_at_most_one_edge(triangle_and_square)
    two_triangles = nx.Graph([("u", "v"), ("v", "w"), ("w", "u"), ("v", "x"), ("x", "u")])
    assert not shares_at_most_one_edge(two_triangles)
    # and the instance that matters: the cuboctahedron, whose edges each carry a triangle and a square
    assert not fc.is_valid(nx.line_graph(nx.hypercube_graph(3)))


def test_zero_is_the_floor_even_when_braces_are_allowed():
    """No valid arrangement sits below the flat sheet, so space is not metastable in this model.

    With the two exclusions above, a valid edge has T = 0, or T = 1 with S = 0. Over every such
    loading the curvature is at most zero, and H = -8 * sum(kappa) is therefore at least zero.
    Equality needs every edge to carry either two squares (the sheet) or one triangle and at
    least one pentagon (a closed braced piece). Both are realised, and nothing beats them.
    """
    best = max(t / 4 - max(0.0, 1 - (2 + t + s) / 4) - max(0.0, 1 - (2 + t + s + p) / 4)
               for t in (0, 1) for s in range(5) for p in range(5) if not (t and s))
    assert best == 0.0
    for g in (fc.square_sheet(6, 6), gc.icosidodecahedron(),
              nx.convert_node_labels_to_integers(nx.line_graph(nx.petersen_graph()))):
        assert fc.is_valid(g) and fc.energy(g) == 0


def test_the_15_point_braced_piece_is_frozen():
    """L(Petersen), 15 points, ten triangles and twelve pentagons, H = 0 like the sheet -- and not
    one valid switch exists out of it. A resting place the chain can enter and never leave."""
    piece = nx.convert_node_labels_to_integers(nx.line_graph(nx.petersen_graph()))
    assert fc.energy(piece) == 0 and fc.moves(piece) == []
