"""The T15 rung 0 analyser, checked against graphs whose answer is known by hand.

The project's rule for a verdict script: build inputs where the right answer is known in advance
and check the script returns it, both when a prediction should hold and when it should fail. The
counting helper is also checked against the named tool, `log_automorphisms`, on a sheet.
"""
from math import exp

import networkx as nx
import numpy as np
import pytest

from graphity.cqg import NO_CAP, torus
from graphity.small_graphs import log_automorphisms, sides_first
from scripts.analyse_t15_rung0 import (from_row, graph_of, laplacian_spectrum, read, renamings,
                                       spectra_differ, to_row, verdict)

C4 = [0.0, 2.0, 2.0, 4.0]                                   # a closed loop of four
P4 = [0.0, 2 - 2 ** 0.5, 2.0, 2 + 2 ** 0.5]                 # an open line of four


def sided(edges, side):
    g = nx.Graph()
    g.add_nodes_from((v, dict(side=s)) for v, s in side.items())
    g.add_edges_from(edges)
    return g


def four_cycle():
    return sided([(0, 1), (1, 2), (2, 3), (3, 0)], {0: 0, 2: 0, 1: 1, 3: 1})


# ------------------------------------------------------------------------------ the counting

def test_a_four_cycle_has_four_renamings_within_sides_and_eight_in_all():
    assert renamings(four_cycle()) == 4
    assert renamings(four_cycle(), same_side=False) == 8


def test_a_line_of_four_is_rigid_within_sides():
    """Reversing the line swaps the ends across sides, so only the identity survives the rule."""
    p4 = sided([(0, 1), (1, 2), (2, 3)], {0: 0, 2: 0, 1: 1, 3: 1})
    assert renamings(p4) == 1
    assert renamings(p4, same_side=False) == 2


def test_two_identical_pieces_may_be_swapped():
    two = sided([(0, 1), (2, 3)], {0: 0, 2: 0, 1: 1, 3: 1})
    assert renamings(two) == 2                       # swap the edges; an edge's ends differ in side
    assert renamings(two, same_side=False) == 8      # 2 per edge, times 2! for the swap


def test_the_empty_graph_has_one_renaming():
    assert renamings(nx.Graph()) == 1


def test_the_helper_agrees_with_the_named_tool_on_a_sheet():
    adj, part = torus(8, 8, NO_CAP)
    sheet = sides_first(adj, part)
    side = (np.arange(64) >= 32).astype(int)
    assert renamings(graph_of(sheet, side)) == round(exp(log_automorphisms(sheet)))


def test_read_of_a_perfect_sheet_finds_no_defect_and_the_tools_number():
    adj, part = torus(8, 8, NO_CAP)
    got = read(adj, part)
    assert got["defect_vertices"] == 0 and got["defect"] == 1 and got["pieces"] == []
    assert got["whole"] == round(exp(log_automorphisms(sides_first(adj, part))))
    back = from_row({k: str(v) for k, v in to_row("sheet", "torus(8,8)", got).items()})
    assert back["whole"] == got["whole"] and back["defect_vertices"] == 0 and back["pieces"] == []


# ------------------------------------------------------------------------------- the spectra

def test_laplacian_of_a_closed_loop_of_four():
    """VISION Update 17: frequencies sqrt 2 twice and 2, i.e. eigenvalues 2, 2, 4."""
    assert laplacian_spectrum(four_cycle()) == pytest.approx(C4)


def test_spectra_differ_by_the_registered_rule():
    assert spectra_differ(C4, P4)
    assert not spectra_differ(C4, [0.0, 2.0, 2.0 + 1e-12, 4.0])      # inside the tolerance
    assert spectra_differ([0.0, 2.0], [0.0, 2.0, 2.0])                # length counts
    assert not spectra_differ([0.0, 0.0, 2.0], [0.0, 2.0])            # zeros do not


# ------------------------------------------------------------------------------- the verdicts

def leftover(source, whole, defect, pieces, defect_vertices=4):
    return dict(object="leftover", source=source, N=64, defect_vertices=defect_vertices,
                whole=whole, defect=defect, pieces=pieces)


def melt(n=64, whole=1):
    return dict(object="melt", source="m", N=n, defect_vertices=n, whole=whole, defect=1, pieces=[])


def test_all_three_holding_is_versions_exist_and_label_type():
    rows = [leftover("a", 2, 4, [("remnant", C4)]), leftover("b", 4, 2, [("twist", P4)]),
            melt(64), melt(96)]
    assert verdict(rows)[1] == "VERSIONS EXIST AND LABEL TYPE"


def test_a_spectrum_shared_across_types_is_versions_exist():
    rows = [leftover("a", 2, 4, [("remnant", C4)]), leftover("b", 4, 2, [("twist", C4)]), melt()]
    lines, out = verdict(rows)
    assert out == "VERSIONS EXIST"
    assert any("CLASH" in ln for ln in lines)


def test_one_type_only_cannot_evaluate_prediction_three():
    rows = [leftover("a", 2, 4, [("remnant", C4)]), leftover("b", 2, 4, [("remnant", C4)]), melt()]
    lines, out = verdict(rows)
    assert out == "VERSIONS EXIST"
    assert any("NOT EVALUABLE" in ln for ln in lines)


def test_every_remnant_rigid_is_no_versions_in_anything_real():
    rows = [leftover("a", 1, 4, [("remnant", C4)]), leftover("b", 1, 1, [("twist", P4)]), melt()]
    assert verdict(rows)[1] == "NO VERSIONS IN ANYTHING REAL"


def test_rigid_as_a_whole_but_not_in_isolation_is_still_no_versions():
    """The whole-arrangement count is the claim as written; the isolated count is reported beside."""
    rows = [leftover("a", 1, 4, [("remnant", C4)]), leftover("b", 1, 4, [("twist", P4)]), melt()]
    lines, out = verdict(rows)
    assert out == "NO VERSIONS IN ANYTHING REAL"
    assert any("defect only > 1 in 2 of 2" in ln for ln in lines)


def test_a_mix_of_rigid_and_not_is_inconclusive():
    rows = [leftover("a", 1, 4, [("remnant", C4)]), leftover("b", 2, 2, [("twist", P4)]), melt()]
    assert verdict(rows)[1] == "INCONCLUSIVE"


def test_a_melt_with_a_symmetry_fails_prediction_two():
    rows = [leftover("a", 2, 4, [("remnant", C4)]), leftover("b", 4, 2, [("twist", P4)]),
            melt(64), melt(96, whole=2)]
    lines, out = verdict(rows)
    assert out == "INCONCLUSIVE"
    assert any("Prediction 2" in ln and "FAILS" in ln for ln in lines)


def test_a_leftover_without_a_defect_takes_no_part():
    rows = [leftover("a", 2, 4, [("remnant", C4)]), leftover("b", 4, 2, [("twist", P4)]),
            leftover("flat", 128, 1, [], defect_vertices=0), melt()]
    lines, out = verdict(rows)
    assert out == "VERSIONS EXIST AND LABEL TYPE"
    assert any("no defect at all" in ln and "flat" in ln for ln in lines)


def test_one_name_with_two_spectra_is_said():
    rows = [leftover("a", 2, 4, [("unnamed: 8 at d=1", C4)]),
            leftover("b", 2, 4, [("unnamed: 8 at d=1", P4)]),
            leftover("c", 2, 4, [("remnant", [0.0, 1.0, 3.0])]), melt()]
    lines, out = verdict(rows)
    assert out == "VERSIONS EXIST AND LABEL TYPE"
    assert any("one name, 2 different spectra" in ln for ln in lines)
