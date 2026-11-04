"""The T15 rung 0b analyser, checked against graphs whose answer is known by hand."""
import numpy as np
import pytest

from graphity.cqg import NO_CAP, torus
from scripts.analyse_t15_resonator import (cycle_adjacency, dense, eigenspaces, laplacian, localisation,
                                           loops_and_cubes, participation_ratio, state_class, verdict)

CFG = dict(localised_at_least=0.5, window=0.25, isolated_eigenvalues=[2.0, 4.0], group_tolerance=1e-9,
           controls=dict(random_sets=0, random_seed=1))


def spaces_of(a):
    return [(lam, b) for lam, b in eigenspaces(laplacian(a)) if lam > 1e-9]


def test_the_isolated_loop_of_four_has_eigenvalues_two_two_four_and_lives_on_itself():
    got = {round(lam, 9): (b.shape[1], localisation(b, [0, 1, 2, 3])) for lam, b in spaces_of(cycle_adjacency(4))}
    assert got == {2.0: (2, pytest.approx(1.0)), 4.0: (1, pytest.approx(1.0))}


def test_two_disjoint_loops_share_degenerate_eigenspaces_and_each_still_lives_on_its_own_points():
    a = np.zeros((8, 8))
    a[:4, :4] = cycle_adjacency(4)
    a[4:, 4:] = cycle_adjacency(4)
    for lam, b in spaces_of(a):
        assert b.shape[1] in (2, 4)                      # degenerate across the two loops
        assert localisation(b, list(range(8))) == pytest.approx(1.0)
        assert localisation(b, [0, 1, 2, 3]) == pytest.approx(1.0)


def test_on_a_perfect_sheet_localisation_is_bounded_by_the_eigenspace_share_and_the_gate_sheet_stays_below_half():
    """On a torus every eigenspace is translation-invariant, so its projector has d/N on the diagonal
    and the most weight any of its modes can put on |S| points is d|S|/N. An 8 x 8 sheet is small and
    symmetric enough for that to exceed 0.5, which is why the pre-registered gate is at 12 x 12."""
    adj, _ = torus(8, 8, NO_CAP)
    pts = [0, 1, 8, 9, 36, 37, 44, 45]
    for lam, b in spaces_of(dense(adj)):
        assert localisation(b, pts) <= b.shape[1] * len(pts) / 64 + 1e-9
    adj, _ = torus(12, 12, NO_CAP)
    pts = [x * 12 + y for x, y in ((0, 0), (0, 1), (1, 0), (1, 1), (6, 6), (6, 7), (7, 6), (7, 7))]
    assert max(localisation(b, pts) for lam, b in spaces_of(dense(adj))) < 0.5


def test_participation_ratio_counts_points():
    v = np.zeros(10)
    v[:4] = 0.5
    assert participation_ratio(v) == pytest.approx(4.0)


def test_loops_and_cubes_are_read_by_isomorphism_from_a_saved_state():
    z = np.load("results/t7d_lam125_n64_adj/N64_rep25.npz")
    loops, cubes = loops_and_cubes(z["adj"])
    assert len(loops) == 2 and all(len(l) == 4 for l in loops) and cubes == []


def rows(*pairs):
    return [dict(eigenvalue=lam, multiplicity=1, loc_loops=loc, loc_each_loop=[loc], loc_loops_collar=loc,
                 loc_cube=None, random_max_here=0.1, pr_best=4.0) for lam, loc in pairs]


def test_state_classes():
    assert state_class(rows((1.9, 0.9), (4.1, 0.8), (3.0, 0.1)), CFG)[0] == "ISOLATED"
    assert state_class(rows((1.9, 0.9), (3.0, 0.7)), CFG)[0] == "RETUNED"     # nothing near 4
    assert state_class(rows((1.5, 0.9), (4.0, 0.9)), CFG)[0] == "RETUNED"     # 1.5 is outside the window of 2
    assert state_class(rows((2.0, 0.4), (4.0, 0.3)), CFG)[0] == "DISSOLVED"


def summary(source, *pairs):
    return (source, rows(*pairs), dict(N=144, n_loops=2, n_cubes=0, random_max=0.2, spread_even=8 / 144))


GATES = dict(loop=True, sheet=True, random=True)


def test_all_isolated_is_resonator_at_the_isolated_frequencies():
    s = [summary("a", (2.0, 0.9), (4.0, 0.9)), summary("b", (2.1, 0.8), (3.9, 0.7))]
    assert verdict(s, GATES, CFG)[1] == "RESONATOR AT THE ISOLATED FREQUENCIES"


def test_localised_but_moved_is_retuned():
    s = [summary("a", (2.0, 0.9), (4.0, 0.9)), summary("b", (2.6, 0.8), (3.2, 0.7))]
    assert verdict(s, GATES, CFG)[1] == "RESONATOR, RETUNED"


def test_nothing_localised_anywhere_is_not_a_resonator():
    s = [summary("a", (2.0, 0.2), (4.0, 0.1)), summary("b", (2.0, 0.3), (4.0, 0.2))]
    assert verdict(s, GATES, CFG)[1] == "NOT A RESONATOR"


def test_a_mix_with_a_dissolved_state_is_inconclusive():
    s = [summary("a", (2.0, 0.9), (4.0, 0.9)), summary("b", (2.0, 0.3), (4.0, 0.2))]
    assert verdict(s, GATES, CFG)[1] == "INCONCLUSIVE"


def test_a_failed_gate_is_inconclusive_whatever_the_states_say():
    s = [summary("a", (2.0, 0.9), (4.0, 0.9))]
    lines, out = verdict(s, dict(loop=True, sheet=True, random=False), CFG)
    assert out == "INCONCLUSIVE" and any("does not discriminate" in ln for ln in lines)
