"""T4: every state at the smallest sizes, whether the switch joins them, and the chain against exact averages."""
from itertools import combinations
from math import exp, factorial

import numpy as np
import pytest

from graphity.cqg import CAP, NO_CAP, run_chain, total_squares, torus
from graphity.small_graphs import (check, circulant, count_labelled_states, explore, one_switch_away,
                                   sides_first)


def brute_force_labelled(n):
    """Independent count: every table with increasing rows, no row fixed, then times n! orderings of the rows.
    Shares nothing with small_graphs.py, in particular not its 'every row pattern is equally likely' step."""
    patterns = sorted(sum(1 << c for c in cols) for cols in combinations(range(n), 4))
    found = 0

    def extend(rows, start):
        nonlocal found
        if len(rows) == n:
            cols = [sum(1 << r for r in range(n) if rows[r] >> c & 1) for c in range(n)]
            if all(bin(c).count("1") == 4 for c in cols) and all(
                    bin(a & b).count("1") <= 2 for a, b in combinations(cols, 2)):
                found += 1
            return
        for i in range(start, len(patterns)):
            if all(bin(patterns[i] & r).count("1") <= 2 for r in rows):
                extend(rows + [patterns[i]], i + 1)

    extend([], 0)
    return found * factorial(n)


@pytest.fixture(scope="module")
def sixteen():
    """The full report for N = 16, worked out once for the whole file."""
    return check(8)


def test_no_states_below_fourteen_vertices():
    for n in (4, 5, 6):
        assert count_labelled_states(n) == (0, None)


def test_counting_formula_against_brute_force():
    assert count_labelled_states(7)[0] == brute_force_labelled(7) == factorial(7) ** 2 // 168 == 151_200


def test_fourteen_vertices_one_class_and_the_chain_cannot_move():
    """N = 14: the biplane graph and nothing else (Q8). No switch from it is valid, so the chain is frozen,
    and that is harmless: there is one class, and our observables cannot tell its labelled copies apart."""
    report = check(7)
    assert report["ergodic"] and report["labelled"] == report["reached"] == 151_200
    (only,) = report["classes"]
    assert (only["squares"], only["in_babies"], only["cubes"], only["symmetries"]) == (21, 14, 0, 168)
    assert only["neighbouring_classes"] == 0
    _, start = count_labelled_states(7)
    assert one_switch_away(start) == []
    part = np.array([0] * 7 + [1] * 7)
    s, x, acc = run_chain(start.copy(), np.flatnonzero(part == 0), 0.0, 0, 50, 1, 1.0, NO_CAP, False)
    assert acc == 0.0 and set(s) == {21}


def test_sixteen_vertices_five_classes_all_joined(sixteen):
    """N = 16: 635 040 000 labelled states in five classes, and the classes reached by switches hold all of them."""
    report = sixteen
    assert report["labelled"] == 635_040_000
    assert report["ergodic"] and report["reached"] == report["labelled"]
    classes = report["classes"]
    assert sorted(c["squares"] for c in classes) == [20, 20, 21, 22, 24]
    (cube,) = [c for c in classes if c["cubes"] == 1]
    assert (cube["squares"], cube["symmetries"], cube["neighbouring_classes"]) == (24, 192, 1)
    assert all(c["neighbouring_classes"] >= 1 and not c["valid_under_cap"] for c in classes)
    assert count_labelled_states(8, CAP) == (0, None)             # the capped model has no states this small


def exact_mean_squares(classes, lam, inv_g):
    """<S> over ALL labelled states, weight exp(-H/g), H = 16 (N - S) + 4 lam X: a finite sum, no sampling."""
    n_vertices = 16
    weights = [c["labelled_states"] * exp(-inv_g * (16 * (n_vertices - c["squares"]) + 4 * lam * c["surplus"]))
               for c in classes]
    return sum(w * c["squares"] for w, c in zip(weights, classes)) / sum(weights)


@pytest.mark.parametrize("lam, inv_g", [(0.0, 0.0), (0.0, 0.08), (0.5, 0.06), (1.0, 0.08)])
def test_chain_matches_the_exact_average_at_sixteen_vertices(sixteen, lam, inv_g):
    """The strongest check we have of the sampler: at N = 16 the right answer is known exactly. It tests the
    proposal, both halves of the energy, the acceptance rule and ergodicity together."""
    classes = sixteen["classes"]
    exact = exact_mean_squares(classes, lam, inv_g)
    if lam == 1.0:
        assert exact == pytest.approx(20.8)                       # every state has H = 0 there: plain counting
    means = []
    for seed in range(12):
        adj, part = torus(4, cap=NO_CAP)                          # the 4-cube
        s, _, _ = run_chain(adj, np.flatnonzero(part == 0), inv_g, 300, 4000, 100 + seed, lam, NO_CAP, seed % 2 == 1)
        means.append(s.mean())
    mean, err = np.mean(means), np.std(means, ddof=1) / np.sqrt(len(means))
    assert err < 0.03
    assert abs(mean - exact) < 5 * err, (mean, err, exact)


def test_explore_finds_the_same_classes_from_any_start():
    _, start = count_labelled_states(8)
    cube, part = torus(4, cap=NO_CAP)
    with pytest.raises(ValueError):                               # the torus numbers its sides like a chessboard
        explore(cube)
    for begin in (start, sides_first(cube, part)):
        classes, joined = explore(begin)
        assert sorted(int(total_squares(adj)) for adj, _ in classes.reps) == [20, 20, 21, 22, 24]
        assert all(joined[k] for k in range(5))


@pytest.mark.parametrize("lam, inv_g", [(0.0, 0.10), (1.0, 0.10), (1.0, 0.25)])
def test_chain_matches_the_exact_average_at_eighteen_vertices(lam, inv_g):
    """As at N = 16, but here the full energy (lam = 1) is not zero on every state, so the local term is tested
    against an exact answer too. The 26 classes come from the recorded exhaustive run, not from a fresh one."""
    import csv
    from pathlib import Path
    from graphity.cqg import is_valid
    table = Path(__file__).resolve().parents[1] / "results" / "ergodicity_small.csv"
    with table.open(newline="") as fh:
        classes = [r for r in csv.DictReader(fh) if r["N"] == "18" and r["cap"] == "none"]
    assert len(classes) == 26 and {r["all_joined"] for r in classes} == {"True"}
    energy = [16 * (18 - int(r["squares"])) + 4 * lam * int(r["surplus"]) for r in classes]
    assert lam == 0.0 or len(set(energy)) > 1                     # the point of going to N = 18
    weights = [int(r["labelled_states"]) * exp(-inv_g * h) for r, h in zip(classes, energy)]
    exact = sum(w * int(r["squares"]) for w, r in zip(weights, classes)) / sum(weights)
    means = []
    for seed in range(12):
        adj = circulant(9)
        assert is_valid(adj, NO_CAP)
        s, _, _ = run_chain(adj, np.arange(9), inv_g, 300, 4000, 300 + seed, lam, NO_CAP, seed % 2 == 1)
        means.append(s.mean())
    mean, err = np.mean(means), np.std(means, ddof=1) / np.sqrt(len(means))
    assert err < 0.03
    assert abs(mean - exact) < 5 * err, (mean, err, exact)


def test_sheet_tube_cube_ladder():
    """VISION Update 7: curling a side of the torus to length 4 adds a quarter of a square per vertex, and the energy
    per vertex is -4 (1 - lam) for every curled side. Large dimensions: sheet 2, tube 1, cube 0."""
    from graphity.cqg import hamiltonian
    for shape, curled in (((8, 8), 0), ((8, 4), 1), ((4, 4), 2)):
        adj, _ = torus(shape[0], shape[1], NO_CAP)
        n = len(adj)
        assert total_squares(adj) / n == 1 + 0.25 * curled
        for lam in (0.0, 0.5, 1.0, 1.5):
            assert hamiltonian(adj, lam) / n == -4 * (1 - lam) * curled


def test_which_arrangements_are_dips():
    """A dip: every valid single switch raises H. The sheet is one for every lam > 0 (cheapest way out 16 lam); the
    4-cube is one only for lam < 1 (cheapest way out 32 (1 - lam)). So for 0 < lam < 1 both are dips and the cube is
    the lower one, and for lam > 1 the cube is no dip at all and cannot be a stable-for-now state above the sheet."""
    from graphity.cqg import hamiltonian
    sheet = sides_first(*torus(6, 6, NO_CAP))
    cube = sides_first(*torus(4, 4, NO_CAP))
    for adj, cheapest in ((sheet, lambda lam: 16 * lam), (cube, lambda lam: 32 * (1 - lam))):
        around = one_switch_away(adj)
        assert len(around) > 40
        for lam in (0.0, 0.5, 0.9, 1.0, 1.1, 1.5):
            way_out = min(hamiltonian(b, lam) - hamiltonian(adj, lam) for b in around)
            assert way_out == pytest.approx(cheapest(lam))


def test_counting_renamings_matches_known_groups_and_factorises_over_pieces():
    """T10's symmetry count, against two groups whose order is known, and its factorisation.

    The 4-cube's full automorphism group has order 2^4 * 4! = 384; half of those swap the two
    sides, so the side-preserving count this project uses is 192 -- which is also what the
    exhaustive enumeration independently records for that class. The 16x10 torus has 16*10
    translations times 2 reflections = 320 side-preserving renamings.
    """
    from math import exp, isclose, lgamma, log

    import numpy as np

    from graphity.cqg import NO_CAP, torus
    from graphity.small_graphs import log_automorphisms, sides_first

    cube, cube_part = torus(4, 4, cap=NO_CAP)
    one = sides_first(cube, cube_part)
    assert isclose(exp(log_automorphisms(one)), 192.0, rel_tol=1e-9)

    sheet, sheet_part = torus(16, 10, cap=NO_CAP)
    assert isclose(exp(log_automorphisms(sides_first(sheet, sheet_part))), 320.0, rel_tol=1e-9)

    for k in (2, 3):
        adj = np.concatenate([cube + 16 * i for i in range(k)])
        part = np.concatenate([cube_part] * k)
        want = k * log(192.0) + lgamma(k + 1)          # A^k * k! for k copies of one shape
        assert isclose(log_automorphisms(sides_first(adj, part)), want, rel_tol=1e-9)


def test_a_melted_graph_has_no_renamings_at_all():
    """Almost every graph is rigid, and that is why the two ways of counting agree for one."""
    import numpy as np

    from graphity.cqg import NO_CAP, run_chain, torus
    from graphity.small_graphs import log_automorphisms, sides_first

    adj, part = torus(6, 6, cap=NO_CAP)
    run_chain(adj, np.flatnonzero(part == 0), 0.0, 200, 1, 77, 1.0, NO_CAP)
    assert log_automorphisms(sides_first(adj, part)) == 0.0
