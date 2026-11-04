"""Checks on the coupled two-copy chain of T75 (graphity.coupled), rule 6."""
import csv
from collections import Counter
from math import exp
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from graphity.coupled import (connected_region, coordinates, damage_mask, damage_radius, fair_sweep_attempts, flaw,
                              lattice_distance, region_radius, run_coupled)
from graphity.cqg import CAP, NO_CAP, hamiltonian, is_valid, run_chain, surplus, torus, total_squares
from graphity.small_graphs import circulant, sides_first

ROOT = Path(__file__).resolve().parents[1]


def brute_hamiltonian(adj, lam):
    """H = 16 (N - S) + 4 lam X from networkx alone, sharing no code with the kernel."""
    g = nx.Graph()
    for u in range(adj.shape[0]):
        for v in adj[u]:
            g.add_edge(u, int(v))
    on_edge, n_squares = Counter(), 0
    for c in nx.simple_cycles(g, length_bound=4):
        if len(c) == 4:
            n_squares += 1
            for i in range(4):
                on_edge[frozenset((c[i], c[(i + 1) % 4]))] += 1
    return 16 * (adj.shape[0] - n_squares) + 4 * lam * sum(max(0, k - 2) for k in on_edge.values())


def melted(l, seed, cap=NO_CAP):
    adj, part = torus(l, cap=cap)
    side_u = np.flatnonzero(part == 0)
    run_chain(adj, side_u, 0.0, 30, 1, seed, 1.0, cap, False)
    return adj, side_u


def exact_mean_squares(n, lam, inv_g):
    with (ROOT / "results" / "ergodicity_small.csv").open(newline="") as fh:
        classes = [r for r in csv.DictReader(fh) if r["N"] == str(n) and r["cap"] == "none" and r["class_id"] != "-1"]
    assert {r["all_joined"] for r in classes} == {"True"}
    weights = [int(r["labelled_states"]) * exp(-inv_g * (16 * (n - int(r["squares"])) + 4 * lam * int(r["surplus"])))
               for r in classes]
    return sum(w * int(r["squares"]) for w, r in zip(weights, classes)) / sum(weights)


def two_starts(n):
    """Two different valid states, numbered sides first, so that both copies share side_u = 0..n/2-1."""
    if n == 16:
        cube, part = torus(4, cap=NO_CAP)
        a = sides_first(cube, part)
    else:
        a = circulant(9)
    b = a.copy()
    run_chain(b, np.arange(n // 2), 0.0, 5, 1, 9, 1.0, NO_CAP, False)
    assert is_valid(a, NO_CAP) and is_valid(b, NO_CAP) and damage_mask(a, b).any()
    return a, b


@pytest.mark.parametrize("n, lam, inv_g", [(16, 0.0, 0.08), (16, 1.25, 0.08), (18, 1.0, 0.10), (18, 1.25, 0.10)])
def test_each_copy_matches_the_exact_average(n, lam, inv_g):
    """Each copy, read alone, must sample exp(-H/g) over all labelled states: the exact <S> at N = 16 and 18 from
    the recorded exhaustive listing (Q9), as tests/test_small_graphs.py checks cqg.run_chain. The copies start in
    different states, so the test sees a copy while the other is elsewhere."""
    exact = exact_mean_squares(n, lam, inv_g)
    means_a, means_b = [], []
    for seed in range(12):
        a, b = two_starts(n)
        out = run_coupled(a, b, np.arange(n // 2), inv_g, 4300, 2 * n, 500 + seed, lam, NO_CAP)
        means_a.append(out["squares_a"][300:].mean())
        means_b.append(out["squares_b"][300:].mean())
    for means in (means_a, means_b):
        mean, err = np.mean(means), np.std(means, ddof=1) / np.sqrt(len(means))
        assert err < 0.03
        assert abs(mean - exact) < 5 * err, (mean, err, exact)


def test_equal_copies_stay_equal_draw_for_draw():
    adj, side_u = melted(8, 3)
    a, b = adj.copy(), adj.copy()
    out = run_coupled(a, b, side_u, 0.1, 50, 128, 11, 1.25)
    assert out["acceptance_a"] > 0.05                             # so moves really happened
    assert (out["damaged"] == 0).all() and np.array_equal(a, b)
    assert out["acceptance_a"] == out["acceptance_b"]


def test_a_copy_alone_is_unaffected_by_how_the_other_was_disturbed():
    """Not draw for draw with run_chain (the neighbour is read by rank), but the same transition probabilities;
    so a damaged copy B must not change what copy A does where A's own proposals are concerned: A's trajectory
    depends only on A and the shared numbers."""
    adj, side_u = melted(8, 4)
    runs = []
    for partner_seed in (1, 2):
        a = adj.copy()
        b, _ = melted(8, partner_seed)
        out = run_coupled(a, b, side_u, 0.2, 20, 128, 21, 1.25)
        runs.append((a, out["squares_a"]))
    assert np.array_equal(runs[0][0], runs[1][0]) and np.array_equal(runs[0][1], runs[1][1])


@pytest.mark.parametrize("lam, cap", [(0.0, NO_CAP), (0.5, NO_CAP), (1.25, NO_CAP), (1.25, CAP)])
def test_incremental_energy_is_exact_and_validity_kept(lam, cap):
    a, side_u = melted(8, 5, cap)
    b, _ = melted(8, 6, cap)
    rng_seed = 31
    for block in range(10):
        out = run_coupled(a, b, side_u, 0.05, 10, 128, rng_seed if block == 0 else -1, lam, cap)
        for adj, s, x in ((a, out["squares_a"], out["surplus_a"]), (b, out["squares_b"], out["surplus_b"])):
            assert is_valid(adj, cap)
            assert s[-1] == total_squares(adj) and x[-1] == surplus(adj)
            assert hamiltonian(adj, lam) == pytest.approx(16 * (64 - s[-1]) + 4 * lam * x[-1])
    assert out["acceptance_a"] > 0.05
    assert hamiltonian(a, lam) == pytest.approx(brute_hamiltonian(a, lam))
    assert hamiltonian(b, lam) == pytest.approx(brute_hamiltonian(b, lam))


def test_same_seed_same_run():
    results = []
    for seed in (7, 7, 8):
        a, part = torus(8, cap=NO_CAP)
        b, _ = flaw(a, 0, 8)
        out = run_coupled(a, b, np.flatnonzero(part == 0), 0.3, 30, 128, seed, 1.25)
        results.append((a, b, out["mask"]))
    assert all(np.array_equal(x, y) for x, y in zip(results[0], results[1]))
    assert not all(np.array_equal(x, y) for x, y in zip(results[0], results[2]))


@pytest.mark.parametrize("l, point, axis", [(8, 0, 0), (8, 27, 1), (16, 100, 0), (16, 255, 1)])
def test_flaw_is_valid_costs_32_and_touches_four_points(l, point, axis):
    flat, _ = torus(l, cap=NO_CAP)
    flawed, touched = flaw(flat, point, l, axis=axis)
    assert is_valid(flawed, NO_CAP) and touched[0] == point
    assert set(np.flatnonzero(damage_mask(flat, flawed))) == set(touched) and len(set(touched)) == 4
    for lam in (1.0, 1.25):                                       # the cheapest move out of the sheet (O22)
        assert hamiltonian(flawed, lam) == 32.0 == brute_hamiltonian(flawed, lam)
    assert lattice_distance(touched, point, l).tolist() == [0, 1, 2, 3]


def test_flaw_refuses_a_graph_that_is_not_the_flat_torus():
    adj, _ = melted(8, 1)
    with pytest.raises(ValueError):
        flaw(adj, 0, 8)


def test_damage_starts_as_the_four_points_of_the_flaw():
    flat, part = torus(16, cap=NO_CAP)
    flawed, touched = flaw(flat, 37, 16)
    out = run_coupled(flat.copy(), flawed, np.flatnonzero(part == 0), 1.0, 1, 0, 1, 1.25)
    assert out["damaged"].tolist() == [4]
    assert set(np.flatnonzero(out["mask"][0])) == set(touched)


def test_radius_on_a_hand_built_mask():
    l = 8
    xy = {tuple(c): i for i, c in enumerate(coordinates(l))}
    assert xy[(1, 2)] == 1 * l + 2                                # the layout of cqg.torus
    mask = np.zeros(l * l, dtype=bool)
    for c in [(0, 0), (1, 0), (0, 2), (1, 2), (7, 0), (4, 4)]:
        mask[xy[c]] = True
    origin = xy[(0, 0)]
    assert damage_radius(mask, origin, l) == 8                    # (4, 4) is 4 + 4 away, the far way round too
    assert damage_radius(np.zeros(l * l, dtype=bool), origin, l) == -1
    near = connected_region(mask, origin, l, reach=2)             # (7, 0) is one step away round the torus
    assert set(np.flatnonzero(near)) == {xy[c] for c in [(0, 0), (1, 0), (0, 2), (1, 2), (7, 0)]}
    assert region_radius(mask, origin, l, reach=2) == 3
    assert region_radius(mask, origin, l, reach=1) == 1           # (0, 2) and (1, 2) are two steps from the rest
    assert region_radius(mask, xy[(3, 3)], l, reach=2) == -1      # the origin itself is not damaged
    assert damage_radius(mask, [xy[(4, 4)], origin], l) == 3      # distance to the nearest of several origins


def test_fair_sweep_and_bad_input():
    assert fair_sweep_attempts(96) == 2 * 96 and fair_sweep_attempts(576) == 6912
    flat, part = torus(8, cap=NO_CAP)
    with pytest.raises(ValueError):
        run_coupled(flat.copy(), torus(10, cap=NO_CAP)[0], np.flatnonzero(part == 0), 1.0, 1, 1, 1)
    with pytest.raises(ValueError):                               # not one side of the bipartition
        run_coupled(flat.copy(), flat.copy(), np.arange(32), 1.0, 1, 1, 1)
