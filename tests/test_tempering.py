"""T5: parallel tempering. The decisive test is against exact averages, where a wrong swap rule would show."""
import csv
from math import exp
from pathlib import Path

import numpy as np
import pytest

from graphity.cqg import NO_CAP, is_valid
from graphity.small_graphs import circulant
from graphity.tempering import temper

ROOT = Path(__file__).resolve().parents[1]
SIDE_U = np.arange(9)


def exact_mean_squares(lam, g):
    with (ROOT / "results" / "ergodicity_small.csv").open(newline="") as fh:
        classes = [r for r in csv.DictReader(fh) if r["N"] == "18" and r["cap"] == "none"]
    weights = [int(r["labelled_states"]) * exp(-(16 * (18 - int(r["squares"])) + 4 * lam * int(r["surplus"])) / g)
               for r in classes]
    return sum(w * int(r["squares"]) for w, r in zip(weights, classes)) / sum(weights)


@pytest.mark.parametrize("lam", [0.0, 1.0])
def test_every_coupling_matches_its_exact_average(lam):
    """N = 18, four couplings at once. Each must give the exact <S> for ITS coupling: swaps move graphs between
    couplings all the time, and any error in the swap rule would pull the averages towards each other."""
    couplings = [20.0, 10.0, 6.0, 4.0]
    means = []
    for seed in range(10):
        graphs = [circulant(9) for _ in couplings]
        out = temper(graphs, SIDE_U, couplings, 450, 5, np.random.SeedSequence([77, seed]), lam, NO_CAP,
                     glauber=seed % 2 == 1, measure_from=50)
        assert all(is_valid(g, NO_CAP) for g in graphs)
        assert (out["swap_rate"] > 0.3).all()                     # so the graphs really did travel
        means.append(out["squares"].mean(axis=1))
    mean, err = np.mean(means, axis=0), np.std(means, axis=0, ddof=1) / np.sqrt(len(means))
    exact = np.array([exact_mean_squares(lam, g) for g in couplings])
    assert np.ptp(exact) > 0.3                                    # the four answers differ, so the test can fail
    assert (err < 0.05).all()
    assert (np.abs(mean - exact) < 5 * err).all(), (mean, err, exact)


def test_same_seed_same_run_and_round_trips_are_counted():
    couplings = [12.0, 9.0, 7.0, 5.5]
    runs = []
    for seed in (5, 5, 6):
        graphs = [circulant(9) for _ in couplings]
        out = temper(graphs, SIDE_U, couplings, 200, 2, np.random.SeedSequence(seed), 1.0, NO_CAP)
        runs.append((out["squares"].copy(), out["swap_rate"].copy(), out["round_trips"], [g.copy() for g in graphs]))
    assert (runs[0][0] == runs[1][0]).all() and (runs[0][1] == runs[1][1]).all() and runs[0][2] == runs[1][2]
    assert all((a == b).all() for a, b in zip(runs[0][3], runs[1][3]))
    assert not (runs[0][0] == runs[2][0]).all()
    assert runs[0][2] > 0 and runs[0][0].shape == (4, 400)


def test_bad_input_is_refused():
    graphs = [circulant(9), circulant(9)]
    with pytest.raises(ValueError):
        temper(graphs, SIDE_U, [5.0, 8.0], 2, 1, np.random.SeedSequence(1), 1.0, NO_CAP)      # coldest first
    with pytest.raises(ValueError):
        temper(graphs, SIDE_U, [8.0, 5.0, 3.0], 2, 1, np.random.SeedSequence(1), 1.0, NO_CAP)  # three couplings, two graphs


def test_a_reading_hook_changes_nothing():
    """T16 reads snapshots through on_round; the run must be bit for bit the run without it."""
    couplings = [12.0, 9.0, 7.0, 5.5]
    seen = []
    runs = []
    for hook in (None, lambda r, gs: seen.append((r, [g.copy() for g in gs]))):
        graphs = [circulant(9) for _ in couplings]
        out = temper(graphs, SIDE_U, couplings, 60, 2, np.random.SeedSequence(9), 1.0, NO_CAP,
                     measure_from=20, on_round=hook)
        runs.append((out["squares"].copy(), out["round_trips"], [g.copy() for g in graphs]))
    assert (runs[0][0] == runs[1][0]).all() and runs[0][1] == runs[1][1]
    assert all((a == b).all() for a, b in zip(runs[0][2], runs[1][2]))
    assert [r for r, _ in seen] == list(range(20, 60))
    assert all((a == b).all() for a, b in zip(seen[-1][1], runs[1][2]))     # the hook saw the final graphs
