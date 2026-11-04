"""T59's two readers of T22's chain: the same first exit draw for draw, consistent clocks, the counted offers."""
import csv
import sys
from pathlib import Path

import numpy as np
import pytest

from graphity.cqg import NO_CAP, is_valid, torus
from graphity.exits import first_exit_clocks, offers_until_exit, run_until_through

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from exact_torus_level import census, rate_per_sweep                      # noqa: E402

G = 1.5


def start(lx=16, ly=4):
    adj, part = torus(lx, ly, NO_CAP)
    return adj, np.flatnonzero(part == 0)


def test_the_first_exit_is_the_same_attempt_in_all_three_readers():
    for lam, seed in ((1.25, 1), (1.25, 2), (1.30, 3), (1.45, 4)):
        adj, side = start()
        _, _, first, _, _ = run_until_through(adj, side, G, lam, NO_CAP, 200000, seed, 0.25)
        adj, side = start()
        clocks = first_exit_clocks(adj, side, G, lam, NO_CAP, 200000, seed, 5)
        adj, side = start()
        offers = offers_until_exit(adj, side, G, lam, NO_CAP, 200000, seed, 1000)
        assert first > 0 and clocks[0] == first and offers[0] == first


def test_the_clocks_are_consistent():
    """A look can only see an exit after it happened; a look every five sweeps sees no earlier than a look every
    sweep; each exit but the one seen came back, so fall-backs are one fewer than exits."""
    hidden = 0
    for seed in range(40):
        adj, side = start()
        first, seen_1, seen_k, exits_1, exits_k, fallbacks = first_exit_clocks(adj, side, G, 1.30, NO_CAP, 50000, seed, 5)
        assert is_valid(adj, NO_CAP)
        sweep_of_first = -(-first // (2 * 64))                      # the sweep in which the first exit was made
        assert 1 <= sweep_of_first <= seen_1 <= seen_k and seen_k % 5 == 0
        assert 1 <= exits_1 <= exits_k and fallbacks == exits_k - 1
        if exits_k == 1:
            assert seen_1 == sweep_of_first and seen_k - seen_1 < 5
        hidden += exits_k - 1
    assert hidden > 0                                               # some exits do come back between looks


def test_a_look_every_sweep_makes_the_two_look_clocks_one():
    for seed in (11, 12, 13):
        adj, side = start()
        first, seen_1, seen_k, exits_1, exits_k, _ = first_exit_clocks(adj, side, G, 1.30, NO_CAP, 50000, seed, 1)
        assert seen_1 == seen_k and exits_1 == exits_k


def test_same_seed_same_clocks():
    out = []
    for _ in range(2):
        adj, side = start()
        out.append(first_exit_clocks(adj, side, G, 1.25, NO_CAP, 50000, 7, 5))
    assert out[0] == out[1]


def test_a_tube_that_never_leaves_reports_no_exit():
    adj, side = start()
    assert first_exit_clocks(adj, side, G, 1.05, NO_CAP, 3, 1, 5) == (-1, -1, -1, 0, 0, 0)
    adj, side = start()
    first, offered, smallest = offers_until_exit(adj, side, G, 1.05, NO_CAP, 3, 1, 2)
    assert first == -1 and offered.shape == (2, 5) and smallest.shape == (2, 2)


def test_the_saved_first_exits_of_t22_come_back_exactly():
    """T22's rows hold the first exit in sweeps; its seed rule is in scripts/run_exits.py."""
    path = ROOT / "results" / "t22_exits_n64_lam125.csv"
    if not path.exists():
        pytest.skip("T22's results are not in this checkout")
    rows = list(csv.DictReader(open(path, newline="")))[:6]
    for r in rows:
        rep = int(r["replica"])
        seed = int(np.random.SeedSequence([20261801, 64, 125, rep]).generate_state(1)[0])
        adj, side = start()
        first = first_exit_clocks(adj, side, G, 1.25, NO_CAP, 100000, seed, 5)[0]
        assert first / 128.0 == float(r["first_exit_sweeps"])


def test_the_waiting_tube_is_offered_what_the_census_counts():
    """Per sweep the perfect torus is offered three exits of kind A and two of kind B (paper 1, Eq. (2)); the
    tally over a long wait must agree with that within its own scatter, and a draw below exp(-cost/g) against an
    A proposal ends the wait, so none smaller than that can have been made before the last one."""
    lam, seed = 1.05, 2560290145                                    # T22's replica 1 at N = 64: sweep 2,481
    adj, side = start()
    first, offered, smallest = offers_until_exit(adj, side, G, lam, NO_CAP, 20000, seed, 500)
    assert first == 317622                                          # 2481.421875 sweeps, as T22 saved it
    whole = first // (500 * 128)
    a, b = offered[:whole, 0].sum(), offered[:whole, 1].sum()
    sweeps = whole * 500
    assert abs(a - 3 * sweeps) < 5 * np.sqrt(3 * sweeps) and abs(b - 2 * sweeps) < 5 * np.sqrt(2 * sweeps)
    assert smallest[:whole, 0].min() >= np.exp(-(32 - 16 * lam) / G)
    assert smallest[:whole, 1].min() >= np.exp(-(64 - 40 * lam) / G)


def test_the_mean_first_exit_is_the_exact_count_where_waits_are_short():
    """At lambda = 1.45 the wait is about twenty sweeps, so 400 tubes cost little: their mean first exit must be
    the reciprocal of the exact rate over every way out (scripts/exact_torus_level.py), within four standard errors."""
    lam = 1.45
    adj, side = start()
    tau = 1.0 / rate_per_sweep(census(adj, side)[0], 64, lam, G)
    waits = []
    for seed in range(400):
        adj, side = start()
        waits.append(first_exit_clocks(adj, side, G, lam, NO_CAP, 5000, 1000 + seed, 5)[0] / 128.0)
    mean, err = np.mean(waits), np.std(waits) / np.sqrt(len(waits))
    assert abs(mean - tau) < 4 * err, (mean, err, tau)
