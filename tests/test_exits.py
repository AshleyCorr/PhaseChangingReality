"""T22's exit counter: consistent bookkeeping, reproducible, and sampling the kernel's ensemble."""
import csv
from math import exp
from pathlib import Path

import numpy as np

from graphity.cqg import NO_CAP, is_valid, torus
from graphity.exits import run_until_through
from graphity.small_graphs import circulant

ROOT = Path(__file__).resolve().parents[1]


def test_a_decay_that_goes_through_has_one_more_exit_than_fallbacks():
    for seed in (1, 2, 3):
        adj, part = torus(16, 4, NO_CAP)
        side_u = np.flatnonzero(part == 0)
        e, f, first, end, through = run_until_through(adj, side_u, 1.5, 1.25, NO_CAP, 20000, seed, 0.25)
        assert through and is_valid(adj, NO_CAP)
        assert e == f + 1 and e >= 1 and 0 < first <= end


def test_same_seed_same_decay():
    out = []
    for _ in range(2):
        adj, part = torus(16, 4, NO_CAP)
        out.append(run_until_through(adj, np.flatnonzero(part == 0), 1.5, 1.25, NO_CAP, 20000, 7, 0.25))
    assert out[0] == out[1]


def test_the_chain_samples_the_kernels_ensemble_at_n18():
    """Run far past any stop (stop_fraction huge never triggers from a non-torus start) and compare the
    distribution's mean S, read at the end of many short runs, with the exact labelled average at N = 18."""
    rows = [r for r in csv.DictReader(open(ROOT / "results" / "ergodicity_small.csv", newline=""))
            if r["N"] == "18" and r["cap"] == "none"]
    lam, g = 1.0, 6.0
    w = [int(r["labelled_states"]) * exp(-(16 * (18 - int(r["squares"])) + 4 * lam * int(r["surplus"])) / g) for r in rows]
    exact = sum(wi * int(r["squares"]) for wi, r in zip(w, rows)) / sum(w)
    from graphity.cqg import total_squares
    s = []
    side_u = np.arange(9)
    for seed in range(300):
        adj = circulant(9)
        run_until_through(adj, side_u, g, lam, NO_CAP, 40, seed, 1e9)
        s.append(total_squares(adj))
    m, e = np.mean(s), np.std(s) / np.sqrt(len(s))
    assert abs(m - exact) < 4 * e + 0.05, (m, e, exact)
