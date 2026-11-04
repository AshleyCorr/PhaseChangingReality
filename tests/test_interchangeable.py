"""The interchangeable-points chain against the exact interchangeable averages at N = 18.

The exact answer weights each class once (times exp(-H/g)); the named ensemble weights it by its number of
labelled graphs. Both are computed from results/ergodicity_small.csv, and the test checks that the two
differ at the chosen settings, so it can fail."""
import csv
from math import exp
from pathlib import Path

import numpy as np
import pytest

from graphity import interchangeable, symmetry
from graphity.cqg import NO_CAP, is_valid, surplus, total_squares
from graphity.small_graphs import circulant

pytestmark = pytest.mark.skipif(not symmetry.available(), reason="igraph not installed")
ROOT = Path(__file__).resolve().parents[1]
PART = np.array([0] * 9 + [1] * 9)


def exact(lam, g):
    rows = [r for r in csv.DictReader(open(ROOT / "results" / "ergodicity_small.csv", newline=""))
            if r["N"] == "18" and r["cap"] == "none"]
    h = [16 * (18 - int(r["squares"])) + 4 * lam * int(r["surplus"]) for r in rows]
    s = [int(r["squares"]) for r in rows]
    w_unl = [exp(-x / g) for x in h]
    w_lab = [int(r["labelled_states"]) * exp(-x / g) for x, r in zip(h, rows)]
    return (sum(a * b for a, b in zip(s, w_unl)) / sum(w_unl), sum(a * b for a, b in zip(s, w_lab)) / sum(w_lab))


def test_canonical_chain_reproduces_the_exact_interchangeable_average():
    lam, g = 1.0, 4.0
    unl, lab = exact(lam, g)
    assert abs(unl - lab) > 0.5                              # the two ensembles differ here, so the test can fail
    means = []
    for seed in range(8):
        adj = circulant(9)
        s, _, _, _ = interchangeable.run(adj, PART, 400, seed, lam, NO_CAP, g=g)
        assert is_valid(adj, NO_CAP)
        means.append(s[50:].mean())
    m, e = np.mean(means), np.std(means, ddof=1) / np.sqrt(len(means))
    assert abs(m - unl) < 4 * e + 0.05, (m, e, unl, lab)
    assert abs(m - lab) > 4 * e, (m, e, unl, lab)


def test_sealed_chain_conserves_energy_exactly():
    lam = 1.0
    adj = circulant(9)
    demons = np.array([40.0, 0.0, 0.0])
    e0 = interchangeable.energy(adj, lam) + demons.sum()
    s, x, _, d = interchangeable.run(adj, PART, 60, 3, lam, NO_CAP, demons=demons)
    h = 16 * (18 - s) + 4 * lam * x
    assert np.allclose(h + d, e0) and (demons >= 0).all()
