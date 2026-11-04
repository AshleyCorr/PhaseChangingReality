"""The interchangeable-points chain at any number of links (graphity.interchangeable_d), checked as rule 6 asks.

Against the exact interchangeable averages at N = 18 (four links; they differ from the named ones there, so the test can
fail); exact energy bookkeeping; exact conservation when sealed; the hard-core rule, degrees and sides preserved at four
and six links; the same seed gives the same run."""
import csv
from math import exp
from pathlib import Path

import numpy as np
import pytest

from graphity import interchangeable_d as ic, symmetry
from graphity.cqg_d import hamiltonian, is_valid, torus
from graphity.small_graphs import circulant

pytestmark = pytest.mark.skipif(not symmetry.available(), reason="igraph not installed")
ROOT = Path(__file__).resolve().parents[1]
PART18 = np.array([0] * 9 + [1] * 9)


def exact(lam, g):
    rows = [r for r in csv.DictReader(open(ROOT / "results" / "ergodicity_small.csv", newline=""))
            if r["N"] == "18" and r["cap"] == "none"]
    h = [16 * (18 - int(r["squares"])) + 4 * lam * int(r["surplus"]) for r in rows]
    s = [int(r["squares"]) for r in rows]
    w_unl = [exp(-x / g) for x in h]
    w_lab = [int(r["labelled_states"]) * exp(-x / g) for x, r in zip(h, rows)]
    return (sum(a * b for a, b in zip(s, w_unl)) / sum(w_unl), sum(a * b for a, b in zip(s, w_lab)) / sum(w_lab))


def test_reproduces_the_exact_interchangeable_average_at_four_links():
    lam, g = 1.0, 4.0
    unl, lab = exact(lam, g)
    assert abs(unl - lab) > 0.5
    means = []
    for seed in range(8):
        adj = circulant(9).astype(np.int64)
        s, _, _, _ = ic.run(adj, PART18, 400, seed, lam, g=g)
        assert is_valid(adj)
        means.append(s[50:].mean())
    m, e = np.mean(means), np.std(means, ddof=1) / np.sqrt(len(means))
    assert abs(m - unl) < 4 * e + 0.05, (m, e, unl, lab)
    assert abs(m - lab) > 4 * e, (m, e, unl, lab)


def test_unweighted_reproduces_the_named_average_at_four_links():
    lam, g = 1.0, 4.0
    unl, lab = exact(lam, g)
    means = []
    for seed in range(8):
        adj = circulant(9).astype(np.int64)
        s, _, _, _ = ic.run(adj, PART18, 400, 100 + seed, lam, g=g, weighted=False)
        means.append(s[50:].mean())
    m, e = np.mean(means), np.std(means, ddof=1) / np.sqrt(len(means))
    assert abs(m - lab) < 4 * e + 0.05, (m, e, unl, lab)


@pytest.mark.parametrize("dims", [[6, 6], [6, 6, 6]])
def test_bookkeeping_conservation_and_constraints(dims):
    lam = 1.02
    adj, part = torus(dims)
    n, deg = adj.shape
    rng = np.random.default_rng(0)
    demons = np.zeros(8)
    demons[0] = 400.0
    e0 = hamiltonian(adj, lam) + demons.sum()
    s, x, a, d = ic.run(adj, part, 30, 5, lam, demons=demons)
    h = hamiltonian(adj, lam)
    dim = deg // 2
    assert abs((16 * (dim * (dim - 1) / 2 * n - s[-1]) + 4 * lam * x[-1]) - h) < 1e-9     # tracked = recomputed
    assert abs(h + demons.sum() - e0) < 1e-9 and (demons >= -1e-12).all()                  # sealed: conserved
    assert is_valid(adj)
    assert all(len(set(adj[v])) == deg for v in range(n))                                   # degrees kept
    assert all(part[w] != part[v] for v in range(n) for w in adj[v])                         # sides kept
    assert a[-1] == symmetry.count(adj, part)


def test_same_seed_same_run():
    lam = 1.02
    runs = []
    for _ in range(2):
        adj, part = torus([6, 6, 6])
        demons = np.array([200.0, 0.0, 0.0, 0.0])
        runs.append((ic.run(adj, part, 10, 11, lam, demons=demons), adj.copy()))
    assert all(np.array_equal(p, q) for p, q in zip(runs[0][0], runs[1][0]))
    assert np.array_equal(runs[0][1], runs[1][1])
