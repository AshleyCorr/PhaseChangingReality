"""The soft form of the owner's square-count rule (graphity.soft_d), checked as rule 6 asks.

At mu = 0 the chain is cqg_d's draw for draw; against the exact averages at N = 18 (four links) with the penalty on,
where they differ from the unpenalized ones by about two squares, so the test can fail; tracked S and X against
recomputation and against networkx; the hard-core rule, degrees and sides preserved at four and six links; the same seed
gives the same run; and the per-link identity H = 4 uncurl + 4 (lam - 1) curl, S - S0 = (curl - uncurl) / 4, on the
flat, slab and rod tori (VISION Update 133)."""
import csv
from math import exp
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from graphity import cqg_d, soft_d
from graphity.small_graphs import circulant

ROOT = Path(__file__).resolve().parents[1]
PART18 = np.array([0] * 9 + [1] * 9)


def per_link(adj):
    """Squares on each link, by networkx: (S, curl, uncurl)."""
    g = nx.Graph()
    for u in range(adj.shape[0]):
        for v in adj[u]:
            g.add_edge(u, int(v))
    flat = adj.shape[1] - 2
    per = [sum(1 for a in g[u] if a != v for b in g[v] if b != u and g.has_edge(a, b)) for u, v in g.edges()]
    return sum(per) // 4, sum(max(c - flat, 0) for c in per), sum(max(flat - c, 0) for c in per)


@pytest.mark.parametrize("dims", [(8, 8), (16, 4), (6, 6, 6), (4, 6, 6)])
@pytest.mark.parametrize("glauber", [False, True])
def test_at_mu_zero_the_chain_is_cqg_d_draw_for_draw(dims, glauber):
    a, part = cqg_d.torus(dims)
    b = a.copy()
    side = np.flatnonzero(part == 0)
    cqg_d.run_chain(a, side, 0.0, 20, 0, 3)                      # melt a little, same stream both ways
    soft_d.run_chain(b, side, 0.0, 20, 0, 3)
    assert np.array_equal(a, b)
    r1 = cqg_d.run_chain(a, side, 1 / 1.5, 30, 40, 5, 1.25, glauber)
    r2 = soft_d.run_chain(b, side, 1 / 1.5, 30, 40, 5, 1.25, 0.0, glauber)
    assert np.array_equal(r1[0], r2[0]) and np.array_equal(r1[1], r2[1]) and r1[2] == r2[2]
    assert np.array_equal(a, b)


def exact(lam, g, mu):
    rows = [r for r in csv.DictReader(open(ROOT / "results" / "ergodicity_small.csv", newline=""))
            if r["N"] == "18" and r["cap"] == "none" and r["class_id"] != "-1"]
    w = [int(r["labelled_states"]) * exp(-(16 * (18 - int(r["squares"])) + 4 * lam * int(r["surplus"])
                                         + mu * (int(r["squares"]) - 18) ** 2) / g) for r in rows]
    return sum(a * int(r["squares"]) for a, r in zip(w, rows)) / sum(w)


@pytest.mark.parametrize("glauber", [False, True])
def test_reproduces_the_exact_soft_average_at_n18(glauber):
    lam, g, mu = 1.0, 4.0, 2.0
    target, unpenalized = exact(lam, g, mu), exact(lam, g, 0.0)
    assert abs(target - unpenalized) > 1.0                      # the penalty moves the answer: the test can fail
    means = []
    for seed in range(8):
        adj = circulant(9).astype(np.int64)
        side = np.flatnonzero(PART18 == 0)
        s, _, _ = soft_d.run_chain(adj, side, 1 / g, 200, 2000, 40 + seed, lam, mu, glauber)
        assert cqg_d.is_valid(adj)
        means.append(s.mean())
    m, e = np.mean(means), np.std(means, ddof=1) / np.sqrt(len(means))
    assert abs(m - target) < 4 * e + 0.02, (m, e, target, unpenalized)
    assert abs(m - unpenalized) > 4 * e, (m, e, target, unpenalized)


@pytest.mark.parametrize("dims,mu", [((8, 8), 0.5), ((6, 6, 6), 0.05), ((4, 6, 6), 2.0)])
def test_bookkeeping_and_constraints(dims, mu):
    lam = 1.25
    adj, part = cqg_d.torus(dims)
    n, deg = adj.shape
    side = np.flatnonzero(part == 0)
    soft_d.run_chain(adj, side, 0.0, 5, 0, 7, lam, mu)          # a random start, away from S0
    for k in range(4):
        s, x, _ = soft_d.run_chain(adj, side, 1 / 2.0, 0, 10, -1, lam, mu)
        brute_s, curl, uncurl = per_link(adj)
        assert s[-1] == cqg_d.total_squares(adj) == brute_s
        assert x[-1] == cqg_d.surplus(adj) == curl
        h = 16 * (soft_d.target(adj) - s[-1]) + 4 * lam * x[-1] + mu * (s[-1] - soft_d.target(adj)) ** 2
        assert abs(h - soft_d.hamiltonian_soft(adj, lam, mu)) < 1e-9
        assert cqg_d.is_valid(adj)
        assert all(len(set(adj[v])) == deg for v in range(n))                       # degrees kept
        assert all(part[w] != part[v] for v in range(n) for w in adj[v])             # sides kept


def test_same_seed_same_run():
    runs = []
    for _ in range(2):
        adj, part = cqg_d.torus((6, 6, 6))
        side = np.flatnonzero(part == 0)
        runs.append((soft_d.run_chain(adj, side, 1 / 1.5, 5, 10, 11, 1.25, 0.3), adj.copy()))
    assert all(np.array_equal(p, q) for p, q in zip(runs[0][0][:2], runs[1][0][:2]))
    assert runs[0][0][2] == runs[1][0][2] and np.array_equal(runs[0][1], runs[1][1])


@pytest.mark.parametrize("dims,curled", [((8, 8, 8), 0), ((4, 8, 16), 1), ((4, 4, 32), 2)])
def test_the_per_link_identity_on_the_ladder(dims, curled):
    adj, _ = cqg_d.torus(dims)
    n = adj.shape[0]
    s, curl, uncurl = per_link(adj)
    assert (curl, uncurl) == (curled * n, 0)
    assert s - soft_d.target(adj) == (curl - uncurl) // 4 == curled * n // 4
    for lam in (1.0, 1.1, 1.25):
        assert abs(cqg_d.hamiltonian(adj, lam) - (4 * uncurl + 4 * (lam - 1) * curl)) < 1e-9
        for mu in (0.0, 1.0, 1 / n):                            # an unbalanced rod pays mu N^2 / 4, a slab mu N^2 / 16
            pen = soft_d.hamiltonian_soft(adj, lam, mu) - cqg_d.hamiltonian(adj, lam)
            assert abs(pen - mu * (curled * n / 4) ** 2) < 1e-6


@pytest.mark.parametrize("dims", [(6, 6, 6), (4, 4, 8)])
def test_anticurl_at_lambda_is_the_toy_at_a_shifted_lambda(dims):
    """The owner's A2 (27 October): anticurl priced at 4 lam a unit, like curl, is lam H(2 - 1/lam) (VISION Update 133)."""
    adj, part = cqg_d.torus(dims)
    side = np.flatnonzero(part == 0)
    seen_curl = seen_uncurl = False
    for k in range(4):
        soft_d.run_chain(adj, side, 0.0 if k % 2 else 1 / 0.8, 1, 0, 20 + k)
        _, curl, uncurl = per_link(adj)
        seen_curl |= curl > 0
        seen_uncurl |= uncurl > 0
        for lam in (1.1, 1.25, 1.4):
            assert abs(4 * lam * uncurl + 4 * (lam - 1) * curl - lam * cqg_d.hamiltonian(adj, 2 - 1 / lam)) < 1e-9
    assert seen_uncurl
