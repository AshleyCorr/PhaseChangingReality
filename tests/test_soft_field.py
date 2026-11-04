"""Rule-6 tests for graphity.soft_field (soft space, stage A; VISION Update 77).

With no field the switches are the present chain draw for draw; the switch kernel's summed energy change equals full
recomputation of H + E_phi; the sources and the field energy against a networkx brute force; leapfrog is reversible;
the sampled field on a fixed torus reproduces the exact Gaussian answers (correlations (g/kappa) L+, equipartition
(N - 1) g / 2, the mean -(q/kappa) L+ (e - mean e)); the lattice resistances of [Cse00] on a large flat torus; the graph
stays valid; same seed, same run.
"""
import networkx as nx
import numpy as np
import pytest

from graphity.cqg import NO_CAP, hamiltonian, is_valid, torus
from graphity.row_growth import _attempts
from graphity.soft_field import (curl_sources, field_energy, hmc_trajectory, laplacian_apply, leapfrog, run_soft,
                                 soft_attempts, total_energy)

LAM = 1.25


def tube(L=12):
    """A 4 x L tube: every link around the curled direction sits on three squares, so every point is a source."""
    return torus(4, L, cap=NO_CAP)


def lplus(adj):
    n = adj.shape[0]
    A = np.zeros((n, n))
    for p in range(n):
        for q in adj[p]:
            A[p, q] = 1.0
    return np.linalg.pinv(np.diag(A.sum(1)) - A)


def test_no_field_is_the_present_chain_draw_for_draw():
    a, part = torus(12, 8, cap=NO_CAP)
    b = a.copy()
    side_u = np.flatnonzero(part == 0).astype(np.int64)
    acc_a = _attempts(a, side_u, 1 / 2.5, 20000, 77, LAM, NO_CAP)
    acc_b, _ = soft_attempts(b, side_u, 1 / 2.5, 20000, 77, LAM, NO_CAP, np.zeros(96), 1.7, 0.0)
    assert acc_a == acc_b and np.array_equal(a, b)


@pytest.mark.parametrize("q", [0.0, 0.8, -1.5])
def test_switch_energy_change_is_exact(q):
    adj, part = tube(12)
    rng = np.random.default_rng(3)
    phi = rng.normal(size=adj.shape[0])
    phi -= phi.mean()
    side_u = np.flatnonzero(part == 0).astype(np.int64)
    e0 = total_energy(adj, phi, 0.9, q, LAM)
    acc, dh = soft_attempts(adj, side_u, 1 / 3.0, 5000, 11, LAM, NO_CAP, phi, 0.9, q)
    assert acc > 0
    assert total_energy(adj, phi, 0.9, q, LAM) - e0 == pytest.approx(dh, abs=1e-8)
    assert is_valid(adj, NO_CAP) and np.all(part[:, None] != part[adj])


def test_sources_and_field_energy_against_networkx():
    adj, part = tube(10)
    side_u = np.flatnonzero(part == 0).astype(np.int64)
    soft_attempts(adj, side_u, 1 / 3.5, 3000, 5, LAM, NO_CAP, np.zeros(40), 1.0, 0.0)   # a rewired state
    G = nx.Graph((p, int(q)) for p in range(adj.shape[0]) for q in adj[p])
    e = np.zeros(adj.shape[0])
    for u, v in G.edges:
        s = sum(1 for a in G[u] if a != v for b in G[v] if b != u and b != a and G.has_edge(a, b))
        if s > 2:
            e[u] += 2 * (LAM - 1) * (s - 2)
            e[v] += 2 * (LAM - 1) * (s - 2)
    assert np.allclose(curl_sources(adj, LAM), e)
    phi = np.random.default_rng(0).normal(size=adj.shape[0])
    brute = 0.35 * sum((phi[u] - phi[v]) ** 2 for u, v in G.edges) + 0.4 * float(e @ phi)
    assert field_energy(adj, phi, 0.7, 0.4, LAM) == pytest.approx(brute)
    assert total_energy(adj, phi, 0.7, 0.4, LAM) == pytest.approx(hamiltonian(adj, LAM) + brute)


def test_flat_torus_has_no_sources_and_a_tube_does():
    adj, _ = torus(8, 8, cap=NO_CAP)
    assert not curl_sources(adj, LAM).any()
    t, _ = tube(8)
    assert np.allclose(curl_sources(t, LAM), 2 * 4 * (LAM - 1) * 0.5)    # two curled links per point, one surplus each


def test_leapfrog_is_reversible_and_keeps_the_mean():
    adj, _ = tube(8)
    rng = np.random.default_rng(1)
    phi = rng.normal(size=32)
    phi -= phi.mean()
    pi = rng.normal(size=32)
    pi -= pi.mean()
    src = curl_sources(adj, LAM)
    p1, m1 = leapfrog(adj, phi, pi, 1.3, 0.7, src, 25, 0.1)
    p2, m2 = leapfrog(adj, p1, -m1, 1.3, 0.7, src, 25, 0.1)
    assert np.allclose(p2, phi, atol=1e-10) and np.allclose(-m2, pi, atol=1e-10)
    assert abs(p1.mean()) < 1e-12


def sample_field(adj, g, kappa, q, n, seed=0, n_leap=15, eps=0.25):
    rng = np.random.default_rng(seed)
    phi = np.zeros(adj.shape[0])
    out = []
    for i in range(n):
        steps = int(rng.integers(max(1, n_leap // 2), n_leap + 1))      # as run_soft draws them
        phi, _ = hmc_trajectory(adj, phi, g, kappa, q, LAM, steps, eps, rng)
        if i >= n // 10:
            out.append(phi.copy())
    return np.array(out)


def test_field_reproduces_the_exact_gaussian_answers():
    adj, _ = torus(6, 6, cap=NO_CAP)
    g, kappa = 1.5, 2.0
    s = sample_field(adj, g, kappa, 0.0, 40000)
    lp = lplus(adj)
    exact_cov = g / kappa * lp
    cov = np.cov(s.T, bias=True)
    assert np.abs(cov - exact_cov).max() < 0.06 * exact_cov.max()
    # equipartition: the field's energy averages (N - 1) g / 2
    e = np.array([0.5 * kappa * x @ laplacian_apply(adj, x) for x in s])
    assert e.mean() == pytest.approx(35 * g / 2, rel=0.03)


def test_field_mean_follows_poisson_with_sources():
    adj, _ = tube(6)
    g, kappa, q = 1.0, 1.5, 2.0
    src = curl_sources(adj, LAM).copy()
    src[0] += 0.0
    s = sample_field(adj, g, kappa, q, 4000)
    exact = -(q / kappa) * lplus(adj) @ (src - src.mean())
    assert np.allclose(s.mean(0), exact, atol=0.05)


def test_lattice_resistances_of_cserti():
    """[Cse00]: on the infinite square lattice the resistance between neighbours is 1/2 and across a square 2/pi."""
    adj, _ = torus(40, 40, cap=NO_CAP)
    lp = lplus(adj)
    r = lambda p, q: lp[p, p] + lp[q, q] - 2 * lp[p, q]
    nb, diag = adj[0, 0], None
    for a in adj[0]:
        for b in adj[a]:
            if b != 0 and b not in adj[0] and len(set(adj[b]) & set(adj[0])) == 2:
                diag = b
    assert r(0, nb) == pytest.approx(0.5, abs=1e-3)
    assert r(0, diag) == pytest.approx(2 / np.pi, abs=2e-3)


def test_same_seed_same_run():
    runs = []
    for s in (9, 9, 10):
        adj, part = tube(10)
        phi, rec = run_soft(adj, part, 2.0, LAM, 1.0, 0.5, 6, 400, 8, 0.2, seed=[s, 1])
        runs.append((phi, rec, adj))
    assert np.array_equal(runs[0][0], runs[1][0]) and runs[0][1] == runs[1][1]
    assert np.array_equal(runs[0][2], runs[1][2])
    assert not np.array_equal(runs[0][0], runs[2][0])
    assert is_valid(runs[0][2], NO_CAP)


def test_waves_travel_with_the_lattice_dispersion():
    """Between refreshes the field obeys the wave equation: a plane wave of wavenumber k on the flat torus oscillates at
    omega^2 = kappa (2 - 2 cos k) for a wave along one axis, so at long wavelengths it travels at sqrt(kappa) links per
    unit time."""
    L, kappa = 32, 2.0
    adj, _ = torus(L, L, cap=NO_CAP)
    x = np.arange(L * L) // L                  # cqg.torus numbers point i at (i // L, i % L)
    k = 2 * np.pi * 3 / L
    phi0 = np.cos(k * x)
    omega = np.sqrt(kappa * (2 - 2 * np.cos(k)))
    eps, n = 0.002, 1500
    phi, _ = leapfrog(adj, phi0, np.zeros(L * L), kappa, 0.0, np.zeros(L * L), n, eps)
    assert np.allclose(phi, np.cos(omega * eps * n) * phi0, atol=1e-4)
    assert omega / k == pytest.approx(np.sqrt(kappa), rel=0.02)
