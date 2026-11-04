"""Link lengths relaxed from curl (graphity.lengths): rule-6 checks and the exact numbers of the brief's step 1."""
import networkx as nx
import numpy as np
import pytest

from graphity import lengths
from graphity.cqg_d import _switch, hamiltonian, is_valid, torus
from graphity.graphs import to_networkx

DIMS = (6, 6, 6)


def switched(n_switches, seed):
    """A flat 6^3 torus with n valid random switches (not dips; a test fixture)."""
    adj, part = torus(DIMS)
    rng = np.random.default_rng(seed)
    side = np.flatnonzero(part == 0)
    done = 0
    while done < n_switches:
        u1, u2 = rng.choice(side, 2, replace=False)
        v1, v2 = adj[u1, rng.integers(6)], adj[u2, rng.integers(6)]
        if v1 == v2 or v1 in adj[u2] or v2 in adj[u1]:
            continue
        _switch(adj, u1, v1, u2, v2)
        if is_valid(adj):
            done += 1
        else:
            _switch(adj, u1, v2, u2, v1)
    return adj, part


def test_flat_torus_has_no_curl_and_no_relaxed_energy():
    adj, _ = torus(DIMS)
    f = lengths.curl_fields(adj, DIMS)
    assert f.shape == (3,) + DIMS and not f.any()
    assert lengths.relaxed_energy(adj, DIMS, -0.5) == 0.0


def test_stiffness_is_step_ones_numbers():
    """K on a unit perturbation of one link: 24 on itself, -4 on its six parallel neighbours, 0 elsewhere
    (sandbox/regge_step1_stiffness.py: 12 d^2 and -4 d^2 for one link and a parallel pair, i.e. K_ee = 24, K_ef = -4)."""
    e = np.zeros(DIMS); e[1, 2, 3] = 1.0
    k = lengths.apply_stiffness(e)
    assert k[1, 2, 3] == 24.0
    neighbours = [(0, 2, 3), (2, 2, 3), (1, 1, 3), (1, 3, 3), (1, 2, 2), (1, 2, 4)]
    assert all(k[p] == -4.0 for p in neighbours)
    assert np.count_nonzero(k) == 7 and k.sum() == 0.0


def test_relax_inverts_the_stiffness_up_to_the_zero_mode():
    rng = np.random.default_rng(0)
    f = rng.random((3,) + DIMS)
    delta, energy = lengths.relax(f, -0.7)
    for a in range(3):
        kd = lengths.apply_stiffness(delta[a])
        assert np.allclose(kd, 0.7 * (f[a] - f[a].mean()))          # K delta = -alpha (f - mean)
        assert abs(delta[a].mean()) < 1e-12
    # E_relax = -(alpha^2/2) f^T K^+ f = (alpha/2) f^T delta
    assert np.isclose(energy, sum(-0.7 / 2 * np.sum(f[a] * delta[a]) for a in range(3)))


def test_two_point_sources_attract_for_either_sign_and_fall_with_distance():
    e = np.zeros((3,) + DIMS); e[0, 0, 0, 0] = 1.0
    one = lengths.relax(e, 0.5)[1]
    prev = -np.inf
    for r in (1, 2, 3):
        two = e.copy(); two[0, r, 0, 0] += 1.0
        for alpha in (0.5, -0.5):
            e_int = lengths.relax(two, alpha)[1] - 2 * one
            assert e_int < 0
        assert e_int > prev                   # weaker (less negative) as r grows
        prev = e_int


def test_curl_fields_against_networkx_square_counts():
    adj, _ = switched(3, 1)
    g = to_networkx(adj)
    f = lengths.curl_fields(adj, DIMS)
    total = 0
    for u, v in g.edges():
        squares = sum(1 for a in g[u] for b in g[v] if a != v and b != u and g.has_edge(a, b))
        total += abs(squares - 4)
    assert f.sum() == total and total > 0


def test_total_energy_is_hamiltonian_plus_relaxed():
    adj, _ = switched(2, 2)
    assert np.isclose(lengths.total_energy(adj, DIMS, 1.25, -0.5),
                      hamiltonian(adj, 1.25) + lengths.relaxed_energy(adj, DIMS, -0.5))


@pytest.mark.parametrize("alpha", [0.0, -0.5])
def test_chain_keeps_the_graph_valid_tracks_the_energy_and_reproduces_its_seed(alpha):
    adj, part = torus(DIMS)
    side = np.flatnonzero(part == 0)
    h, e, acc = lengths.run_chain_relaxed(adj, side, DIMS, 1 / 20.0, 2, 3, 7, 1.25, alpha)
    assert is_valid(adj) and acc > 0
    assert np.isclose(h[-1], hamiltonian(adj, 1.25))
    assert np.isclose(e[-1], lengths.relaxed_energy(adj, DIMS, alpha))
    adj2, part2 = torus(DIMS)
    h2, e2, acc2 = lengths.run_chain_relaxed(adj2, side, DIMS, 1 / 20.0, 2, 3, 7, 1.25, alpha)
    assert np.array_equal(adj, adj2) and np.array_equal(h, h2) and acc == acc2


def test_chain_at_alpha_zero_never_pays_a_relaxed_energy():
    adj, part = torus(DIMS)
    side = np.flatnonzero(part == 0)
    _, e, _ = lengths.run_chain_relaxed(adj, side, DIMS, 1 / 20.0, 1, 2, 3, 1.25, 0.0)
    assert not e.any()
