"""The local spark (graphity.spark; PREREGISTRATION T26): distances, local stores, and the hot patch."""
import numpy as np

from graphity.cqg import NO_CAP, hamiltonian, is_valid, torus
from graphity.spark import distances_from, hot_patch, local_stores, patch_vertices


def test_distances_on_the_flat_torus_are_the_lattice_ones():
    adj, part = torus(12, 12, NO_CAP)
    d = distances_from(adj, 0)
    assert d[0] == 0
    assert (d >= 0).all() and d.max() == 12          # a 12 x 12 torus: at most 6 + 6 steps
    assert sorted(d[adj[0]]) == [1, 1, 1, 1]
    assert (d[part == 0] % 2 == 0).all() and (d[part == 1] % 2 == 1).all()   # bipartite


def test_local_stores_share_the_energy_over_the_patch_side_zero_only():
    adj, part = torus(12, 12, NO_CAP)
    stores = local_stores(adj, part, 0, 2, 30.0)
    inside = patch_vertices(adj, 0, 2)
    holders = np.flatnonzero(stores > 0)
    assert set(holders) <= set(inside) and (part[holders] == 0).all()
    assert abs(stores.sum() - 30.0) < 1e-9 and np.allclose(stores[holders], 30.0 / len(holders))
    everywhere = local_stores(adj, part, 0, None, 144.0)
    assert (everywhere[part == 0] == 2.0).all() and (everywhere[part == 1] == 0).all()


def test_hot_patch_delivers_the_budget_inside_the_radius_and_stays_valid():
    lam = 1.25
    adj, part = torus(12, 12, NO_CAP)
    before = adj.copy()
    rng = np.random.default_rng(3)
    got = hot_patch(adj, part, 0, 3, 64.0, lam, NO_CAP, rng)
    assert got == 64.0
    assert abs(hamiltonian(adj, lam) - 64.0) < 1e-9
    assert is_valid(adj, NO_CAP)
    changed = np.flatnonzero((adj != before).any(axis=1))
    d = distances_from(before, 0)
    assert (d[changed] <= 3).all()                    # nothing outside the patch was touched


def test_hot_patch_is_reproducible_from_its_seed():
    lam = 1.25
    a1, part = torus(12, 12, NO_CAP); a2 = a1.copy()
    hot_patch(a1, part, 0, 3, 48.0, lam, NO_CAP, np.random.default_rng(11))
    hot_patch(a2, part, 0, 3, 48.0, lam, NO_CAP, np.random.default_rng(11))
    assert (a1 == a2).all()


def test_hot_patch_never_exceeds_the_budget():
    lam = 1.25
    adj, part = torus(12, 12, NO_CAP)
    got = hot_patch(adj, part, 0, 2, 10.0, lam, NO_CAP, np.random.default_rng(5), max_tries=5000)
    assert got <= 10.0 and abs(hamiltonian(adj, lam) - got) < 1e-9
