"""The per-vertex store option of the bath kernel (PREREGISTRATION T26; ASSUMPTIONS Q22) and stream carry-on.

Three things: the shared-bath path is bit for bit what it was before the option existed (a pinned run);
with by_vertex the energy is conserved exactly, only a move made from a vertex can spend that vertex's
store, and stores of side-1 vertices are never touched; and seed < 0 carries the stream on, so two
blocks equal one run of their combined length.
"""
import hashlib

import numpy as np

from graphity.cqg import NO_CAP, hamiltonian, torus
from graphity.sealed import run_sealed_bath

# Recorded on 2026-09-24 from the kernel as committed before `by_vertex` and `seed < 0` were added:
# 16 x 4 torus, 32 demons with 40 units in the first, 400 sweeps, seed 2024, lambda 1.25.
PINNED = "a6453d572db03e604c5a24e19678fdc558d6171481dd27b90cf8bdda33e4a461"


def test_the_shared_bath_is_bit_for_bit_what_it_was():
    adj, part = torus(16, 4, NO_CAP)
    side_u = np.flatnonzero(part == 0)
    demons = np.zeros(32)
    demons[0] = 40.0
    s, x, mean, tot, acc = run_sealed_bath(adj, side_u, demons, 400, 2024, 1.25, NO_CAP)
    h = hashlib.sha256(np.concatenate([s, x, adj.ravel()]).tobytes() + tot.tobytes()).hexdigest()
    assert h == PINNED
    assert int(s[-1]) == 67 and int(x[-1]) == 20 and float(tot[-1]) == 52.0


def test_per_vertex_stores_conserve_energy_and_leave_side_one_alone():
    lam = 1.25
    adj, part = torus(12, 12, NO_CAP)
    n = adj.shape[0]
    side_u = np.flatnonzero(part == 0)
    stores = np.zeros(n)
    hot = side_u[:6]
    stores[hot] = 8.0
    e0 = hamiltonian(adj, lam) + stores.sum()
    s, x, mean, tot, _ = run_sealed_bath(adj, side_u, stores, 300, 31, lam, NO_CAP, by_vertex=True)
    h = 16.0 * (n - s) + 4.0 * lam * x
    assert np.allclose(h + tot, e0)
    assert abs(hamiltonian(adj, lam) + stores.sum() - e0) < 1e-9
    assert (stores >= -1e-12).all()
    assert (stores[part == 1] == 0.0).all()          # never chosen as u1, never touched


def test_only_a_move_from_the_hot_vertex_can_spend_its_store():
    """With energy in one vertex's store and none anywhere else, the first uphill move must be made from
    that vertex; a shared bath would let any vertex spend it."""
    lam = 1.25
    adj, part = torus(12, 12, NO_CAP)
    n = adj.shape[0]
    side_u = np.flatnonzero(part == 0)
    hot = int(side_u[17])
    stores = np.zeros(n)
    stores[hot] = 100.0
    s, x, mean, tot, _ = run_sealed_bath(adj, side_u, stores, 50, 5, lam, NO_CAP, by_vertex=True)
    assert s[-1] < n                                   # the sheet has been damaged, so the store was spent
    spent = stores.sum() < 100.0
    assert spent
    # Every store other than the hot vertex's can only have gained (a move paid into it), never gone
    # negative, and the hot vertex's store is the only one below its starting value.
    others = np.delete(np.arange(n), hot)
    assert (stores[others] >= 0.0).all()
    assert stores[hot] < 100.0


def test_seed_below_zero_carries_the_stream_on():
    lam = 1.25
    a1, part = torus(16, 4, NO_CAP)
    a2 = a1.copy()
    side_u = np.flatnonzero(part == 0)
    d1 = np.zeros(2 * 64); d1[0] = 12.0
    d2 = d1.copy()
    s_one, x_one, _, tot_one, _ = run_sealed_bath(a1, side_u, d1, 200, 9, lam, NO_CAP)
    s_a, x_a, _, tot_a, _ = run_sealed_bath(a2, side_u, d2, 120, 9, lam, NO_CAP)
    s_b, x_b, _, tot_b, _ = run_sealed_bath(a2, side_u, d2, 80, -1, lam, NO_CAP)
    assert (np.concatenate([s_a, s_b]) == s_one).all() and (np.concatenate([x_a, x_b]) == x_one).all()
    assert np.allclose(np.concatenate([tot_a, tot_b]), tot_one)
    assert (a1 == a2).all() and np.allclose(d1, d2)
