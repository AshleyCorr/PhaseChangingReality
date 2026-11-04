"""The pieces of scripts/explore_bianconi_action.py that must be exact: the boundary maps, the count of squares, and
the identity that turns three determinants into two."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import explore_bianconi_action as ba                           # noqa: E402
from graphity.cqg import NO_CAP, run_chain, torus, total_squares   # noqa: E402


def warmed(seed=5):
    adj, part = torus(8, 8, NO_CAP)
    run_chain(adj, np.flatnonzero(part == 0), 1.0 / 6.0, 0, 200, seed, 1.0, NO_CAP, False)
    return adj, part


def cases():
    return [torus(8, 8, NO_CAP), torus(16, 4, NO_CAP), torus(4, 4, NO_CAP), warmed()]


def test_the_boundary_of_a_boundary_is_zero_and_the_squares_are_the_models():
    for adj, part in cases():
        b1, b2 = ba.boundaries(adj, part)
        assert b1.shape == (adj.shape[0], 2 * adj.shape[0])
        assert np.abs(b1 @ b2).max() == 0.0
        assert b2.shape[1] == total_squares(adj)             # every 4-cycle, once
        assert (np.abs(b2).sum(axis=0) == 4).all() and (np.abs(b1).sum(axis=0) == 2).all()


def test_three_determinants_are_twice_two():
    for adj, part in cases():
        b1, b2 = ba.boundaries(adj, part)
        l0, l1, l2 = b1 @ b1.T, b1.T @ b1 + b2 @ b2.T, b2.T @ b2
        for c0 in (0.1, 1.0, 10.0):
            direct = ba.logdet(l0, c0) + ba.logdet(l1, c0) + ba.logdet(l2, c0)
            assert abs(direct - ba.action_parts(adj, part, (c0,))[3][c0]) < 1e-8 * max(1.0, abs(direct))


def test_the_flat_torus_has_the_lattice_spectrum():
    adj, part = torus(8, 8, NO_CAP)
    b1, _ = ba.boundaries(adj, part)
    k = 2.0 * np.pi * np.arange(8) / 8
    lattice = np.sort((4.0 - 2.0 * np.cos(k)[:, None] - 2.0 * np.cos(k)[None, :]).ravel())
    assert np.allclose(np.sort(np.linalg.eigvalsh(b1 @ b1.T)), lattice, atol=1e-9)


def test_sides_found_by_walking_agree_with_the_given_ones_up_to_a_swap():
    for adj, part in cases():
        found = ba.side_of(adj)
        assert (found == part).all() or (found == 1 - part).all()
        b1a, b2a = ba.boundaries(adj, part)
        b1b, b2b = ba.boundaries(adj, None)
        assert b1a.shape == b1b.shape and b2a.shape == b2b.shape
