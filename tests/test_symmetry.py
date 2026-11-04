"""The fast symmetry count must equal the slow, independent one (Q15) wherever both can run."""
from math import exp

import numpy as np
import pytest

from graphity import symmetry
from graphity.cqg import NO_CAP, run_chain, torus
from graphity.small_graphs import log_automorphisms, sides_first

pytestmark = pytest.mark.skipif(not symmetry.available(), reason="igraph not installed")


def slow(adj, part):
    return round(exp(log_automorphisms(sides_first(adj, part))))


@pytest.mark.parametrize("lx, ly", [(4, 4), (4, 6), (6, 6), (8, 4), (8, 6)])
def test_perfect_tori(lx, ly):
    adj, part = torus(lx, ly, NO_CAP)
    assert symmetry.count(adj, part) == slow(adj, part)


def test_melted_graphs():
    for seed in (1, 2, 3):
        adj, part = torus(6, 6, NO_CAP)
        run_chain(adj, np.flatnonzero(part == 0), 1 / 30.0, 0, 50 * seed, seed, 1.0, NO_CAP, False)
        assert symmetry.count(adj, part) == slow(adj, part)


def test_known_values():
    adj, part = torus(4, 4, NO_CAP)
    assert symmetry.count(adj, part) == 192          # the 4-cube, sides fixed (Q15, TASKS T10)
    adj, part = torus(16, 4, NO_CAP)
    assert symmetry.count(adj, part) == 128          # the tube: 2N


def test_canonical_key_is_blind_to_renaming_within_sides_and_sees_everything_else():
    rng = np.random.default_rng(5)
    adj, part = torus(6, 6, NO_CAP)
    run_chain(adj, np.flatnonzero(part == 0), 1 / 30.0, 0, 30, 4, 1.0, NO_CAP, False)
    side0, side1 = np.flatnonzero(part == 0), np.flatnonzero(part == 1)
    perm = np.empty(36, dtype=np.int64)
    perm[side0] = rng.permutation(side0)
    perm[side1] = rng.permutation(side1)
    renamed = np.empty_like(adj)
    renamed[perm] = perm[adj]
    assert symmetry.canonical_key(adj, part) == symmetry.canonical_key(renamed, part)
    flat, _ = torus(6, 6, NO_CAP)
    assert symmetry.canonical_key(adj, part) != symmetry.canonical_key(flat, part)
