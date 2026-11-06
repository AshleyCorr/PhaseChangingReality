"""The loose band (graphity.loose_band): validity, locality and reproducibility of the loosening."""
import networkx as nx
import numpy as np
import pytest

from graphity.cqg import NO_CAP, is_valid, torus, total_squares
from graphity.loose_band import band_mask, loose_band_tube, loosen


def _graph(adj):
    return nx.Graph([(u, int(v)) for u in range(len(adj)) for v in adj[u] if v >= 0])


@pytest.mark.parametrize("L,k", [(32, 2), (48, 4), (64, 8)])
def test_loosened_tube_stays_valid_bipartite_and_four_regular(L, k):
    adj, part, in_band, made = loose_band_tube(L, k, np.random.default_rng(1))
    assert made == k
    assert is_valid(adj, NO_CAP)
    g = _graph(adj)
    assert nx.is_bipartite(g)
    assert all(d == 4 for _, d in g.degree())
    col = nx.bipartite.color(g)          # networkx picks which side is 0; the two colourings must agree up to a swap
    same = [col[i] == part[i] for i in range(len(adj))]
    assert all(same) or not any(same)


def test_squares_are_lost_only_inside_the_band():
    L, k = 32, 4
    adj0, _ = torus(L, 4, NO_CAP)
    adj, part, in_band, made = loose_band_tube(L, k, np.random.default_rng(3))
    assert total_squares(adj) < total_squares(adj0)
    # every link outside the band is unchanged
    outside = np.flatnonzero(~in_band)
    for u in outside:
        assert set(adj0[u].tolist()) == set(adj[u].tolist())


def test_band_mask_is_centred_and_circular():
    """Exact: the band is the columns strictly within width/2 of the centre, so width 8 gives 7 columns (29 to 35
    at L = 64): the look of 5 November printed 'of 32 band points' for what are 28; T107's registration says 7."""
    m = band_mask(64, 8)
    assert m.sum() == 7 * 4
    cols = set((np.flatnonzero(m) // 4).tolist())
    assert cols == set(range(29, 36))
    m2 = band_mask(64, 8, centre=2)
    assert set((np.flatnonzero(m2) // 4).tolist()) == {63, 0, 1, 2, 3, 4, 5}
    assert band_mask(32, 4).sum() == 3 * 4 and band_mask(48, 6).sum() == 5 * 4 and band_mask(128, 16).sum() == 15 * 4


def test_same_seed_reproduces_and_another_differs():
    a1, *_ = loose_band_tube(32, 4, np.random.default_rng(7))
    a2, *_ = loose_band_tube(32, 4, np.random.default_rng(7))
    a3, *_ = loose_band_tube(32, 4, np.random.default_rng(8))
    assert np.array_equal(a1, a2)
    assert not np.array_equal(a1, a3)


def test_k_zero_is_the_packed_tube():
    adj0, _ = torus(32, 4, NO_CAP)
    adj, part, in_band, made = loose_band_tube(32, 0, np.random.default_rng(0))
    assert made == 0 and np.array_equal(adj, adj0)


def test_loosen_refuses_switches_that_break_the_cap_rule():
    """With cap = 2 (two shared neighbours at most, the hard-core rule) every made switch keeps is_valid."""
    L = 32
    adj, part = torus(L, 4, NO_CAP)
    in_band = band_mask(L, 4)
    made = loosen(adj, part, np.random.default_rng(11), in_band, 6)
    assert made == 6 and is_valid(adj, NO_CAP)
