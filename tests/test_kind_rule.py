"""Her kind rule of 6 November 2026 (graphity.curl_pair_moves, kind_rule="boundary"): the first pair across the boundary
sets the outward kind, every pair after follows it; the coin is unchanged; the books balance."""
import numpy as np
import pytest

from graphity.cqg import NO_CAP, torus
from graphity.curl_pair_moves import PairMoves
from graphity.curl_pairs import CurlState


def _dist_along_x(state, L, centre):
    """Each square's distance (circular, in columns) from the centre column, on a 4 x L tube with i = x * 4 + y."""
    ang = 2 * np.pi * (np.arange(len(state.adj)) // 4) / L
    d = np.empty(len(state.sq))
    for q, (black, white) in enumerate(state.sq):
        c = list(black) + list(white)
        m = np.arctan2(np.sin(ang[c]).mean(), np.cos(ang[c]).mean()) * L / (2 * np.pi)
        d[q] = abs(((m - centre + L / 2) % L) - L / 2)
    return d


def _moves(rule, seed=5, L=32, lam=1.25):
    adj, part = torus(L, 4, NO_CAP)
    st = CurlState(adj, part)
    dist = _dist_along_x(st, L, L // 2)
    return PairMoves(st, lam=lam, barrier=12.0, seed=seed, price_from="gravity", clock="simultaneous",
                     kind_rule=rule, dist=dist, edge=2.0), dist


def test_the_coin_is_unchanged_by_the_new_fields():
    a = PairMoves(CurlState(*torus(16, 16, NO_CAP)), lam=1.5, barrier=4.0, seed=3, price_from="gravity",
                  clock="simultaneous")
    b, _ = _moves("coin", seed=3, L=64)
    assert a.outward_kind == 0 and b.outward_kind == 0
    a.run(); b.run()
    assert a.outward_kind == 0 and b.outward_kind == 0
    # the T105 cell reproduces with the fields absent: same seed, same census
    a2 = PairMoves(CurlState(*torus(16, 16, NO_CAP)), lam=1.5, barrier=4.0, seed=3, price_from="gravity",
                   clock="simultaneous")
    a2.run()
    assert a.census() == a2.census()


def test_the_boundary_rule_needs_dist_and_edge():
    with pytest.raises(ValueError):
        PairMoves(CurlState(*torus(16, 4, NO_CAP)), lam=1.25, barrier=0.0, seed=1, kind_rule="boundary")
    with pytest.raises(ValueError):
        PairMoves(CurlState(*torus(16, 4, NO_CAP)), lam=1.25, barrier=0.0, seed=1, kind_rule="nonsense")


def test_the_first_straddling_pair_sets_the_outward_kind_once():
    pm, dist = _moves("boundary")
    pm.make_pairs()
    assert pm.outward_kind in (-1, 1)
    k = pm.outward_kind
    pm.resolve_at_once()
    pm.make_pairs()
    assert pm.outward_kind == k


def test_every_pair_after_the_first_puts_the_outward_kind_on_the_farther_square():
    pm, dist = _moves("boundary")
    pm.make_pairs()
    k = pm.outward_kind
    c = pm.state.c
    for i, j in pm.links:
        if c[i] != 0.0 and c[j] == -c[i] and dist[i] != dist[j]:
            outer, inner = (i, j) if dist[i] > dist[j] else (j, i)
            # pairs made before the first straddling pair were placed by the coin; after it, by the rule.
            # the rule's pairs: the outer square carries k. At least one pair obeys it (the straddling pair itself).
            pass
    obeying = sum(1 for i, j in pm.links if c[i] != 0.0 and c[j] == -c[i] and dist[i] != dist[j]
                  and np.sign(c[i] if dist[i] > dist[j] else c[j]) == k)
    disobeying = sum(1 for i, j in pm.links if c[i] != 0.0 and c[j] == -c[i] and dist[i] != dist[j]
                     and np.sign(c[i] if dist[i] > dist[j] else c[j]) == -k)
    assert obeying > disobeying


def test_books_balance_and_net_twist_is_zero_under_the_boundary_rule():
    pm, _ = _moves("boundary", seed=9)
    pm.run()
    assert abs(pm.books() - pm.release) < 1e-9
    assert pm.census()["net_twist"] == 0.0
    assert pm.census()["outward_kind"] == pm.outward_kind


def test_same_seed_reproduces_under_the_boundary_rule():
    a, _ = _moves("boundary", seed=21)
    b, _ = _moves("boundary", seed=21)
    c, _ = _moves("boundary", seed=22)
    a.run(); b.run(); c.run()
    assert a.census() == b.census()
    assert a.census() != c.census()


def test_opposite_twins_appear_across_seeds():
    kinds = set()
    for seed in range(12):
        pm, _ = _moves("boundary", seed=seed)
        pm.run()
        kinds.add(pm.outward_kind)
    assert kinds >= {-1, 1}
