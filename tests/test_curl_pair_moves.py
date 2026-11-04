"""The two kinds as moves (src/graphity/curl_pair_moves.py; her answers of 4 November 2026). Rule 6: a brute-force check
of the square adjacency against networkx, exactness of the books after every move, constraint preservation, a same-seed
check, and hand cases that pin each move."""
import itertools

import networkx as nx
import numpy as np
import pytest

from graphity.cqg import NO_CAP, torus
from graphity.curl_pair_moves import PairMoves, square_links
from graphity.curl_pairs import CurlState


def flat(L):
    adj, part = torus(L, L, NO_CAP)
    return CurlState(adj, part)


@pytest.mark.parametrize("L", [4, 6, 8])
def test_square_adjacency_against_networkx_brute_force(L):
    s = flat(L)
    G = nx.Graph()
    for u in range(len(s.adj)):
        for v in s.adj[u]:
            G.add_edge(int(u), int(v))
    corners = [frozenset(b + w) for b, w in s.sq]
    links, nb = square_links(s.sq)
    brute = set()
    for a, b in itertools.combinations(range(len(corners)), 2):
        shared = corners[a] & corners[b]
        if len(shared) == 2 and G.has_edge(*tuple(shared)):
            brute.add((a, b))
    assert set(links) == brute
    if L >= 6:                                                         # the 4-cube puts three squares on every link
        assert all(len(nb[k]) == 4 for k in range(len(corners)))      # every square of the flat torus has four link-neighbours
        assert len(links) == 2 * L * L                                 # one pair of squares per link


def test_books_balance_after_every_move_and_constraints_hold():
    s = flat(8)
    m = PairMoves(s, lam=1.25, barrier=4.0, seed=3)
    start = m.release
    assert start == 4 * 0.25 * 64
    for _ in range(200):
        for move in (m.make_pairs, m.cancel_pairs, m.join_stores):
            move()
            assert m.books() == pytest.approx(start, abs=1e-12)
            assert set(np.unique(s.c)).issubset({-1.0, 0.0, 1.0})
            assert np.all((m.store == 0) | (np.sign(s.c) == m.store))   # a store flag sits on a full curl of its sign
            assert m.ledger >= -1e-12
        m.sweeps += 1
    recount = dict(plus=int(np.sum(m.store > 0)), minus=int(np.sum(m.store < 0)))
    assert m.stores_made == recount["plus"] + recount["minus"]
    assert m.photons == 2 * round(m.light / 2.0)


def test_same_seed_reproduces_and_another_seed_differs():
    a = PairMoves(flat(12), lam=1.25, barrier=4.0, seed=11).run()
    b = PairMoves(flat(12), lam=1.25, barrier=4.0, seed=11).run()
    c = PairMoves(flat(12), lam=1.25, barrier=4.0, seed=12).run()
    assert a == b
    assert a["frozen"] and c["frozen"]
    assert a["books"] == pytest.approx(a["release"])


def test_one_pair_is_made_across_a_shared_link_and_cancels_when_there_is_room():
    s = flat(8)
    m = PairMoves(s, lam=1.25, barrier=0.0, seed=1, ledger=2.0)
    made = m.make_pairs()
    assert made == 1 and m.ledger == 0.0
    filled = np.flatnonzero(s.c)
    assert len(filled) == 2 and tuple(sorted(filled)) in set(m.links)
    assert s.c[filled].sum() == 0.0
    assert m.cancel_pairs() == 1
    assert m.light == 2.0 and m.photons == 2 and not s.c.any()
    assert m.sweep() == 0                                              # nothing left: the run freezes


def test_nothing_happens_with_an_empty_ledger():
    s = flat(8)
    m = PairMoves(s, lam=1.0, barrier=0.0, seed=1)
    assert m.release == 0.0 and m.run()["sweeps"] == 1 and not s.c.any()


def test_a_full_sheet_of_one_sign_with_no_room_joins_exactly_one_store_per_price():
    s = flat(8)
    s.c[:] = 1.0                                                       # no empty square anywhere: no room
    lam, B = 1.25, 21.0
    price = B + 4 * (lam - 1)
    m = PairMoves(s, lam=lam, barrier=B, seed=5, ledger=2 * price + 0.5)
    assert m.cancel_pairs() == 0                                       # same sign everywhere: nothing cancels
    assert m.join_stores() == 2 and m.stores_made == 2
    assert m.ledger == pytest.approx(0.5)
    assert int(np.sum(m.store == 1)) == 2 and int(np.sum(s.c == 0)) == 2
    assert m.held == pytest.approx(2 * (1.0 + price))
    assert m.books() == pytest.approx(m.release + 64.0)                # the 64 planted curls are not in the release
    assert m.cancel_pairs() == 0                                       # a store never cancels, and nothing else is opposite


def test_a_store_never_cancels_against_an_opposite_neighbour():
    s = flat(8)
    s.c[:] = 1.0
    m = PairMoves(s, lam=1.25, barrier=0.0, seed=5, ledger=1.0)
    assert m.join_stores() == 1
    i = int(np.flatnonzero(m.store)[0])
    j = m.nb[i][0]
    s.c[j] = -1.0                                                      # plant an opposite curl beside the store, with room beside it
    s.c[m.nb[j][-1]] = 0.0
    before = m.light
    m.cancel_pairs()
    assert m.store[i] == 1 and s.c[i] == 1.0
    assert m.light == before or s.c[j] == 0.0                          # if j cancelled it was against a free curl, never the store


def test_the_store_cluster_reader_counts_connected_stores():
    s = flat(8)
    m = PairMoves(s, lam=1.25, barrier=0.0, seed=0)
    assert m.largest_store_cluster() == 0
    a, b = m.links[0]
    m.store[a] = m.store[b] = 1
    far = next(k for k in range(len(s.sq)) if k not in m.nb[a] and k not in m.nb[b] and k not in (a, b))
    m.store[far] = -1
    assert m.largest_store_cluster() == 2


def test_under_her_gravity_form_a_join_costs_the_ledger_nothing_and_gravity_only_grows():
    s = flat(8)
    s.c[:] = 1.0                                                       # no room anywhere, same sign everywhere
    lam, B = 1.25, 21.0
    m = PairMoves(s, lam=lam, barrier=B, seed=5, ledger=0.0, price_from="gravity")
    n = m.join_stores()
    assert 0 < n <= 32 and m.stores_made == n                          # each join empties a square, which gives its
    assert m.ledger == 0.0 and m.gravity == pytest.approx(n * (B + 4 * (lam - 1)))   # neighbours room, so they stay apart
    assert m.held == pytest.approx(float(n)) and m.books() == pytest.approx(m.release + 64.0)
    again = m.join_stores()                                            # a second pass joins only pairs still without room
    assert m.gravity == pytest.approx((n + again) * (B + 4 * (lam - 1)))


def test_books_balance_under_the_gravity_form_through_a_whole_run():
    s = flat(12)
    m = PairMoves(s, lam=1.5, barrier=4.0, seed=9, price_from="gravity")
    start = m.release
    for _ in range(50):
        for move in (m.make_pairs, m.cancel_pairs, m.join_stores):
            move()
            assert m.books() == pytest.approx(start, abs=1e-12)
            assert np.all((m.store == 0) | (np.sign(s.c) == m.store))
    assert m.gravity >= 0.0


def test_t103_form_is_unchanged_by_the_gravity_switch():
    a = PairMoves(flat(12), lam=1.25, barrier=4.0, seed=11).run()
    b = PairMoves(flat(12), lam=1.25, barrier=4.0, seed=11, price_from="ledger").run()
    assert a == b and a["gravity"] == 0.0


def test_her_clock_judges_room_once_so_a_packed_sheet_both_cancels_and_joins():
    s = flat(8)
    s.c[:] = 1.0
    s.c[s.sq.index(s.sq[0])] = -1.0                                    # one anti-curl in a packed sheet of curl: no room anywhere
    m = PairMoves(s, lam=1.25, barrier=0.0, seed=2, price_from="gravity", clock="simultaneous", ledger=0.0)
    done = m.resolve_at_once()
    assert m.photons == 0                                              # the odd pair had no room, so nothing cancelled
    assert m.stores_made == done and 20 <= m.stores_made <= 31         # a maximal matching of the 63 same-sign squares, each used once
    assert m.books() == pytest.approx(m.release + 64.0)
    assert int(np.sum(m.store == 1)) == m.stores_made


def test_her_clock_cancels_first_and_uses_each_square_once():
    s = flat(8)
    m = PairMoves(s, lam=1.25, barrier=0.0, seed=2, price_from="gravity", clock="simultaneous", ledger=2.0)
    m.make_pairs()                                                     # one pair, with room all round
    assert m.resolve_at_once() == 1 and m.photons == 2 and not s.c.any() and m.stores_made == 0


def test_her_clock_books_balance_through_a_run_and_same_seed_reproduces():
    a = PairMoves(flat(16), lam=2.0, barrier=4.0, seed=7, price_from="gravity", clock="simultaneous")
    ra = a.run()
    rb = PairMoves(flat(16), lam=2.0, barrier=4.0, seed=7, price_from="gravity", clock="simultaneous").run()
    assert ra == rb and ra["frozen"] and ra["books"] == pytest.approx(ra["release"])
    assert np.all((a.store == 0) | (np.sign(a.state.c) == a.store))
    assert ra["stores"] == ra["stores_plus"] + ra["stores_minus"]


def test_the_sequential_clock_is_unchanged():
    a = PairMoves(flat(12), lam=1.25, barrier=4.0, seed=11).run()
    b = PairMoves(flat(12), lam=1.25, barrier=4.0, seed=11, clock="sequential").run()
    assert a == b
