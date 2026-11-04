"""The curl-and-anti-curl state on the squares (src/graphity/curl_pairs.py; VISION Update 156): the exact fact it rests
on, and its readers against hand cases."""
import pytest

from graphity.cqg import NO_CAP, torus
from graphity.curl_pairs import ANTIMATTER, MATTER, CurlState, daughter_front, squares, wiring_is_paired


def test_every_square_of_the_flat_torus_has_one_black_and_one_white_diagonal():
    adj, part = torus(8, 8, NO_CAP)
    sq = squares(adj, part)
    assert len(sq) == 64                                   # one square per point on the flat torus
    assert wiring_is_paired(adj, part)
    for black, white in sq:
        assert part[black[0]] == part[black[1]] == 0 and part[white[0]] == part[white[1]] == 1
        assert not (adj[black[0]] == black[1]).any()        # a diagonal is never a link


def test_the_four_cube_is_paired_too():
    adj, part = torus(4, 4, NO_CAP)
    assert len(squares(adj, part)) == 24 and wiring_is_paired(adj, part)


def test_the_wiring_alone_holds_no_surplus():
    adj, part = torus(8, 8, NO_CAP)
    s = CurlState(adj, part)
    assert s.net_curl() == 0 and s.stored() == 0 and s.census()["empty"] == 64


def test_a_pair_nets_zero_and_a_full_curl_is_the_only_stable_store():
    adj, part = torus(8, 8, NO_CAP)
    s = CurlState(adj, part)
    s.set_pair(0, 1, 0.5)
    assert s.net_curl() == 0 and s.stored() == 1.0 and s.stable() == 0
    assert s.census()["matter"] == 1 and s.census()["antimatter"] == 1
    s.set_curl(2, 1.0)
    assert s.net_curl() == 1.0 and s.stable() == 1
    assert s.stored_pairs(MATTER) == [s.sq[0][0], s.sq[2][0]] and s.stored_pairs(ANTIMATTER) == [s.sq[1][1]]


def test_the_rules_are_enforced():
    adj, part = torus(8, 8, NO_CAP)
    s = CurlState(adj, part)
    with pytest.raises(ValueError):
        s.set_pair(0, 0, 0.5)
    with pytest.raises(ValueError):
        s.set_pair(0, 1, 1.5)
    with pytest.raises(ValueError):
        s.set_curl(0, -2.0)


def test_daughter_front_matches_the_arithmetic_written_before_it():
    for lam, s in ((1.10, 0.2), (1.25, 0.5), (1.50, 1.0), (1.75, 1.0)):
        r = daughter_front(L=8, R=4, lam=lam)
        assert abs(r["pair_size"] - s) < 1e-12
        assert r["pairs_made"] == 8 * 4 and r["frozen"] == 0
        assert abs(r["surplus_energy"] - s * 8) < 1e-12 and abs(r["gross"] - s * 8 * 4) < 1e-12
        assert abs(r["surplus_fraction"] - 1 / 4) < 1e-12
        assert r["photons"] == 2 * 8 * 3 and abs(r["photon_energy"] - 2 * s * 8 * 3) < 1e-12
    assert abs(daughter_front(8, 4, 1.75)["light"] - 8 * 4 * (4 * 0.75 - 2.0)) < 1e-12
    assert daughter_front(8, 4, 1.25)["light"] == 0.0


def test_daughter_front_surplus_is_the_boundary_row_only():
    r = daughter_front(L=16, R=8, lam=1.25)
    assert abs(r["surplus_curl"] - 0.5 * 16) < 1e-12          # the inside nets the last row's matter alone
