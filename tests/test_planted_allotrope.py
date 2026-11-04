"""scripts/build_planted_allotrope.py: the finite Fig. 7 background, the planted allotropes, and their cheapest switches.

The square counts are checked against networkx (an independent count of 4-cycles), the group relations are checked
on the permutations themselves, and the switch enumeration is checked on the flat torus against the wall on the record
(every switch out of flat space costs at least 32; ASSUMPTIONS O22, O51).
"""
import importlib.util
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("build_planted_allotrope", SCRIPTS / "build_planted_allotrope.py")
bpa = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bpa)

from graphity.cqg_d import hamiltonian, is_valid, surplus, total_squares   # noqa: E402


@pytest.fixture(scope="module")
def objs():
    return bpa.build_all(240, 14)


def squares_by_networkx(adj):
    g = nx.Graph()
    g.add_edges_from((int(u), int(v)) for u in range(adj.shape[0]) for v in adj[u])
    assert nx.is_bipartite(g)
    return sum(1 for c in nx.simple_cycles(g, length_bound=4) if len(c) == 4)


def _power(p, k):
    out = tuple(range(len(p)))
    for _ in range(k):
        out = bpa._mul(out, p)
    return out


@pytest.mark.parametrize("size", sorted(bpa.GROUPS))
def test_generators_satisfy_the_relations(size):
    a, b = (tuple(x) for x in bpa.GROUPS[size])
    e = tuple(range(len(a)))
    c = bpa._mul(bpa._inv(b), a)
    assert _power(a, 6) == e and all(_power(a, k) != e for k in (1, 2, 3))
    assert _power(b, 4) == e and _power(b, 2) != e
    assert _power(c, 2) == e and c != e
    # both odd permutations, so the sign 2-colours the Cayley graph
    for p in (a, b):
        seen, s = set(), 1
        for i in range(len(p)):
            j, n = i, 0
            while j not in seen:
                seen.add(j)
                j = p[j]
                n += 1
            s *= -1 if n and n % 2 == 0 else 1
        assert s == -1


# (name, N, S, H, points by squares touched, squares on the links of the planted points)
EXPECTED = [
    ("hyperbolic_background", 240, 180, 960, {3: 240}, None),
    ("hyperbolic_fold", 234, 171, 1008, {2: 18, 3: 216}, (1, 1, 1, 1)),
    ("hyperbolic_handle", 228, 162, 1056, {2: 36, 3: 192}, (1, 1, 1, 1)),
    ("flat_torus", 196, 196, 0, {4: 196}, None),
    ("flat_handle", 188, 182, 96, {3: 24, 4: 164}, (1, 1, 2, 2)),
]


@pytest.mark.parametrize("name,n,s,h,hist,pattern", EXPECTED)
def test_each_object_is_a_valid_state_with_the_stated_numbers(objs, name, n, s, h, hist, pattern):
    adj, part, bg = objs[name]
    assert adj.shape == (n, 4)
    assert is_valid(adj)
    assert bpa._two_colour(adj) is not None
    assert all(part[u] != part[v] for u in range(n) for v in adj[u])
    assert int(total_squares(adj)) == s == squares_by_networkx(adj)
    assert int(surplus(adj)) == 0
    assert float(hamiltonian(adj, 1.0)) == h == 16 * (n - s)
    c = bpa.square_counts(adj)
    assert {int(k): int(v) for k, v in enumerate(np.bincount(c)) if v} == hist
    if pattern is not None:
        reg, _ = bpa.region(adj, bg)
        assert {bpa.link_pattern(adj, int(v)) for v in reg} == {pattern}


def test_background_points_all_look_like_fig7(objs):
    adj, _, _ = objs["hyperbolic_background"]
    assert {bpa.link_pattern(adj, v) for v in range(adj.shape[0])} == {(1, 1, 2, 2)}
    # every point is on exactly one hexagon walked by slot 0
    assert all(len(set(bpa.hexagon(adj, g))) == 6 for g in range(adj.shape[0]))


def test_building_is_deterministic(objs):
    again = bpa.build_all(240, 14)
    for name in objs:
        assert np.array_equal(objs[name][0], again[name][0])


def _cheapest(objs, name):
    adj, part, bg = objs[name]
    reg, _ = bpa.region(adj, bg)
    near = np.flatnonzero((part == 0) & (bpa.distances(adj, reg if len(reg) else [0]) <= 1))
    if not len(reg):
        near = near[:1]
    return bpa.switches(adj, part, near)


def test_flat_torus_wall_is_the_one_on_the_record(objs):
    mv = _cheapest(objs, "flat_torus")
    assert min(m[0] for m in mv) == 32.0


def test_walls_of_the_hyperbolic_objects(objs):
    bg = _cheapest(objs, "hyperbolic_background")
    assert min(m[0] for m in bg) == 0.0 and sum(1 for m in bg if m[0] == 0) == 8
    fold = _cheapest(objs, "hyperbolic_fold")
    assert min(m[0] for m in fold) == 0.0                 # no downhill single switch out of the fold
    handle = _cheapest(objs, "hyperbolic_handle")
    assert min(m[0] for m in handle) < 0                  # the handle has one
