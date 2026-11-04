"""scripts/exact_torus_counting.py: the lattice tori weighed two ways at lambda = 1, every single switch out of each,
and the tube's zero-cost switches read from their positions. Each count is checked against an independent rebuild in
networkx where that is affordable (rule 6)."""
import math
import sys
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from networkx.algorithms.isomorphism import GraphMatcher

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import exact_flat_torus_widths as flat                            # noqa: E402
import exact_torus_counting as e                                  # noqa: E402
from graphity import symmetry                                     # noqa: E402
from graphity.cqg import NO_CAP, is_valid                         # noqa: E402

pytestmark = pytest.mark.skipif(not symmetry.available(), reason="igraph is not installed")


# ---------------------------------------------------------------- an independent rebuild in networkx

def nx_torus(a, b):
    """The a x b lattice torus built by networkx, not by cqg.torus; side (x + y) mod 2."""
    g = nx.grid_2d_graph(a, b, periodic=True)
    return g, {p: (p[0] + p[1]) % 2 for p in g}


def squares_and_surplus(nb):
    """S and X counted from the neighbour sets: S_e = the pairs (a, b), a a neighbour of u other than v and b one of
    v other than u, with a and b joined."""
    s4 = x = 0
    for u in nb:
        for v in nb[u]:
            if u < v:
                s_e = sum(1 for a in nb[u] if a != v for b in nb[v] if b != u and b in nb[a])
                assert s_e <= 3                                   # Q2: the hard-core rule stops an edge at three
                s4 += s_e
                x += max(s_e - 2, 0)
    return s4 // 4, x


def hard_core_ok(nb):
    """No two points share more than two neighbours."""
    common = Counter()
    for w in nb:
        ns = sorted(nb[w])
        for i in range(len(ns)):
            for j in range(i + 1, len(ns)):
                common[ns[i], ns[j]] += 1
    return max(common.values()) <= 2


def brute_histogram(a, b):
    """Every single switch (u1, v1), (u2, v2) -> (u1, v2), (u2, v1), u1 and u2 on side 0, applied to neighbour sets
    and kept if the hard-core rule holds; returns ({(dS, dX): count}, pairs tried). The cost at any lambda is
    -16 dS + 4 lambda dX, so comparing the pairs checks every lambda at once."""
    g, side = nx_torus(a, b)
    nb = {p: set(g[p]) for p in g}
    s0, x0 = squares_and_surplus(nb)
    edges = [(u, v) if side[u] == 0 else (v, u) for u, v in g.edges()]
    hist, tried = Counter(), 0
    for i in range(len(edges)):
        u1, v1 = edges[i]
        for j in range(i + 1, len(edges)):
            u2, v2 = edges[j]
            if u1 == u2 or v1 == v2 or v2 in nb[u1] or v1 in nb[u2]:
                continue
            tried += 1
            for p, q, r in ((u1, v1, v2), (v1, u1, u2), (u2, v2, v1), (v2, u2, u1)):
                nb[p].remove(q)
                nb[p].add(r)
            if hard_core_ok(nb):
                s, x = squares_and_surplus(nb)
                hist[s - s0, x - x0] += 1
            for p, q, r in ((u1, v2, v1), (v2, u1, u2), (u2, v1, v2), (v1, u2, u1)):
                nb[p].remove(q)
                nb[p].add(r)
    return dict(sorted(hist.items())), tried


def nx_side_preserving_automorphisms(a, b):
    g, side = nx_torus(a, b)
    nx.set_node_attributes(g, side, "side")
    gm = GraphMatcher(g, g, node_match=lambda p, q: p["side"] == q["side"])
    return sum(1 for _ in gm.isomorphisms_iter())


def ring_bundle(length, sigma):
    """The 4 x L tube with ring L - 1 joined to ring 0 through sigma: (x, L - 1) - (sigma[x], 0); built here by hand."""
    adj = np.empty((4 * length, 4), dtype=np.int64)
    inv = [sigma.index(x) for x in range(4)]
    for x in range(4):
        for y in range(length):
            up = e.index(sigma[x], 0, length) if y == length - 1 else e.index(x, y + 1, length)
            down = e.index(inv[x], length - 1, length) if y == 0 else e.index(x, y - 1, length)
            adj[e.index(x, y, length)] = (e.index(x + 1, y, length), e.index(x - 1, y, length), up, down)
    return adj


def same_class(adj1, part1, adj2, part2):
    """True when a renaming that keeps each side (side 0 to side 0) carries one graph onto the other."""
    return symmetry.canonical_key(adj1, part1) == symmetry.canonical_key(adj2, part2)


# ---------------------------------------------------------------- the renaming counts

@pytest.mark.parametrize("n", [64, 96, 144])
def test_rectangles_have_2N_renamings_and_the_square_4N(n):
    shapes = e.tori_of(n)
    assert shapes == {64: [(4, 16), (8, 8)], 96: [(4, 24), (6, 16), (8, 12)],
                      144: [(4, 36), (6, 24), (8, 18), (12, 12)]}[n]
    for a, b in shapes:
        r = e.counts(a, b)
        assert r["H"] == 0
        assert r["S_per_point"] == (1.25 if a == 4 else 1.0)
        assert r["A"] == (4 * n if a == b else 2 * n)


def test_the_renaming_counts_agree_with_networkx_at_64():
    for a, b in e.tori_of(64):
        assert nx_side_preserving_automorphisms(a, b) == e.counts(a, b)["A"]


@pytest.mark.parametrize("length", [6, 8, 12])
def test_the_tube_renamings_are_the_lattice_maps(length):
    """(x, y) -> (+-x + i, +-y + j) with i + j even: all keep edges and sides, all differ, and there are A of them."""
    adj, part = e.build(4, length)
    edges = e.edge_set(adj)
    maps = e.tube_symmetries(length)
    for p in maps:
        assert {frozenset(int(p[v]) for v in ed) for ed in edges} == edges
        assert np.array_equal(part[p], part)
    assert len({tuple(p) for p in maps}) == len(maps) == 8 * length == symmetry.count(adj, part)


# ---------------------------------------------------------------- every single switch

@pytest.mark.parametrize("a, b", [(4, 12), (6, 8)])
def test_switch_costs_match_a_networkx_brute_force(a, b):
    adj, part = e.build(a, b)
    rows, tried = e.every_switch(adj, part)
    pairs, tried_nx = brute_histogram(a, b)
    assert tried == tried_nx and len(rows) == sum(pairs.values())
    assert Counter((int(ds), int(dx)) for ds, dx in rows[:, 4:6]) == pairs     # dS and dX separately
    at_1 = Counter()
    for (ds, dx), k in pairs.items():
        at_1[-16 * ds + 4 * dx] += k
    assert e.cost_histogram(adj, part) == dict(sorted(at_1.items()))
    for u1, v1, u2, v2, _ds, _dx in rows[:50]:                    # the kernel's own check agrees on the switched graph
        new = adj.copy()
        e._switch(new, u1, v1, u2, v2)
        assert is_valid(new, NO_CAP)


def test_the_recorded_histograms_at_48():
    adj, part = e.build(4, 12)
    assert e.cost_histogram(adj, part) == {0: 24, 16: 144, 24: 96, 28: 48, 32: 720, 36: 96, 40: 960, 48: 528}
    adj, part = e.build(6, 8)
    assert e.cost_histogram(adj, part) == {16: 24, 32: 288, 40: 288, 52: 96, 56: 216, 64: 1296}


def test_the_look_at_144_is_reproduced():
    """sandbox/out/torus_free_energy_one_switch_2026-10-23.txt, N = 144."""
    want = {(4, 36): (38232, {0: 72, 16: 432, 24: 288, 28: 144, 32: 9072, 36: 288, 40: 16704, 48: 8496}),
            (6, 24): (38160, {16: 72, 32: 864, 40: 864, 52: 288, 56: 576, 60: 144, 64: 31464}),
            (8, 18): (38160, {32: 864, 40: 864, 52: 576, 56: 1224, 60: 144, 64: 30456}),
            (12, 12): (38160, {32: 864, 40: 864, 52: 576, 56: 1152, 60: 288, 64: 30384})}
    hists = {}
    for (a, b), (pairs, hist) in want.items():
        adj, part = e.build(a, b)
        _rows, tried = e.every_switch(adj, part)
        hists[a, b] = e.cost_histogram(adj, part)
        assert tried == pairs and hists[a, b] == hist
    assert e.free_energy(hists[4, 36], 0.5) == pytest.approx(-2.145, abs=5e-4)
    assert e.free_energy(hists[4, 36], 0.5) == pytest.approx(-0.5 * math.log(73), rel=1e-9)
    assert e.free_energy(hists[6, 24], 10.0) == pytest.approx(-48.117, abs=5e-4)


@pytest.mark.parametrize("length", [8, 16, 24])
def test_a_6_x_L_torus_has_3L_switches_at_16_and_none_cheaper(length):
    adj, part = e.build(6, length)
    hist = e.cost_histogram(adj, part)
    assert min(hist) == 16 and hist[16] == 3 * length


# ---------------------------------------------------------------- the tube's zero-cost switches

@pytest.mark.parametrize("length", [8, 10, 12, 16, 24, 36, 40, 48])
def test_the_tube_has_2L_zero_cost_switches(length):
    zero = e.zero_cost_switches(length)
    assert len(zero) == 2 * length and all((ds, dx) == (0, 0) for *_r, ds, dx in zero)


def test_the_short_tube_has_more():
    """At L = 6 twelve more zero-cost switches exchange two ring edges three rings apart (dS = -2, dX = -8)."""
    zero = e.zero_cost_switches(6)
    assert Counter((ds, dx) for *_r, ds, dx in zero) == {(0, 0): 12, (-2, -8): 12}
    for u1, v1, u2, v2, ds, dx in zero:
        if (ds, dx) == (-2, -8):
            (x1, y1), (x2, y2) = e.xy(u1, 6), e.xy(v1, 6)
            y3 = (y1 + 3) % 6
            assert y1 == y2 and (x1 - x2) % 4 in (1, 3)           # a ring edge (x, y) - (x +- 1, y) is removed,
            assert (e.xy(u2, 6), e.xy(v2, 6)) == ((x2, y3), (x1, y3))   # and the matching one on ring y + 3;
            # added: (x, y) - (x, y + 3) and (x +- 1, y + 3) - (x +- 1, y), links three rings along


@pytest.mark.parametrize("length", [8, 12])
def test_each_zero_cost_switch_crosses_two_opposite_links_of_one_junction(length):
    out, orbits, tube_faces = e.read_zero_cost(length)
    seen = set()
    for r in out:
        lows = []
        for p, q in r["removed"]:                                  # each removed edge is a link, ring y to y + 1
            lo, hi = (p, q) if (q[1] - p[1]) % length == 1 else (q, p)
            assert lo[0] == hi[0] and (hi[1] - lo[1]) % length == 1
            lows.append(lo)
        (xa, y), (xb, y_) = sorted(lows)
        assert y == y_ and xb == xa + 2                            # the same junction, opposite points of the ring
        y1 = (y + 1) % length
        added = {frozenset(ed) for ed in r["added"]}
        assert added == {frozenset({(xa, y), (xb, y1)}), frozenset({(xb, y), (xa, y1)})}   # the two links crossed
        seen.add((y, xa % 2))
        assert (r["dS"], r["dX"]) == (0, 0)
        assert len(r["lost"]) == len(r["gained"]) == 4
        assert all({c[1] for c in sq} == {y, y1} for sq in r["lost"] + r["gained"])
        assert not r["tube"] and r["A"] == 4 * length
        fixed = [x for x in range(4) if r["monodromy"][x] == x]
        assert fixed == [x for x in range(4) if x % 2 != xa % 2]   # the reflection fixing the uncrossed pair
        assert r["lines"] == [length, length, 2 * length]
        assert r["surface"] == (True, 0, False)
    assert len(seen) == 2 * length                                 # every junction, both pairs, once
    assert len(orbits) == 1 and tube_faces == (True, 0, True)


# ---------------------------------------------------------------- the family reachable at zero cost

@pytest.mark.parametrize("length", [8, 10, 12, 24])
def test_the_zero_cost_family_of_the_tube_is_three_classes(length):
    n = 4 * length
    classes, hit, moves = e.reach(length, cap=50)
    assert not hit and len(classes) == 3
    order = sorted(classes, key=lambda k: classes[k][0])
    part = e.build(4, length)[1]
    reps = [classes[k][1] for k in order]
    assert [symmetry.count(r, part) for r in reps] == [2 * n, n, 2 * n]
    assert sum(2 * n / symmetry.count(r, part) for r in reps) == 4  # labelled copies, family over tube
    tube, reflected, half = reps
    assert same_class(tube, part, ring_bundle(length, [0, 1, 2, 3]), part)
    assert same_class(reflected, part, ring_bundle(length, [2, 1, 0, 3]), part)
    assert same_class(reflected, part, ring_bundle(length, [0, 3, 2, 1]), part)     # the two reflections: one class
    assert same_class(half, part, ring_bundle(length, [2, 3, 0, 1]), part)
    d = length                                                      # the half-turn closure is a lattice torus
    fg = flat.graph(4, 2, d)
    assert same_class(half, part, fg, np.array([(i // d + i % d) % 2 for i in range(4 * d)]))
    assert [e.describe(r, part, length)["surface"] for r in reps] == [(True, 0, True), (True, 0, False),
                                                                     (True, 0, True)]
    k0, k1, k2 = order
    assert moves[k0] == {k1: 2 * length}
    assert moves[k1] == {k0: length, k2: length}
    assert moves[k2] == {k1: 2 * length}
    for c in order:                                                 # each move is counted the same from both ends
        for t, k in moves[c].items():
            assert k * symmetry.count(classes[t][1], part) == moves[t][c] * symmetry.count(classes[c][1], part)
    # the three tie at every lambda: equal S and X, and every zero-cost switch between them has dS = dX = 0
    assert len({(int(e.total_squares(r)), int(e.surplus(r))) for r in reps}) == 1
    for k in order:
        assert all((int(r[4]), int(r[5])) == (0, 0) for r in classes[k][2] if e.cost(int(r[4]), int(r[5])) == 0)


def test_the_renamings_that_fix_a_zero_cost_switch():
    """At L = 8, the {0, 2} switch at junction 0 is fixed by four renamings: each leaves y alone and acts on the ring
    as the identity, the half turn, or one of the two reflections x -> -x, x -> 2 - x."""
    length = 8
    sw = [r for r in e.zero_cost_switches(length)
          if sorted(e.xy(v, length) for v in r[:4]) == [(0, 0), (0, 1), (2, 0), (2, 1)]]
    assert len(sw) == 1
    stab = e.switch_stabilizer(length, sw[0])
    assert len(stab) == 4
    actions = set()
    for s in stab:
        assert all(q[1] == p[1] for p, q in s.items())             # y fixed
        ring = tuple(s[x, 0][0] for x in range(4))
        assert all(s[x, y][0] == ring[x] for x in range(4) for y in range(length))
        actions.add(ring)
    assert actions == {(0, 1, 2, 3), (2, 3, 0, 1), (0, 3, 2, 1), (2, 1, 0, 3)}


@pytest.mark.parametrize("length", [8, 12])
def test_at_lambda_1_25_each_family_class_has_12L_exits_of_cost_12(length):
    """Every class of the zero-cost family: nothing below 0, nothing between 0 and 12, 12L switches at 12, 8L at 14."""
    for h in e.family_exits(length, 1.25):
        assert h == {0: 2 * length, 12: 12 * length, 14: 8 * length}


def test_the_short_tube_family_has_a_fourth_class():
    classes, hit, _ = e.reach(6, cap=50)
    part = e.build(4, 6)[1]
    extra = [e.describe(rep, part, 6) for _d, rep, _r in classes.values() if not e.ring_maps(rep, 6)]
    assert not hit and len(classes) == 4 and len(extra) == 1
    assert (extra[0]["A"], extra[0]["S"], extra[0]["X"]) == (8, 28, 16)
    a_tube = symmetry.count(*e.build(4, 6))                         # labelled copies, family over tube: 1 + 6 + 2 + 1
    assert sorted(a_tube // symmetry.count(rep, part) for _d, rep, _r in classes.values()) == [1, 1, 2, 6]


def test_the_4_cube_family_has_five_classes():
    """At L = 4 the tube is the 4-cube; its zero-cost family is five classes (not read beyond the count)."""
    classes, hit, _ = e.reach(4, cap=50)
    assert not hit and len(classes) == 5


def test_the_family_free_energy_starts_at_minus_g_ln_4():
    fam = e.family_free_energy(12, couplings=(0.5, 1.0))
    assert fam[0.5] == pytest.approx(-0.5 * math.log(4), rel=1e-9)
    assert fam[1.0] == pytest.approx(-math.log(4), rel=1e-4)
    adj, part = e.build(4, 12)
    assert e.free_energy(e.cost_histogram(adj, part), 0.5) == pytest.approx(-0.5 * math.log(25), rel=1e-9)
