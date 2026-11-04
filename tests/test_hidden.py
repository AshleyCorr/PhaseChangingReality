"""The hidden count of a region (graphity.hidden). It is exact, so every check here is an equality.

Two enumerators written here, sharing no code with the module, say what the right answers are: `all_subsets`,
the simplest there is, for small regions; and `reference`, which only skips wirings with wrong degrees, for
regions of 12 to 14 points. Both ask `is_valid` and `hamiltonian` about every graph they build.
"""
import itertools
from collections import Counter
from functools import lru_cache
from math import factorial

import numpy as np
import pytest

from graphity.cqg import NO_CAP, hamiltonian, is_valid, run_chain, surplus, torus, total_squares
import graphity.hidden as hidden
from graphity.hidden import hidden_count, rewire, shapes, valid_wirings

LAM = 1.25


def block(ly, bx, by):
    """The bx x by block in the corner of a torus whose second side is ly (vertex = x * ly + y)."""
    return [x * ly + y for x in range(bx) for y in range(by)]


def ball(adj, anchors, radius):
    """The vertices within `radius` steps of any of `anchors`."""
    seen = set(anchors)
    for _ in range(radius):
        seen |= {int(w) for v in seen for w in adj[v]}
    return sorted(seen)


@lru_cache(maxsize=None)
def warm_graph():
    """A valid graph that is not a ground state: the 8 x 8 torus after 300 sweeps at g = 6 (H = 341 at lam 1.25)."""
    adj, part = torus(8, 8, NO_CAP)
    run_chain(adj, np.flatnonzero(part == 0), 1 / 6.0, 300, 1, 7, LAM, NO_CAP, False)
    return adj, part


@lru_cache(maxsize=None)
def cases():
    """name -> (adj, part, region). The tests read these arrays and never write to them."""
    flat, fpart = torus(8, 8)
    wide, wpart = torus(16, 16)
    tube, tpart = torus(16, 4, NO_CAP)
    cube, cpart = torus(4, 4, NO_CAP)
    warm, mpart = warm_graph()
    return {
        "flat 2x2": (flat, fpart, block(8, 2, 2)),
        "flat 2x3": (flat, fpart, block(8, 2, 3)),
        "flat 3x3": (flat, fpart, block(8, 3, 3)),                          # 5 vertices of one side, 4 of the other
        "flat row of three": (flat, fpart, [0, 1, 2]),                      # needs 1, 2, 1; sides 2 against 1
        "flat plus": (flat, fpart, ball(flat, (9,), 1)),                    # needs 4, 1, 1, 1, 1; sides 1 against 4
        "flat 2x2 and a far point": (flat, fpart, block(8, 2, 2) + [36]),   # vertex 36 needs 0
        "flat ball of radius 1": (wide, wpart, ball(wide, (0, 1, 16, 17), 1)),   # 12 points round a square
        "tube, two columns": (tube, tpart, list(range(8))),
        "tube, three columns": (tube, tpart, list(range(12))),
        "4-cube, half": (cube, cpart, list(range(8))),
        "warm 2x3": (warm, mpart, block(8, 2, 3)),
        "warm 3x3": (warm, mpart, block(8, 3, 3)),
        "warm ball of a vertex": (warm, mpart, ball(warm, (0,), 1)),
        "warm ball of a link": (warm, mpart, ball(warm, (0, int(warm[0, 0])), 1)),
        "warm ball of radius 2": (warm, mpart, ball(warm, (9,), 2)),        # 14 points, needs of 1, 2 and 4
    }


SMALL = ["flat 2x2", "flat 2x3", "flat 3x3", "flat row of three", "flat plus", "flat 2x2 and a far point",
         "tube, two columns", "4-cube, half", "warm 2x3", "warm 3x3", "warm ball of a vertex", "warm ball of a link"]
LARGER = ["flat ball of radius 1", "tube, three columns", "warm ball of radius 2"]
EVERY = SMALL + LARGER


def interior(adj, part, region):
    """The region's own links, as increasing (side-0 vertex, side-1 vertex) pairs."""
    inside = set(region)
    return tuple(sorted((v, int(w)) for v in region if part[v] == 0 for w in adj[v] if int(w) in inside))


def built(adj, region, wiring):
    """The graph with the region's inside replaced by `wiring`; None unless every vertex ends with four links."""
    inside = set(region)
    rows = {v: [int(w) for w in adj[v] if int(w) not in inside] for v in region}
    for u, w in wiring:
        rows[u].append(w)
        rows[w].append(u)
    if any(len(row) != 4 for row in rows.values()):
        return None
    new = adj.copy()
    for v in region:
        new[v] = rows[v]
    return new


def tidy(d_h):
    return 0.0 if abs(d_h) <= 1e-9 else round(d_h, 9)


def all_subsets(adj, part, region, lam):
    """(levels, same-energy wirings) from every set of opposite-side pairs inside the region, of the right size."""
    region = sorted(set(region))
    pairs = [(u, w) for u in region if part[u] == 0 for w in region if part[w] == 1]
    h0 = hamiltonian(adj, lam)
    levels, same = Counter(), set()
    for wiring in itertools.combinations(pairs, len(interior(adj, part, region))):
        new = built(adj, region, wiring)
        if new is not None and is_valid(new, NO_CAP):
            d_h = tidy(hamiltonian(new, lam) - h0)
            levels[d_h] += 1
            if d_h == 0.0:
                same.add(wiring)
    return dict(sorted(levels.items())), same


def reference(adj, part, region, lam):
    """The same two answers for larger regions: each side-0 vertex in turn takes its partners among the side-1
    vertices that still need links. Nothing is skipped but wrong degrees; validity and energy are asked of
    `is_valid` and `hamiltonian` for every wiring that is left."""
    region = sorted(set(region))
    inside = set(region)
    need = {v: sum(1 for w in adj[v] if int(w) in inside) for v in region}
    givers = [v for v in region if part[v] == 0]
    takers = [v for v in region if part[v] == 1]
    h0 = hamiltonian(adj, lam)
    levels, same = Counter(), set()

    def place(i, wiring):
        if i == len(givers):
            new = None if any(need[w] for w in takers) else built(adj, region, wiring)
            if new is not None and is_valid(new, NO_CAP):
                d_h = tidy(hamiltonian(new, lam) - h0)
                levels[d_h] += 1
                if d_h == 0.0:
                    same.add(tuple(sorted(wiring)))
            return
        for partners in itertools.combinations([w for w in takers if need[w] > 0], need[givers[i]]):
            for w in partners:
                need[w] -= 1
            place(i + 1, wiring + [(givers[i], w) for w in partners])
            for w in partners:
                need[w] += 1

    place(0, [])
    return dict(sorted(levels.items())), same


def first_script_count(adj, region, lam):
    """The count as scripts/hidden_count.py defines it: any pairs inside the region, right degrees, same energy."""
    region = sorted(region)
    inside = set(region)
    n_links = sum(1 for v in region for w in adj[v] if int(w) in inside) // 2
    h0 = hamiltonian(adj, lam)
    same = []
    for wiring in itertools.combinations(itertools.combinations(region, 2), n_links):
        new = built(adj, region, wiring)
        if new is not None and abs(hamiltonian(new, lam) - h0) <= 1e-9:
            same.append(wiring)
    return same


def check(name, lam, enumerator):
    adj, part, region = cases()[name]
    levels, same = enumerator(adj, part, region, lam)
    r = hidden_count(adj, part, region, lam, keep=10 ** 6)
    assert r["levels"] == levels
    assert r["valid"] == sum(levels.values())
    assert r["count"] == levels[0.0] == len(same)
    assert len(r["wirings"]) == len(set(r["wirings"])) and set(r["wirings"]) == same
    return r


@pytest.mark.parametrize("name", SMALL)
def test_agrees_with_every_subset_of_pairs(name):
    check(name, LAM, all_subsets)


@pytest.mark.parametrize("lam", [0.0, 0.5, 1.0, 2.0])
@pytest.mark.parametrize("name", ["tube, two columns", "4-cube, half", "warm 2x3", "warm ball of a link"])
def test_agrees_with_every_subset_of_pairs_at_other_lambda(name, lam):
    check(name, lam, all_subsets)


@pytest.mark.parametrize("name", ["flat 2x3", "flat row of three", "flat plus", "tube, two columns", "4-cube, half",
                                  "warm ball of a link"])
def test_the_two_enumerators_agree_with_each_other(name):
    adj, part, region = cases()[name]
    assert reference(adj, part, region, LAM) == all_subsets(adj, part, region, LAM)


@pytest.mark.parametrize("name", LARGER)
def test_agrees_with_the_reference_on_12_to_14_points(name):
    """About 25 s in all, most of it the three columns of the tube: the reference is plain Python."""
    r = check(name, LAM, reference)
    assert r["valid"] > 4000                       # these are the cases with many wirings to lose or invent


def test_the_odd_shapes_are_what_they_claim_to_be():
    """Vertices that need one link back, sides of different sizes, a vertex that needs none."""
    def needs(name):
        adj, part, region = cases()[name]
        inside = set(region)
        return [(int(part[v]), sum(1 for w in adj[v] if int(w) in inside)) for v in region]
    assert needs("flat row of three") == [(0, 1), (1, 2), (0, 1)]
    assert sorted(needs("flat plus")) == [(0, 4)] + [(1, 1)] * 4
    assert needs("flat 2x2 and a far point")[-1] == (0, 0)
    assert Counter(side for side, _ in needs("flat 3x3")) == {0: 5, 1: 4}
    assert {need for _, need in needs("warm ball of radius 2")} == {1, 2, 4}
    for name, links in [("flat row of three", 2), ("flat plus", 4), ("flat 2x2 and a far point", 4)]:
        r = hidden_count(*cases()[name], LAM)
        assert (r["count"], r["valid"], r["links"], r["levels"]) == (1, 1, links, {0.0: 1})


def test_flat_blocks_have_only_their_own_wiring():
    for name, points, cut in [("flat 2x2", 4, 8), ("flat 2x3", 6, 10), ("flat 3x3", 9, 12)]:
        r = hidden_count(*cases()[name], LAM)
        assert (r["count"], r["points"], r["cut"], r["energy"]) == (1, points, cut, 0.0)


@pytest.mark.parametrize("name", EVERY)
def test_the_original_wiring_is_always_counted(name):
    adj, part, region = cases()[name]
    r = hidden_count(adj, part, region, LAM, keep=10 ** 6)
    assert r["count"] >= 1 and r["levels"][0.0] == r["count"] and r["square_levels"][(0, 0)] >= 1
    assert interior(adj, part, region) in r["wirings"]
    assert r["points"] == len(region) and r["links"] == len(interior(adj, part, region))
    assert r["cut"] == sum(1 for v in region for w in adj[v] if int(w) not in set(region))
    assert r["energy"] == hamiltonian(adj, LAM)


@pytest.mark.parametrize("name", EVERY)
def test_every_wiring_found_is_valid_and_its_energy_is_the_full_one(name):
    """The search reads the energy from the edges near the region only, and never calls is_valid. Here every
    wiring it finds is rebuilt as a whole graph, and both are asked of the kernel's own functions."""
    adj, part, region = cases()[name]
    found = valid_wirings(adj, part, region)
    r = hidden_count(adj, part, region, LAM)
    assert len(found) == r["valid"] == len({links for links, _, _ in found})
    s0, x0, h0 = total_squares(adj), surplus(adj), hamiltonian(adj, LAM)
    tally = Counter()
    for links, d_squares, d_surplus in found:
        new = rewire(adj, region, links)
        assert is_valid(new, NO_CAP)
        assert np.array_equal(np.sort(new, axis=1), np.sort(built(adj, region, links), axis=1))
        assert total_squares(new) - s0 == d_squares
        assert surplus(new) - x0 == d_surplus
        assert abs(hamiltonian(new, LAM) - h0 - (-16 * d_squares + 4 * LAM * d_surplus)) < 1e-9
        tally[(d_squares, d_surplus)] += 1
    assert tally == r["square_levels"]


@pytest.mark.parametrize("lam", [0.0, 0.5, 1.0, 1.25, 2.0])
@pytest.mark.parametrize("name", EVERY)
def test_the_shorter_search_finds_the_same_wirings(name, lam):
    """same_energy_only prunes by the energy still in reach; it must lose no tie and invent none, at any lam."""
    adj, part, region = cases()[name]
    full = hidden_count(adj, part, region, lam, keep=10 ** 6)
    short = hidden_count(adj, part, region, lam, keep=10 ** 6, same_energy_only=True)
    assert short["count"] == full["count"] and short["wirings"] == full["wirings"]
    assert short["valid"] is None and short["levels"] == {0.0: full["count"]}
    assert sum(short["square_levels"].values()) == full["count"]
    for key in ("points", "cut", "links", "sealed", "renamings", "energy"):
        assert short[key] == full[key]


def test_the_first_script_counted_wirings_the_model_does_not_have():
    """Two links of the warm graph that share no square: (0, 1) and (4, 33). The model can only swap partners
    across the sides, (0, 33) and (4, 1), and that breaks the hard-core rule here. The first script also allows
    (0, 4) and (1, 33), which join two vertices of one side, and so it counts two where the model has one."""
    adj, part = warm_graph()
    region = [0, 1, 4, 33]
    assert [int(part[v]) for v in region] == [0, 1, 0, 1] and interior(adj, part, region) == ((0, 1), (4, 33))
    loose = first_script_count(adj, region, LAM)
    assert sorted(loose) == [((0, 1), (4, 33)), ((0, 4), (1, 33))]
    swapped = built(adj, region, [(0, 33), (4, 1)])
    assert swapped is not None and not is_valid(swapped, NO_CAP)
    r = hidden_count(adj, part, region, LAM)
    assert (r["count"], r["valid"]) == (1, 1) and len(loose) > r["count"]


def renamed(wiring, swaps):
    return tuple(sorted((swaps.get(u, u), swaps.get(w, w)) for u, w in wiring))


def test_a_ball_in_flat_space_counts_the_renamings_of_its_sealed_points():
    """The 12 points within one step of the square (0, 1, 16, 17) of the flat 16 x 16 torus. The square's own four
    points have all their links inside the ball. Exchanging the names of 0 and 17, or of 1 and 16, or both, gives
    the same graph under other names: valid, same energy, a different interior wiring. Those four are the whole
    count: nothing else ties. (The count is by names; see the module docstring.)"""
    adj, part, region = cases()["flat ball of radius 1"]
    r = hidden_count(adj, part, region, LAM, keep=100)
    assert (r["points"], r["cut"], r["links"], r["sealed"], r["renamings"]) == (12, 16, 16, (2, 2), 4)
    assert (r["count"], r["valid"]) == (4, 4576)
    original = interior(adj, part, region)
    expected = {renamed(original, {**a, **b}) for a in ({}, {0: 17, 17: 0}) for b in ({}, {1: 16, 16: 1})}
    assert len(expected) == 4 and set(r["wirings"]) == expected


@pytest.mark.parametrize("name", EVERY)
def test_renaming_sealed_points_always_gives_another_tie(name):
    adj, part, region = cases()[name]
    r = hidden_count(adj, part, region, LAM, keep=10 ** 6)
    inside = set(region)
    sealed = [[v for v in region if part[v] == side and all(int(w) in inside for w in adj[v])] for side in (0, 1)]
    assert r["sealed"] == (len(sealed[0]), len(sealed[1]))
    assert r["renamings"] == factorial(len(sealed[0])) * factorial(len(sealed[1]))
    original = interior(adj, part, region)
    for new0 in itertools.permutations(sealed[0]):
        for new1 in itertools.permutations(sealed[1]):
            swaps = dict(zip(sealed[0] + sealed[1], new0 + new1))
            assert renamed(original, swaps) in r["wirings"]
    assert r["count"] % factorial(len(sealed[0])) == 0 and r["count"] % factorial(len(sealed[1])) == 0


@pytest.mark.parametrize("lam", [0.0, 1.0, 1.25])
@pytest.mark.parametrize("name", EVERY)
def test_folding_the_renamings_changes_no_number(name, lam):
    """fold_renamings visits one wiring of each class of renamings of the sealed side-1 points and counts it
    k1! times. Every number must come out the same, and the wirings it lists, renamed in all k1! ways, must be
    the full list of ties, each once."""
    adj, part, region = cases()[name]
    full = hidden_count(adj, part, region, lam, keep=10 ** 6)
    folded = hidden_count(adj, part, region, lam, keep=10 ** 6, fold_renamings=True)
    short = hidden_count(adj, part, region, lam, keep=10 ** 6, fold_renamings=True, same_energy_only=True)
    for key in full:
        if key != "wirings":
            assert folded[key] == full[key]
    assert short["count"] == full["count"] and short["wirings"] == folded["wirings"]
    inside = set(region)
    sealed1 = [v for v in region if part[v] == 1 and all(int(w) in inside for w in adj[v])]
    spread = [renamed(w, dict(zip(sealed1, new))) for w in folded["wirings"] for new in itertools.permutations(sealed1)]
    assert len(spread) == len(set(spread)) == full["count"] and set(spread) == set(full["wirings"])


def sealed_points(adj, part, region):
    inside = set(region)
    return [[v for v in region if part[v] == side and all(int(w) in inside for w in adj[v])] for side in (0, 1)]


def shapes_the_plain_way(adj, part, region, wirings):
    """Classes of `wirings` under every renaming of the sealed points of side 0 and of side 1, tried one by one."""
    sealed0, sealed1 = sealed_points(adj, part, region)
    reached, n_shapes = set(), 0
    for wiring in wirings:
        if wiring not in reached:
            n_shapes += 1
            for new0 in itertools.permutations(sealed0):
                for new1 in itertools.permutations(sealed1):
                    reached.add(renamed(wiring, dict(zip(sealed0 + sealed1, new0 + new1))))
    assert reached == set(wirings)                 # renaming a tie gives a tie, so the list is closed
    return n_shapes


@pytest.mark.parametrize("lam", [0.0, 1.0, 1.25])
@pytest.mark.parametrize("name", EVERY)
def test_shapes_against_every_renaming_tried_one_by_one(name, lam):
    adj, part, region = cases()[name]
    full = hidden_count(adj, part, region, lam, keep=10 ** 6)
    expected = shapes_the_plain_way(adj, part, region, full["wirings"])
    assert full["shapes"] == expected == shapes(adj, part, region, full["wirings"])
    assert 1 <= expected <= full["count"] and expected * full["renamings"] >= full["count"]
    for same in (False, True):                     # from one wiring of each folded class, and with nothing listed
        folded = hidden_count(adj, part, region, lam, keep=0, same_energy_only=same, fold_renamings=True)
        assert folded["shapes"] == expected and folded["wirings"] == []
    assert hidden_count(adj, part, region, lam, keep=0, same_energy_only=True)["shapes"] == expected


def test_a_flat_window_hides_only_the_names_of_its_sealed_points():
    """The 12-point ball again: four same-energy wirings by name, one shape. So do the flat blocks."""
    r = hidden_count(*cases()["flat ball of radius 1"], LAM)
    assert (r["count"], r["renamings"], r["shapes"]) == (4, 4, 1)
    adj, part = torus(16, 16)
    for bx, by, count in [(3, 3, 1), (4, 4, 4), (4, 5, 36)]:
        r = hidden_count(adj, part, block(16, bx, by), LAM, keep=0, same_energy_only=True, fold_renamings=True)
        assert (r["count"], r["shapes"]) == (count, 1)


def test_the_tube_hides_more_than_names():
    """Two columns of the tube have no sealed point, so their four ties are four shapes; three columns have
    sixteen ties, four renamings and four shapes."""
    two = hidden_count(*cases()["tube, two columns"], LAM)
    assert (two["count"], two["renamings"], two["shapes"]) == (4, 1, 4)
    three = hidden_count(*cases()["tube, three columns"], LAM, keep=0)
    assert (three["count"], three["renamings"], three["shapes"]) == (16, 4, 4)


def test_shapes_is_none_when_it_cannot_be_worked_out(monkeypatch):
    adj, part, region = cases()["tube, three columns"]
    monkeypatch.setattr(hidden, "SHAPES_KEEP", 15)             # sixteen ties were found, fifteen kept
    assert hidden_count(adj, part, region, LAM)["shapes"] is None
    assert hidden_count(adj, part, region, LAM, keep=16)["shapes"] == 4
    assert hidden_count(adj, part, region, LAM, fold_renamings=True)["shapes"] == 4     # eight classes: all kept
    monkeypatch.setattr(hidden, "SHAPES_KEEP", 5000)
    monkeypatch.setattr(hidden, "SHAPES_WORK", 31)             # 16 ties x 2! renamings to try = 32
    assert hidden_count(adj, part, region, LAM)["shapes"] is None
    assert shapes(adj, part, region, hidden_count(adj, part, region, LAM, keep=16)["wirings"]) is None


def test_tube_windows():
    """Two columns of the 4 x 16 tube have no sealed point and still four ties; three columns have sixteen."""
    two = hidden_count(*cases()["tube, two columns"], LAM)
    assert (two["count"], two["valid"], two["sealed"], two["cut"]) == (4, 24, (0, 0), 8)
    assert two["levels"] == {0.0: 4, 12.0: 16, 24.0: 4}
    three = hidden_count(*cases()["tube, three columns"], LAM)
    assert (three["count"], three["valid"], three["sealed"], three["cut"]) == (16, 9216, (2, 2), 8)


def test_larger_windows_by_the_shorter_search():
    """16 and 20 points, where the full search has too many wirings to visit: a 4 x 4 and a 4 x 5 block of the
    flat torus. Every tie is a renaming of the sealed points (2 + 2, and 3 + 3)."""
    adj, part = torus(16, 16)
    for bx, by, sealed in [(4, 4, (2, 2)), (4, 5, (3, 3))]:
        r = hidden_count(adj, part, block(16, bx, by), LAM, keep=0, same_energy_only=True, fold_renamings=True)
        assert (r["points"], r["sealed"], r["count"]) == (bx * by, sealed, r["renamings"])
    unfolded = hidden_count(adj, part, block(16, 4, 4), LAM, keep=0, same_energy_only=True)
    assert unfolded["count"] == 4


def test_keep_limits_the_list_and_not_the_count():
    adj, part, region = cases()["tube, three columns"]
    everything = hidden_count(adj, part, region, LAM, keep=100)
    some = hidden_count(adj, part, region, LAM, keep=5)
    none = hidden_count(adj, part, region, LAM, keep=0)
    assert len(everything["wirings"]) == 16 and some["wirings"] == everything["wirings"][:5] and none["wirings"] == []
    assert some["count"] == none["count"] == 16 and some["levels"] == none["levels"] == everything["levels"]


def test_row_order_and_the_way_the_region_is_given_do_not_matter_and_inputs_are_untouched():
    rng = np.random.default_rng(3)
    for name in ["flat 3x3", "tube, two columns", "warm ball of a link", "flat ball of radius 1"]:
        adj, part, region = cases()[name]
        before = hidden_count(adj, part, region, LAM, keep=100)
        shuffled = rng.permuted(adj, axis=1)
        assert not np.array_equal(shuffled, adj)
        keep_adj, keep_part, keep_region = shuffled.copy(), part.copy(), list(region)
        mixed = [region[i] for i in rng.permutation(len(region))] + region[:2]      # out of order, two named twice
        for same in (False, True):
            after = hidden_count(shuffled, part, np.array(mixed), LAM, keep=100, same_energy_only=same)
            assert after["wirings"] == before["wirings"] and after["count"] == before["count"]
            if not same:
                assert after == before
        assert valid_wirings(shuffled, part, iter(mixed)) == valid_wirings(adj, part, region)
        assert np.array_equal(shuffled, keep_adj) and np.array_equal(part, keep_part) and region == keep_region


def test_regions_with_nothing_inside():
    adj, part = torus(8, 8)
    for region, cut in [([], 0), ([5], 4), ([0, 9, 18], 12)]:          # no two of these vertices are linked
        for same in (False, True):
            r = hidden_count(adj, part, region, LAM, same_energy_only=same)
            assert (r["count"], r["points"], r["cut"], r["links"]) == (1, len(region), cut, 0)
            assert r["levels"] == {0.0: 1} and r["wirings"] == [()] and r["sealed"] == (0, 0)


def test_bad_input_is_refused():
    adj, part = torus(8, 8)
    with pytest.raises(ValueError, match="side"):
        hidden_count(adj, np.zeros(64, dtype=np.int64), [0, 1], LAM)
    with pytest.raises(ValueError, match="does not have"):
        hidden_count(adj, part, [0, 64], LAM)
    broken = adj.copy()
    broken[0, 0] = broken[0, 1]                                        # vertex 0 linked twice to one neighbor
    with pytest.raises(ValueError, match="valid"):
        hidden_count(broken, part, [0, 1], LAM)
    with pytest.raises(ValueError, match="free slot"):
        rewire(adj, [0, 1, 8, 9], [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1)])
