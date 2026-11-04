"""scripts/exact_sixloop_walls.py: what a price on loops of six does to the tube, by complete listing at 48 points
(ASSUMPTIONS O124). Pins the numbers the record quotes."""
import sys
from pathlib import Path

import networkx as nx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import exact_sixloop_walls as w                                   # noqa: E402
from graphity.cqg import NO_CAP, is_valid                         # noqa: E402
from read_endpoint_width import twisted_tube                      # noqa: E402

LENGTH, LAM = 12, 1.25


@pytest.fixture(scope="module")
def six():
    adj, side = twisted_tube(LENGTH, 2), w.side_of(LENGTH)
    return adj, side, w.switches(adj, side)


def test_the_tube_six_around_is_stuck_below_four_elevenths(six):
    _adj, _side, moves = six
    table = w.kinds(moves)
    assert sum(table.values()) == 2472 and len(table) == 34
    assert table[(-4, 0, -176)] == 96                                        # the switch that removes the most
    assert table[(-1, 3, -59)] == 48 and table[(-2, 0, -24)] == 96 and table[(-2, 2, -94)] == 48
    assert w.stuck_below(table, LAM) == pytest.approx(64 / 176)              # 4/11
    assert min(w.cost(k, LAM, 0.0) for k in table) == 20.0                   # with no price: a re-curling switch
    assert table[(0, 4, 48)] == 72                                           # it breaks no square: 16 lambda, not 32
    assert all(w.cost(k, LAM, 0.30) > 0 for k in table) and any(w.cost(k, LAM, 0.37) < 0 for k in table)


def test_the_cheapest_way_out_is_long_range_above_a_price_and_local_below(six):
    adj, _side, moves = six
    g = nx.Graph((i, int(j)) for i in range(adj.shape[0]) for j in adj[i])
    dist = dict(nx.all_pairs_shortest_path_length(g))

    def apart(move):
        u1, v1, u2, v2 = move
        return min(dist[a][b] for a in (u1, v1) for b in (u2, v2))

    far = {apart(m[0]) for m in moves if m[1:] == (-4, 0, -176)}
    assert min(far) >= 5                                                     # its two links are far apart
    assert {apart(m[0]) for m in moves if m[1:] == (-1, 3, -59)} == {2}      # the local ones are not
    assert {apart(m[0]) for m in moves if m[1:] == (-2, 0, -24)} == {2}
    assert {apart(m[0]) for m in moves if m[1:] == (-2, 2, -94)} == {3}
    # 64 - 176 mu against 31 - 59 mu: they cross at mu = 33/117
    assert w.cost((-4, 0, -176), LAM, 0.25) > w.cost((-1, 3, -59), LAM, 0.25)
    assert w.cost((-4, 0, -176), LAM, 0.30) < w.cost((-1, 3, -59), LAM, 0.30)


def test_eight_around_is_stuck_at_every_price_and_the_tube_only_below_a_tenth():
    eight = w.kinds(w.switches(twisted_tube(LENGTH, 4), w.side_of(LENGTH)))
    assert w.stuck_below(eight, LAM) == float("inf")
    assert min(w.cost(k, LAM, 0.0) for k in eight) == 32.0 and (-2, 0, 12) in eight
    tube = w.kinds(w.switches(twisted_tube(LENGTH, 0), w.side_of(LENGTH)))
    assert w.stuck_below(tube, LAM) == pytest.approx(14 / 132)
    assert min(w.cost(k, LAM, 0.0) for k in tube if w.cost(k, LAM, 0.0) != 0.0) == 12.0   # paper 1's twelve


def test_after_the_push_the_change_can_go_on_without_breaking_more_squares():
    adj, push = w.pushed(LENGTH)
    assert push == (-4, 0, -176) and is_valid(adj, NO_CAP)
    table = w.kinds(w.switches(adj, w.side_of(LENGTH)))
    assert table[(4, 0, 176)] == 1                                           # one switch straight back
    keepers = {k[2]: c for k, c in table.items() if k[0] == 0 and k[1] == 0 and k[2] < 0}
    assert keepers == {-124: 2, -56: 2, -41: 2, -32: 4, -12: 4}             # squares kept, six-loops removed
    assert all(w.cost((0, 0, de), LAM, 0.05) < 0 for de in keepers)          # downhill at any price above zero
