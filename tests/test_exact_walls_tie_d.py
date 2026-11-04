"""scripts/exact_walls_tie_d.py: the direction tie of VISION Update 30, T = sum_v d(v) (D - d(v)), and the walls
priced with it. At kappa = 0 it must be exact_walls_d.py; the tie of the ladder's rungs is known in closed form; and
a switch's change of T, recomputed on the whole graph, is checked against an independent count with networkx."""
import importlib.util
import sys
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tie = _load("exact_walls_tie_d")
walls_d = _load("exact_walls_d")

from graphity.cqg_d import _switch, is_valid, torus     # noqa: E402


def _by_squares(kinds):
    """Collapse the tie script's kinds to exact_walls_d's (dS, dX) kinds."""
    out = Counter()
    for (ds, dx, _, _, _), ways in kinds.items():
        out[(ds, dx)] += ways
    return out


@pytest.mark.parametrize("dims", [[16, 4], [4, 4, 6], [4, 4, 18]])
def test_kappa_zero_is_exact_walls_d(dims):
    adj, part = torus(dims)
    k = tie.kinds(adj, part)
    ref = {(ds, dx): ways for _, ds, dx, ways in walls_d.walls(adj, part, 1.25, top=10 ** 6)}
    assert dict(_by_squares(k)) == ref
    # and the priced wall at kappa = 0 is the model's: O49's 96 - 64 lam out of 4 x 4 x 18, paper 1's 12 out of 16 x 4
    lam = 1.25
    expected = {(16, 4): 12.0, (4, 4, 6): 96 - 80 * lam, (4, 4, 18): 96 - 64 * lam}[tuple(dims)]
    assert tie.wall(k, lam, 0.0)[0] == pytest.approx(expected)
    assert tie.wall(k, lam, 0.0, plus=True)[0] == pytest.approx(expected)


def test_null_switch_of_the_tube_is_counted_and_left_out():
    adj, part = torus([16, 4])
    k = tie.kinds(adj, part)
    nulls = [q for q in k if tie.is_null(q)]
    assert len(nulls) == 1 and k[nulls[0]] == 2
    assert tie.wall(k, 1.25, 0.0)[1] != nulls[0]


@pytest.mark.parametrize("dims", [[6, 6], [6, 6, 8], [6, 6, 6, 6], [4, 4], [4, 4, 4], [4, 4, 4, 4]])
def test_flat_torus_and_single_cube_have_no_tie(dims):
    adj, _ = torus(dims)
    n, dim, h, t, tp, hist = tie.rung(adj, 1.25)
    assert t == 0 and tp == 0
    assert list(hist) == [dim if min(dims) > 4 else 0]


@pytest.mark.parametrize("dims, curled", [([4, 4, 8], 2), ([4, 4, 18], 2), ([4, 12, 12], 1), ([4, 4, 4, 12], 3),
                                          ([4, 4, 8, 8], 2), ([4, 8, 8, 8], 1)])
def test_partly_curled_torus_carries_c_times_d_open_per_point(dims, curled):
    """Every point of a torus with c directions curled has d = D - c, so T = N (D - c) c. Six links: 4 x 4 x L (d = 1)
    gives 1 x (3 - 1) = 2 per point and 4 x 12 x 12 (d = 2) gives 2 x 1 = 2 as well; eight links: 4 x 4 x 4 x 12 and
    4 x 8 x 8 x 8 give 3 per point, 4 x 4 x 8 x 8 gives 4."""
    adj, _ = torus(dims)
    n, dim, h, t, tp, hist = tie.rung(adj, 1.25)
    assert hist == {dim - curled: n}
    assert t == tp == n * (dim - curled) * curled
    assert h / n == pytest.approx(4 * 0.25 * curled)          # the additive ladder, 4 (lam - 1) per curled direction


def _d_networkx(adj):
    """d(v) counted independently: pairs of v's neighbors with no common neighbor other than v."""
    g = nx.Graph()
    for u in range(adj.shape[0]):
        for v in adj[u]:
            g.add_edge(u, int(v))
    d = np.zeros(adj.shape[0], dtype=np.int64)
    for v in g:
        nb = sorted(g[v])
        for i in range(len(nb)):
            for j in range(i + 1, len(nb)):
                common = (set(g[nb[i]]) & set(g[nb[j]])) - {v}
                if not common:
                    d[v] += 1
    return d


def test_change_of_tie_matches_networkx_on_every_switch_of_one_torus():
    """Every valid switch out of 4 x 4 x 6, its change of T, T+ and the d-histogram recounted with networkx: the
    tallies, ways included, are the script's."""
    adj, part = torus([4, 4, 6])
    dim = 3
    k = tie.kinds(adj, part)
    d0 = _d_networkx(adj)
    t0 = int(np.sum(d0 * (dim - d0)))
    h0 = np.bincount(d0, minlength=16)
    ref = Counter()
    u1 = int(np.flatnonzero(part == 0)[0])
    for u2 in np.flatnonzero(part == 0):
        u2 = int(u2)
        if u2 == u1:
            continue
        for v1 in adj[u1]:
            v1 = int(v1)
            if v1 in adj[u2]:
                continue
            for v2 in adj[u2]:
                v2 = int(v2)
                if v2 == v1 or v2 in adj[u1]:
                    continue
                t = adj.copy()
                _switch(t, u1, v1, u2, v2)
                if not is_valid(t):
                    continue
                d = _d_networkx(t)
                dt = int(np.sum(d * (dim - d))) - t0
                dtp = int(np.sum(np.maximum(0, d * (dim - d)))) - t0
                dh = np.bincount(d, minlength=16) - h0
                ref[(dt, dtp, tuple((int(i), int(c)) for i, c in enumerate(dh) if c != 0))] += 1
    mine = Counter()
    for (_, _, dt, dtp, change), ways in k.items():
        mine[(dt, dtp, change)] += ways
    assert mine == ref


def test_positive_set_is_the_intersection_of_half_lines():
    # at lam = 1: A costs 16 - 8 kappa (positive below 2); B costs 4 + 2 kappa (always); C costs -4 + 4 kappa (above 1)
    fake = Counter({(-1, 0, -8, 0, ((4, 1),)): 1, (0, 1, 2, 2, ((2, 1),)): 1, (0, -1, 4, 4, ((1, 1),)): 1})
    assert tie.positive_set(fake, 1.0) == (1.0, 2.0, True)
    for kap in np.linspace(0, 4, 401):
        inside = 1.0 < kap < 2.0
        assert inside == all(tie.cost(q, 1.0, kap) > 0 for q in fake)
        assert (tie.wall(fake, 1.0, kap)[0] > 0) == inside
    # a kind that is downhill and does not change T leaves no kappa at which the state is stuck
    assert tie.positive_set(Counter({(1, 0, 0, 0, ((3, 1),)): 1}), 1.0) is None
