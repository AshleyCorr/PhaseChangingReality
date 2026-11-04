"""Rule-6 tests for graphity.row_growth (PREREGISTRATION T76, "Runner and tests").

An inserted pair keeps the graph valid (simple, two-sided, four links per point, the hard-core rule); a pair inserted
into a flat torus leaves it flat and adds exactly 2W points; no insertion where the corridor holds a curled link or
where the grid is broken across the boundary; growth off reproduces cqg.run_chain on the same seed; same-seed
reproducibility; and D and the curled-link share checked by brute force with networkx on a grown state.
"""
import itertools

import networkx as nx
import numpy as np
import pytest

from graphity.cqg import NO_CAP, _switch, hamiltonian, is_valid, run_chain, surplus, torus, total_squares
from graphity.row_growth import (GrowthState, can_insert, corridor_clear, curled_points, edge_census, grid_intact,
                                 growth_step, insert_rows, row_points, run_growth, settle_graph, stream_seeds)

LAM = 1.25


def two_sided(state):
    return bool(np.all(state.part[:, None] != state.part[state.adj]))


def labels_consistent(state):
    """part = (x + y) % 2 and `at` inverts `coords`."""
    x, y = state.coords[:, 0], state.coords[:, 1]
    return (np.array_equal(state.part, (x + y) % 2)
            and np.array_equal(state.at[x, y], np.arange(state.n)))


def is_flat_grid(state):
    """Every point linked exactly to its four grid neighbours."""
    lx, ly = state.lx, state.ly
    for p in range(state.n):
        x, y = state.coords[p]
        want = {state.at[(x + 1) % lx, y], state.at[(x - 1) % lx, y],
                state.at[x, (y + 1) % ly], state.at[x, (y - 1) % ly]}
        if set(state.adj[p].tolist()) != want:
            return False
    return True


def to_nx(adj):
    g = nx.Graph()
    g.add_nodes_from(range(adj.shape[0]))
    for u in range(adj.shape[0]):
        for v in adj[u]:
            g.add_edge(u, int(v))
    return g


def brute_census(adj):
    """D and the curled share from every 4-cycle networkx finds."""
    g = to_nx(adj)
    per_edge = {tuple(sorted(e)): 0 for e in g.edges()}
    for cyc in nx.simple_cycles(g, length_bound=4):
        if len(cyc) == 4:
            for a, b in zip(cyc, cyc[1:] + cyc[:1]):
                per_edge[tuple(sorted((a, b)))] += 1
    d = sum(max(0, 2 - s) for s in per_edge.values())
    curled = sum(1 for s in per_edge.values() if s > 2)
    return d, curled / len(per_edge)


def brute_corridor(state, axis, i, reach=3):
    """Condition (ii) from networkx distances: no link with an endpoint within `reach` of rows i, i + 1 is curled."""
    g = to_nx(state.adj)
    src = list(row_points(state, axis, i)) + list(row_points(state, axis, i + 1))
    dist = nx.multi_source_dijkstra_path_length(g, [int(s) for s in src], cutoff=reach)
    near = set(dist)
    for u, v in g.edges():
        if (u in near or v in near):
            s_e = sum(1 for a in g[v] if a != u for b in g[u] if b != v and g.has_edge(a, b))
            if s_e > 2:
                return False
    return True


def warm_state(lx, ly, g, sweeps, seed):
    """A flat torus with some defects from the chain at coupling g."""
    st = GrowthState.from_torus(lx, ly)
    run_chain(st.adj, st.side_u(), 1.0 / g, 0, sweeps, seed, LAM, NO_CAP)
    assert is_valid(st.adj, NO_CAP)
    return st


# ------------------------------------------------------------------------- the flat torus and one insertion

@pytest.mark.parametrize("lx,ly", [(8, 8), (12, 6), (6, 10)])
def test_from_torus_labels(lx, ly):
    st = GrowthState.from_torus(lx, ly)
    assert st.n == lx * ly and (st.lx, st.ly) == (lx, ly)
    assert labels_consistent(st) and is_flat_grid(st) and two_sided(st)
    adj, part = torus(lx, ly, cap=NO_CAP)
    assert np.array_equal(st.adj, adj) and np.array_equal(st.part, part)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("i", [0, 3, 7])          # 7 is the wrap boundary of an 8-long axis
def test_insertion_into_flat_torus_stays_flat_and_adds_2w(axis, i):
    st = GrowthState.from_torus(8, 8)
    w = st.at.shape[1 - axis]
    n0 = st.n
    assert can_insert(st, axis, i) == (True, "ok")
    new = insert_rows(st, axis, i)
    assert st.n == n0 + 2 * w and len(new) == 2 * w
    assert st.at.shape[axis] == 10 and st.at.shape[1 - axis] == 8
    assert is_valid(st.adj, NO_CAP) and two_sided(st) and labels_consistent(st)
    d, curled, links = edge_census(st.adj)
    assert d == 0 and curled == 0 and links == 2 * st.n
    assert total_squares(st.adj) == st.n and surplus(st.adj) == 0 and hamiltonian(st.adj, LAM) == 0.0
    assert is_flat_grid(st)


def test_flat_torus_allows_every_boundary_and_grows_to_a_flat_torus():
    st = GrowthState.from_torus(8, 6)
    for axis in (0, 1):
        for i in range(st.at.shape[axis]):
            assert can_insert(st, axis, i) == (True, "ok")
    pairs = growth_step(st, np.random.default_rng(1))
    assert pairs == 8 + 6                            # every boundary of the step's start, once
    assert (st.lx, st.ly) == (8 + 2 * 8, 6 + 2 * 6)
    assert is_valid(st.adj, NO_CAP) and two_sided(st) and labels_consistent(st) and is_flat_grid(st)
    assert edge_census(st.adj)[:2] == (0, 0)


def test_insertion_into_a_rewired_state_stays_valid():
    """Insertions wherever allowed, in a state the chain has damaged, keep every constraint."""
    st = warm_state(12, 12, 2.5, 30, 5)
    rng = np.random.default_rng(7)
    total = 0
    for _ in range(3):
        run_chain(st.adj, st.side_u(), 1.0 / 2.5, 0, 2, -1, LAM, NO_CAP)
        total += growth_step(st, rng)
        assert is_valid(st.adj, NO_CAP) and two_sided(st) and labels_consistent(st)
        assert (st.adj >= 0).all() and all(len(set(r)) == 4 for r in st.adj.tolist())
    assert total > 0


# ------------------------------------------------------------------------- refusals

def test_no_insertion_anywhere_on_a_curled_torus():
    """A side of 4 wraps round into squares: the links round it carry three each, so no corridor is clear."""
    st = GrowthState.from_torus(4, 12)
    assert edge_census(st.adj)[1] > 0
    for axis in (0, 1):
        for i in range(st.at.shape[axis]):
            ok, why = can_insert(st, axis, i)
            assert not ok and why == "curled"
    assert growth_step(st, np.random.default_rng(0)) == 0


def local_curl():
    """A 12 x 12 flat torus with one switch, found by search, that leaves links on three squares round x = 5, 6."""
    st = GrowthState.from_torus(12, 12)
    at = st.at
    _switch(st.adj, at[4, 6], at[5, 6], at[7, 5], at[6, 5])
    assert is_valid(st.adj, NO_CAP)
    return st


def test_no_insertion_where_the_corridor_holds_a_curled_link():
    st = local_curl()
    flagged = curled_points(st.adj)
    assert flagged.any()
    assert grid_intact(st, 0, 5)                     # the grid is whole across x = 5 | 6 ...
    assert can_insert(st, 0, 5) == (False, "curled")  # ... but curled links sit in the corridor
    assert can_insert(st, 0, 2) == (False, "curled")  # row 3 is 2 steps from a curled point
    assert can_insert(st, 0, 10) == (True, "ok")       # rows 10, 11: 4 or more steps away
    assert can_insert(st, 0, 11) == (True, "ok")
    for axis in (0, 1):                             # the numba corridor agrees with networkx everywhere
        for i in range(st.at.shape[axis]):
            assert corridor_clear(st, axis, i) == brute_corridor(st, axis, i)


def test_no_insertion_where_the_grid_is_broken():
    st = GrowthState.from_torus(12, 12)
    at = st.at
    u1, v1, u2, v2 = at[2, 4], at[3, 4], at[8, 8], at[8, 9]      # one link across x = 2 | 3, one far away
    assert st.part[u1] == st.part[u2] == 0
    _switch(st.adj, u1, v1, u2, v2)
    assert is_valid(st.adj, NO_CAP)
    assert not grid_intact(st, 0, 2)
    assert can_insert(st, 0, 2) == (False, "broken")
    with pytest.raises(ValueError):
        insert_rows(st, 0, 2)


def test_corridor_matches_networkx_on_damaged_states():
    for seed in (1, 2):
        st = warm_state(10, 12, 3.0, 20, seed)
        flagged = curled_points(st.adj)
        for axis in (0, 1):
            for i in range(st.at.shape[axis]):
                assert corridor_clear(st, axis, i, flagged=flagged) == brute_corridor(st, axis, i)


def test_growth_step_offers_each_boundary_once_and_not_new_ones():
    """On a flat torus every original boundary is taken exactly once; the boundaries made by the step are not."""
    st = GrowthState.from_torus(6, 6)
    assert growth_step(st, np.random.default_rng(3)) == 12
    assert (st.lx, st.ly) == (18, 18)


def test_stop_at_ends_the_step():
    st = GrowthState.from_torus(8, 8)
    pairs = growth_step(st, np.random.default_rng(0), stop_at=64 + 40)
    assert pairs == 3 and st.n >= 104                # two pairs add at most 16 + 20 points, a third at least 16


# ------------------------------------------------------------------------- the chain and the driver

@pytest.mark.parametrize("lx,ly,fair,sweeps", [(12, 8, 5, 5), (16, 16, 3, 8)])
def test_growth_off_reproduces_run_chain(lx, ly, fair, sweeps):
    """At N = 96 a fair sweep is one sweep; at N = 256, 3 fair sweeps are exactly 8 sweeps of 2N attempts."""
    n = lx * ly
    assert sum(2 * n * n * k // 96 - 2 * n * n * (k - 1) // 96 for k in range(1, fair + 1)) == 2 * n * sweeps
    a = GrowthState.from_torus(lx, ly)
    recs = run_growth(a, 2.0, LAM, fair, 0, 8, seed=42)
    chain_seed, _ = stream_seeds(42)
    adj, part = torus(lx, ly, cap=NO_CAP)
    s, x, _ = run_chain(adj, np.flatnonzero(part == 0), 1.0 / 2.0, 0, sweeps, chain_seed, LAM, NO_CAP)
    assert np.array_equal(a.adj, adj)
    assert recs[-1]["S"] == s[-1] and recs[-1]["X"] == x[-1]
    assert [r["phase"] for r in recs] == ["start"] + ["settle"] * fair
    assert all(r["N"] == n and r["rows_added"] == 0 for r in recs)


def test_settle_graph_is_run_growths_settling_on_a_torus():
    """The unlabelled settling (for starts that are not a labelled torus) gives run_growth's settling readings and
    final wiring on the same seed, field by field except the grid sizes."""
    a = GrowthState.from_torus(16, 12)
    ra = run_growth(a, 2.5, LAM, 7, 0, 8, seed=[5, 0, 3])
    adj, part = torus(16, 12, cap=NO_CAP)
    rb = settle_graph(adj, part, 2.5, LAM, 7, seed=[5, 0, 3])
    assert np.array_equal(a.adj, adj)
    drop = ("Lx", "Ly")
    keep = [{k: v for k, v in r.items() if k not in drop} for r in ra]
    assert keep == [{k: v for k, v in r.items() if k not in drop} for r in rb]
    assert is_valid(adj, NO_CAP) and np.all(part[:, None] != part[adj])


def test_fair_clock_carries_the_remainder():
    st = GrowthState.from_torus(16, 16)
    recs = run_growth(st, 2.0, LAM, 6, 0, 8, seed=1)
    att = [r["attempts"] for r in recs[1:]]
    assert np.array_equal(np.cumsum(att), (2 * 256 * 256 * np.arange(1, 7)) // 96)


def test_same_seed_same_run_and_different_seed_different_run():
    a, b, c = (GrowthState.from_torus(8, 8) for _ in range(3))
    ra = run_growth(a, 2.5, LAM, 5, 6, 8, seed=[20267601, 0])
    rb = run_growth(b, 2.5, LAM, 5, 6, 8, seed=[20267601, 0])
    rc = run_growth(c, 2.5, LAM, 5, 6, 8, seed=[20267601, 1])
    assert ra == rb and np.array_equal(a.adj, b.adj) and np.array_equal(a.coords, b.coords)
    assert ra != rc


def test_driver_grows_records_and_stops_at_the_cap():
    st = GrowthState.from_torus(8, 8)
    recs = run_growth(st, 1.5, LAM, 3, 50, 4, seed=9)
    assert recs[0]["phase"] == "start" and recs[0]["attempts"] == 0
    grow = [r for r in recs if r["phase"] == "grow"]
    assert grow and st.n >= 4 * 64 and recs[-1]["N"] == st.n
    assert len(grow) < 50                             # the cap stopped it before the last fair sweep
    n = 64
    for r in recs[1:]:                                # N only grows, and by whole rows
        assert r["N"] >= n
        n = r["N"]
        assert r["rho"] == r["D"] / r["N"]
    assert is_valid(st.adj, NO_CAP) and two_sided(st) and labels_consistent(st)


def test_census_matches_networkx_on_a_grown_state():
    st = GrowthState.from_torus(8, 8)
    run_growth(st, 6.0, LAM, 30, 3, 8, seed=11)
    assert st.n > 64
    d, curled, links = edge_census(st.adj)
    bd, bshare = brute_census(st.adj)
    assert d == bd and curled / links == pytest.approx(bshare)
    assert d == 4 * (st.n - total_squares(st.adj)) + surplus(st.adj)     # D - X = sum_e (2 - S_e) = 4 (N - S)
    assert d > 0                                      # the check is not vacuous


def test_brute_census_on_known_states():
    assert brute_census(torus(8, 8, cap=NO_CAP)[0]) == (0, 0.0)
    adj = torus(4, 4, cap=NO_CAP)[0]                  # the 4-cube: three squares on every link
    assert brute_census(adj) == (0, 1.0) and edge_census(adj) == (0, 32, 32)


# --- T76 Amendment 2: each clear offer is taken with chance q ---------------------------------------------------

def test_q_zero_adds_nothing_and_q_one_is_the_registered_rate():
    from graphity.row_growth import GrowthState, growth_step
    st = GrowthState.from_torus(12)
    assert growth_step(st, np.random.default_rng(1), q=0.0) == 0
    assert st.n == 144
    a, b = GrowthState.from_torus(12), GrowthState.from_torus(12)
    pa = growth_step(a, np.random.default_rng(7))
    pb = growth_step(b, np.random.default_rng(7), q=1.0)
    assert pa == pb and a.n == b.n and np.array_equal(a.adj, b.adj)


def test_q_half_takes_about_half_the_offers_on_a_clean_sheet():
    from graphity.row_growth import GrowthState, growth_step
    pairs = [growth_step(GrowthState.from_torus(12), np.random.default_rng(s), q=0.5) for s in range(40)]
    mean = float(np.mean(pairs))
    assert 0.5 * 24 * 0.5 < mean < 0.5 * 24 * 1.5   # 24 boundaries offered; about half taken
