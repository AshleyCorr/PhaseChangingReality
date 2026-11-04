"""T74's kernel (graphity.layer_outside; PREREGISTRATION T74 with Amendments 2 and 3): conservation to 1e-9 at every
row (F + stores + daughters + E_out + far - fed = F0), T71's path draw for draw with no outside, the curl-edge flow capped
at c and always downhill without overshoot, nothing crossing at c = 0, the reservoir isolated while fed and while
digesting, digestion ending at balance (or at its limit), the equal background as the common mean, a warm background
feeding the slab and a cold one draining it at most c a sweep, the even take never driving a store below zero,
same-seed reproducibility, and a valid end state by brute force. Small sizes only."""
import networkx as nx
import numpy as np
import pytest

from graphity import contraction as C
from graphity import layer_feed as LF
from graphity import layer_outside as LO
from graphity.cqg_d import is_valid

LAM = 1.25
A = 4.0 * (LAM - 1.0)
PHASE = -1          # the phase is the last field of an outside row
FAR, EDGE_IN, FLOW, G_OUT, G_IN, N_IN, G_BG = 16, 17, 18, 19, 20, 21, 22


def brute_ok(state):
    g = nx.Graph()
    for u in range(state.n):
        for v in state.adj[u]:
            g.add_edge(u, int(v))
    assert g.number_of_nodes() == state.n and all(d == 6 for _, d in g.degree()) and nx.is_bipartite(g)
    assert is_valid(state.adj)


def outside_for(st, feed="steady", c_over_r=2.0, g_bg=0.0, n_fed=120, every=1, n_digest_max=60, n_unfed=120,
                rho_unfed=1.0 / 64, rho_fed=0.0, g_bg_fed=0.0):
    """Amendments 2 and 3 by default: isolated while fed and while digesting, relaxing at 1/64 once unfed toward g_bg
    (a number or LO.EQUAL)."""
    R = st.n / 16.0
    return LO.Outside(n_out=st.n, c=c_over_r * R, n_fed=n_fed, feed=feed, amount=R * every, every=every,
                      rho_fed=rho_fed, g_bg_fed=g_bg_fed, rho_unfed=rho_unfed, g_bg_unfed=g_bg,
                      n_digest_max=n_digest_max, n_unfed=n_unfed)


def run(st, o, seed, **kw):
    kw.setdefault("gate", A * st.n)
    kw.setdefault("release_to", "daughter")
    kw.setdefault("log_every", 1)
    return LO.run_layer_outside(st, LAM, None, seed, outside=o, **kw)


def assert_conserved(first, rows):
    for res, row in zip(LO.conservation_residuals(first["f0"], rows, first["e_out0"]), rows):
        assert abs(res) < 1e-9 * max(1.0, row[5], abs(row[FAR]))


@pytest.mark.parametrize("move_set", LF.MOVE_SETS)
@pytest.mark.parametrize("release_to", LF.RELEASE_TO)
def test_no_outside_is_t71_draw_for_draw(move_set, release_to):
    st = C.build((4, 6, 6))
    args = dict(rate=1.0 / 16, n_sweeps=150, seed=11, gate=A * st.n, leak=0.0, move_set=move_set, max_width=8,
                log_every=7, release_to=release_to)
    a, rows_a, first_a = LF.run_layer_feed(st, LAM, **args)
    b, rows_b, first_b = LO.run_layer_outside(st, LAM, outside=None, **args)
    assert rows_a == rows_b and first_a == first_b
    assert np.array_equal(a.adj, b.adj) and np.array_equal(a.stores, b.stores) and a.dims == b.dims


def test_no_outside_is_t71_with_leak_and_the_feed_stopping():
    st = C.build((4, 6, 6))
    args = dict(rate=0.1, n_sweeps=100, seed=5, gate=0.0, leak=0.02, feed_stops_at_split=True, move_set="model",
                log_every=1)
    _, rows_a, first_a = LF.run_layer_feed(st, LAM, **args)
    _, rows_b, first_b = LO.run_layer_outside(st, LAM, outside=None, **args)
    assert rows_a == rows_b and first_a == first_b


def test_the_length_is_set_by_the_phases():
    st = C.build((4, 6, 6))
    with pytest.raises(AssertionError):
        LO.run_layer_outside(st, LAM, 100, 1, outside=outside_for(st))
    with pytest.raises(AssertionError):
        LO.run_layer_outside(st, LAM, None, 1, outside=None)


@pytest.mark.parametrize("feed,every", [("steady", 1), ("meals", 16), ("meals", 64)])
@pytest.mark.parametrize("g_bg", [0.0, 4 * A, LO.EQUAL])
def test_conservation_and_same_seed(feed, every, g_bg):
    st = C.build((4, 6, 6))
    o = outside_for(st, feed=feed, every=every, g_bg=g_bg, n_fed=128, n_digest_max=64, n_unfed=128)
    a, rows_a, first_a = run(st, o, 21)
    b, rows_b, first_b = run(st, o, 21)
    assert rows_a == rows_b and first_a == first_b
    assert np.array_equal(a.adj, b.adj) and np.array_equal(a.stores, b.stores)
    assert_conserved(first_a, rows_a)
    assert all(row[12] == 0 for row in rows_a)                 # the clean guarantee holds on the new path too
    assert all(s >= -1e-12 for s in a.stores)                  # no store ever below zero
    assert rows_a[-1][13] >= 1                                 # the slab burped at least once
    for sweep, e in first_a["births"]:
        assert e == pytest.approx(A * st.n, abs=1e-9)          # every daughter a N_slab
    # the fed total follows the schedule: meals at sweeps 1, 1 + every, ...; 128 is a multiple of every, so every
    # feed delivers the same energy
    assert rows_a[-1][5] == pytest.approx(o.amount * len(range(1, 129, every)))
    assert rows_a[-1][5] == pytest.approx(128 * st.n / 16.0)
    t = first_a["t_unfed"]
    assert o.n_fed <= t <= o.n_fed + o.n_digest_max and rows_a[-1][0] == t + o.n_unfed
    assert [r[PHASE] for r in rows_a] == [1] * 128 + [2] * (t - 128) + [3] * o.n_unfed
    brute_ok(a)


def test_conservation_with_a_nonzero_start_in_the_reservoir():
    st = C.build((4, 6, 6))
    o = LO.Outside(n_out=st.n, c=4.0, n_fed=50, feed="meals", amount=40.0, every=10, e_out0=123.0, rho_fed=0.05,
                   g_bg_fed=1.0, rho_unfed=0.05, g_bg_unfed=0.0, n_digest_max=30, n_unfed=60)
    _, rows, first = run(st, o, 2, log_every=5)
    assert_conserved(first, rows)


@pytest.mark.parametrize("c_over_r", [2.0, 0.5])
@pytest.mark.parametrize("g_bg", [0.0, 4 * A, LO.EQUAL])
def test_the_edge_flow_is_capped_and_runs_downhill_without_overshoot(c_over_r, g_bg):
    st = C.build((4, 6, 6))
    o = outside_for(st, feed="meals", every=32, c_over_r=c_over_r, g_bg=g_bg, n_fed=128, n_digest_max=64,
                    n_unfed=128)
    _, rows, first = run(st, o, 8)
    assert len(rows) == first["t_unfed"] + o.n_unfed
    assert first["max_abs_flow"] <= o.c + 1e-12
    crossed_in = crossed_out = False
    for row in rows:
        flow, g_out, g_in, n_in = row[FLOW], row[G_OUT], row[G_IN], row[N_IN]
        assert abs(flow) <= o.c + 1e-12
        assert flow * (g_out - g_in) >= -1e-12                 # never uphill
        if abs(g_out - g_in) > 1e-12:
            assert flow != 0.0 or o.c == 0.0
        # after the flow the two means have not crossed
        g_out_after = g_out - flow / o.n_out
        g_in_after = g_in + flow / n_in
        assert (g_out_after - g_in_after) * (g_out - g_in) >= -1e-9
        crossed_in |= flow > 0
        crossed_out |= flow < 0
    assert crossed_in
    if g_bg == 0.0:
        assert crossed_out                                     # a cold background draws the slab's energy back out


def test_at_zero_cap_nothing_crosses_and_digestion_runs_to_its_limit():
    st = C.build((4, 6, 6))
    o = LO.Outside(n_out=st.n, c=0.0, n_fed=100, feed="meals", amount=500.0, every=10, rho_fed=1.0 / 64,
                   g_bg_fed=4 * A, rho_unfed=1.0 / 64, g_bg_unfed=4 * A, n_digest_max=40, n_unfed=60)
    final, rows, first = run(st, o, 3)
    assert all(row[EDGE_IN] == 0.0 and row[FLOW] == 0.0 for row in rows)
    assert all(row[4] == 0.0 for row in rows) and np.all(final.stores == 0.0)
    assert all(row[3] == first["f0"] for row in rows) and rows[-1][13] == 0     # nothing paid, nothing born
    assert first["digest_balanced"] is False and first["t_unfed"] == 140       # never balanced: the limit ends it
    assert_conserved(first, rows)


def test_the_reservoir_is_isolated_while_fed_and_while_digesting():
    """Amendments 2 (item 1) and 3 (item 1): no relaxation until the unfed phase; the meals are the reservoir's only
    input, and digestion ends when the two means agree within 1 %."""
    st = C.build((4, 6, 6))
    o = outside_for(st, feed="meals", every=16, c_over_r=0.5, g_bg=0.0, n_fed=64, n_digest_max=400, n_unfed=64)
    _, rows, first = run(st, o, 4)
    assert all(r[FAR] == 0.0 for r in rows if r[PHASE] in (1, 2))
    assert any(r[PHASE] == 2 for r in rows)                       # c = R/2 leaves a backlog to digest
    t = first["t_unfed"]
    assert first["digest_balanced"] is True
    at = next(r for r in rows if r[0] == t)
    g_out, g_in = at[15] / o.n_out, at[4] / at[1]
    assert abs(g_out - g_in) <= 0.01 * g_in
    before = next(r for r in rows if r[0] == t - 1)               # and it was not balanced a sweep earlier
    assert abs(before[15] / o.n_out - before[4] / before[1]) > 0.01 * before[4] / before[1]
    assert first["s_unfed_start"] == at[4]
    assert first["g_common"] == pytest.approx((at[15] + at[4]) / (o.n_out + at[1]))
    assert next(r for r in rows if r[0] == t + 1)[FAR] > 0.0      # cold, unfed: the reservoir drains to far space
    stop = next(r for r in rows if r[0] == 64)
    assert stop[5] == pytest.approx(stop[15] + stop[EDGE_IN])     # fed = what the reservoir holds + what crossed


def test_a_fed_phase_that_ends_balanced_has_no_digestion():
    st = C.build((4, 6, 6))
    o = outside_for(st, feed="steady", c_over_r=1e6, g_bg=0.0, n_fed=50, n_unfed=30)
    _, rows, first = run(st, o, 6, gate=1e18, move_set="layer")
    assert first["t_unfed"] == 50 and first["digest_balanced"] is True
    assert not any(r[PHASE] == 2 for r in rows) and rows[-1][0] == 80


def test_an_equal_background_is_the_common_mean_and_moves_nothing_once_balanced():
    """Amendment 3, item 2: equal is the common mean of reservoir and slab stores at the start of the unfed phase. With
    the two sides balanced exactly there (a cap large enough to equalize every sweep, no move able to fire), nothing
    crosses the edge in the unfed phase, the slab's stores stay as they were and far space is never touched."""
    st = C.build((4, 6, 6))
    o = outside_for(st, feed="steady", c_over_r=1e6, g_bg=LO.EQUAL, n_fed=50, n_unfed=100)
    _, rows, first = run(st, o, 6, gate=1e18, move_set="layer")
    start = next(r for r in rows if r[0] == first["t_unfed"])
    assert first["g_common"] == pytest.approx((start[15] + start[4]) / (o.n_out + start[1]))
    unfed = [r for r in rows if r[PHASE] == 3]
    assert len(unfed) == 100 and all(r[G_BG] == first["g_common"] for r in unfed)
    assert all(abs(r[FLOW]) <= 1e-9 for r in unfed)
    assert all(abs(r[4] - start[4]) <= 1e-9 * start[4] for r in unfed)
    assert all(abs(r[FAR]) <= 1e-9 * start[5] for r in rows)


def test_an_equal_background_after_digestion_moves_at_most_the_last_percent():
    """With a backlog (c = R/4) digestion runs until the two means agree within 1 %; the equal background is then the
    common mean, which lies between them, so the slab's stores move by no more than that last 1 % in the unfed
    phase (no move able to fire)."""
    st = C.build((4, 6, 6))
    o = outside_for(st, feed="steady", c_over_r=0.25, g_bg=LO.EQUAL, n_fed=50, n_digest_max=4096, n_unfed=200)
    _, rows, first = run(st, o, 6, gate=1e18, move_set="layer")
    assert first["digest_balanced"] is True and first["t_unfed"] > 50     # there was a backlog to digest
    s0 = first["s_unfed_start"]
    assert all(abs(r[4] - s0) <= 0.01 * s0 for r in rows if r[PHASE] == 3)
    assert_conserved(first, rows)


def test_warm_background_unfed_feeds_the_slab():
    """Warm surroundings (g_bg = 4a) once unfed: the flow only ever runs into the slab, so stores + F never decreases
    in the unfed phase, and the stores end more than 5 % above their value at its start: fed by its surroundings. The
    layer moves are held off by a gate no pool can reach, so that a refold's spending is not in the test."""
    st = C.build((4, 6, 6))
    o = outside_for(st, feed="steady", c_over_r=0.5, g_bg=4 * A, n_fed=40, n_digest_max=200, n_unfed=200)
    _, rows, first = run(st, o, 13, gate=1e18, move_set="model")
    unfed = [r for r in rows if r[PHASE] == 3]
    assert len(unfed) == 200 and all(r[FLOW] >= -1e-12 for r in unfed)
    held = [r[3] + r[4] for r in unfed]
    assert all(b >= a - 1e-9 for a, b in zip(held[:-1], held[1:]))
    for a, b in zip(unfed[:-1], unfed[1:]):
        if b[9] == a[9]:                                       # no clean switch fired: the stores never fell
            assert b[4] >= a[4] - 1e-9
    assert rows[-1][4] > 1.05 * first["s_unfed_start"]
    assert rows[-1][FAR] < 0                                   # the far ledger was the source


def test_cold_background_unfed_the_slab_loses_at_most_c_a_sweep():
    st = C.build((4, 6, 6))
    o = outside_for(st, feed="steady", c_over_r=0.5, g_bg=0.0, n_fed=120, n_digest_max=200, n_unfed=280)
    _, rows, first = run(st, o, 17, gate=1e18, move_set="layer")
    stores = [r[4] for r in rows if r[PHASE] == 3]
    assert stores[-1] < 0.5 * first["s_unfed_start"]           # it dissolves
    assert all(a - b <= o.c + 1e-9 for a, b in zip(stores[:-1], stores[1:]))


def test_take_evenly_is_even_when_it_can_be_and_never_negative():
    s = np.array([3.0, 3.0, 3.0, 3.0])
    LO.take_evenly(s, 2.0)
    assert np.allclose(s, 2.5)
    s = np.array([0.0, 0.1, 5.0, 5.0])
    LO.take_evenly(s, 2.0)
    assert s[0] == 0.0 and s[1] == 0.0 and np.allclose(s[2:], 5.0 - 0.95) and s.sum() == pytest.approx(8.1)


def test_edge_flow_equalizes_when_the_cap_allows():
    stores = np.zeros(10)
    flow, g_out, g_in = LO.edge_flow(30.0, 5, stores, c=1e9)
    assert flow == pytest.approx(20.0)                         # 30 shared over 15 stores: 10 stay, 20 cross
    flow, _, _ = LO.edge_flow(30.0, 5, stores, c=3.0)
    assert flow == 3.0
    flow, _, _ = LO.edge_flow(0.0, 5, np.full(10, 3.0), c=100.0)
    assert flow == pytest.approx(-10.0)                        # 30 over 15 stores: 10 leave the slab
