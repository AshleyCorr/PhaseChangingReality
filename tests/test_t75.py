"""T75 as amended: the distances on the starting graph, the general flaw, the verdict words on synthetic records, the
runner on a tiny made-up set of saved states, and the config writer (graphity.coupled, scripts/run_t75.py,
scripts/analyse_t75.py, scripts/make_t75_configs.py)."""
import csv
import importlib.util
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from graphity.coupled import (cheapest_switch_at, damage_mask, diameter, fair_sweep_schedule, flaw_at, flaw_candidates,
                              graph_distances, joined_region, lattice_distance, read_records)
from graphity.cqg import NO_CAP, hamiltonian, is_valid, run_chain, torus

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


run_t75 = _load("run_t75")
analyse = _load("analyse_t75")
make = _load("make_t75_configs")


def to_nx(adj):
    g = nx.Graph()
    for u in range(adj.shape[0]):
        for v in adj[u]:
            g.add_edge(u, int(v))
    return g


# ---------------------------------------------------------------- distances

def test_graph_distances_and_diameter_match_networkx():
    adj, part = torus(10, cap=NO_CAP)
    run_chain(adj, np.flatnonzero(part == 0), 0.0, 3, 1, 5, 1.0, NO_CAP, False)   # a few random moves: not flat
    g = to_nx(adj)
    sources = np.array([0, 17, 42], dtype=np.int64)
    want = nx.multi_source_dijkstra_path_length(g, set(sources.tolist()))
    got = graph_distances(adj, sources)
    assert all(got[v] == want[v] for v in range(100))
    assert diameter(adj) == nx.diameter(g)
    flat, _ = torus(12, cap=NO_CAP)
    assert diameter(flat) == 12                                   # 6 + 6, the far corner
    assert (graph_distances(flat, np.array([5])) == lattice_distance(np.arange(144), 5, 12)).all()


def test_diameter_of_a_graph_in_two_pieces_is_minus_one():
    a, _ = torus(6, cap=NO_CAP)
    two = np.concatenate([a, a + 36])
    assert diameter(two) == -1 and graph_distances(two, np.array([0]))[40] == -1


def test_joined_region_and_jumps_on_a_hand_built_mask():
    l = 12
    flat, _ = torus(l, cap=NO_CAP)
    at = lambda x, y: (x % l) * l + y % l                         # noqa: E731
    anchors = np.array([at(0, 0), at(1, 0), at(0, 2), at(1, 2)], dtype=np.int64)
    mask = np.zeros(l * l, dtype=bool)
    mask[anchors] = True
    chain = [at(3, 0), at(5, 0)]                                  # each 2 steps from the one before: joined
    far = at(5, 4)                                                # 4 + 2 = 6 from (1, 2); 4 from (5, 0): not joined
    mask[chain] = True
    mask[far] = True
    region = joined_region(flat, mask, anchors, 2)
    assert set(np.flatnonzero(region)) == set(anchors.tolist()) | set(chain)
    damaged, radius, joined, jump = read_records(flat, mask[None, :], anchors, 2)
    assert damaged[0] == 7 and joined[0] == 4 and radius[0] == 6 and not jump[0]   # (5, 0) is 4 from (1, 0)
    mask[at(6, 7)] = True                                         # 5 + 5 = 10 from the nearest anchor: farther than 2 x 4
    assert read_records(flat, mask[None, :], anchors, 2)[3][0]
    lone = np.zeros(l * l, dtype=bool)
    lone[at(6, 6)] = True                                         # damage, none of it joined to the flaw
    d, r, j, jump = read_records(flat, lone[None, :], anchors, 2)
    assert (d[0], j[0], bool(jump[0])) == (1, -1, True)
    none = read_records(flat, np.zeros((1, l * l), dtype=bool), anchors, 2)
    assert none[0][0] == 0 and none[1][0] == -1 and not none[3][0]


def test_unreachable_damage_counts_far_and_jumps():
    a, _ = torus(6, cap=NO_CAP)
    two = np.concatenate([a, a + 36])
    mask = np.zeros(72, dtype=bool)
    mask[[0, 1, 50]] = True
    _, radius, joined, jump = read_records(two, mask[None, :], np.array([0, 1, 6, 7]), 2)
    assert radius[0] == 72 and joined[0] == 0 and jump[0]


def test_fair_schedule_carries_the_remainder():
    sched, fair = fair_sweep_schedule(512, 2000)
    assert sched[0] == 0 and len(sched) == 2001 and fair[-1] == 2000
    assert sched.sum() == (2 * 512 * 512 * 2000) // 96          # 5461.33 a fair sweep, nothing lost to rounding
    assert set(sched[1:].tolist()) == {5461, 5462}
    sched, fair = fair_sweep_schedule(1024, 10, record_every=5)
    assert fair.tolist() == [0, 5, 10] and sched.sum() == (2 * 1024 * 1024 * 10) // 96


# ---------------------------------------------------------------- the flaw on any graph

@pytest.mark.parametrize("l, point", [(8, 0), (12, 77), (16, 255)])
def test_flaw_at_on_a_flat_torus(l, point):
    flat, _ = torus(l, cap=NO_CAP)
    cands = flaw_candidates(flat, point)
    assert cands == flaw_candidates(flat, point) and cands and all(point in c for c in cands)
    flawed, four, dh = flaw_at(flat, point)
    assert dh == 32.0
    assert four == cands[0] or four in cands
    assert is_valid(flawed, NO_CAP) and point in four
    assert hamiltonian(flawed, 1.25) - hamiltonian(flat, 1.25) == 32.0
    assert set(np.flatnonzero(damage_mask(flat, flawed))) == set(four)
    with pytest.raises(ValueError):
        flaw_at(flat, point, cost=31.0)
    assert flaw_at(flat, point, cost=None)[2] == 32.0           # the first valid candidate, whatever it costs


def test_flaw_at_on_a_saved_kind_of_state():
    """A flaw beside a flaw: the general builder still finds a valid switch costing exactly 32 where it can."""
    flat, _ = torus(16, cap=NO_CAP)
    once, _, _ = flaw_at(flat, 0)
    twice, four, _ = flaw_at(once, 8 * 16 + 8)
    assert is_valid(twice, NO_CAP) and hamiltonian(twice, 1.25) - hamiltonian(once, 1.25) == 32.0


# ---------------------------------------------------------------- the flaw of Amendment 3: the cheapest switch at a point

def nx_energy(g, lam):
    """H = 16 (N - S) + 4 lam X from networkx alone."""
    on_edge, n_squares = {}, 0
    for c in nx.simple_cycles(g, length_bound=4):
        if len(c) == 4:
            n_squares += 1
            for i in range(4):
                e = frozenset((c[i], c[(i + 1) % 4]))
                on_edge[e] = on_edge.get(e, 0) + 1
    return 16 * (g.number_of_nodes() - n_squares) + 4 * lam * sum(max(0, k - 2) for k in on_edge.values())


def nx_switched(g, p, q, x, y):
    h = g.copy()
    h.remove_edges_from([(p, q), (x, y)])
    h.add_edges_from([(p, y), (x, q)])
    return h


def nx_valid(h):
    """Simple and 4-regular, and no two points with more than two common neighbours (the hard-core rule, Q2)."""
    if any(d != 4 for _, d in h.degree()):
        return False
    nodes = list(h.nodes)
    nb = {v: set(h[v]) for v in nodes}
    return all(len(nb[a] & nb[b]) <= 2 for i, a in enumerate(nodes) for b in nodes[i + 1:])


def brute_cheapest(adj, point, lam=1.25, reach=3, avoid=(), above=None):
    """Every switch of an edge at `point` with an edge whose nearer end is within `reach` steps, touching none of
    `avoid`, from networkx alone: returns the cheapest cost and the set of switches attaining it."""
    g = to_nx(adj)
    dist = nx.single_source_shortest_path_length(g, point)
    base = nx_energy(g, lam)
    costs = {}
    for q in g[point]:
        for x, y in g.edges:
            for xx, yy in ((x, y), (y, x)):
                if min(dist.get(xx, 99), dist.get(yy, 99)) > reach or dist.get(xx, 1) % 2:
                    continue                                    # xx must sit on point's side
                if xx == point or yy == q or g.has_edge(point, yy) or g.has_edge(xx, q):
                    continue
                if set(avoid) & {point, q, xx, yy}:
                    continue
                h = nx_switched(g, point, q, xx, yy)
                if nx_valid(h):
                    c = nx_energy(h, lam) - base
                    if above is None or c > above + 1e-9:
                        costs[(point, q, xx, yy)] = c
    best = min(costs.values())
    return best, {k for k, c in costs.items() if abs(c - best) < 1e-9}


def product_like():
    """A small stand-in for a burp's product: a 4 x 10 tube (one direction curled, 40 points) after a few random
    moves, so that it is neither flat nor regular."""
    adj, part = torus(4, 10, cap=NO_CAP)
    run_chain(adj, np.flatnonzero(part == 0), 0.3, 2, 1, 17, 1.25, NO_CAP, False)
    return adj


@pytest.mark.parametrize("make_graph, points", [(lambda: torus(8, cap=NO_CAP)[0], [0, 27]),
                                                (lambda: torus(6, 8, cap=NO_CAP)[0], [5]),
                                                (product_like, [0, 13, 22])])
def test_cheapest_switch_matches_brute_force(make_graph, points):
    adj = make_graph()
    assert is_valid(adj, NO_CAP)
    for point in points:
        switched, four, cost = cheapest_switch_at(adj, point)
        best, argmin = brute_cheapest(adj, point)
        assert cost == pytest.approx(best) and four in argmin
        assert four == min(argmin, key=lambda s: sorted([tuple(sorted(s[:2])), tuple(sorted(s[2:]))]))  # the tie rule
        assert is_valid(switched, NO_CAP) and point in four
        assert nx_energy(to_nx(switched), 1.25) - nx_energy(to_nx(adj), 1.25) == pytest.approx(cost)


def test_cheapest_switch_avoiding_a_knot_matches_brute_force():
    """Amendment 4: beside a knot the flaw is the cheapest switch touching no knot point. A made-up knot (a flaw in a
    10 x 10 torus); every start 1 or 2 steps from it; checked against networkx with the same exclusion."""
    flat, _ = torus(10, cap=NO_CAP)
    knotted, _ = flaw_at(flat, 0)[:2]
    knots = run_t75.knot_points(knotted)
    to_knot = graph_distances(knotted, knots)
    starts = np.flatnonzero((to_knot >= 1) & (to_knot <= 2))
    assert starts.size
    cheaper_if_touching = 0
    for point in starts[:6]:
        point = int(point)
        _, four, cost = cheapest_switch_at(knotted, point, avoid=knots)
        best, argmin = brute_cheapest(knotted, point, avoid=knots.tolist())
        assert not set(four) & set(knots.tolist())
        assert cost == pytest.approx(best) and four in argmin
        assert four == min(argmin, key=lambda s: sorted([tuple(sorted(s[:2])), tuple(sorted(s[2:]))]))
        cheaper_if_touching += cheapest_switch_at(knotted, point)[2] < cost
    assert cheaper_if_touching > 0                                # so the exclusion is doing something here


def test_a_cost_free_switch_is_skipped_for_the_cheapest_positive_one():
    """Amendment 5: next to a made-up knot some switches cost nothing or less (moving or healing the knot). Where the
    plain cheapest switch costs 0 or less, the flaw taken is the cheapest costing more than zero, checked against
    networkx with the same rule."""
    flat, _ = torus(10, cap=NO_CAP)
    knotted, _ = flaw_at(flat, 0)[:2]
    seen = 0
    for point in range(100):
        if cheapest_switch_at(knotted, point)[2] > 0:
            continue
        _, four, cost = cheapest_switch_at(knotted, point, above=0.0)
        best, argmin = brute_cheapest(knotted, point, above=0.0)
        assert cost > 0 and cost == pytest.approx(best) and four in argmin
        seen += 1
        if seen == 3:
            break
    assert seen > 0                                               # so the test met a cost-free or downhill switch


def test_the_runner_never_takes_a_cost_free_flaw(tmp_path):
    flat, part = torus(16, cap=NO_CAP)
    knotted, _ = flaw_at(flat, 5)[:2]
    for placement in ("beside", "far"):
        got = run_t75.place_flaw(knotted, run_t75.knot_points(knotted), placement, 3, 1.25, 3)
        assert got is not None and got[2] > 0


def test_cheapest_switch_on_a_flat_torus_costs_32():
    flat, _ = torus(10, cap=NO_CAP)
    assert cheapest_switch_at(flat, 0)[2] == 32.0                # O22: every cheapest move out of flat space


# ---------------------------------------------------------------- the verdict words, on synthetic records

def records(r_list, damaged=None, jumps=()):
    damaged = damaged or [4] * len(r_list)
    return [dict(fair_sweep=t, R=r, damaged=d, jump=t in jumps) for t, (r, d) in enumerate(zip(r_list, damaged))]


D = 40                                                            # D/4 = 10


def test_replica_flags():
    healed = analyse.replica_reading(records([0, 1, -1], [4, 6, 0]), D)
    assert healed["healed"] and not healed["confined"] and not healed["reached"]
    stays = analyse.replica_reading(records([0, 2, 3, 3]), D)
    assert stays["confined"] and not stays["healed"]
    spread = analyse.replica_reading(records([0, 1, 3, 5, 7, 9, 11, 30], [4, 6, 10, 20, 30, 40, 50, 60]), D)
    assert spread["reached"] and not spread["travels"] and spread["t_reach"] == 6
    assert spread["speed"] == pytest.approx(2.0)                  # R = 3, 5, 7, 9, 11 at t = 2..6
    moves = analyse.replica_reading(records([0, 3, 6, 9, 12], [4, 8, 8, 6, 8]), D)
    assert moves["travels"] and moves["reached"] and moves["speed"] == pytest.approx(3.0)
    late_jump = analyse.replica_reading(records([0, 3, 6, 9, 12, 13], jumps={5}), D)
    early_jump = analyse.replica_reading(records([0, 3, 6, 9, 12, 13], jumps={4}), D)
    assert not late_jump["jumped"] and early_jump["jumped"]       # at or before t_reach only
    leap = analyse.replica_reading(records([0, 3, 20]), D)
    assert leap["reached"] and leap["speed"] == pytest.approx(17.0)
    one = analyse.replica_reading(records([0, 0, 20]), D)
    assert one["reached"] and np.isnan(one["speed"])              # the stretch from R = 3 is a single record


def cell(kind, n_kind, other="confined", n=32, speed=2.0):
    make_one = {"healed": records([0, -1], [4, 0]), "confined": records([0, 2]),
                "spreads": records([0, 3, 3 + speed, 3 + 2 * speed, 3 + 3 * speed, 3 + 4 * speed],
                                   [4, 20, 20, 20, 20, 20]),
                "travels": records([0, 3, 3 + speed, 3 + 2 * speed, 3 + 3 * speed, 3 + 4 * speed]),
                "jumps": records([0, 2], jumps={1}), "mixed": records([0, 5])}
    reads = [analyse.replica_reading(make_one[kind], D) for _ in range(n_kind)]
    reads += [analyse.replica_reading(make_one[other], D) for _ in range(n - n_kind)]
    return analyse.cell_word(reads)


def test_cell_words():
    assert cell("healed", 24)["word"] == "HEALS" and cell("healed", 23)["word"] == "MIXED"
    assert cell("confined", 32)["word"] == "CONFINED"
    assert cell("spreads", 24, "mixed")["word"] == "SPREADS"
    assert cell("travels", 24, "mixed")["word"] == "TRAVELS"
    j = cell("jumps", 8, "mixed")
    assert j["word"] == "JUMPS" and j["jumps"]
    assert cell("jumps", 7, "mixed")["word"] == "MIXED"
    s = cell("spreads", 30, "mixed", speed=2.5)
    assert s["speed"] == pytest.approx(2.5) and s["n_speed"] == 30


def test_signal_words():
    assert analyse.signal_word(cell("spreads", 32, speed=2.0), cell("travels", 32, speed=2.3)) == "ONE SPEED"
    assert analyse.signal_word(cell("spreads", 32, speed=2.0), cell("spreads", 32, speed=3.0)) == "SEVERAL SPEEDS"
    assert analyse.signal_word(cell("healed", 32), cell("confined", 32)) == "NO SIGNAL"
    assert analyse.signal_word(cell("spreads", 32), cell("confined", 32)).startswith("UNSCORED")


def test_one_speed_compares_the_single_far_flaw_with_the_pair():
    """Amendment 7: the cross word reads 'far' against 'pair'; 'beside' is reported as a cell and is not in it."""
    def rows(placement, r_list, damaged):
        return [dict(N=512, g=1.5, placement=placement, replica=k, D=D, fair_sweep=t, damaged=d, R=r, jump=False,
                     flaw_cost=20.0) for k in range(32) for t, (r, d) in enumerate(zip(r_list, damaged))]
    spread = ([0, 3, 5, 7, 9, 11], [4, 20, 20, 20, 20, 20])
    faster = ([0, 3, 5.4, 7.8, 10.2, 12.6], [8, 20, 20, 20, 20, 20])
    healed = ([0, -1], [4, 0])
    cells, signal = analyse.verdicts(rows("far", *spread) + rows("pair", *faster) + rows("beside", *healed))
    assert cells[(512, 1.5, "beside")]["word"] == "HEALS" and signal == {(512, 1.5): "ONE SPEED"}
    cells, signal = analyse.verdicts(rows("far", *spread) + rows("pair", *healed) + rows("beside", *spread))
    assert signal[(512, 1.5)].startswith("UNSCORED")              # beside spreading does not stand in for the pair
    cells, signal = analyse.verdicts(rows("far", *spread) + rows("beside", *spread))
    assert signal == {}                                           # no pair cell, no cross word


# ---------------------------------------------------------------- the runner, on made-up saved states

def fake_states(tmp_path, run="fake", l=16, n_states=3):
    """Saved states in run_scrap_freeze's format: a flat torus with a 'knot' (a flaw, whose points are not at d = 2)
    in states 0 and 2, and a perfectly flat one as state 1 (no knot, so no 'beside' placement)."""
    flat, part = torus(l, cap=NO_CAP)
    d = tmp_path / (run + "_00_adj")
    d.mkdir()
    for rep in range(n_states):
        adj = flat if rep == 1 else flaw_at(flat, (3 * rep) * l + 5)[0]
        np.savez(d / ("N%d_rep%d_tc0.npz" % (l * l, rep)), adj=adj, part=part, lam=1.25, t_cool=0, replica=rep)
    return tmp_path


def fake_config(tmp_path, placement, replicas=2, name=None, fair_sweeps=4):
    cfg = {"name": name or "t75_fake_%s" % placement, "section": "T75", "states_dir": str(tmp_path), "states": "fake",
           "N": 256, "lambda": 1.25, "g": 1.5, "placement": placement, "replicas": replicas,
           "fair_sweeps": fair_sweeps, "record_every": 1, "n_ref": 96, "step": 2, "flaw_reach": 3, "seed": 99}
    path = tmp_path / (cfg["name"] + ".json")
    path.write_text(json.dumps(cfg))
    return path


def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def test_load_states_in_replica_order(tmp_path):
    states = run_t75.load_states(fake_states(tmp_path), "fake")
    assert [s[3] for s in states] == [0, 1, 2] and all(s[1].shape == (256, 4) for s in states)
    assert run_t75.knot_points(states[1][1]).size == 0 and run_t75.knot_points(states[0][1]).size > 0


def independent_cost(state_path, flaw, lam=1.25):
    z = np.load(state_path)
    g = to_nx(np.asarray(z["adj"], dtype=np.int64))
    p, q, x, y = flaw
    return nx_energy(nx_switched(g, p, q, x, y), lam) - nx_energy(g, lam)


def saved_states(tmp_path, reps=(1, 4, 5)):
    """Three of T51's own saved end states at N = 512 (results/t51_l128_10k_00_adj), copied in as the run 'fake':
    replicas 1 and 5 carry knots (8 and 12 knot points), replica 4 has none. A made-up knot will not do here: next to
    a flaw written into a flat torus, the cheapest switch at every point 1 or 2 steps away touches the flaw itself, so
    nothing is ever 'beside' it (which the beside rule of Amendment 3 excludes, and the real knots allow)."""
    import shutil
    src = ROOT / "results" / "t51_l128_10k_00_adj"
    d = tmp_path / "fake_00_adj"
    d.mkdir()
    for rep in reps:
        shutil.copy(src / ("N512_rep%d_tc10000.npz" % rep), d)
    return tmp_path


def test_runner_beside_excludes_touching_and_records_the_cost(tmp_path):
    saved_states(tmp_path)
    cfg_path = fake_config(tmp_path, "beside")
    cfg = json.loads(cfg_path.read_text())
    cfg["N"] = 512
    cfg_path.write_text(json.dumps(cfg))
    out = tmp_path / "out"
    run_t75.main(cfg_path, out)
    rows = read_csv(out / "t75_fake_beside.csv")
    meta = json.loads((out / "t75_fake_beside.meta.json").read_text())
    assert len(rows) == 2 * 5 and {r["state_replica"] for r in rows} == {"1", "5"}   # replica 4 has no knot
    first = [r for r in rows if r["fair_sweep"] == "0"]
    assert all(r["damaged"] == "4" and r["R"] == "0" and r["attempts"] == "0" and r["use"] == "0" for r in first)
    assert all(r["knot_distance"] in ("1", "2") for r in rows)  # Amendment 3, item 2: touching the knot is excluded
    for r in first:                                              # Amendment 4: the flaw touches no knot point
        adj = [s[1] for s in run_t75.load_states(tmp_path, "fake") if s[0].name == r["state"]][0]
        assert not set(int(v) for v in r["flaw"].split()) & set(run_t75.knot_points(adj).tolist())
    states = run_t75.load_states(tmp_path, "fake")
    for r in first:                                              # the recorded cost, recomputed from networkx alone
        flaw = [int(v) for v in r["flaw"].split()]
        assert float(r["flaw_cost"]) == pytest.approx(independent_cost(tmp_path / "fake_00_adj" / r["state"], flaw))
        adj = [s[1] for s in states if s[0].name == r["state"]][0]
        to_knot = graph_distances(adj, run_t75.knot_points(adj))
        assert min(to_knot[v] for v in flaw) == int(r["knot_distance"])
    assert int(rows[4]["attempts"]) == (2 * 512 * 512 * 4) // 96
    assert [p["replica"] for p in meta["replicas_plan"]] == [0, 1] and meta["skipped_disconnected"] == []
    with pytest.raises(FileExistsError):                          # results are append-only
        run_t75.main(cfg_path, out)


def test_beside_never_takes_a_flaw_that_touches_the_knot():
    """On a made-up knot every nearby cheapest switch touches it (which Amendment 3 excluded, leaving nothing); under
    Amendment 4 the cheapest switch touching no knot point is taken instead, and it is beside the knot."""
    flat, _ = torus(16, cap=NO_CAP)
    knotted, _ = flaw_at(flat, 5)[:2]
    knots = run_t75.knot_points(knotted)
    to_knot = graph_distances(knotted, knots)
    touching = [cheapest_switch_at(knotted, int(p))[1] for p in np.flatnonzero((to_knot >= 1) & (to_knot <= 2))]
    assert touching and all(min(to_knot[v] for v in four) == 0 for four in touching)
    got = run_t75.place_flaw(knotted, knots, "beside", 1, 1.25, 3)
    assert got is not None
    flawed, four, cost, near, parts = got
    assert not set(four) & set(knots.tolist()) and near in (1, 2) and is_valid(flawed, NO_CAP) and parts == [cost]


def test_runner_far_and_same_seed_same_rows(tmp_path):
    fake_states(tmp_path)
    run_t75.main(fake_config(tmp_path, "far", replicas=3, name="t75_a"), tmp_path / "a")
    run_t75.main(fake_config(tmp_path, "far", replicas=3, name="t75_b"), tmp_path / "b")
    a, b = read_csv(tmp_path / "a" / "t75_a.csv"), read_csv(tmp_path / "b" / "t75_b.csv")
    assert a == b and {r["state_replica"] for r in a} == {"0", "1", "2"}
    assert all(int(r["knot_distance"]) >= 6 or r["knot_distance"] == "-1" for r in a)
    cells, signal = analyse.verdicts(analyse.read_rows([tmp_path / "a" / "t75_a.csv"]))
    assert list(cells) == [(256, 1.5, "far")] and cells[(256, 1.5, "far")]["replicas"] == 3 and signal == {}
    assert cells[(256, 1.5, "far")]["cost_min"] > 0


def test_reuse_fills_32_with_different_placements_and_skips_a_disconnected_state(tmp_path):
    fake_states(tmp_path)
    a, pa = torus(8, cap=NO_CAP)                                 # a fourth state of 256 points, in four pieces
    knot, _, _ = cheapest_switch_at(a, 0)
    pieces = np.concatenate([knot, a + 64, a + 128, a + 192])
    np.savez(tmp_path / "fake_00_adj" / "N256_rep3_tc0.npz", adj=pieces, part=np.concatenate([pa] * 4), lam=1.25,
             t_cool=0, replica=3)
    run_t75.main(fake_config(tmp_path, "far", replicas=32, name="t75_reuse", fair_sweeps=1), tmp_path / "out")
    rows = read_csv(tmp_path / "out" / "t75_reuse.csv")
    meta = json.loads((tmp_path / "out" / "t75_reuse.meta.json").read_text())
    first = [r for r in rows if r["fair_sweep"] == "0"]
    assert len(first) == 32 and len(meta["replicas_plan"]) == 32
    assert {r["state_replica"] for r in first} == {"0", "1", "2"}   # 3 is in pieces and skipped
    assert meta["skipped_disconnected"] == [str(tmp_path / "fake_00_adj" / "N256_rep3_tc0.npz")]
    assert [r["state_replica"] for r in first[:6]] == ["0", "1", "2", "0", "1", "2"]
    assert [r["use"] for r in first[:6]] == ["0", "0", "0", "1", "1", "1"] and first[-1]["use"] == "10"
    assert [p["use"] for p in meta["replicas_plan"]] == [int(r["use"]) for r in first]
    placements = {(r["state"], frozenset(r["flaw"].split())) for r in first}
    assert len(placements) == 32                                  # every reuse a different placement


def knotted_torus(l=24, at=5):
    flat, part = torus(l, cap=NO_CAP)
    return flaw_at(flat, at)[0], part


def test_the_pair_is_two_positive_far_flaws_four_steps_apart():
    adj, _ = knotted_torus()
    knots = run_t75.knot_points(adj)
    to_knot = graph_distances(adj, knots)
    for seed in (1, 2, 3):
        twice, eight, cost, near, (c1, c2) = run_t75.place_pair(adj, knots, seed, 1.25, 3)
        first, second = eight[:4], eight[4:]
        assert len(set(eight)) == 8 and is_valid(twice, NO_CAP)
        assert min(graph_distances(adj, np.array(first))[list(second)]) == 4   # nearest points 4 apart
        assert min(to_knot[list(eight)]) >= 6 and near == min(to_knot[list(eight)])
        assert c1 > 0 and c2 > 0 and cost == pytest.approx(c1 + c2)
        g0 = to_nx(adj)
        g1 = nx_switched(g0, *first)
        g2 = nx_switched(g1, *second)
        assert nx_energy(g1, 1.25) - nx_energy(g0, 1.25) == pytest.approx(c1)
        assert nx_energy(g2, 1.25) - nx_energy(g1, 1.25) == pytest.approx(c2)   # priced on the once-flawed state
        assert set(np.flatnonzero(damage_mask(adj, twice))) == set(eight)


def test_runner_pair_rows_carry_eight_points_and_both_costs(tmp_path):
    d = tmp_path / "fake_00_adj"
    d.mkdir()
    flat, part = torus(24, cap=NO_CAP)
    for rep, adj in enumerate([knotted_torus()[0], flat]):
        np.savez(d / ("N576_rep%d_tc0.npz" % rep), adj=adj, part=part, lam=1.25, t_cool=0, replica=rep)
    cfg_path = fake_config(tmp_path, "pair", replicas=3, fair_sweeps=1)
    cfg = json.loads(cfg_path.read_text())
    cfg["N"] = 576
    cfg_path.write_text(json.dumps(cfg))
    run_t75.main(cfg_path, tmp_path / "out")
    rows = read_csv(tmp_path / "out" / "t75_fake_pair.csv")
    first = [r for r in rows if r["fair_sweep"] == "0"]
    assert len(first) == 3 and [r["state_replica"] for r in first] == ["0", "1", "0"]
    for r in first:
        assert len(r["flaw"].split()) == 8 and r["damaged"] == "8" and r["R"] == "0"
        assert float(r["flaw_cost"]) == pytest.approx(float(r["flaw_cost_1"]) + float(r["flaw_cost_2"]))
        assert float(r["flaw_cost_1"]) > 0 and float(r["flaw_cost_2"]) > 0


def test_runner_refuses_when_a_pass_adds_nothing(tmp_path):
    flat, part = torus(16, cap=NO_CAP)
    d = tmp_path / "fake_00_adj"
    d.mkdir()
    np.savez(d / "N256_rep0_tc0.npz", adj=flat, part=part, lam=1.25, t_cool=0, replica=0)
    with pytest.raises(ValueError):                               # a flat torus has no knot, so nothing is beside one
        run_t75.main(fake_config(tmp_path, "beside"), tmp_path / "out")
    assert not (tmp_path / "out").exists()                        # refused before anything was written


# ---------------------------------------------------------------- the config writer

def test_make_configs_needs_three_couplings_and_never_rewrites(tmp_path):
    with pytest.raises(ValueError):
        make.main([1.0, 1.5], tmp_path)
    paths = make.main([1.0, 1.5, 2.25], tmp_path)
    assert len(paths) == 18 and len({p.name for p in paths}) == 18
    cfgs = [json.loads(p.read_text()) for p in paths]
    assert {c["N"] for c in cfgs} == {512, 1024} and {c["placement"] for c in cfgs} == {"beside", "far", "pair"}
    assert {c["states"] for c in cfgs} == {"t51_l128_10k", "t51_l256_300k"}
    assert all(c["replicas"] == 32 and c["fair_sweeps"] == 2000 and c["record_every"] == 1 for c in cfgs)
    assert all(c["flaw_reach"] == 3 and c["lambda"] == 1.25 for c in cfgs)
    assert len({c["seed"] for c in cfgs}) == 18 and "t75_n512_far_g2p25" in {c["name"] for c in cfgs}
    assert "t75_n1024_pair_g1" in {c["name"] for c in cfgs}
    before = paths[0].read_text()
    make.main([1.0, 1.5, 2.25], tmp_path)
    assert paths[0].read_text() == before
