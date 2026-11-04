"""T51's runner, reading rules, configs and queue file, tested before any run (PREREGISTRATION T51 with its
Amendments 1 and 2).

The runs here are test-sized and loosened: a 4 x 24 tube (N = 96, where a fair sweep is exactly one sweep), opened
warm at g = 1.75 so that it opens within a few hundred sweeps, into pytest's temporary folders.
"""
import csv
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from graphity.cqg import NO_CAP, run_chain, torus
from graphity.dimension import local_dimension

ROOT = Path(__file__).resolve().parents[1]
LAM, G_WARM, SEED, N = 1.25, 1.75, 3, 96


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


run = _load("run_scrap_freeze")
a = _load("analyse_t51")
race = _load("run_scrap_race")
t37 = _load("analyse_t37")


def _cfg(name, t_cool, **over):
    cfg = dict(name=name, section="T51", side=[24, 4], g_hot=G_WARM, g_cold=0.25, t_cool=t_cool, hold=100, block=50,
               n_ref=96, open_check_every=50, open_stop_flat=0.90, open_max_d1_piece=8, open_cap=6000, seed=SEED,
               replicas=2, replica_ids=[0, 1], save_adjacency=True)
    cfg["lambda"] = LAM
    cfg.update(over)
    return cfg


def _run(cfg, out_dir):
    path = Path(out_dir) / (cfg["name"] + ".json")
    path.write_text(json.dumps(cfg))
    run.main(str(path), str(out_dir))
    return list(csv.DictReader(open(Path(out_dir) / (cfg["name"] + ".csv"), newline="")))


def _open(rep, look=run.has_opened, cap=6000, max_d1_piece=8, stop_flat=0.90):
    return run.open_tube(24, LAM, G_WARM, run.stream_seed(SEED, N, rep), 50, stop_flat, max_d1_piece, cap, look)


def _graph(adj):
    g = nx.Graph()
    g.add_nodes_from(range(adj.shape[0]))
    g.add_edges_from((u, int(v)) for u in range(adj.shape[0]) for v in adj[u] if v >= 0)
    return g


def _largest_d1_by_the_definition(adj):
    """The largest connected piece of points at d = 1, straight from the amendment's words, with networkx."""
    d = local_dimension(adj)
    pieces = nx.connected_components(_graph(adj).subgraph([v for v in range(adj.shape[0]) if d[v] == 1]))
    return max((len(piece) for piece in pieces), default=0)


def _triple(r):
    """(columns, other pieces, energy) of a row of the file or of a reading."""
    return int(r["columns"]), int(r["others"]), float(r["h"])


def _last_row(rows, rep, t_cool):
    """The row at the end of the hold of one replica at one cooling time."""
    return [r for r in rows if r["final"] == "True" and r["replica"] == str(rep) and r["t_cool"] == str(t_cool)][0]


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    """One small run of the runner (a quench and a 200-sweep cooling, two replicas), and three opened sheets: the
    two replicas of that run and replica 2, which opens with a resting piece of eight points at d = 1."""
    out = tmp_path_factory.mktemp("t51")
    cfg = _cfg("t51_smoke", [0, 200])
    return dict(out=out, cfg=cfg, rows=_run(cfg, out), sheets={rep: _open(rep) for rep in (0, 1, 2)})


# ---- the runner -------------------------------------------------------------------------------------------------

def test_rows_follow_the_protocol(small):
    for rep in (0, 1):
        mine = [r for r in small["rows"] if r["replica"] == str(rep)]
        assert [r["phase"] for r in mine] == ["opening"] + ["hold"] * 2 + ["cooling"] * 4 + ["hold"] * 2
        assert [r["t_cool"] for r in mine] == [""] + ["0"] * 2 + ["200"] * 6
        assert [r["block"] for r in mine] == ["0", "1", "2", "1", "2", "3", "4", "5", "6"]
        assert [r["final"] for r in mine] == ["False", "False", "True"] + ["False"] * 5 + ["True"]
        assert all(r["opened"] == "True" and r["N"] == "96" and r["L"] == "24" for r in mine)
        first = mine[0]
        assert float(first["flat_share"]) >= 0.90 and int(first["open_sweeps"]) % 50 == 0
        assert int(first["largest_d1"]) <= 8
        assert float(first["g"]) == G_WARM
        assert len(set(r["open_sweeps"] for r in mine)) == 1
        cooled = mine[3:]
        assert [float(r["g"]) for r in cooled] == run.schedule(G_WARM, 0.25, 200, 100, 50)
        assert [r["fair_sweep"] for r in cooled] == ["50", "100", "150", "200", "250", "300"]
        assert [r["chain_sweep"] for r in cooled] == [r["fair_sweep"] for r in cooled]     # N = 96: the same clock
        assert [float(r["g"]) for r in mine[1:3]] == [0.25, 0.25]


def test_same_config_and_seed_give_the_same_file_byte_for_byte(small, tmp_path):
    _run(small["cfg"], tmp_path)
    assert (tmp_path / "t51_smoke.csv").read_bytes() == (small["out"] / "t51_smoke.csv").read_bytes()
    saved = sorted((small["out"] / "t51_smoke_adj").glob("*.npz"))
    assert [f.name for f in saved] == ["N96_rep%d_tc%d.npz" % (rep, t) for rep in (0, 1) for t in (0, 200)]
    for f in saved:
        assert (np.load(f)["adj"] == np.load(tmp_path / "t51_smoke_adj" / f.name)["adj"]).all()


def test_every_cooling_time_starts_from_the_same_opened_sheet(small, tmp_path):
    """The paired design: the opening knows nothing of the cooling time, in one config or across two."""
    alone = _run(_cfg("t51_smoke_alone", [200]), tmp_path)

    def pick(rows, phase_is_opening):
        return [r for r in rows if (r["phase"] == "opening") == phase_is_opening and r["t_cool"] in ("", "200")]
    assert pick(alone, True) == pick(small["rows"], True)          # the same opened sheet, read the same
    assert pick(alone, False) == pick(small["rows"], False)        # and a cooling does not depend on its companions
    for rep in (0, 1):
        sheet, part, sweeps, opened = small["sheets"][rep]
        first = [r for r in small["rows"] if r["phase"] == "opening" and r["replica"] == str(rep)][0]
        assert opened and int(first["open_sweeps"]) == sweeps and _triple(first) == _triple(run.reading(sheet, LAM))
        for t_cool in (0, 200):                                    # replay each cooling from that one sheet
            adj = sheet.copy()
            gs = run.schedule(G_WARM, 0.25, t_cool, 100, 50)
            last = list(run.cool(adj, part, gs, 50, run.stream_seed(SEED, N, rep, t_cool), LAM))[-1][2]
            kept = np.load(small["out"] / "t51_smoke_adj" / ("N96_rep%d_tc%d.npz" % (rep, t_cool)))
            assert (kept["adj"] == adj).all() and (kept["part"] == part).all()
            assert (int(kept["t_cool"]), int(kept["replica"]), float(kept["lam"])) == (t_cool, rep, LAM)
            assert _triple(_last_row(small["rows"], rep, t_cool)) == _triple(last)
    again = _open(0)
    assert (again[0] == small["sheets"][0][0]).all() and again[2] == small["sheets"][0][2]


def test_streams_are_separate_and_the_opening_stream_ignores_the_cooling_time():
    s = run.stream_seed
    assert s(SEED, N, 0) == s(SEED, N, 0) and 0 <= s(SEED, N, 0) < 2 ** 32
    different = [s(SEED, N, 0), s(SEED, N, 1), s(SEED, 192, 0), s(SEED + 1, N, 0),
                 s(SEED, N, 0, 0), s(SEED, N, 0, 1000), s(SEED, N, 1, 0), s(SEED, N, 1, 1000)]
    assert len(set(different)) == len(different)                   # the quench, t_cool = 0, has its own stream too


def test_looking_does_not_disturb_the_chain(small):
    def look_hard(adj, stop_flat, max_d1_piece):                   # the whole reading first, then the stop rule
        run.reading(adj, LAM)
        return run.has_opened(adj, stop_flat, max_d1_piece)
    looked, part, sweeps, opened = _open(0, look=look_hard)
    assert opened and (looked == small["sheets"][0][0]).all()
    blind, _, blind_sweeps, blind_opened = _open(0, look=lambda adj, flat, piece: False, cap=sweeps)   # never looks
    assert blind_sweeps == sweeps and not blind_opened and (blind == looked).all()
    straight, part2 = torus(24, 4, NO_CAP)                         # nor does pausing: one call of the whole length
    side_u = np.flatnonzero(part2 == 0)
    run_chain(straight, side_u, 1.0 / G_WARM, 0, sweeps, run.stream_seed(SEED, N, 0), LAM, NO_CAP, False)
    assert (straight == looked).all()
    gs = run.schedule(G_WARM, 0.25, 200, 100, 50)
    read_all, read_none = looked.copy(), looked.copy()
    readings = list(run.cool(read_all, part, gs, 50, 11, LAM))
    list(run.cool(read_none, part, gs, 50, 11, LAM, read=lambda adj, lam: None))
    assert (read_all == read_none).all() and len(readings) == 6 and readings[-1][2] == run.reading(read_all, LAM)


def test_schedule_in_fair_sweeps():
    gs = run.schedule(1.25, 0.25, 1000, 5000, 250)
    assert len(gs) == 4 + 20 and run.cooling_blocks(1000, 250) == 4
    assert abs(gs[0] - 1.25 * 0.2 ** 0.25) < 1e-12 and abs(gs[3] - 0.25) < 1e-12 and gs[4:] == [0.25] * 20
    ratios = [gs[i + 1] / gs[i] for i in range(3)]
    assert max(ratios) - min(ratios) < 1e-12                       # the same factor each block
    assert run.schedule(1.25, 0.25, 0, 5000, 250) == [0.25] * 20   # the quench: straight to the hold
    assert run.cooling_blocks(0, 250) == 0
    assert len(run.schedule(1.25, 0.25, 300000, 5000, 250)) == 1200 + 20
    for t_cool in (300, 1000, 3000, 10000, 30000, 100000, 300000):  # T25's schedule whenever there is a cooling
        assert run.schedule(1.25, 0.25, t_cool, 5000, 250) == race.schedule(1.25, 0.25, t_cool, 5000, 250)


def test_the_fair_clock():
    assert run.chain_sweeps(1, 96, 96) == 1 and run.chain_sweeps(250, 96, 96) == 250      # T19's and T25's size
    assert run.chain_sweeps(250, 512, 96) == 1333                  # 1,333.33 rounded
    assert run.chain_sweeps(250, 1024, 96) == 2667                 # 2,666.67 rounded
    assert run.chain_sweeps(250, 2048, 96) == 5333                 # 5,333.33 rounded


def test_the_runner_keeps_the_fair_clock_at_a_larger_size(tmp_path):
    """At N = 192 a fair sweep is two sweeps of the chain, and the runner runs them, not only labels them. Kept warm
    (g from 4 to 2) so that the chain moves; both halves of the stop rule are switched off (no flat point asked for,
    a piece at d = 1 of any size allowed) so that the tube "opens" at its first reading."""
    cfg = _cfg("t51_smoke_192", [100], side=[48, 4], g_hot=4.0, g_cold=2.0, hold=50, open_check_every=10,
               open_stop_flat=0.0, open_max_d1_piece=192, replica_ids=[0])
    rows = _run(cfg, tmp_path)
    assert [r["phase"] for r in rows] == ["opening", "cooling", "cooling", "hold"] and rows[0]["open_sweeps"] == "10"
    assert [(r["fair_sweep"], r["chain_sweep"]) for r in rows[1:]] == [("50", "100"), ("100", "200"), ("150", "300")]
    kept = np.load(tmp_path / "t51_smoke_192_adj" / "N192_rep0_tc100.npz")["adj"]
    gs = run.schedule(4.0, 2.0, 100, 50, 50)
    replay = {}
    for block_sweeps in (100, 50):
        adj, part, _, _ = run.open_tube(48, LAM, 4.0, run.stream_seed(SEED, 192, 0), 10, 0.0, 192, 6000)
        list(run.cool(adj, part, gs, block_sweeps, run.stream_seed(SEED, 192, 0, 100), LAM))
        replay[block_sweeps] = adj
    assert (kept == replay[100]).all() and not (kept == replay[50]).all()


def test_at_96_points_the_cooling_is_t25s_loop():
    """T25's own loop (scripts/run_scrap_race.main), written out, against `cool` from the same graph and seed."""
    gs = race.schedule(G_WARM, 0.25, 200, 100, 50)
    theirs, part = torus(24, 4, NO_CAP)
    side_u = np.flatnonzero(part == 0)
    for b, g in enumerate([G_WARM] + gs):
        if b:
            run_chain(theirs, side_u, 1.0 / g, 0, 50, 7 if b == 1 else -1, LAM, NO_CAP, False)
    ours, _ = torus(24, 4, NO_CAP)
    list(run.cool(ours, part, run.schedule(G_WARM, 0.25, 200, 100, 50), run.chain_sweeps(50, N, 96), 7, LAM))
    assert (ours == theirs).all() and not (ours == torus(24, 4, NO_CAP)[0]).all()


def test_the_runner_refuses_to_overwrite(small, tmp_path):
    with pytest.raises(FileExistsError):                           # the finished run is never run over
        run.main(str(small["out"] / "t51_smoke.json"), str(small["out"]))
    cfg = _cfg("t51_smoke_taken", [0])
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfg))
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "t51_smoke_taken.csv").write_text("kept\n")
    with pytest.raises(FileExistsError):
        run.main(str(path), str(tmp_path / "a"))
    assert (tmp_path / "a" / "t51_smoke_taken.csv").read_text() == "kept\n"
    adj_dir = tmp_path / "b" / "t51_smoke_taken_adj"
    adj_dir.mkdir(parents=True)
    (adj_dir / "N96_rep1_tc0.npz").write_bytes(b"kept")
    with pytest.raises(FileExistsError):                           # a saved graph in the way stops it before any sweep
        run.main(str(path), str(tmp_path / "b"))
    assert (adj_dir / "N96_rep1_tc0.npz").read_bytes() == b"kept" and not list((tmp_path / "b").glob("*.csv*"))
    adj, part = torus(8, 8, NO_CAP)
    run.save_final(tmp_path / "c", 64, 0, 0, adj, part, LAM)
    with pytest.raises(FileExistsError):
        run.save_final(tmp_path / "c", 64, 0, 0, adj, part, LAM)


def test_a_tube_that_does_not_open_gets_one_row_and_is_not_followed(tmp_path):
    rows = _run(_cfg("t51_smoke_shut", [0, 200], g_hot=1.25, open_cap=100, replica_ids=[0]), tmp_path)
    assert len(rows) == 1
    r = rows[0]
    assert (r["phase"], r["opened"], r["final"]) == ("opening", "False", "True")
    assert (r["open_sweeps"], r["t_cool"], r["block"]) == ("100", "", "0")
    assert float(r["flat_share"]) < 0.90 or int(r["largest_d1"]) > 8
    assert not (tmp_path / "t51_smoke_shut_adj").exists()
    s = a.summarize(rows)
    assert s["table"] == {} and s["openings"][(24, 0)]["opened"] is False and s["verdict"] == "NOT READ"


# ---- the stop rule of Amendments 1 and 2 ------------------------------------------------------------------------

def _capped_tube(columns):
    """A stretch of tube `columns` columns long, made by hand: rings of four points, each ring joined to the next,
    with one more ring at either end standing for the sheet round it. The points of the stretch are at d = 1 (the two
    links along the tube close no square); the end rings have three links (the empty slot is -1) and are at d = 0."""
    rings = columns + 2
    adj = np.full((4 * rings, 4), -1, dtype=np.int64)
    for y in range(rings):
        for x in range(4):
            adj[4 * y + x, 0] = 4 * y + (x + 1) % 4
            adj[4 * y + x, 1] = 4 * y + (x - 1) % 4
            if y + 1 < rings:
                adj[4 * y + x, 2] = 4 * (y + 1) + x
            if y >= 1:
                adj[4 * y + x, 3] = 4 * (y - 1) + x
    return adj


def _side_by_side(*graphs):
    """Graphs set side by side in one array, sharing no link (an empty slot, -1, stays empty)."""
    out, start = [], 0
    for g in graphs:
        out.append(np.where(g >= 0, g + start, -1))
        start += g.shape[0]
    return np.vstack(out)


def test_the_stop_rule_on_numbers():
    assert run.stop_rule(0.95, 8, 0.90, 8) and run.stop_rule(0.90, 0, 0.90, 8)   # a resting piece of eight is allowed
    assert not run.stop_rule(0.95, 12, 0.90, 8)                    # flat enough, but three columns of tube are left
    assert not run.stop_rule(0.95, 9, 0.90, 8) and not run.stop_rule(0.89, 8, 0.90, 8)
    assert not run.stop_rule(0.95, 8, 0.90, 4)                     # Amendment 1's limit, which Amendment 2 replaced


def test_stretches_of_tube_made_by_hand():
    """A flat 16 x 16 sheet beside stretches of tube: with three columns left it has not opened; with a piece of
    eight and a piece of four it has."""
    flat, _ = torus(16, 16, NO_CAP)                                # 256 points at d = 2
    three, two, one = _capped_tube(3), _capped_tube(2), _capped_tube(1)
    assert [int((local_dimension(t) == 1).sum()) for t in (three, two, one)] == [12, 8, 4]
    assert [int((local_dimension(t) == 0).sum()) for t in (three, two, one)] == [8, 8, 8]
    with_three = _side_by_side(flat, three)                        # 93 % flat, twelve points of tube in one piece
    assert run.flat_and_tube(with_three) == (256 / 276, 12) and _largest_d1_by_the_definition(with_three) == 12
    assert not run.has_opened(with_three, 0.90, 8)
    assert run.has_opened(with_three, 0.90, 12)                    # the flat share alone would have stopped here
    with_two_and_one = _side_by_side(flat, two, one)               # 90 % flat; pieces of eight and of four at d = 1
    assert run.flat_and_tube(with_two_and_one) == (256 / 284, 8)
    assert _largest_d1_by_the_definition(with_two_and_one) == 8
    assert run.has_opened(with_two_and_one, 0.90, 8)
    assert not run.has_opened(with_two_and_one, 0.90, 4)           # Amendment 1's limit would have waited for it
    assert run.flat_and_tube(flat) == (1.0, 0) and run.has_opened(flat, 0.90, 8)
    assert not run.has_opened(_side_by_side(flat, two, one, one), 0.90, 8)     # 86 % flat: not flat enough


def test_a_real_three_column_stretch_keeps_the_opening_going():
    """With the flat threshold lowered to 80 %, so that at 96 points three columns of tube can still be there when it
    is met: replica 1 of the test seed is 83 % flat after 550 sweeps with twelve points of tube in one piece. The
    flat share alone stops there; the rule at eight goes on until the stretch has shrunk."""
    early, _, early_sweeps, early_opened = _open(1, stop_flat=0.80, max_d1_piece=N)        # the piece rule cannot bind
    flat, largest = run.flat_and_tube(early)
    assert early_opened and flat >= 0.80 and largest == _largest_d1_by_the_definition(early) and largest >= 12
    assert not run.has_opened(early, 0.80, 8)
    capped, _, capped_sweeps, capped_opened = _open(1, stop_flat=0.80, cap=early_sweeps)   # the rule at eight, cut off
    assert (capped == early).all() and capped_sweeps == early_sweeps and not capped_opened
    sheet, _, sweeps, opened = _open(1, stop_flat=0.80)
    flat, largest = run.flat_and_tube(sheet)
    assert opened and sweeps > early_sweeps and flat >= 0.80 and largest <= 8
    assert largest == _largest_d1_by_the_definition(sheet)


def test_a_resting_piece_of_eight_does_not_hold_the_opening_up(small):
    """Replica 2 of the test seed is 92 % flat after 350 sweeps with one piece of eight points at d = 1. The rule at
    eight ends the opening there, and the piece is read as an other; Amendment 1's limit of four went on."""
    sheet, _, sweeps, opened = small["sheets"][2]
    flat, largest = run.flat_and_tube(sheet)
    assert opened and flat >= 0.90 and largest == _largest_d1_by_the_definition(sheet) == 8
    first = run.reading(sheet, LAM)
    assert (first["columns"], first["others"], first["n_d1"]) == (0, 1, 8)
    later, _, later_sweeps, later_opened = _open(2, max_d1_piece=4)
    assert later_opened and later_sweeps > sweeps and run.flat_and_tube(later)[1] <= 4


def test_every_opening_that_reports_opened_meets_both_conditions(small, tmp_path):
    rows = small["rows"] + _run(_cfg("t51_smoke_more", [0], replica_ids=[2]), tmp_path)
    first = [r for r in rows if r["phase"] == "opening"]
    assert [r["replica"] for r in first] == ["0", "1", "2"] and all(r["opened"] == "True" for r in first)
    for r in first:
        sheet, _, sweeps, opened = small["sheets"][int(r["replica"])]
        assert opened and int(r["open_sweeps"]) == sweeps
        assert float(r["flat_share"]) == float((local_dimension(sheet) == 2).mean()) >= 0.90
        assert int(r["largest_d1"]) == _largest_d1_by_the_definition(sheet) <= 8
        assert run.has_opened(sheet, 0.90, 8)
        sooner = _open(int(r["replica"]), cap=sweeps - 50)         # the first such reading: one reading sooner, not yet
        assert sooner[2] == sweeps - 50 and not sooner[3]
    assert [int(r["largest_d1"]) for r in first] == [4, 4, 8]      # two single columns, and one resting piece of eight
    with_column = small["sheets"][0][0]                            # it holds one curled column: four points at d = 1
    assert run.reading(with_column, LAM)["columns"] == 1 and run.flat_and_tube(with_column)[1] == 4


# ---- the leftover reader ----------------------------------------------------------------------------------------

def _by_the_definition(adj):
    """Columns and other pieces straight from the pre-registered words, with networkx."""
    d = local_dimension(adj)
    g = _graph(adj)
    pieces = list(nx.connected_components(g.subgraph([v for v in g if d[v] != 2])))
    columns = sum(1 for p in pieces if len(p) == 4 and all(d[v] == 1 for v in p))
    return columns, len(pieces) - columns


def test_reading_the_tube_the_sheet_and_a_mixture_made_by_hand():
    tube, _ = torus(24, 4, NO_CAP)                                 # every point at d = 1, one piece, 4(lambda - 1) each
    assert run.reading(tube, LAM) == dict(columns=0, others=1, n_d1=96, largest_d1=96, flat_share=0.0, h=96.0)
    assert run.flat_and_tube(tube) == (0.0, 96)
    flat, _ = torus(8, 8, NO_CAP)
    assert run.reading(flat, LAM) == dict(columns=0, others=0, n_d1=0, largest_d1=0, flat_share=1.0, h=0.0)
    short, _ = torus(6, 4, NO_CAP)                                 # 24 points at d = 1: energy 24
    cube, _ = torus(4, 4, NO_CAP)                                  # the 4-cube, 16 points at d = 0: energy 32
    mixed = np.vstack([flat, short + 64, cube + 88])               # side by side, sharing no link
    assert run.reading(mixed, LAM) == dict(columns=0, others=2, n_d1=24, largest_d1=24, flat_share=64 / 104, h=56.0)
    assert _by_the_definition(mixed) == (0, 2)


def test_the_reader_is_t37s_and_agrees_with_the_definition(small):
    graphs = [np.load(f)["adj"] for f in sorted((small["out"] / "t51_smoke_adj").glob("*.npz"))]
    graphs += [small["sheets"][rep][0] for rep in (0, 1, 2)]
    columns_seen = 0
    for adj in graphs:
        r = run.reading(adj, LAM)
        assert (r["columns"], r["others"]) == t37.leftovers(adj) == _by_the_definition(adj)
        h, flat, _ = t37.end_state(adj, LAM)
        assert r["h"] == h and r["flat_share"] == flat
        assert (r["flat_share"], r["largest_d1"]) == run.flat_and_tube(adj)
        assert r["largest_d1"] == _largest_d1_by_the_definition(adj)
        assert r["n_d1"] == int((local_dimension(adj) == 1).sum())
        columns_seen += r["columns"]
    assert columns_seen > 0                                        # so the four-points-at-d-1 rule itself was used
    assert a.FLAT == t37.CLEAN == 0.90


# ---- the analysis -----------------------------------------------------------------------------------------------

def _replica(length, t_cool, rep, c0, c1, h0=40.0, h1=8.0, o0=2, o1=0, flat=0.99):
    """The rows the analysis reads for one replica at one cooling time: the opening, one block on the way that must
    be ignored, and the end of the hold."""
    base = dict(N=str(4 * length), L=str(length), lam="1.25", replica=str(rep), open_sweeps="30000", opened="True")
    opening = dict(base, t_cool="", phase="opening", block="0", columns=str(c0), others=str(o0), n_d1=str(4 * c0),
                   largest_d1="4", flat_share="0.91", h=str(h0), final="False")
    on_the_way = dict(base, t_cool=str(t_cool), phase="hold", block="23", columns="99", others="99", n_d1="396",
                      largest_d1="4", flat_share="0.5", h="999.0", final="False")
    end = dict(base, t_cool=str(t_cool), phase="hold", block="24", columns=str(c1), others=str(o1), n_d1=str(4 * c1),
               largest_d1="4", flat_share=str(flat), h=str(h1), final="True")
    return [opening, on_the_way, end]


def _verdict_rows(c_10k, c_30k, c_100k, c_300k, c0=4):
    """L = 256, c0 columns at the end of every opening; 80 replicas, the 300,000 cell only the first 40."""
    rows = []
    for rep in range(80):
        rows += _replica(256, 10000, rep, c0, c_10k(rep)) + _replica(256, 30000, rep, c0, c_30k(rep))
        rows += _replica(256, 100000, rep, c0, c_100k(rep))
        if rep < 40:
            rows += _replica(256, 300000, rep, c0, c_300k(rep))
    return rows


def test_survival_is_a_ratio_of_sums_and_leaves_out_the_melted():
    rows = []
    for rep in range(10):
        rows += _replica(256, 10000, rep, c0=4, c1=3)
    rows += _replica(256, 10000, 10, c0=4, c1=40, h1=600.0, flat=0.50)        # MELTED OR DEFECTED
    s = a.summarize(rows)
    c = s["table"][(256, 10000)]
    assert (c["n"], c["kept"], c["melted"]) == (11, 10, 1) and s["clash"] == []
    assert c["S"][0] == 0.75 and c["S"][1] < 1e-12                 # 30 of 40 columns; identical replicas, no spread
    assert c["S_E"][0] == 0.2 and c["others_ratio"][0] == 0.0
    assert abs(c["frozen"][0] - 8.0 / 1024) < 1e-15 and c["frozen_all"][0] > 5 * c["frozen"][0]
    assert abs(c["per_column"][0] - 3 / 256) < 1e-15
    assert a.ratio_se([1, 0], [1, 3])[0] == 0.25                   # the ratio of the sums, not the mean of the ratios


def test_the_resampling_error_is_fixed_and_of_the_right_size():
    values = list(range(40))
    ratio, se = a.ratio_se(values, [1] * 40)                       # with ones below, the ratio is a plain mean
    assert ratio == 19.5 and a.ratio_se(values, [1] * 40) == (ratio, se)
    assert abs(se / (np.std(values) / math.sqrt(40)) - 1.0) < 0.10
    assert all(math.isnan(v) for v in a.ratio_se([], [])) and math.isnan(a.ratio_se([1, 2], [0, 0])[0])
    assert a.mean_se([1.0, 3.0]) == (2.0, 1.0) and math.isnan(a.mean_se([1.0])[1])


def test_a_broken_pairing_is_reported():
    rows = _replica(256, 10000, 0, c0=4, c1=3) + _replica(256, 30000, 0, c0=5, c1=3)
    assert a.summarize(rows)["clash"] == [(256, 0)]


def test_p1_the_fair_clock_on_ratios():
    assert a.score_p1({30000: 0.99, 100000: 0.47})[1] is True
    assert a.score_p1({30000: 0.79, 100000: 0.67})[1] is True                    # 0.20 away counts as within
    assert a.score_p1({30000: 1.19, 100000: 0.27})[1] is True
    lines, holds = a.score_p1({30000: 0.78, 100000: 0.47})                       # the first ratio alone outside
    assert holds is False and [ok for _, _, ok in lines.values()] == [False, True]
    lines, holds = a.score_p1({30000: 1.00, 100000: 0.68})                       # the second ratio alone outside
    assert holds is False and [ok for _, _, ok in lines.values()] == [True, False]
    assert a.score_p1({30000: 1.20, 100000: 0.47})[1] is False
    assert a.score_p1({30000: 0.99})[1] is None                                  # a cell missing: NOT READ
    assert a.score_p1({30000: None, 100000: None})[1] is None                    # C(10,000) zero: NOT READ
    assert a.score_p1({30000: 0.50})[1] is None                                  # NOT READ while a cell is missing
    assert a.P1_TARGETS == {30000: 0.99, 100000: 0.47} and (a.P1_L, a.P1_BASE, a.P1_WINDOW) == (256, 10000, 0.20)


def test_p1_end_to_end_needs_no_count_at_the_end_of_the_opening():
    # over 80 replicas: C(10,000) = 320, C(30,000) = 320, C(100,000) = 160, so the ratios are 1.0 and 0.5
    good = _verdict_rows(lambda r: 4, lambda r: 4, lambda r: 2, lambda r: 2)
    s = a.summarize(good)
    assert s["p1"] is True and [line[0] for line in s["p1_lines"].values()] == [1.0, 0.5]
    assert [s["p1_ratios"][t]["n"] for t in (30000, 100000)] == [80, 80]
    assert (s["p1_ratios"][100000]["c_a"], s["p1_ratios"][100000]["c_b"]) == (320, 160)
    assert s["p1_ratios"][100000]["ratio"] == s["ratios"][10000]["R"]            # the same number as R(10,000)
    # the same coolings with no column at all at the end of any opening: S is undefined, P1 and the verdict stand
    born_later = a.summarize(_verdict_rows(lambda r: 4, lambda r: 4, lambda r: 2, lambda r: 2, c0=0))
    assert math.isnan(born_later["table"][(256, 10000)]["S"][0])
    assert born_later["p1"] is True and born_later["verdict"] == s["verdict"] == "GENTLE"
    # each ratio outside its window on its own
    assert a.summarize(_verdict_rows(lambda r: 4, lambda r: 2, lambda r: 2, lambda r: 2))["p1"] is False  # 0.5 for 0.99
    assert a.summarize(_verdict_rows(lambda r: 4, lambda r: 4, lambda r: 4, lambda r: 2))["p1"] is False  # 1.0 for 0.47
    for gone in ("10000", "30000", "100000"):                                    # a missing cell: NOT READ
        short = a.summarize([r for r in good if r["t_cool"] != gone])
        assert short["p1"] is None and "a cell is missing" in "\n".join(a.p1_text(short))
    # a replica melted in the 30,000 cell is in neither sum of that ratio (with it the ratio would be 720 / 324)
    s = a.summarize(good + _replica(256, 30000, 80, 4, 400, flat=0.2) + _replica(256, 10000, 80, 4, 4))
    assert s["p1_ratios"][30000]["n"] == 80 and s["p1_lines"][30000][0] == 1.0 and s["p1"] is True


def test_p1_says_so_plainly_when_nothing_survived_the_10000_cooling():
    s = a.summarize(_verdict_rows(lambda r: 0, lambda r: 4, lambda r: 2, lambda r: 2))
    assert s["p1"] is None and all(c["ratio"] is None and c["c_a"] == 0 for c in s["p1_ratios"].values())
    text = "\n".join(a.p1_text(s))
    assert "NOT READ" in text and text.count("C(10000) is zero") == 2 and "no column survived" in text
    assert s["verdict"] == "CLIFF"                                 # the verdict's own zero rule


def test_p2_the_size_check():
    agree = {128: (0.012, 0.002), 256: (0.014, 0.002), 512: (0.011, 0.001)}
    pairs, holds = a.score_p2(agree)
    assert holds is True and set(pairs) == {(128, 256), (128, 512), (256, 512)}
    assert abs(pairs[(128, 256)][1] - math.hypot(0.002, 0.002)) < 1e-15
    assert a.score_p2({**agree, 512: (0.004, 0.001)})[1] is False                # 0.008 and 0.010 against 0.0045
    assert a.score_p2({128: (0.012, 0.002), 256: (0.014, 0.002)})[1] is None
    rows = []
    for rep in range(8):
        rows += _replica(128, 10000, rep, 2, 1 + rep % 2) + _replica(256, 10000, rep, 4, 2 + 2 * (rep % 2))
        rows += _replica(512, 10000, rep, 8, 4 + 4 * (rep % 2))
    assert a.summarize(rows)["p2"] is True                         # the same 3 / 256 per column of tube at each size
    rows = [r for r in rows if r["L"] != "512"]
    for rep in range(8):
        rows += _replica(512, 10000, rep, 8, 1 + rep % 2)                        # four times fewer per column
    assert a.summarize(rows)["p2"] is False


def test_the_verdict_from_the_decade_ratios():
    assert a.verdict(0.5, 0.5) == "GENTLE" and a.verdict(0.25, 0.85) == "GENTLE"
    assert a.verdict(0.24, 0.5) == "CLIFF" and a.verdict(0.9, 0.1) == "CLIFF" and a.verdict(0.0, 0.0) == "CLIFF"
    assert a.verdict(0.86, 0.99) == "FROZEN" and a.verdict(1.2, 1.0) == "FROZEN"
    assert a.verdict(0.5, 0.9) == "MIXED" and a.verdict(0.86, 0.85) == "MIXED"
    assert a.verdict(None, 0.5) == "CLIFF" and a.verdict(0.5, None) == "CLIFF"   # S(10,000) or S(30,000) zero
    assert a.decade_ratio(0.8, 0.4) == 0.5 and a.decade_ratio(0.0, 0.0) is None and a.decade_ratio(0.5, 0.0) == 0.0


def test_the_verdict_end_to_end_on_the_replicas_both_cells_share():
    # the last 40 replicas lose every column at 30,000; the 300,000 cell does not have them, so R(30,000) drops them
    s = a.summarize(_verdict_rows(lambda r: 4, lambda r: 4 if r < 40 else 0, lambda r: 2, lambda r: 2))
    assert s["ratios"][10000]["R"] == 0.5 and s["ratios"][10000]["n"] == 80
    assert s["ratios"][30000]["R"] == 0.5 and s["ratios"][30000]["n"] == 40 and s["ratios"][30000]["unpaired"] == 1.0
    assert s["verdict"] == "GENTLE" and s["verdict_unpaired"] == "MIXED"
    assert a.summarize(_verdict_rows(lambda r: 4, lambda r: 4, lambda r: 0, lambda r: 2))["verdict"] == "CLIFF"
    assert a.summarize(_verdict_rows(lambda r: 4, lambda r: 4, lambda r: 4, lambda r: 4))["verdict"] == "FROZEN"
    assert a.summarize(_verdict_rows(lambda r: 4, lambda r: 4, lambda r: 2, lambda r: 4))["verdict"] == "MIXED"
    zero = a.summarize(_verdict_rows(lambda r: 4, lambda r: 0, lambda r: 2, lambda r: 0))
    assert zero["ratios"][30000]["R"] is None and zero["verdict"] == "CLIFF"     # S(30,000) = 0
    melted = _verdict_rows(lambda r: 4, lambda r: 4, lambda r: 2, lambda r: 2)
    melted += _replica(256, 100000, 80, 4, 400, flat=0.2) + _replica(256, 10000, 80, 4, 4)
    s = a.summarize(melted)                                         # a melted replica is in neither cell of the ratio
    assert s["ratios"][10000]["n"] == 80 and s["ratios"][10000]["R"] == 0.5 and s["table"][(256, 100000)]["melted"] == 1
    fewer = [r for r in _verdict_rows(lambda r: 4, lambda r: 4, lambda r: 2, lambda r: 2)
             if not (r["t_cool"] == "300000" and int(r["replica"]) >= 7)]
    assert a.summarize(fewer)["ratios"][30000]["n"] == 7           # whatever the cells share: no count is built in
    no_300k = [r for r in _verdict_rows(lambda r: 4, lambda r: 4, lambda r: 2, lambda r: 2) if r["t_cool"] != "300000"]
    assert a.summarize(no_300k)["verdict"] == "NOT READ"


def test_the_analysis_reads_what_the_runner_writes(small, capsys):
    rows = a.load(small["out"])
    assert rows == small["rows"]
    s = a.summarize(rows)
    assert set(s["table"]) == {(24, 0), (24, 200)} and all(c["n"] == 2 for c in s["table"].values())
    assert s["clash"] == [] and all(o["opened"] for o in s["openings"].values())
    for (length, t_cool), c in s["table"].items():
        last = [r for r in rows if r["final"] == "True" and r["t_cool"] == str(t_cool)]
        assert len(last) == 2 and c["n"] == c["kept"] + c["melted"]
    assert a.check_saved(small["out"], rows) == (4, [])            # the saved graphs say what their rows say
    a.main(str(small["out"]))
    printed = capsys.readouterr().out
    assert "VERDICT: NOT READ" in printed and "4, of which 0 differ" in printed


# ---- the configs and the queue file -----------------------------------------------------------------------------

SETS = {    # files' common stem: (L, cooling times, seed, files, replicas per file, replicas covered, the seeded set)
    "t51_l256_fast": (256, [0, 1000, 3000, 10000, 30000], 20261005, 20, 4, 80, 80),
    "t51_l256_100k": (256, [100000], 20261005, 20, 4, 80, 80),
    "t51_l256_300k": (256, [300000], 20261005, 40, 1, 40, 80),      # the first 40 of the 80, one to a job
    "t51_l128_10k": (128, [10000], 20261006, 1, 40, 40, 40),
    "t51_l512_10k": (512, [10000], 20261007, 10, 4, 40, 40),
}


def test_the_configs_are_the_pre_registered_cells():
    paths = sorted((ROOT / "configs").glob("t51_*.json"))
    assert len(paths) == 91
    ids, files = defaultdict(list), defaultdict(int)
    for p in paths:
        cfg = json.loads(p.read_text())
        stem = p.stem.rsplit("_", 1)[0]
        length, t_cool, seed, _, per_file, _, in_set = SETS[stem]
        assert cfg["name"] == p.stem and cfg["section"] == "T51"
        assert cfg["_purpose"].startswith("PREREGISTRATION.md T51 (written 2026-10-05, before any run): ")
        assert cfg["_purpose"].endswith(" Not exploratory.")
        assert cfg["side"] == [length, 4] and cfg["t_cool"] == t_cool and cfg["seed"] == seed
        assert (cfg["lambda"], cfg["g_hot"], cfg["g_cold"]) == (1.25, 1.25, 0.25)
        assert (cfg["n_ref"], cfg["hold"], cfg["block"]) == (96, 5000, 250)
        assert (cfg["open_check_every"], cfg["open_stop_flat"], cfg["open_cap"]) == (100, 0.90, 600000)
        assert cfg["open_max_d1_piece"] == 8                       # Amendments 1 and 2
        assert cfg["replicas"] == in_set and cfg["save_adjacency"] is True and len(cfg["replica_ids"]) == per_file
        ids[stem] += cfg["replica_ids"]
        files[stem] += 1
    for stem, (_, _, _, n_files, _, covered, _) in SETS.items():
        assert files[stem] == n_files and ids[stem] == list(range(covered))      # disjoint, complete, in order


def test_the_queue_file_names_every_config_once():
    text = (ROOT / "cloud" / "queue" / "2026-10-05_t51.txt").read_text()
    jobs = [ln.split() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    assert len(jobs) == 91 and "Amendments 1 and 2" in text
    assert all(len(j) == 2 and j[0] == "run_scrap_freeze" for j in jobs)
    assert sorted(j[1] for j in jobs) == sorted(p.stem for p in (ROOT / "configs").glob("t51_*.json"))
    assert (ROOT / "scripts" / "run_scrap_freeze.py").exists()
