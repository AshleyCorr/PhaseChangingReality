"""T53's runner, analyzer, configs and queue file, written before any run (PREREGISTRATION T53)."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from graphity.cqg_d import is_valid, run_chain, torus

ROOT = Path(__file__).resolve().parents[1]
LENGTHS, LAMBDAS, COUPLINGS = (18, 36, 72), (1.25, 1.40), (1.5, 2.0, 2.5, 3.0, 3.5)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


runner = _load("run_curled_bath_d")
t53 = _load("analyse_t53")
t48 = _load("analyse_t48")

TINY = dict(name="t53_tiny", section="T53", dims=[4, 4, 6], g=3.0, replicas=2, n_sweeps=60, record_every=20,
            seed=11, save_adjacency=True, **{"lambda": 1.4})


def _run(tmp_path, out, cfg=TINY):
    path = tmp_path / (cfg["name"] + ".json")
    path.write_text(json.dumps(cfg))
    runner.main(str(path), str(tmp_path / out))
    return tmp_path / out


def _rows(path):
    return list(csv.DictReader(open(path, newline="", encoding="utf-8")))


# ---------------------------------------------------------------- the runner

def test_same_config_and_seed_give_the_same_file_and_graphs(tmp_path):
    a, b = _run(tmp_path, "a"), _run(tmp_path, "b")
    assert (a / "t53_tiny.csv").read_text() == (b / "t53_tiny.csv").read_text()
    finals = []
    for rep in range(2):
        ga, gb = np.load(a / "t53_tiny_adj" / ("rep%d.npz" % rep)), np.load(b / "t53_tiny_adj" / ("rep%d.npz" % rep))
        assert np.array_equal(ga["adj"], gb["adj"]) and int(ga["seed"]) == int(gb["seed"])
        assert is_valid(ga["adj"])
        finals.append(ga["adj"])
    assert not np.array_equal(finals[0], finals[1])                 # the replicas have their own streams
    assert runner.seed_of(11, 96, 0) != runner.seed_of(11, 96, 1) != runner.seed_of(12, 96, 1)


def test_the_file_has_the_columns_and_rows_the_analyzer_reads(tmp_path):
    out = _run(tmp_path, "a")
    rows = _rows(out / "t53_tiny.csv")
    assert list(rows[0])[:6] == ["N", "dims", "lam", "g", "replica", "sweep"]
    for column in ["d%d" % k for k in range(8)] + ["damaged_d", "largest_open_d", "open_regions", "n_open2", "pieces",
                                                   "h", "acceptance", "final"]:
        assert column in rows[0]
    assert [(r["replica"], r["sweep"]) for r in rows] == [(str(k), str(s)) for k in (0, 1) for s in (0, 20, 40, 60)]
    assert [r["final"] for r in rows] == ["False", "False", "False", "True"] * 2
    start = rows[0]                                                 # the exact start: every point at d = 1
    assert (start["N"], start["dims"], start["d1"], start["n_open2"], start["pieces"]) == ("96", "4 4 6", "96", "0", "1")
    assert float(start["h"]) == pytest.approx(8 * (1.4 - 1.0) * 96)
    for r in rows:                                                  # the columns agree with each other
        assert sum(int(r["d%d" % k]) for k in range(8)) == 96
        assert float(r["h"]) == pytest.approx(float(r["h_per_vertex"]) * 96)
        assert int(r["n_open2"]) == sum(int(r["d%d" % k]) for k in range(2, 8))
    meta = json.loads((out / "t53_tiny.meta.json").read_text())
    assert meta["config"]["seed"] == 11 and "T53" in meta["preregistration"]
    s = t53.summarize(rows)                                         # and the analyzer reads the file as it is
    assert sum(s["cells"][(6, 1.4, 3.0)].values()) == 2 and len(s["notes"][(6, 1.4, 3.0)]) == 2


def test_reading_the_graph_does_not_disturb_the_chain():
    finals = []
    for read in (runner.reading, lambda adj, lam: {}):              # the real reading, and no reading at all
        adj, part = torus([4, 4, 6])
        steps = list(runner.follow(adj, np.flatnonzero(part == 0), 1.4, 3.0, 60, 20, 4321, read))
        assert [s[0] for s in steps] == [20, 40, 60]
        finals.append(adj)
    assert np.array_equal(finals[0], finals[1])
    adj, part = torus([4, 4, 6])                                    # blocks carry the stream: one call makes the same chain
    run_chain(adj, np.flatnonzero(part == 0), 1.0 / 3.0, 0, 60, 4321, 1.4, False)
    assert np.array_equal(adj, finals[0])
    assert not np.array_equal(adj, torus([4, 4, 6])[0])             # and the chain did move
    before = adj.copy()
    runner.reading(adj, 1.4)
    assert np.array_equal(adj, before)


def test_readers_on_hand_built_graphs():
    flat, _ = torus([6, 6, 8])                                      # flat: every point at d = 3, one region, one piece
    r = runner.reading(flat, 1.25)
    assert (r["d3"], r["open_regions"], r["n_open2"], r["pieces"], r["largest_open_d"], r["damaged_d"]) == (288, 1, 288, 1, 288, 0)
    assert r["h"] == 0.0
    start, _ = torus([4, 4, 18])                                    # the exact start: every point at d = 1, no region
    r = runner.reading(start, 1.25)
    assert (r["d1"], r["open_regions"], r["n_open2"], r["pieces"], r["largest_open_d"], r["damaged_d"]) == (288, 0, 0, 1, 0, 0)
    assert r["h"] == pytest.approx(8 * 0.25 * 288)
    two, _ = runner.start_of({"gas": {"dims": [4, 4, 6], "copies": 2}})[:2]      # two disjoint tori: two pieces
    assert runner.whole_pieces(two) == 2 and runner.whole_pieces(start) == 1 and runner.whole_pieces(flat) == 1


def test_open_regions_count_pieces_of_sixteen_or_more_at_d_two_or_more():
    adj, _ = torus([4, 4, 18])
    layer = np.arange(288) % 18                                     # the position along the open direction
    d = np.ones(288, dtype=np.int64)
    d[layer <= 1] = 2                                               # two layers side by side: one piece of 32
    d[layer == 9] = 3                                               # one layer: a piece of exactly 16, which counts
    d[layer == 14] = 2
    d[14] = 1                                                       # one layer less a point: 15, too small to count
    assert runner.open_regions_of(adj, d) == (2, 63)
    d[layer == 5] = 9                                               # d >= 2 as written: a damaged layer counts too
    assert runner.open_regions_of(adj, d) == (3, 79)
    d[layer == 10] = 2                                              # joining the layer at 9: still three pieces
    assert runner.open_regions_of(adj, d) == (3, 95)
    assert runner.open_regions_of(adj, np.ones(288, dtype=np.int64)) == (0, 0)


def test_refuses_to_overwrite(tmp_path):
    out = _run(tmp_path, "a")
    text = (out / "t53_tiny.csv").read_text()
    with pytest.raises(FileExistsError):
        _run(tmp_path, "a")
    assert (out / "t53_tiny.csv").read_text() == text
    kept = tmp_path / "b" / "t53_tiny_adj"                          # a saved graph alone stops the run before it starts
    kept.mkdir(parents=True)
    (kept / "rep1.npz").write_bytes(b"kept")
    with pytest.raises(FileExistsError):
        _run(tmp_path, "b")
    assert (kept / "rep1.npz").read_bytes() == b"kept"
    assert sorted(p.name for p in (tmp_path / "b").iterdir()) == ["t53_tiny_adj"]
    with pytest.raises(ValueError):                                 # a part block would be dropped silently
        _run(tmp_path, "c", dict(TINY, name="t53_tiny_c", n_sweeps=50))


# ---------------------------------------------------------------- the analyzer, on made-up rows

def _row(length, lam, g, rep, sweep, d, largest=0, regions=0, pieces=1, h=0.0):
    r = dict(dims="4 4 %d" % length, N=16 * length, lam=lam, g=g, replica=rep, sweep=sweep, largest_open_d=largest,
             open_regions=regions, n_open2=sum(d[2:]), pieces=pieces, h=h)
    r.update({"d%d" % k: (d[k] if k < len(d) else 0) for k in range(8)})
    return r


def _last(d, largest=0):
    return [_row(18, 1.25, 2.0, 0, 0, [0, 288]), _row(18, 1.25, 2.0, 0, 1000, d, largest)]


def test_every_branch_of_the_replica_rule_and_agreement_with_t48():
    cases = [([0, 288], 0, "STAYS"),
             ([0, 200, 88], 0, "STAYS"),                            # some points moved; the start rung still holds the most
             ([0, 144, 144], 0, "STAYS"),                           # a tie goes to the lower rung, as in T48
             ([0, 100, 188], 0, "ADVANCES"),
             ([0, 0, 20, 268], 100, "ADVANCES"),                    # open everywhere but in pieces: not one space
             ([0, 0, 145, 143], 143, "ADVANCES"),                   # one point short of half at d = 3
             ([0, 0, 144, 144], 144, "OPENS"),                      # exactly half, in one piece of half
             ([0, 0, 0, 217, 71], 200, "OPENS"),                    # one point short of a quarter damaged
             ([0, 0, 0, 216, 72], 216, "DAMAGED"),                  # exactly a quarter damaged wins over open
             ([0, 0, 0, 0, 100, 100, 50, 38], 0, "DAMAGED")]
    for d, largest, want in cases:
        blocks = _last(d, largest)
        assert t53.class_of(blocks) == want, (d, largest)
        assert t53.class_of(blocks) == t48.read_replica(blocks)[0]
    late = _last([0, 0, 0, 288], 288) + [_row(18, 1.25, 2.0, 0, 2000, [0, 288])]
    assert t53.class_of(late) == "STAYS"                            # the last reading decides, not an earlier one


def test_cell_majority_and_mixed():
    assert t53.majority_of(["OPENS"] * 5 + ["DAMAGED"] * 3) == "OPENS"
    assert t53.majority_of(["OPENS"] * 4 + ["DAMAGED"] * 4) == "MIXED"          # half is not a majority
    assert t53.majority_of(["STAYS"] * 3 + ["ADVANCES"] * 3 + ["DAMAGED"] * 2) == "MIXED"
    assert t53.majority_of(["STAYS"]) == "STAYS"


def test_every_branch_of_the_window_verdict():
    stays, adv, opens, dam = ["STAYS"] * 8, ["ADVANCES"] * 8, ["OPENS"] * 8, ["DAMAGED"] * 8
    mixed = ["ADVANCES"] * 4 + ["DAMAGED"] * 4
    assert t53.window_verdict({1.5: stays, 2.0: adv, 2.5: opens, 3.0: dam, 3.5: dam}) == "OPENS IN A WINDOW"
    assert t53.window_verdict({1.5: opens[:5] + dam[:3], 2.0: dam}) == "OPENS IN A WINDOW"
    assert t53.window_verdict({1.5: stays, 2.0: adv, 2.5: mixed, 3.0: dam}) == "ADVANCES ONLY"
    assert t53.window_verdict({1.5: opens[:4] + adv[:4], 2.0: adv}) == "ADVANCES ONLY"     # no OPENS majority anywhere
    assert t53.window_verdict({1.5: stays, 2.0: stays, 2.5: dam, 3.0: dam}) == "DAMAGED"
    assert t53.window_verdict({1.5: dam[:5] + stays[:3]}) == "DAMAGED"
    assert t53.window_verdict({1.5: stays, 2.0: stays}) == "STAYS"                         # nothing moved anywhere
    assert t53.window_verdict({1.5: stays, 2.0: mixed, 2.5: dam}) == "STAYS"               # moved, without a DAMAGED majority
    # "anything moved" is read replica by replica: one damaged replica among stayers is a g at which something moved
    assert t53.window_verdict({1.5: stays[:7] + dam[:1], 2.0: dam}) == "STAYS"
    assert t53.moved_at({1.5: stays, 2.0: stays[:7] + dam[:1], 2.5: dam}) == [2.0, 2.5]


def _opening(length, lam, g, rep, regions_at_tenth, end="ADVANCES"):
    """A replica with a tenth of its points at d >= 2 first at sweep 2000, in `regions_at_tenth` regions."""
    n = 16 * length
    tenth = -(-n // 10)
    last = {"ADVANCES": ([0, 0, n], 0), "OPENS": ([0, 0, 0, n], n), "STAYS": ([0, n], 0),
            "DAMAGED": ([0, 0, 0, 0, n], 0)}[end]
    return [_row(length, lam, g, rep, 0, [0, n]),
            _row(length, lam, g, rep, 1000, [0, n - tenth + 1, tenth - 1], regions=7),      # just short of a tenth
            _row(length, lam, g, rep, 2000, [0, n - tenth, tenth], regions=regions_at_tenth),
            _row(length, lam, g, rep, 3000, [0, 0, n], regions=1),
            _row(length, lam, g, rep, 4000, last[0], last[1], regions=1)]


def _several(long_, short):
    rows = []
    for length, values in ((72, long_), (18, short)):
        for rep, v in enumerate(values):
            rows += _opening(length, 1.25 if rep % 2 else 1.4, 2.0 + 0.5 * (rep % 3), rep, *v)
    return t53.summarize(rows)


def test_every_branch_of_one_place_or_several():
    s = _several([(3,), (2,), (2, "OPENS")], [(1,), (1,), (2,)])
    assert {k: sorted(v) for k, v in s["regions"].items()} == {72: [2, 2, 3], 18: [1, 1, 2]}    # read at the first tenth
    assert s["several"] == "SEVERAL"
    assert _several([(1,), (1,), (4,)], [(1,)])["several"] == "ONE"                         # median below 2 at L = 72
    assert _several([(2,), (2,)], [(2,), (2,)])["several"] == "ONE"                         # not larger than at L = 18
    assert _several([(1,), (2,)], [(1,)])["several"] == "ONE"                               # a median of 1.5 is not 2
    assert _several([(3,)], [(3, "STAYS")])["several"] == "ONE"                             # nothing opened at L = 18
    assert _several([(3, "DAMAGED"), (3, "STAYS")], [(1,)])["several"] == "ONE"             # nothing opened at L = 72
    s = _several([(3,), (3,), (1, "DAMAGED"), (1, "STAYS")], [(1,), (5, "DAMAGED")])        # only OPENS and ADVANCES count
    assert s["regions"] == {72: [3, 3], 18: [1]} and s["several"] == "SEVERAL"
    assert t53.several_verdict({72: [2, 2], 18: [1, 2]}) == "SEVERAL"                       # 2 against 1.5
    assert t53.several_verdict({}) == "ONE"
    assert t53.regions_at_first_tenth(_last([0, 288])) is None


def test_summarize_puts_cells_windows_and_verdicts_together():
    rows = []
    for rep in range(8):
        rows += _opening(18, 1.25, 1.5, rep, 1, "STAYS")
        rows += _opening(18, 1.25, 2.0, rep, 1, "ADVANCES" if rep < 5 else "DAMAGED")
        rows += _opening(18, 1.4, 2.5, rep, 1, "OPENS" if rep < 4 else "ADVANCES")
        rows += _opening(72, 1.4, 3.0, rep, 2, "OPENS")
        rows += _opening(72, 1.25, 3.5, rep, 1, "DAMAGED")
    s = t53.summarize(rows)
    assert s["cells"][(18, 1.25, 2.0)] == {"ADVANCES": 5, "DAMAGED": 3}
    assert s["majority"] == {(18, 1.25, 1.5): "STAYS", (18, 1.25, 2.0): "ADVANCES", (18, 1.4, 2.5): "MIXED",
                             (72, 1.4, 3.0): "OPENS", (72, 1.25, 3.5): "DAMAGED"}
    assert s["window"] == {(18, 1.25): "ADVANCES ONLY", (18, 1.4): "STAYS", (72, 1.4): "OPENS IN A WINDOW",
                           (72, 1.25): "DAMAGED"}
    assert s["moved"][(18, 1.25)] == [2.0] and s["several"] == "SEVERAL"
    assert s["regions_by_lam"][(72, 1.4)] == [2] * 8 and (72, 1.25) not in s["regions_by_lam"]


def test_reported_items_half_times_pause_pieces_and_energy():
    def series(modal_rungs, pieces=1, h=0.0):
        shape = {1: [0, 288], 2: [0, 100, 188], 3: [0, 0, 100, 188]}
        return [_row(18, 1.4, 2.5, 0, 1000 * i, shape[m], pieces=pieces, h=h) for i, m in enumerate(modal_rungs)]
    assert t53.pause_of(series([1] * 3 + [2] * 20 + [3] * 2)) == ("pauses", 20)
    assert t53.pause_of(series([1] * 3 + [2] * 19 + [3] * 2)) == ("straight through", 19)
    assert t53.pause_of(series([1] * 3 + [2] * 10 + [1] + [2] * 12 + [3])) == ("straight through", 12)   # in a row
    assert t53.pause_of(series([1, 3] + [2] * 30)) == ("straight through", 0)       # only the wait before d = 3 counts
    assert t53.pause_of(series([1] * 3 + [2] * 25)) == ("rests there", 25)
    assert t53.pause_of(series([1] * 30)) == ("neither", 0)
    r = t53.report_of(series([1] * 3 + [2] * 20 + [3] * 2, pieces=2, h=144.0))
    assert r == dict(half_d2=3000, half_d3=23000, pause=("pauses", 20), pieces=2, h_per_point=0.5)
    r = t53.report_of(series([1] * 4))
    assert (r["half_d2"], r["half_d3"]) == (None, None)
    assert t53.modal_rung(_row(18, 1.4, 2.5, 0, 0, [0, 100, 100, 88])) == 1                 # the lower rung on a tie


# ---------------------------------------------------------------- the configs and the queue file

def test_the_thirty_configs_and_the_queue_file():
    table = [(length, lam, g) for length in LENGTHS for lam in LAMBDAS for g in COUPLINGS]
    names = ["t53_l%d_lam%d_g%d" % (length, round(lam * 100), round(g * 10)) for length, lam, g in table]
    assert sorted(p.stem for p in (ROOT / "configs").glob("t53_*.json")) == sorted(names) and len(set(names)) == 30
    seeds = []
    for k, (name, (length, lam, g)) in enumerate(zip(names, table)):
        cfg = json.loads((ROOT / "configs" / (name + ".json")).read_text())
        assert cfg["name"] == name and cfg["section"] == "T53"
        assert (cfg["dims"], cfg["lambda"], cfg["g"]) == ([4, 4, length], lam, g)
        assert (cfg["replicas"], cfg["n_sweeps"], cfg["record_every"], cfg["save_adjacency"]) == (8, 200000, 1000, True)
        assert cfg["seed"] == 20265301 + k                          # in order: L, then lambda, then g
        assert set(cfg) == {"_purpose", "name", "section", "dims", "lambda", "g", "replicas", "n_sweeps",
                            "record_every", "seed", "save_adjacency"}
        purpose = cfg["_purpose"]
        assert purpose.startswith("PREREGISTRATION.md T53 (written 2026-10-05, before any run)")
        assert "Not exploratory." in purpose and "exploratory" not in purpose.replace("Not exploratory.", "")
        assert purpose.endswith("Every six-link result carries VISION Update 24's caveat: the reproduction gate is open.")
        assert "4 x 4 x %d torus" % length in purpose and "lambda %.2f" % lam in purpose and "g = %.1f" % g in purpose
        seeds.append(cfg["seed"])
    assert seeds == list(range(20265301, 20265331))
    lines = (ROOT / "cloud" / "queue" / "2026-10-05_t53.txt").read_text().splitlines()
    jobs = [ln.split() for ln in lines if ln.strip() and not ln.startswith("#")]
    assert jobs == [["run_curled_bath_d", name] for name in names]
    assert (ROOT / "scripts" / "run_curled_bath_d.py").exists()
    assert any("T53" in ln and "before any run" in ln for ln in lines if ln.startswith("#"))
