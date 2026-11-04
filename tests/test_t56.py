"""T56's runner (scripts/run_curled_bath_tie_d.py), its configs and its tie table, written before any run
(PREREGISTRATION T56)."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from graphity.cqg_d import is_valid

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


tied = _load("run_curled_bath_tie_d")
untied = _load("run_curled_bath_d")

TINY = dict(name="t56_tiny", section="T56", dims=[4, 6, 6], g=2.5, tie="all_at_the_last", replicas=2, n_sweeps=60,
            record_every=20, seed=19, save_adjacency=True, **{"lambda": 1.2})


def _run(module, tmp_path, out, cfg):
    path = tmp_path / (cfg["name"] + ".json")
    path.write_text(json.dumps(cfg))
    module.main(str(path), str(tmp_path / out))
    return tmp_path / out


def _rows(path):
    return list(csv.DictReader(open(path, newline="", encoding="utf-8")))


def test_the_tie_table_is_the_owners_shape_and_none_is_zero():
    assert np.allclose(tied.table_of("all_at_the_last", 1.25), [0.0, 1.0, 2.0, 0.0])
    assert np.allclose(tied.table_of("all_at_the_last", 1.10), [0.0, 0.4, 0.8, 0.0])
    assert np.allclose(tied.table_of("none", 1.25), [0.0, 0.0, 0.0, 0.0])
    assert np.allclose(tied.table_of([0, 1, -1, 0], 1.25), [0.0, 1.0, -1.0, 0.0])
    with pytest.raises(ValueError):
        tied.table_of([0, 1, 2], 1.25)


def test_same_config_and_seed_give_the_same_file_and_valid_graphs(tmp_path):
    a, b = _run(tied, tmp_path, "a", TINY), _run(tied, tmp_path, "b", TINY)
    assert (a / "t56_tiny.csv").read_text() == (b / "t56_tiny.csv").read_text()
    for rep in range(2):
        ga, gb = np.load(a / "t56_tiny_adj" / ("rep%d.npz" % rep)), np.load(b / "t56_tiny_adj" / ("rep%d.npz" % rep))
        assert np.array_equal(ga["adj"], gb["adj"]) and is_valid(ga["adj"])
    meta = json.loads((a / "t56_tiny.meta.json").read_text())
    assert np.allclose(meta["ftab"], [0.0, 0.8, 1.6, 0.0])


def test_with_tie_none_it_is_t53s_runner_draw_for_draw(tmp_path):
    cfg_t = dict(TINY, tie="none", name="t56_none")
    cfg_u = {k: v for k, v in cfg_t.items() if k != "tie"}
    cfg_u = dict(cfg_u, name="t56_none", section="T53")
    a = _run(tied, tmp_path, "tied", cfg_t)
    b = _run(untied, tmp_path, "untied", cfg_u)
    ra, rb = _rows(a / "t56_none.csv"), _rows(b / "t56_none.csv")
    assert len(ra) == len(rb)
    shared = [k for k in rb[0] if k in ra[0]]
    for x, y in zip(ra, rb):
        assert all(x[k] == y[k] for k in shared), (x, y)
    assert all(float(r["t_tie"]) == 0.0 for r in ra)
    for rep in range(2):
        assert np.array_equal(np.load(a / "t56_none_adj" / ("rep%d.npz" % rep))["adj"],
                              np.load(b / "t56_none_adj" / ("rep%d.npz" % rep))["adj"])


def test_the_file_has_the_columns_the_analyzer_reads_and_the_tie_total(tmp_path):
    out = _run(tied, tmp_path, "a", TINY)
    rows = _rows(out / "t56_tiny.csv")
    assert list(rows[0])[:7] == ["N", "dims", "lam", "g", "tie", "replica", "sweep"]
    for col in ("d0", "d3", "damaged_d", "largest_open_d", "open_regions", "pieces", "h", "t_tie", "final"):
        assert col in rows[0]
    assert sum(1 for r in rows if r["final"] == "True") == 2
    # at sweep 0 the slab is all d = 2 and the tie total is 2a per point
    assert all(int(r["d2"]) == 144 and abs(float(r["t_tie"]) - 2 * 0.8 * 144) < 1e-9 for r in rows if r["sweep"] == "0")


def test_refuses_to_overwrite(tmp_path):
    _run(tied, tmp_path, "a", TINY)
    with pytest.raises(FileExistsError):
        _run(tied, tmp_path, "a", TINY)


def test_the_t56_configs_if_present():
    cfgs = sorted((ROOT / "configs").glob("t56_*.json"))
    if not cfgs:
        pytest.skip("no T56 configs yet")
    names = set()
    for p in cfgs:
        c = json.loads(p.read_text())
        assert c["name"] == p.stem and c["section"] == "T56" and c["name"] not in names
        names.add(c["name"])
        assert c["tie"] in ("all_at_the_last", "none") and c["replicas"] == 8
        assert c["n_sweeps"] % c["record_every"] == 0 and c["save_adjacency"] is True
        assert len(c["dims"]) == 3 and 4 in c["dims"]


# ---------------------------------------------------------------- the analyzer

a56 = _load("analyse_t56")


def _blocks(n, pieces, largest, d3, damaged, sweeps=(1000, 2000)):
    rows = []
    for s in sweeps:
        rows.append(dict(N=str(n), dims="4 8 8", sweep=str(s), pieces=str(pieces), largest_open_d=str(largest),
                         d3=str(d3), damaged_d=str(damaged), h="0", t_tie="0", n_open3=str(d3), open_regions3="1"))
    return rows


def test_geometry_of_reads_slab_and_rod():
    assert a56.geometry_of(dict(dims="4 8 8")) == "slab"
    assert a56.geometry_of(dict(dims="4 4 18")) == "rod"
    assert a56.length_of(dict(dims="4 4 18")) == 18


def test_one_space_needs_one_piece_and_nine_tenths_in_a_majority():
    good = _blocks(256, 1, 240, 250, 0)
    seams = _blocks(256, 2, 200, 250, 0)
    assert a56.one_space([good, good, seams]) == "ONE SPACE"
    assert a56.one_space([good, seams, seams]) == "OPEN WITH SEAMS"
    assert a56.one_space([_blocks(256, 1, 220, 250, 0)]) == "OPEN WITH SEAMS"     # 220 < 0.9 * 256


def test_control_verdict_every_branch():
    assert a56.control_verdict([("OPENS", "ADVANCES")] * 3 + [("OPENS", "OPENS")]) == "THE TIE MADE THE DIFFERENCE"
    assert a56.control_verdict([("OPENS", "OPENS")] * 3 + [("OPENS", "ADVANCES")]) == "NO DIFFERENCE"
    assert a56.control_verdict([("OPENS", "ADVANCES")] * 2 + [("ADVANCES", "OPENS")] * 2) == "MIXED"
    assert a56.control_verdict([]) == "NO CONTROL"


def test_several_verdict_uses_the_two_stage_one_lengths():
    assert a56.several_verdict({8: [1, 1, 1], 12: [2, 3, 2]}, (8, 12)) == "SEVERAL"
    assert a56.several_verdict({8: [2, 2], 12: [2, 2]}, (8, 12)) == "ONE"
    assert a56.several_verdict({8: [1]}, (8, 12)) == "ONE"


def test_held_out_rule_and_score():
    opens = {8: {2.0, 2.5, 3.0}, 12: {2.0, 2.5}}
    damage = {12: {2.0: 0.02, 2.5: 0.2}}
    assert a56.held_out_prediction(opens, damage, (8, 12)) == [2.0]
    assert a56.held_out_score([2.0], {2.0, 2.5}) == "HOLDS"
    assert a56.held_out_score([2.0], {2.5, 3.0}) == "MISSES"
    assert a56.held_out_prediction({}, {}, (8, 12)) == []


def test_summarize_on_a_tiny_run_gives_windows_and_a_control(tmp_path):
    out = _run(tied, tmp_path, "a", TINY)
    cfg_u = dict(TINY, tie="none", name="t56_tiny_untied")
    _run(tied, tmp_path, "a", cfg_u)
    s = a56.summarize(a56.load(tmp_path / "a"))
    assert ("slab", 6, 1.2) in s["windows"]
    assert s["control"] == "MIXED"       # one tied-untied pair at lambda = 1.20: fewer than the three the rule needs either way
    assert "slab" in s["several"] and ("slab", 1.2) in s["held_out"]
