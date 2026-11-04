"""T61 (PREREGISTRATION.md): the sealed, drained, cooled tube and its reading. Written before any run."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import analyse_t61 as a                                           # noqa: E402
import make_t61_configs                                           # noqa: E402
import run_burp_heat as r                                         # noqa: E402
from graphity.cqg import torus                                    # noqa: E402
from read_endpoint_width import twisted_tube                      # noqa: E402

TINY = dict(name="t61_tiny", section="T61", **{"lambda": 1.25}, lengths=[12], stores_per_point=[2, 0.25], replicas=2,
            hold_sweeps_at_192=1500, leak=0.05, drain_stop=0.5, cold_sweeps=300, cold_g=0.05, save_adjacency=True,
            seed=7)


def test_room_on_tori_whose_answer_is_known():
    assert r.room(twisted_tube(24, 2)) == (22.0, 30.0, 38.0, 0.0)          # flat and six around: no point sees a sheet
    assert r.room(torus(12, 16)[0]) == (25.0, 41.0, 61.0, 1.0)             # wide: every point does
    assert r.room(torus(8, 12)[0])[3] == 1.0 and r.room(torus(6, 16)[0])[3] == 0.0
    assert r.wall(1.25) == 12.0 and r.wall(1.40) == 8.0


def test_a_replica_is_reproducible_and_its_energy_books_balance():
    one = r.replica(12, 24, 1.25, 12345, 1500, 0.05, 0.5, 300, 0.05)
    two = r.replica(12, 24, 1.25, 12345, 1500, 0.05, 0.5, 300, 0.05)
    assert one[0] == two[0] and np.array_equal(one[1], two[1])
    fields = one[0]
    assert fields["drift"] < 1e-9 and fields["valid"] and fields["spark"] == 12.0
    assert fields["left_in_stores"] <= 0.5 and fields["drained"] > 0
    assert sum(fields["d%d" % j] for j in range(7)) == 48
    other = r.replica(12, 24, 1.25, 54321, 1500, 0.05, 0.5, 300, 0.05)
    assert other[0] != fields or not np.array_equal(other[1], one[1])


def test_the_runner_writes_the_same_file_twice_saves_graphs_and_refuses_to_overwrite(tmp_path):
    cfg = tmp_path / "t61_tiny.json"
    cfg.write_text(json.dumps(TINY))
    for name in ("a", "b"):
        r.main(str(cfg), str(tmp_path / name))
    text = (tmp_path / "a" / "t61_tiny.csv").read_text()
    assert text == (tmp_path / "b" / "t61_tiny.csv").read_text()
    rows = list(csv.DictReader(text.splitlines()))
    assert len(rows) == 4 and {row["stores"] for row in rows} == {"96", "12"}
    assert len({row["seed"] for row in rows}) == 4
    saved = sorted(p.name for p in (tmp_path / "a" / "t61_tiny_adj").iterdir())
    assert saved == ["N48_C12_rep0.npz", "N48_C12_rep1.npz", "N48_C96_rep0.npz", "N48_C96_rep1.npz"]
    meta = json.loads((tmp_path / "a" / "t61_tiny.meta.json").read_text())
    assert meta["config"] == TINY and "git_commit" in meta
    with pytest.raises(BaseException):
        r.main(str(cfg), str(tmp_path / "a"))
    assert (tmp_path / "a" / "t61_tiny.csv").read_text() == text


def row(sees3, n=96, d1=0, flat=False, width="", drift=0.0, valid=True, h_end=14.0, near5=38.0):
    return dict(N=str(n), d1=str(d1), sees3=str(sees3), flat=str(flat), width=str(width), drift=str(drift),
                valid=str(valid), h_end=str(h_end), near5=str(near5))


def test_outcomes_majorities_and_verdict_words():
    assert [a.outcome(row(s)) for s in (0.0, 0.049, 0.05, 0.49, 0.5, 1.0)] == [
        "TUBE", "TUBE", "PATCHWORK", "PATCHWORK", "WIDE", "WIDE"]
    assert a.outcome(row(0.0, d1=25)) == "NOT CONVERTED" and a.outcome(row(0.0, d1=24)) == "TUBE"
    c = a.cell([row(0.0, flat=True, width=6)] * 5 + [row(0.3)] * 2 + [row(0.9)] * 1)
    assert c["majority"] == "TUBE" and c["converted"] == 8 and c["gate"] and c["flat"] == 5 and c["widths"] == {6: 5}
    assert c["wide"] == 0.125 and c["tube"] == 0.625
    assert a.cell([row(0.0)] * 4 + [row(0.3)] * 4)["majority"] == "MIXED"
    assert not a.cell([row(0.0)] * 8 + [row(0.0, d1=96)])["gate"]              # one in nine did not convert
    assert not a.cell([row(0.0, drift=1e-3)] * 8)["gate"] and not a.cell([row(0.0, valid=False)] * 8)["gate"]
    assert a.verdict(["TUBE", "TUBE"]) == "STAYS A TUBE" and a.verdict(["TUBE", "PATCHWORK"]) == "PATCHWORK"
    assert a.verdict(["PATCHWORK", "WIDE"]) == "WIDENS" and a.verdict(["TUBE", "MIXED"]) == "MIXED"


def test_the_held_out_numbers_come_from_the_two_smaller_sizes_only():
    cells = {(1.25, 96, 0.25): dict(wide=0.2, tube=0.5, gate=True), (1.25, 192, 0.25): dict(wide=0.4, tube=0.1, gate=True),
             (1.25, 384, 0.25): dict(wide=1.0, tube=0.0, gate=True), (1.25, 96, 2.0): dict(wide=0.0, tube=1.0, gate=True)}
    p = a.predict(cells)
    assert list(p) == ["1.25|0.25"]                                            # 2.0 has one stage-1 size only
    assert p["1.25|0.25"]["wide"] == pytest.approx([0.3, 0.05, 0.55])
    assert p["1.25|0.25"]["tube"] == pytest.approx([0.3, 0.05, 0.55])
    assert a.held_out_misses(dict(wide=0.5, tube=0.1, gate=True), p["1.25|0.25"]) == []
    assert len(a.held_out_misses(dict(wide=0.6, tube=0.0, gate=False), p["1.25|0.25"])) == 3


def test_configs_and_the_guard_on_the_held_out_size(tmp_path, monkeypatch):
    stage1 = make_t61_configs.configs()
    assert [c["name"] for c in stage1] == ["t61_lam125_n96", "t61_lam125_n192", "t61_lam140_n96", "t61_lam140_n192"]
    assert all(c["replicas"] == 24 and c["stores_per_point"] == [2, 1, 0.5, 0.333333, 0.25] for c in stage1)
    assert [int(round(x * 192)) for x in stage1[1]["stores_per_point"]] == [384, 192, 96, 64, 48]
    held = make_t61_configs.configs(held_out=True)
    assert [c["name"] for c in held][:5] == ["t61_lam125_n384_c768", "t61_lam125_n384_c384", "t61_lam125_n384_c192",
                                             "t61_lam125_n384_c128", "t61_lam125_n384_c96"]
    assert len(held) == 10 and all(len(c["stores_per_point"]) == 1 and c["lengths"] == [96] for c in held)
    monkeypatch.setattr(make_t61_configs, "PREDICTION", tmp_path / "none.json")
    with pytest.raises(SystemExit):
        make_t61_configs.main("--held-out")
    seeds = {r.seed_of(20266100, n, lam, c, rep) for n in (96, 192) for lam in (1.25, 1.4) for c in (24, 48)
             for rep in range(3)}
    assert len(seeds) == 24
