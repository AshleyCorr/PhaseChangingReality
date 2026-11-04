"""T62 (PREREGISTRATION.md): the tube six around under a price on loops of six; its runner and its reading.
Written before any run. The analyzer is tested on synthetic rows only."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import analyse_t62 as a                                           # noqa: E402
import make_t62_configs                                           # noqa: E402
import run_burp_heat                                              # noqa: E402
import run_sixloop_widen as r                                     # noqa: E402
from graphity.cqg import NO_CAP, is_valid, torus                  # noqa: E402
from graphity.sixloop import excess6                              # noqa: E402
from read_endpoint_width import is_flat, twisted_tube, width      # noqa: E402

TINY = dict(name="t62_tiny", section="T62", **{"lambda": 1.25}, mu=0.25, g=2.5, lengths=[12], replicas=2,
            hold_sweeps_at_192=600, cool_sweeps_at_192=400, cool_to=0.5, cold_sweeps=100, cold_g=0.05,
            save_adjacency=True, seed=7)


def test_the_start_is_the_tube_six_around_and_the_balls_are_right():
    for length in (12, 24):
        adj, side = r.six_around(length)
        assert np.array_equal(adj, twisted_tube(length, 2)) and is_flat(adj) and width(adj) == 6
        assert is_valid(adj, NO_CAP) and excess6(adj, side) == 15 * 4 * length
        assert {int(v) for v in side} == {x * length + y for x in range(4) for y in range(length) if (x + y) % 2 == 0}
    near = r.balls(r.six_around(24)[0])
    assert set(map(tuple, near)) == {(22, 30, 38)}                             # six around: no point sees a sheet
    wide = r.balls(torus(12, 16, NO_CAP)[0])
    assert set(map(tuple, wide)) == {(25, 41, 61)}                             # an endless sheet's own numbers
    for adj in (r.six_around(24)[0], torus(8, 12, NO_CAP)[0], torus(6, 16, NO_CAP)[0]):
        near = r.balls(adj)
        expected = run_burp_heat.room(adj)                                     # T61's reader, written separately
        assert (near[:, 0].mean(), near[:, 1].mean(), near[:, 2].mean()) == pytest.approx(expected[:3])
        assert float((near[:, 0] == 25).mean()) == pytest.approx(expected[3])


def test_a_replica_is_reproducible_keeps_its_books_and_the_control_stays_put():
    one = r.replica(12, 1.25, 0.25, 2.5, 12345, 600, 400, 0.5, 100, 0.05)
    two = r.replica(12, 1.25, 0.25, 2.5, 12345, 600, 400, 0.5, 100, 0.05)
    assert one[0] == two[0] and np.array_equal(one[1], two[1])
    f = one[0]
    assert f["books"] and f["valid"] and f["e6_start"] == 720 and sum(f["d%d" % j] for j in range(7)) == 48
    assert f["h_end"] == 16.0 * (48 - f["s_end"]) + 5.0 * f["x_end"]
    other = r.replica(12, 1.25, 0.25, 2.5, 54321, 600, 400, 0.5, 100, 0.05)
    assert other[0] != f or not np.array_equal(other[1], one[1])
    still = r.replica(12, 1.25, 0.0, 1.5, 99, 600, 400, 0.5, 100, 0.05)[0]     # no price: six around is stuck
    assert still["flat"] and still["sees3"] == 0.0 and still["left_sweep"] == -1 and still["e6_end"] == 720


def test_the_runner_writes_the_same_file_twice_saves_graphs_and_refuses_to_overwrite(tmp_path):
    cfg = tmp_path / "t62_tiny.json"
    cfg.write_text(json.dumps(TINY))
    for name in ("a", "b"):
        r.main(str(cfg), str(tmp_path / name))
    text = (tmp_path / "a" / "t62_tiny.csv").read_text()
    assert text == (tmp_path / "b" / "t62_tiny.csv").read_text()
    rows = list(csv.DictReader(text.splitlines()))
    assert len(rows) == 2 and len({row["seed"] for row in rows}) == 2 and {row["hold_sweeps"] for row in rows} == {"600"}
    assert sorted(p.name for p in (tmp_path / "a" / "t62_tiny_adj").iterdir()) == ["N48_rep0.npz", "N48_rep1.npz"]
    meta = json.loads((tmp_path / "a" / "t62_tiny.meta.json").read_text())
    assert meta["config"] == TINY and "git_commit" in meta
    with pytest.raises(BaseException):
        r.main(str(cfg), str(tmp_path / "a"))
    assert (tmp_path / "a" / "t62_tiny.csv").read_text() == text


def test_configs_and_the_guard_on_the_held_out_size(tmp_path, monkeypatch):
    stage1 = make_t62_configs.configs()
    assert len(stage1) == 20 and stage1[0]["name"] == "t62_mu000_g150_n96" and stage1[-1]["name"] == "t62_mu030_g200_n192"
    assert all(c["replicas"] == 24 and c["lambda"] == 1.25 and len(c["lengths"]) == 1 for c in stage1)
    assert {(c["mu"], c["g"]) for c in stage1} == {(mu, g) for mu in a.MUS for g in a.GS}
    held = make_t62_configs.configs(held_out=True)
    assert len(held) == 10 and all(c["lengths"] == [96] for c in held)
    monkeypatch.setattr(make_t62_configs, "PREDICTION", tmp_path / "none.json")
    with pytest.raises(SystemExit):
        make_t62_configs.main("--held-out")
    seeds = {r.seed_of(20266200, n, 1.25, mu, g, rep) for n in (96, 192) for mu in a.MUS for g in a.GS for rep in range(3)}
    assert len(seeds) == 60


FIELDS = ["N", "mu", "g", "replica", "sees3", "e6_start", "e6_end", "h_end", "phi_min", "left_sweep", "valid", "books",
          "flat"]


def write_cell(folder, mu, g, n, sees3=0.0, valid=True, books=True, replicas=range(24)):
    path = folder / (make_t62_configs.name_of(mu, g, n) + ".csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for rep in replicas:
            w.writerow(dict(N=n, mu=mu, g=g, replica=rep, sees3=sees3, e6_start=15 * n, e6_end=0, h_end=16.0,
                            phi_min=0.95, left_sweep=500 if sees3 else -1, valid=valid, books=books, flat=False))


def write_design(folder, sizes, wide_at=(), skip=()):
    for n in sizes:
        for mu in a.MUS:
            for g in a.GS:
                if (mu, g, n) not in skip:
                    write_cell(folder, mu, g, n, sees3=0.8 if (mu, g) in wide_at else 0.0)


def test_outcomes_majorities_and_verdict_words():
    row = lambda s: dict(sees3=str(s))                                         # noqa: E731
    assert [a.outcome(row(s)) for s in (0.0, 0.049, 0.05, 0.49, 0.5, 1.0)] == [
        "TUBE", "TUBE", "PATCHWORK", "PATCHWORK", "WIDE", "WIDE"]
    assert a.verdict(["TUBE", "TUBE"]) == "STAYS SIX AROUND" and a.verdict(["TUBE", "WIDE"]) == "WIDENS"
    assert a.verdict(["PATCHWORK", "TUBE"]) == "PATCHWORK" and a.verdict(["TUBE", "MIXED"]) == "MIXED"
    assert a.verdict([]) == "NOT READ"


def test_a_complete_clean_stage_1_is_read_and_gives_ten_numbers(tmp_path, capsys, monkeypatch):
    write_design(tmp_path, a.STAGE1, wide_at={(0.20, 2.0), (0.25, 1.5), (0.25, 2.0)})
    assert a.design_problems(a.load(tmp_path), a.STAGE1) == []
    assert a.main(str(tmp_path)) == 0
    out = capsys.readouterr().out
    assert out.count("CONTROL") == 4 and out.count("holds") == 4 and "NOT READ" not in out and "not read" not in out
    assert "VERDICT mu=0.20 N=96: WIDENS" in out and "VERDICT mu=0.10 N=192: STAYS SIX AROUND" in out
    monkeypatch.setattr(a, "PREDICTION", tmp_path / "prediction.json")
    assert a.main(str(tmp_path), "--write-prediction") == 0
    fixed = json.loads((tmp_path / "prediction.json").read_text())
    assert len(fixed) == 10 and fixed["0.20|2.00"]["wide"] == [1.0, 0.75, 1.0] and fixed["0.20|1.50"]["tube"] == [1.0, 0.75, 1.0]


def test_a_cell_is_not_read_when_the_control_at_its_coupling_moves(tmp_path, capsys):
    write_design(tmp_path, a.STAGE1, wide_at={(0.20, 2.0), (0.25, 2.0)}, skip={(0.0, 2.0, 96)})
    write_cell(tmp_path, 0.0, 2.0, 96, sees3=0.3)                              # the bath alone moves the tube at g = 2
    a.main(str(tmp_path))
    out = capsys.readouterr().out
    assert "CONTROL g=2.00 N=96: FAILS" in out and "CONTROL g=1.50 N=96: holds" in out
    assert "VERDICT mu=0.20 N=96: STAYS SIX AROUND | not read: g=2.00" in out   # only g = 1.5 is read there
    assert "VERDICT mu=0.20 N=192: WIDENS" in out


def test_no_word_on_incomplete_or_failed_data(tmp_path, capsys, monkeypatch):
    bad, short = (0.25, 1.5, 96), (0.10, 2.0, 192)
    write_design(tmp_path, a.STAGE1, skip={bad, short})
    write_cell(tmp_path, *bad, sees3=0.9, valid=False)                         # 24 graphs that would read WIDE, none valid
    write_cell(tmp_path, *short, replicas=range(23))
    a.main(str(tmp_path))
    out = capsys.readouterr().out
    assert "gate FAIL" in out and "WIDENS" not in out
    assert "VERDICT mu=0.25 N=96: STAYS SIX AROUND | not read: g=1.50 (its gates fail)" in out
    assert "not the replicas 0 to 23 once each" in out
    monkeypatch.setattr(a, "PREDICTION", tmp_path / "prediction.json")
    with pytest.raises(SystemExit):
        a.main(str(tmp_path), "--write-prediction")
    assert not (tmp_path / "prediction.json").exists()


def test_the_held_out_size_is_scored_only_when_complete(tmp_path, capsys, monkeypatch):
    write_design(tmp_path, a.STAGE1, wide_at={(0.25, 2.0)})
    monkeypatch.setattr(a, "PREDICTION", tmp_path / "prediction.json")
    a.main(str(tmp_path), "--write-prediction")
    write_cell(tmp_path, 0.25, 2.0, a.HELD_OUT, sees3=0.8)                     # one held-out cell, and no control
    capsys.readouterr()
    assert a.main(str(tmp_path)) == 1
    out = capsys.readouterr().out
    assert "AS PREDICTED" not in out and out.count("HELD-OUT") == 5 and "HELD-OUT mu=0.25 N=384: NOT READ" in out
    for mu in a.MUS:
        for g in a.GS:
            if (mu, g) != (0.25, 2.0):
                write_cell(tmp_path, mu, g, a.HELD_OUT, sees3=0.8 if (mu, g) == (0.30, 1.5) else 0.0)
    assert a.main(str(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "HELD-OUT mu=0.25 N=384: AS PREDICTED | verdict there: WIDENS" in out
    assert "HELD-OUT mu=0.30 N=384: NOT AS PREDICTED AT g=1.50" in out
