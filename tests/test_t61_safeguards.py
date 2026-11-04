"""T61's analyzer gives no word on incomplete or failed data (amendment of 11 October 2026, from the outside review's
three synthetic cases: a cell of invalid graphs, a stage 1 with settings missing, a held-out size with one cell).
Synthetic rows only; no result of T61 is opened."""
import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyse_t61 as a                                           # noqa: E402

FIELDS = ["N", "lam", "stores", "stores_per_point", "replica", "d1", "sees3", "flat", "width", "drift", "valid",
          "h_end", "near5"]


def write_cell(folder, lam, n, ratio, sees3=0.0, valid=True, replicas=range(24), name=None):
    """One cell's rows, appended to the stage-1 file of (lambda, N) or written to a held-out file of its own."""
    stores = int(round(ratio * n))
    tag = "t61_lam%d_n%d" % (round(100 * lam), n)
    path = folder / ((name or (tag + "_c%d" % stores if n == a.HELD_OUT else tag)) + ".csv")
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for rep in replicas:
            w.writerow(dict(N=n, lam=lam, stores=stores, stores_per_point=ratio, replica=rep, d1=0, sees3=sees3,
                            flat=False, width="", drift=0.0, valid=valid, h_end=14.0, near5=38.0))


def write_design(folder, sizes, skip=(), **kw):
    for lam in a.LAMBDAS:
        for n in sizes:
            for ratio in a.RATIOS:
                if (lam, n, ratio) not in skip:
                    write_cell(folder, lam, n, ratio, **kw)


def test_a_complete_clean_stage_1_is_read_and_gives_ten_numbers(tmp_path, capsys, monkeypatch):
    write_design(tmp_path, a.STAGE1)
    assert a.design_problems(a.load(tmp_path), a.STAGE1) == []
    assert a.main(str(tmp_path)) == 0
    out = capsys.readouterr().out
    assert out.count("STAYS A TUBE") == 4 and "NOT READ" not in out
    monkeypatch.setattr(a, "PREDICTION", tmp_path / "prediction.json")
    assert a.main(str(tmp_path), "--write-prediction") == 0
    assert len(json.loads((tmp_path / "prediction.json").read_text())) == 10


def test_a_cell_of_invalid_graphs_gets_no_word_and_no_prediction(tmp_path, capsys, monkeypatch):
    bad = (1.25, 96, 2.0)
    write_design(tmp_path, a.STAGE1, skip=(bad,))
    write_cell(tmp_path, *bad, sees3=0.9, valid=False)                 # 24 graphs that would read WIDE, none valid
    assert a.main(str(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "gate FAIL" in out and "WIDENS" not in out
    assert "VERDICT lambda=1.25 N=96: NOT READ (gate fails with 2 stores per point)" in out
    assert "VERDICT lambda=1.25 N=192: STAYS A TUBE" in out          # the other size is untouched
    monkeypatch.setattr(a, "PREDICTION", tmp_path / "prediction.json")
    with pytest.raises(SystemExit):
        a.main(str(tmp_path), "--write-prediction")
    assert not (tmp_path / "prediction.json").exists()


def test_a_stage_1_with_settings_missing_gets_no_word_and_no_prediction(tmp_path, capsys, monkeypatch):
    for n in a.STAGE1:
        write_cell(tmp_path, 1.25, n, 0.25)                           # one amount of room, one lambda
    assert a.main(str(tmp_path)) == 0
    out = capsys.readouterr().out
    assert out.count("NOT READ (missing cell") == 4
    monkeypatch.setattr(a, "PREDICTION", tmp_path / "prediction.json")
    with pytest.raises(SystemExit):
        a.main(str(tmp_path), "--write-prediction")
    assert not (tmp_path / "prediction.json").exists()


def test_replicas_must_be_the_24_once_each_and_numbers_finite(tmp_path, capsys):
    short, twice, blank = (1.25, 96, 1.0), (1.25, 192, 0.5), (1.40, 96, 0.25)
    write_design(tmp_path, a.STAGE1, skip=(short, twice, blank))
    write_cell(tmp_path, *short, replicas=range(23))
    write_cell(tmp_path, *twice, replicas=list(range(23)) + [0])
    write_cell(tmp_path, *blank, sees3="nan")
    problems = a.design_problems(a.load(tmp_path), a.STAGE1)
    assert len(problems) == 3 and sum("not the replicas 0 to 23 once each" in p for p in problems) == 2
    assert sum("missing or not finite" in p for p in problems) == 1
    a.main(str(tmp_path))
    out = capsys.readouterr().out
    assert "UNREADABLE" in out and out.count("NOT READ") == 3 and "VERDICT lambda=1.40 N=192: STAYS A TUBE" in out


def test_a_cell_outside_the_design_is_a_problem(tmp_path):
    write_design(tmp_path, a.STAGE1)
    write_cell(tmp_path, 1.25, 96, 0.75)
    assert a.design_problems(a.load(tmp_path), a.STAGE1) == [
        "a cell outside the registered design: lambda=1.25 N=96 stores per point 0.75"]
    write_cell(tmp_path, 1.10, 96, 0.25, name="t61_lam110_n96")
    assert a.strays(a.load(tmp_path)) == ["a lambda outside the registered design: 1.10"]


def test_the_held_out_size_is_scored_only_when_all_five_cells_of_a_lambda_are_complete(tmp_path, capsys, monkeypatch):
    write_design(tmp_path, a.STAGE1)
    monkeypatch.setattr(a, "PREDICTION", tmp_path / "prediction.json")
    a.main(str(tmp_path), "--write-prediction")
    write_cell(tmp_path, 1.25, a.HELD_OUT, 0.25)                      # one held-out cell: the review's third case
    capsys.readouterr()
    assert a.main(str(tmp_path)) == 1
    out = capsys.readouterr().out
    assert "AS PREDICTED" not in out and out.count("HELD-OUT") == 2 and out.count("NOT READ (missing cell") == 2
    for ratio in a.RATIOS[:-1]:                                        # complete lambda = 1.25; 1.40 stays empty
        write_cell(tmp_path, 1.25, a.HELD_OUT, ratio)
    assert a.main(str(tmp_path)) == 1
    out = capsys.readouterr().out
    assert "HELD-OUT lambda=1.25 N=384: AS PREDICTED | verdict there: STAYS A TUBE" in out
    assert "HELD-OUT lambda=1.40 N=384: NOT READ (missing cell" in out
    for ratio in a.RATIOS[:-1]:
        write_cell(tmp_path, 1.40, a.HELD_OUT, ratio)
    write_cell(tmp_path, 1.40, a.HELD_OUT, 0.25, valid=False)         # a failed gate: a miss, and no verdict word
    assert a.main(str(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "HELD-OUT lambda=1.40 N=384: NOT AS PREDICTED AT 0.25 (gate) | verdict there: NOT READ (a gate fails)" in out


def test_the_committed_prediction_has_all_ten_numbers():
    fixed = json.loads(a.PREDICTION.read_text(encoding="utf-8"))
    assert sorted(fixed) == sorted("%.2f|%g" % (lam, ratio) for lam in a.LAMBDAS for ratio in a.RATIOS)
