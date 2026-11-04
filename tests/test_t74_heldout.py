"""T74's held-out size (PREREGISTRATION T74, Amendment 6): the L = 16 configs from `make_t74_configs.py --held-out`,
the committed prediction file, and the scorer `t74_heldout_score.py`. Fast: nothing is run."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import make_t74_configs as M     # noqa: E402
import t74_heldout_score as S    # noqa: E402

PRED = json.loads((ROOT / "configs" / "t74_heldout_prediction.json").read_text())


def test_held_out_configs_differ_from_stage_1_only_where_they_must():
    stage = {c["name"].replace("_L12_", "_L16_"): c for c in M.cells() if c["L"] == 12}
    held = list(M.held_out_cells())
    assert len(held) == 54
    assert [c["seed"] for c in held] == list(range(20267901, 20267955))
    stage_seeds = {c["seed"] for c in M.cells()}
    assert not stage_seeds & {c["seed"] for c in held}
    for c in held:
        s = stage[c["name"]]
        assert c["L"] == 16
        diff = {k for k in c if c[k] != s[k]}
        assert diff <= {"name", "L", "seed", "_purpose", "written"}


def test_held_out_configs_match_the_committed_files():
    for c in M.held_out_cells():
        p = ROOT / "configs" / (c["name"] + ".json")
        assert json.loads(p.read_text()) == c


def test_held_out_refuses_without_the_prediction_file(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "PREDICTION", tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError):
        M.main(tmp_path, held_out=True)
    assert not list(tmp_path.iterdir())


def test_the_prediction_file_is_complete():
    names = {c["name"] for c in M.held_out_cells()}
    assert len(PRED["periods"]) == 18
    assert all(p["low"] <= p["fit"] <= p["high"] for p in PRED["periods"])
    fed = set(PRED["fed"]["must_burps"]) | set(PRED["fed"]["six"])
    assert len(PRED["fed"]["must_burps"]) == 48 and len(PRED["fed"]["six"]) == 6 and fed == names
    unfed = set(PRED["unfed"]["must_repeat"]) | set(PRED["unfed"]["others"])
    assert len(PRED["unfed"]["must_repeat"]) == 33 and len(PRED["unfed"]["others"]) == 21 and unfed == names
    assert PRED["unfed"]["others_range"] == [14, 21]
    assert len(PRED["timing"]) == 12


def fake_cells(period_shift=0.0, drop=None, fed_word=None):
    """A cell per L = 16 config whose reads repeat the prediction exactly."""
    fit = {(p["lam"], p["feed"], p["cap"]): p["fit"] for p in PRED["periods"]}
    words = dict(PRED["unfed"]["must_repeat"], **PRED["unfed"]["others"])
    cells = {}
    for c in M.held_out_cells():
        n = c["name"]
        if n == drop:
            continue
        lam, feed, cap = S.setting(n)
        f = fit[(lam, feed, cap)] + (period_shift if (lam, feed, cap) == ("1.25", "steady", "c2R") else 0.0)
        cells[n] = dict(n=16, fed_word=fed_word if fed_word and n == fed_word[0] else "BURPS",
                        unfed_word=words[n], periods=[f] * 10)
        if fed_word and n == fed_word[0]:
            cells[n]["fed_word"] = fed_word[1]
    return cells


def test_score_repeats_when_everything_repeats():
    # constant periods give cv 0, so the timing words are not tested here, only that the read is complete
    out = S.score(PRED, fake_cells())
    assert out["size"] == "HOLDS AT L = 16" and not out["misses"] and not out["not_read"]
    assert out["fed"][0] == "REPEATS STAGE 1" and out["unfed"][0] == "REPEATS STAGE 1"


def test_score_names_a_period_outside_its_range():
    out = S.score(PRED, fake_cells(period_shift=1.0))
    assert out["size"] == "MISSES AT L = 16" and out["misses"] == [("1.25", "steady", "c2R")]


def test_score_does_not_read_a_missing_cell():
    name = "t74_lam125_L16_steady_c2R_cold"
    out = S.score(PRED, fake_cells(drop=name))
    assert out["size"] == "NOT READ" and out["not_read"] == [name]


def test_score_fed_changes_on_never_burps():
    six = PRED["fed"]["six"][0]
    assert S.score(PRED, fake_cells(fed_word=(six, "MIXED")))["fed"][0] == "REPEATS STAGE 1"
    assert S.score(PRED, fake_cells(fed_word=(six, "NEVER BURPS")))["fed"][0] == "CHANGES"
    must = PRED["fed"]["must_burps"][0]
    assert S.score(PRED, fake_cells(fed_word=(must, "MIXED")))["fed"][0] == "CHANGES"
