"""T71's held-out size (PREREGISTRATION T71, Amendment 4): the helpers of scripts/t71_heldout_numbers.py, the prediction
table it wrote, and the nine L = 16 configs. Fast: nothing is resampled and nothing is run."""
import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import t71_heldout_numbers as H    # noqa: E402

ARMS = ("model", "layer", "layerref")
LAMS = (110, 125, 140)


@pytest.mark.parametrize("lam, rate, expected", [(1.10, 1 / 64, 52), (1.10, 1 / 16, 13), (1.25, 1 / 64, 128),
                                                  (1.25, 1 / 16, 32), (1.40, 1 / 64, 205), (1.40, 1 / 16, 52)])
def test_the_gate_is_reached_at_the_same_sweep_at_every_size(lam, rate, expected):
    a = 4.0 * (lam - 1.0)
    assert [H.t_gate(L, rate, a * 4 * L * L) for L in (8, 12, 16)] == [expected] * 3


def test_clip_and_band_on_a_tiny_input():
    assert H.clip(1.3, "slab_share") == 1 and H.clip(-0.2, "daughters") == 0 and H.clip(7.0, "period") == 7.0
    lo, hi = H.band(10.0, [-1.0, 0.0, 0.0, 2.0], 0.0)
    assert (lo, hi) == (9.0, 12.0)


def test_write_prediction_round_trips(tmp_path):
    rows = [dict(cell="t71_model_lam110_l16 r=1/16", statistic="period, median", rule="A", predicted=5.0, low=4.0,
                 high=6.0, scored="yes", units="sweeps", note="x"),
            dict(cell="c", statistic="majority word", rule="word", predicted="CYCLES", scored="yes")]
    path = tmp_path / "p.csv"
    H.write_prediction(path, rows)
    back = list(csv.DictReader(open(path, newline="")))
    assert list(back[0]) == H.COLUMNS
    assert (back[0]["predicted"], back[0]["low"], back[0]["high"]) == ("5", "4", "6")
    assert back[1]["predicted"] == "CYCLES" and back[1]["low"] == ""
    assert H.cell_name(("layer", "stores", 1.25, 0.015625)) == "t71_layerref_lam125_l16 r=1/64"


def test_the_committed_prediction_table():
    rows = list(csv.DictReader(open(ROOT / "configs" / "t71_heldout_prediction.csv", newline="")))
    assert rows and list(rows[0]) == H.COLUMNS
    cells = {r["cell"] for r in rows}
    assert len(cells) == 18 and not any("control" in c for c in cells)       # the control is left out (option (c))
    words = [r for r in rows if r["statistic"] == "majority word"]
    assert len(words) == 18 and all(r["scored"] == "yes" for r in words)
    assert {r["predicted"] for r in words} == {"CYCLES", "OSCILLATES"}
    for r in rows:
        assert r["scored"] in ("yes", "no") and r["rule"] in ("A", "B", "C", "D", "word", "reported")
        if r["low"] and r["high"]:
            assert float(r["low"]) <= float(r["high"])
            if r["predicted"] not in ("",) and r["rule"] in ("A", "B", "C", "D"):
                assert float(r["low"]) - 1e-9 <= float(r["predicted"]) <= float(r["high"]) + 1e-9, r
    # rule A and D numbers that can fail, plus rule B's median and latest replica in 18 cells: the 81 of section 3
    sampled = [r for r in rows if r["scored"] == "yes" and r["rule"] in ("A", "D")]
    rule_b = [r for r in rows if r["rule"] == "B" and r["statistic"] != "first split, earliest replica"]
    assert (len(sampled), len(rule_b)) == (45, 36)


def test_the_held_out_configs_change_only_name_sizes_seed_purpose_and_written():
    used = {}
    for p in (ROOT / "configs").glob("*.json"):
        if p.name.endswith("_l16.json") and p.name.startswith("t71_"):
            continue
        try:
            s = json.loads(p.read_text(encoding="utf-8")).get("seed")
        except (ValueError, AttributeError):
            continue
        used.setdefault(s, p.name)
    seeds = []
    for arm in ARMS:
        for lam in LAMS:
            old = json.loads((ROOT / "configs" / ("t71_%s_lam%d.json" % (arm, lam))).read_text())
            new = json.loads((ROOT / "configs" / ("t71_%s_lam%d_l16.json" % (arm, lam))).read_text())
            assert list(new) == list(old)
            assert sorted(k for k in old if old[k] != new[k]) == ["_purpose", "name", "seed", "sizes", "written"]
            assert new["name"] == old["name"] + "_l16" and new["sizes"] == [16] and new["seed"] == old["seed"] + 12
            assert new["seed"] not in used, (new["seed"], used.get(new["seed"]))
            seeds.append(new["seed"])
    assert sorted(seeds) == [20267113, 20267114, 20267115, 20267117, 20267118, 20267119, 20267121, 20267122, 20267123]
    manifest = (ROOT / "cloud" / "queue" / "2026-10-25_t71_heldout.txt").read_text().splitlines()
    jobs = [ln.split() for ln in manifest if ln.strip() and not ln.lstrip().startswith("#")]
    assert jobs == [["run_layer_feed", "t71_%s_lam%d_l16" % (arm, lam)] for lam in LAMS for arm in ARMS]


def test_the_held_out_scorer_rules():
    import t71_heldout_score as S
    word = dict(rule="word", predicted="CYCLES", low="", high="")
    assert S.holds(word, "CYCLES") is True and S.holds(word, "ONE BURP") is False
    num = dict(rule="A", predicted="5", low="4", high="6")
    assert S.holds(num, 6.0) is True and S.holds(num, 6.1) is False and S.holds(num, None) is None
    exact = dict(rule="C", predicted="1", low="1", high="1")
    assert S.holds(exact, 1.0 + 1e-12) is True and S.holds(exact, 0.99) is False
    upper = dict(rule="B", predicted="", low="52", high="65")
    assert S.holds(upper, 65) is True and S.holds(upper, 66) is False
    reps = [dict(word="CYCLES", first_split=13, spent_first=False, clean_total=2, period=None),
            dict(word="CYCLES", first_split=21, spent_first=False, clean_total=0, period=5.0),
            dict(word="ONE BURP", first_split=None, spent_first=True, clean_total=1, period=4.0)]
    assert S.value("majority word", reps) == "CYCLES"
    assert (S.value("first split, earliest replica", reps), S.value("first split, latest replica", reps)) == (13, 21)
    assert S.value("period, median", reps) == 4.5 and S.value("SPENT FIRST count", reps) == 1
    assert S.value("clean moves in all", reps) == 3 and S.value("replicas with a clean move", reps) == 2
