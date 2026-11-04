"""T64 (PREREGISTRATION.md): T63's design on fresh seeds, with the control's rule rewritten. The configs, and the
reading on synthetic rows. Written before any run of T64."""
import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import analyse_t63 as t63                                         # noqa: E402
import analyse_t64 as a                                           # noqa: E402
import make_t63_configs                                           # noqa: E402
import make_t64_configs                                           # noqa: E402
import run_sixloop_widen as r                                     # noqa: E402

FIELDS = ["N", "mu", "g", "replica", "start", "sees3", "x_end", "e6_start", "e6_end", "h_end", "phi_min", "curl_sweep",
          "valid", "books", "flat"]


def write_cell(folder, mu, n, sees3=0.0, leftovers=0, leftover_sees3=0.4, flat=None, valid=True, start="tube"):
    """24 replicas. The first `leftovers` rest on a state with a leftover: not flat, a little curl left, their own
    sees3. The others have `sees3` and are exactly flat if `flat` (by default: when they are six around)."""
    flat = (sees3 < 0.05) if flat is None else flat
    path = folder / (make_t64_configs.name_of(mu, n) + ".csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for rep in range(24):
            left = rep < leftovers
            w.writerow(dict(N=n, mu=mu, g=1.5, replica=rep, start=start, sees3=leftover_sees3 if left else sees3,
                            x_end=10 if left else 0, e6_start=18 * n, e6_end=0, h_end=34.0 if left else (0.0 if flat else 16.0),
                            phi_min=0.93, curl_sweep=-1 if left else 1500, valid=valid, books=True,
                            flat=(not left) and flat))


def write_design(folder, sizes, wide_at=(), leftovers=5, skip=()):
    for n in sizes:
        for mu in a.MUS:
            if (mu, n) not in skip:
                write_cell(folder, mu, n, sees3=0.8 if mu in wide_at else 0.0, leftovers=leftovers if mu == 0.0 else 0)


def test_the_configs_are_t63s_design_on_fresh_seeds(tmp_path, monkeypatch):
    new, old = make_t64_configs.configs(), make_t63_configs.configs()
    assert [c["name"] for c in new][:5] == ["t64_mu0000_n96", "t64_mu0025_n96", "t64_mu0050_n96", "t64_mu0100_n96",
                                            "t64_mu0200_n96"]
    assert len(new) == 10 and len(make_t64_configs.configs(held_out=True)) == 5
    changed = ("name", "section", "seed", "_purpose")
    for c, d in zip(new, old):
        assert {k: v for k, v in c.items() if k not in changed} == {k: v for k, v in d.items() if k not in changed}
        assert c["section"] == "T64" and c["seed"] == 20266400 != d["seed"]
    cells = [(n, mu, rep) for n in (96, 192, 384) for mu in a.MUS for rep in range(24)]
    fresh = {r.seed_of(20266400, n, 1.25, mu, 1.5, rep) for n, mu, rep in cells}
    used = {r.seed_of(20266300, n, 1.25, mu, 1.5, rep) for n, mu, rep in cells}
    assert len(fresh) == len(cells) and not fresh & used                    # no seed of T63's is run again
    monkeypatch.setattr(make_t64_configs, "PREDICTION", tmp_path / "none.json")
    with pytest.raises(SystemExit):
        make_t64_configs.main("--held-out")


def test_the_control_holds_at_paper_1s_own_rate_where_t63s_rule_did_not(tmp_path, capsys):
    write_design(tmp_path, a.STAGE1, wide_at={0.10, 0.20}, leftovers=5)       # 19 of 24 exactly flat and six around
    groups = a.load(tmp_path)
    cells = {k: t63.cell(v) for k, v in groups.items()}
    assert a.control_counts(groups[(0.0, 96)]) == (24, 19, 19)
    assert a.control_fails(groups, cells, 96) == []
    assert t63.control_fails(groups, cells, 96) != []                          # T63's nine tenths refuses the same rows
    assert a.main(str(tmp_path)) == 0
    out = capsys.readouterr().out
    assert out.count("holds") == 2 and "NOT READ" not in out
    assert "CONTROL N=96: holds | 24 replicas, 19 exactly flat, 19 exactly flat and six around" in out
    assert "VERDICT mu=0.100 N=96: WIDENS" in out and "VERDICT mu=0.000 N=192: ENDS SIX AROUND" in out
    assert "VERDICT mu=0.025 N=192: ENDS SIX AROUND" in out


def test_no_word_when_the_control_does_not_reproduce_paper_1(tmp_path, capsys, monkeypatch):
    write_design(tmp_path, a.STAGE1, wide_at={0.10}, skip={(0.0, 96), (0.0, 192), (0.05, 192)})
    write_cell(tmp_path, 0.0, 96, leftovers=10)                               # only 14 of 24 end exactly flat
    write_cell(tmp_path, 0.0, 192, sees3=0.3, flat=True)                      # exactly flat, and not six around
    write_cell(tmp_path, 0.05, 192, sees3=0.9, valid=False)
    assert a.main(str(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "CONTROL N=96: FAILS (the control: only 14 of 24 end exactly flat and six around)" in out
    assert "the control: only 0 of its 24 exactly flat end states are six around" in out
    assert "VERDICT mu=0.100 N=96: NOT READ" in out and "VERDICT mu=0.100 N=192: NOT READ" in out
    assert "VERDICT mu=0.000 N=96: ENDS SIX AROUND" in out                    # the control's own word is still given
    monkeypatch.setattr(a, "PREDICTION", tmp_path / "prediction.json")
    with pytest.raises(SystemExit):
        a.main(str(tmp_path), "--write-prediction")                           # no numbers from a failed control
    assert not (tmp_path / "prediction.json").exists()


def test_the_boundaries_of_the_control_rule(tmp_path):
    for leftovers, holds in ((9, True), (10, False)):                         # 15 of 24 is 0.625; 14 of 24 is 0.583
        folder = tmp_path / str(leftovers)
        folder.mkdir()
        write_design(folder, (96,), leftovers=leftovers)
        groups = a.load(folder)
        cells = {k: t63.cell(v) for k, v in groups.items()}
        assert (a.control_fails(groups, cells, 96) == []) is holds


def test_stage_1_gives_five_numbers_and_the_held_out_size_is_scored_only_when_complete(tmp_path, capsys, monkeypatch):
    write_design(tmp_path, a.STAGE1, wide_at={0.10, 0.20})
    monkeypatch.setattr(a, "PREDICTION", tmp_path / "prediction.json")
    assert a.main(str(tmp_path), "--write-prediction") == 0
    fixed = json.loads((tmp_path / "prediction.json").read_text())
    assert sorted(fixed) == ["0.000", "0.025", "0.050", "0.100", "0.200"] and fixed["0.100"]["wide"] == [1.0, 0.75, 1.0]
    assert fixed["0.000"]["tube"] == [pytest.approx(19 / 24), pytest.approx(19 / 24 - 0.25), 1.0]
    with pytest.raises(SystemExit):
        a.main(str(tmp_path), "--write-prediction")                           # never overwritten
    write_cell(tmp_path, 0.10, a.HELD_OUT, sees3=0.8)                          # one held-out cell, and no control
    capsys.readouterr()
    assert a.main(str(tmp_path)) == 1
    out = capsys.readouterr().out
    assert "AS PREDICTED" not in out and out.count("HELD-OUT") == 5
    for mu in a.MUS:
        if mu != 0.10:
            write_cell(tmp_path, mu, a.HELD_OUT, sees3=0.3 if mu == 0.20 else 0.0, leftovers=5 if mu == 0.0 else 0,
                       flat=False if mu == 0.20 else None)
    assert a.main(str(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "HELD-OUT mu=0.100 N=384: AS PREDICTED | verdict there: WIDENS" in out
    assert "HELD-OUT mu=0.200 N=384: NOT AS PREDICTED" in out and "verdict there: PATCHWORK" in out
    assert "HELD-OUT mu=0.000 N=384: AS PREDICTED | verdict there: ENDS SIX AROUND" in out
