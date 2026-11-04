"""T63 (PREREGISTRATION.md): the curled tube itself under a price on loops of six. The runner's start, and the
reading on synthetic rows. Written before any run."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import analyse_t63 as a                                           # noqa: E402
import make_t63_configs                                           # noqa: E402
import run_sixloop_widen as r                                     # noqa: E402
from graphity.cqg import NO_CAP, hamiltonian, is_valid, torus     # noqa: E402
from graphity.sixloop import excess6                              # noqa: E402
from read_endpoint_width import twisted_tube                      # noqa: E402


def test_the_start_is_the_curled_tube_and_the_control_ends_six_around():
    for length in (12, 24):
        adj, side = r.seam_shifted(length, r.STARTS["tube"])
        assert np.array_equal(adj, twisted_tube(length, 0)) and is_valid(adj, NO_CAP)
        assert hamiltonian(adj, 1.25) == 4 * length and excess6(adj, side) == 18 * 4 * length
        assert hamiltonian(torus(length, 4, NO_CAP)[0], 1.25) == 4 * length  # the same energy as the kernel's own tube
    fields, adj, _side = r.replica(12, 1.25, 0.0, 1.5, 4, 12000, 2000, 0.5, 200, 0.05, "tube")   # no price: paper 1
    assert fields["start"] == "tube" and fields["books"] and fields["valid"] and fields["e6_start"] == 864
    assert fields["curl_sweep"] > 0 and fields["x_end"] == 0                 # the burp happened
    assert fields["sees3"] == 0.0                                            # and it ended six around, not wide
    six = r.replica(12, 1.25, 0.0, 1.5, 4, 600, 400, 0.5, 100, 0.05)[0]      # the default start is T62's
    assert six["start"] == "six" and six["curl_sweep"] == 0 and six["e6_start"] == 720


def test_configs_and_the_guard_on_the_held_out_size(tmp_path, monkeypatch):
    stage1 = make_t63_configs.configs()
    assert [c["name"] for c in stage1][:5] == ["t63_mu0000_n96", "t63_mu0025_n96", "t63_mu0050_n96", "t63_mu0100_n96",
                                               "t63_mu0200_n96"]
    assert len(stage1) == 10 and all(c["start"] == "tube" and c["g"] == 1.5 and c["replicas"] == 24 for c in stage1)
    assert len(make_t63_configs.configs(held_out=True)) == 5
    monkeypatch.setattr(make_t63_configs, "PREDICTION", tmp_path / "none.json")
    with pytest.raises(SystemExit):
        make_t63_configs.main("--held-out")
    seeds = {r.seed_of(20266300, n, 1.25, mu, 1.5, rep) for n in (96, 192) for mu in a.MUS for rep in range(3)}
    assert len(seeds) == 30


FIELDS = ["N", "mu", "g", "replica", "start", "sees3", "x_end", "e6_start", "e6_end", "h_end", "phi_min", "curl_sweep",
          "valid", "books", "flat"]


def write_cell(folder, mu, n, sees3=0.0, valid=True, replicas=range(24), curled=0, start="tube"):
    path = folder / (make_t63_configs.name_of(mu, n) + ".csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for i, rep in enumerate(replicas):
            w.writerow(dict(N=n, mu=mu, g=1.5, replica=rep, start=start, sees3=sees3, x_end=n if i < curled else 0,
                            e6_start=18 * n, e6_end=0, h_end=16.0, phi_min=0.93, curl_sweep=-1 if i < curled else 1500,
                            valid=valid, books=True, flat=False))


def write_design(folder, sizes, wide_at=(), skip=()):
    for n in sizes:
        for mu in a.MUS:
            if (mu, n) not in skip:
                write_cell(folder, mu, n, sees3=0.8 if mu in wide_at else 0.0)


def test_outcomes_and_words():
    row = lambda s, x=0: dict(sees3=str(s), x_end=str(x), N="96")                # noqa: E731
    assert [a.outcome(row(s)) for s in (0.0, 0.05, 0.49, 0.5)] == ["TUBE", "PATCHWORK", "PATCHWORK", "WIDE"]
    assert a.outcome(row(0.9, x=25)) == "NOT CONVERTED" and a.outcome(row(0.9, x=24)) == "WIDE"
    assert [a.verdict(m) for m in ("WIDE", "PATCHWORK", "TUBE", "MIXED")] == [
        "WIDENS", "PATCHWORK", "ENDS SIX AROUND", "MIXED"]


def test_a_complete_clean_stage_1_is_read_and_gives_five_numbers(tmp_path, capsys, monkeypatch):
    write_design(tmp_path, a.STAGE1, wide_at={0.10, 0.20})
    assert a.design_problems(a.load(tmp_path), a.STAGE1) == []
    assert a.main(str(tmp_path)) == 0
    out = capsys.readouterr().out
    assert out.count("CONTROL") == 2 and out.count("holds") == 2 and "NOT READ" not in out
    assert "VERDICT mu=0.100 N=96: WIDENS" in out and "VERDICT mu=0.025 N=192: ENDS SIX AROUND" in out
    monkeypatch.setattr(a, "PREDICTION", tmp_path / "prediction.json")
    assert a.main(str(tmp_path), "--write-prediction") == 0
    fixed = json.loads((tmp_path / "prediction.json").read_text())
    assert sorted(fixed) == ["0.000", "0.025", "0.050", "0.100", "0.200"] and fixed["0.100"]["wide"] == [1.0, 0.75, 1.0]


def test_no_word_when_the_control_does_not_end_six_around_or_data_are_bad(tmp_path, capsys, monkeypatch):
    write_design(tmp_path, a.STAGE1, wide_at={0.10}, skip={(0.0, 96), (0.05, 192), (0.20, 192), (0.025, 192)})
    write_cell(tmp_path, 0.0, 96, sees3=0.3)                                  # the control did not reproduce paper 1
    write_cell(tmp_path, 0.05, 192, sees3=0.9, valid=False)                   # invalid graphs
    write_cell(tmp_path, 0.20, 192, sees3=0.9, curled=5)                      # too few converted
    write_cell(tmp_path, 0.025, 192, start="six")                             # the wrong start
    a.main(str(tmp_path))
    out = capsys.readouterr().out
    assert "CONTROL N=96: FAILS" in out and "VERDICT mu=0.100 N=96: NOT READ" in out
    assert "VERDICT mu=0.100 N=192: WIDENS" in out                            # N = 192 has its own control
    assert "VERDICT mu=0.050 N=192: NOT READ (its gates fail)" in out
    assert "VERDICT mu=0.200 N=192: NOT READ (its gates fail)" in out
    assert "did not start from the curled tube" in out
    monkeypatch.setattr(a, "PREDICTION", tmp_path / "prediction.json")
    with pytest.raises(SystemExit):
        a.main(str(tmp_path), "--write-prediction")


def test_the_held_out_size_is_scored_only_when_complete(tmp_path, capsys, monkeypatch):
    write_design(tmp_path, a.STAGE1, wide_at={0.10, 0.20})
    monkeypatch.setattr(a, "PREDICTION", tmp_path / "prediction.json")
    a.main(str(tmp_path), "--write-prediction")
    write_cell(tmp_path, 0.10, a.HELD_OUT, sees3=0.8)                          # one held-out cell, and no control
    capsys.readouterr()
    assert a.main(str(tmp_path)) == 1
    out = capsys.readouterr().out
    assert "AS PREDICTED" not in out and out.count("HELD-OUT") == 5
    for mu in a.MUS:
        if mu != 0.10:
            write_cell(tmp_path, mu, a.HELD_OUT, sees3=0.3 if mu == 0.20 else 0.0)
    assert a.main(str(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "HELD-OUT mu=0.100 N=384: AS PREDICTED | verdict there: WIDENS" in out
    assert "HELD-OUT mu=0.200 N=384: NOT AS PREDICTED" in out and "verdict there: PATCHWORK" in out
