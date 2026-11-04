"""The per-birth reader of T71 on a tiny synthetic file (written with tmp_path; nothing in results/)."""
import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import read_t71_births as B        # noqa: E402

COLS = ["lam", "L", "N_slab", "rate", "gate", "leak", "move_set", "release_to", "feed_stops_at_split", "replica",
        "seed", "sweep", "N", "dims", "F", "stores", "fed", "leaked", "n_split", "n_contract", "n_clean", "open",
        "curled", "broken", "daughters", "daughter_energy", "residual", "first_split", "first_clean", "first_contract"]


def _row(replica, sweep, n_contract, daughters, daughter_energy):
    # lam = 1.25, N_slab = 10: a = 1 per point, a N_slab = 10
    return dict(lam=1.25, L=6, N_slab=10, rate=0.0625, gate=10.0, leak=0.0, move_set="model", release_to="daughter",
                feed_stops_at_split=0, replica=replica, seed=1, sweep=sweep, N=10, dims="4x6x6", F=0.0, stores=0.0,
                fed=0.0, leaked=0.0, n_split=daughters, n_contract=n_contract, n_clean=0, open=0, curled=10, broken=0,
                daughters=daughters, daughter_energy=daughter_energy, residual=0.0, first_split="", first_clean="",
                first_contract="")


def _write(tmp_path, rows, tie=None):
    path = tmp_path / "t71_test.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    cfg = dict(name="t71_test", lambdas=[1.25])
    if tie is not None:
        cfg["tie"] = tie
    (tmp_path / "t71_test.meta.json").write_text(json.dumps(dict(config=cfg)))
    return path


def test_births_are_counted_from_the_first_logged_row_and_binned_against_a_n_slab(tmp_path):
    rows = [
        # replica 0: born before its first logged row (10 = exactly a N_slab), then a refold, then a birth of 12.5
        # (1.25 a N_slab), then no refold and a birth of 14 (above 1.25), then a refold and a short birth of 5
        _row(0, 10, 0, 1, 10.0),
        _row(0, 12, 1, 1, 10.0),
        _row(0, 15, 1, 2, 22.5),
        _row(0, 20, 1, 3, 36.5),
        _row(0, 22, 2, 3, 36.5),
        _row(0, 25, 2, 4, 41.5),
        # replica 1: never a daughter
        _row(1, 10, 0, 0, 0.0),
        _row(1, 25, 0, 0, 0.0),
    ]
    path = _write(tmp_path, rows)
    cells = B.main([str(path)])
    assert len(cells) == 1
    (c,) = cells.values()
    assert c["n"] == 2 and c["a_n"] == pytest.approx(10.0)
    assert c["births"] == 4 and c["first_row_births"] == 1
    assert c["exact"] == 1
    assert c["above"] == 2 and c["above_band"] == 1 and c["above_125"] == 1 and c["largest"] == pytest.approx(1.4)
    assert c["below"] == 1 and c["shortest"] == pytest.approx(0.5)
    assert c["refolds"] == 2 and c["pairs"] == 3           # refolds before births 2 and 4, none before birth 3
    assert c["anomalies"] == 0 and c["flagged"] == 0
    r0, r1 = c["replicas"]
    assert r0["energies"] == pytest.approx([10.0, 12.5, 14.0, 5.0])
    assert r1["births"] == 0 and r1["shortest"] is None


def test_a_row_adding_two_daughters_is_flagged_and_split_evenly(tmp_path):
    rows = [_row(0, 10, 0, 0, 0.0), _row(0, 12, 0, 2, 20.0)]
    path = _write(tmp_path, rows)
    (c,) = B.main([str(path)]).values()
    assert c["births"] == 2 and c["flagged"] == 1 and c["exact"] == 2 and c["first_row_births"] == 0


def test_a_tie_in_the_meta_is_reported(tmp_path, capsys):
    path = _write(tmp_path, [_row(0, 10, 0, 1, 10.0)], tie="all_at_last")
    B.main([str(path)])
    assert "names a tie" in capsys.readouterr().out
    assert B.tie_of(str(tmp_path / "absent.meta.json")) is None


def test_no_paths_is_refused():
    with pytest.raises(SystemExit):
        B.main([])
