"""The T71 runner and analyzer together on a tiny cell (scratch output; nothing in results/)."""
import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyse_t71 as A            # noqa: E402
import run_layer_feed as R         # noqa: E402


def _run(tmp_path, name, **over):
    cfg = dict(name=name, section="T71", written="2026-10-18", lambdas=[1.25], sizes=[6], rates=[0.0625],
               gate="a * N_slab", leak=0.0, n_sweeps=60, replicas=2, seed=71, move_set="layer",
               feed_stops_at_split=True, max_width=8, log_every=1)
    cfg.update(over)
    path = tmp_path / (name + ".json")
    path.write_text(json.dumps(cfg))
    R.main(str(path), str(tmp_path))
    return list(csv.DictReader(open(tmp_path / (name + ".csv"), newline="")))


def test_the_runner_conserves_energy_and_the_analyzer_reads_the_gate_and_the_release(tmp_path):
    rows = _run(tmp_path, "t71_test_a")
    assert all(abs(float(r["residual"])) < 1e-9 for r in rows)
    assert (tmp_path / "t71_test_a.meta.json").exists()
    cells = A.read_cells(rows)
    assert len(cells) == 1
    (cell,) = cells.values()
    assert cell["n"] == 2
    for rep in cell["replicas"]:
        assert rep["first_split"] >= 32                        # the pool of 72 points reaches 144 at 2a / r = 32 sweeps
        assert rep["release"] == pytest.approx(144.0)           # a N_slab, credited at the split
        assert rep["word"] in ("OSCILLATES", "OPENS AND STAYS") and not rep["spent_first"]
    assert cell["median_release"] == pytest.approx(144.0)


def test_a_gate_the_stores_never_reach_reads_never_splits(tmp_path):
    rows = _run(tmp_path, "t71_test_b", gate="100 * N_slab", n_sweeps=20, move_set="model")
    (cell,) = A.read_cells(rows).values()
    assert cell["majority"] == "NEVER SPLITS" and cell["words"] == {"NEVER SPLITS": 2}
    assert cell["median_first_split"] is None
    assert all(int(r["broken"]) == 0 for r in rows)


def test_the_runner_refuses_to_overwrite(tmp_path):
    _run(tmp_path, "t71_test_c", n_sweeps=5)
    with pytest.raises(FileExistsError):
        _run(tmp_path, "t71_test_c", n_sweeps=5)


def test_the_daughter_mode_reads_cycles_with_the_period_and_the_energy_per_daughter(tmp_path):
    rows = _run(tmp_path, "t71_test_d", release_to="daughter", feed_stops_at_split=False, n_sweeps=240)
    assert all(abs(float(r["residual"])) < 1e-9 for r in rows)
    (cell,) = A.read_cells(rows).values()
    assert cell["majority"] == "CYCLES"
    assert cell["median_per_daughter"] == pytest.approx(144.0)
    assert cell["median_daughters"] >= 2 and cell["median_period"] is not None
    for rep in cell["replicas"]:
        assert rep["first_split"] >= 32 and rep["release"] == pytest.approx(144.0)


def test_logging_at_events_reads_the_same_as_logging_every_sweep(tmp_path):
    """The reading (word, release, births, period, shares) must not depend on log_every once events are logged."""
    dense = _run(tmp_path, "t71_test_e1", release_to="daughter", feed_stops_at_split=False, n_sweeps=240, log_every=1)
    sparse = _run(tmp_path, "t71_test_e50", release_to="daughter", feed_stops_at_split=False, n_sweeps=240, log_every=50)
    assert len(sparse) < len(dense) / 3
    (c1,) = A.read_cells(dense).values()
    (c50,) = A.read_cells(sparse).values()
    for r1, r50 in zip(c1["replicas"], c50["replicas"]):
        for key in ("word", "first_split", "first_clean", "spent_first", "release", "refolds", "daughters",
                    "per_daughter", "period", "end_width", "end_F"):
            assert r1[key] == r50[key], key
        assert r1["open_share"] == pytest.approx(r50["open_share"], abs=1e-12)
        assert r1["slab_share"] == pytest.approx(r50["slab_share"], abs=1e-12)
    # and with events switched off the sparse log cannot read the release exactly, which is why events are logged
    off = _run(tmp_path, "t71_test_off", release_to="daughter", feed_stops_at_split=False, n_sweeps=240, log_every=50,
               log_events=False)
    assert len(off) < len(sparse)
