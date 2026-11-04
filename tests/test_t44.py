"""T44's analyzer on hand-made blocks, and the runner with "kappa" end to end (PREREGISTRATION T44, before any run)."""
import csv
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _block(sweep, h, d, total, kappa=2.0):
    row = dict(kappa=kappa, N=100, C=50, spark=44.0, replica=0, lam=1.25, sweep=sweep, h_per_vertex=h, total=total)
    row.update({"d%d" % k: (d[k] if k < len(d) else 0) for k in range(8)})
    return row


def test_outcomes_and_pattern():
    a = _load("analyse_t44")
    together = [_block(0, 3.0, [100], 44), _block(200, 1.5, [55, 0, 0, 45], 150), _block(400, 0.05, [0, 0, 5, 95], 300)]
    out, pattern, stages, _ = a.read_replica(together, 1.25, 100)
    assert (out, pattern) == ("FLAT", "TOGETHER")
    assert stages[1] == stages[2] == stages[3] == (300 - 44) / 100      # all three reached at the same block
    stepwise = [_block(0, 3.0, [100], 44), _block(200, 2.0, [10, 80, 10], 100), _block(400, 0.05, [0, 0, 5, 95], 300)]
    assert a.read_replica(stepwise, 1.25, 100)[:2] == ("FLAT", "STEPWISE")
    assert a.read_replica([_block(0, 3.0, [100], 44), _block(200, 3.0, [90, 10], 44)], 1.25, 100)[0] == "STUCK"
    assert a.read_replica([_block(0, 3.0, [100], 44), _block(200, 2.0, [40, 20, 10, 0, 0, 30], 44)], 1.25, 100)[0] == "MELTED"
    assert a.read_replica([_block(0, 3.0, [100], 44), _block(200, 1.0, [20, 20, 40, 20], 44)], 1.25, 100)[0] == "PARTLY OPEN"


def test_verdict_needs_a_flat_majority_in_some_cell():
    a = _load("analyse_t44")
    flat = [_block(0, 3.0, [100], 44), _block(200, 0.05, [0, 0, 5, 95], 300)]
    stuck = [_block(0, 3.0, [100], 44), _block(200, 3.0, [100], 44)]
    rows = []
    for rep, blocks in enumerate([flat, flat, stuck]):
        rows += [dict(r, replica=rep) for r in blocks]
    s = a.summarize(rows)
    assert s["rows"][2.0] == "ONE PUSH OPENS ALL"
    rows = []
    for rep, blocks in enumerate([flat, stuck, stuck]):
        rows += [dict(r, replica=rep) for r in blocks]
    assert a.summarize(rows)["rows"][2.0] == "NOT ALL"


def test_runner_with_kappa_writes_the_tie_and_conserves(tmp_path):
    m = _load("run_sealed_curled_d")
    cfg = dict(name="smoke_t44", section="T44", gas={"dims": [4, 4, 4], "copies": 2}, capacities=["N/2"], sparks=[44.0],
               replicas=1, n_sweeps=40, record_every=20, seed=3, kappa=2.0)
    cfg["lambda"] = 1.25
    p = tmp_path / "smoke_t44.json"
    p.write_text(json.dumps(cfg))
    m.main(str(p), str(tmp_path))
    rows = list(csv.DictReader(open(tmp_path / "smoke_t44.csv", newline="")))
    assert rows[0]["kappa"] == "2.0" and rows[0]["tie_T"] == "0"
    assert all(float(r["drift"]) < 1e-8 for r in rows)
    a = _load("analyse_t44")
    s = a.summarize(rows)
    assert list(s["rows"]) == [2.0]
