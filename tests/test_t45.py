"""T45's analyzer and the runner with a table tie, before any run (PREREGISTRATION T45)."""
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


def _rows(spark, rep, final_d, h):
    base = dict(tie_label="A", lam=1.4, N=100, C=50, spark=spark, replica=rep, total=spark)
    r0 = dict(base, sweep=0, h_per_vertex=4.8, **{"d%d" % k: (100 if k == 0 else 0) for k in range(8)})
    r1 = dict(base, sweep=200, h_per_vertex=h, **{"d%d" % k: (final_d[k] if k < len(final_d) else 0) for k in range(8)})
    return [r0, r1]


def test_verdicts():
    a = _load("analyse_t45")
    flat, stuck = [0, 0, 5, 95], [100]
    rows = []
    for rep in range(3):
        rows += _rows(3.21, rep, stuck, 4.8) + _rows(13.35, rep, flat, 0.05)
    assert a.summarize(rows)["verdict"][("A", 1.4)] == "PUSHED THROUGH"
    rows = []
    for rep in range(3):
        rows += _rows(3.21, rep, flat, 0.05)
    assert a.summarize(rows)["verdict"][("A", 1.4)] == "ONE PUSH OPENS ALL"
    rows = []
    for rep in range(3):
        rows += _rows(3.21, rep, stuck, 4.8)
    assert a.summarize(rows)["verdict"][("A", 1.4)] == "NOT ALL"


def test_runner_with_a_table_tie_conserves(tmp_path):
    m = _load("run_sealed_curled_d")
    cfg = dict(name="smoke_t45", section="T45", gas={"dims": [4, 4, 4], "copies": 2}, capacities=["N/2"],
               sparks=[3.21], replicas=1, n_sweeps=40, record_every=20, seed=5, tie_label="A",
               ftable_per_a=[0.0, 1.0, 1.528, 0.0])
    cfg["lambda"] = 1.4
    p = tmp_path / "smoke_t45.json"
    p.write_text(json.dumps(cfg))
    m.main(str(p), str(tmp_path))
    rows = list(csv.DictReader(open(tmp_path / "smoke_t45.csv", newline="")))
    assert rows[0]["tie_label"] == "A" and float(rows[0]["tie_T"]) == 0.0
    assert all(float(r["drift"]) < 1e-8 for r in rows)
    assert list(_load("analyse_t45").summarize(rows)["verdict"]) == [("A", 1.4)]
