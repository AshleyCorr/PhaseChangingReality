"""scripts/accept_inbox.py: a fetched run passes only when it matches its committed config, and nothing is overwritten."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("accept_inbox", ROOT / "scripts" / "accept_inbox.py")
accept = importlib.util.module_from_spec(spec)
spec.loader.exec_module(accept)


def _layout(tmp_path, cfg, recorded, rows="a,drift\n1,0.0\n2,1e-9\n"):
    configs, results, inbox = tmp_path / "configs", tmp_path / "results", tmp_path / "inbox"
    for d in (configs, results, inbox / "t99_x" / "t99_x_adj"):
        d.mkdir(parents=True)
    (configs / "t99_x.json").write_text(json.dumps(cfg))
    run = inbox / "t99_x"
    (run / "t99_x.csv").write_text(rows)
    (run / "t99_x.meta.json").write_text(json.dumps({"config": recorded, "python": "3.12"}))
    (run / "t99_x_adj" / "rep0.npz").write_bytes(b"x")
    return configs, results, run


def test_a_matching_run_passes_and_moves(tmp_path):
    cfg = {"name": "t99_x", "seed": 5, "lambda": 1.3}
    configs, results, run = _layout(tmp_path, cfg, dict(cfg))
    ok, reasons, report = accept.check(run, configs, results)
    assert ok and not reasons and report["rows"] == 2 and report["max_drift"] == 1e-9
    accept.move(run, results)
    assert (results / "t99_x.csv").is_file() and (results / "t99_x_adj" / "rep0.npz").is_file() and not run.exists()


def test_a_changed_config_fails(tmp_path):
    cfg = {"name": "t99_x", "seed": 5, "lambda": 1.3}
    configs, results, run = _layout(tmp_path, cfg, dict(cfg, seed=6))
    ok, reasons, _ = accept.check(run, configs, results)
    assert not ok and "differs" in reasons[0]


def test_nothing_is_overwritten_and_an_empty_csv_fails(tmp_path):
    cfg = {"name": "t99_x"}
    configs, results, run = _layout(tmp_path, cfg, dict(cfg), rows="a\n")
    (results / "t99_x.meta.json").write_text("{}")
    ok, reasons, _ = accept.check(run, configs, results)
    assert not ok
    assert any("no rows" in r for r in reasons) and any("already exists" in r for r in reasons)
