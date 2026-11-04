"""The runner must not infer a thermometer spacing from its energy coefficient."""
import csv
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "sealed_budget", Path(__file__).resolve().parents[1] / "scripts/run_sealed_sheet_budget.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.mark.parametrize("step", [None, 1.0])
def test_runner_reports_measured_mean_and_only_explicit_temperature_proxy(tmp_path, monkeypatch, step):
    # Use a known demon trace to distinguish the correct unit-level proxy from
    # the old, unjustified 4*lambda=5 spacing at lambda=1.25.
    def trace(*args):
        return np.array([36, 36]), np.zeros(2), np.array([0.5, 0.5]), np.zeros(2), 0.0
    monkeypatch.setattr(runner, "run_sealed", trace)
    cfg = dict(name="thermometer", sides=[[6, 6]], budgets_per_point=[1],
               replicas=1, seed=1, n_sweeps=2, **{"lambda": 1.25})
    if step is not None:
        cfg["thermometer_step"] = step
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    runner.main(path, tmp_path)
    with (tmp_path / "thermometer.csv").open(newline="") as f:
        row = next(csv.DictReader(f))
    assert float(row["demon_mean"]) == 0.5
    if step is None:
        assert math.isnan(float(row["temperature"]))
    else:
        assert float(row["temperature"]) == pytest.approx(1 / math.log(3))
