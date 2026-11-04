"""T66 (PREREGISTRATION T66): the healing-time runner, its analyser and its configs."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import analyse_t66 as an                                   # noqa: E402
import make_t66_configs as mk                              # noqa: E402
from run_heal_time import run_cell                         # noqa: E402


def test_run_cell_conserves_energy_and_reproduces():
    a = list(run_cell([6, 6, 6], 1.25, 1.0, 4 * 216, 40, 10, 7))
    b = list(run_cell([6, 6, 6], 1.25, 1.0, 4 * 216, 40, 10, 7))
    assert [r[0] for r in a] == [0, 10, 20, 30, 40]
    assert all(r[4] < 1e-6 for r in a)                       # drift
    assert [r[1] for r in a] == [r[1] for r in b]            # same seed, same census
    assert a[0][1] == dict(damaged=0, flat=216, curled=0)    # flat start
    assert abs(a[0][3] - 216.0) < 1e-9                       # E = rho N in the stores at sweep 0
    c = list(run_cell([6, 6, 6], 1.25, 1.0, 4 * 216, 40, 10, 8))
    assert [r[1] for r in c] != [r[1] for r in a]            # a different seed gives different draws


def test_two_direction_torus_runs_at_four_links():
    a = list(run_cell([6, 8], 1.25, 1.0, 4 * 48, 20, 10, 3))
    assert a[0][1] == dict(damaged=0, flat=48, curled=0)
    assert all(r[4] < 1e-6 for r in a)


def _rows(n, dam, replica=0, every=10):
    return pd.DataFrame(dict(N=n, D=3, rho=1.0, replica=replica, sweep=[i * every for i in range(len(dam))],
                             damaged=dam, store_mean=0.1))


def test_replica_stats_peak_and_censoring():
    peak, ps, t, healed = an.replica_stats(_rows(216, [0, 5, 30, 20, 10, 2, 1]), 216)
    assert (peak, ps, t, healed) == (30, 20, 30, True)      # scar level 3 at N = 216; reached 30 sweeps after the peak
    peak, ps, t, healed = an.replica_stats(_rows(216, [0, 5, 30, 20, 10, 8, 7]), 216)
    assert (peak, ps, t, healed) == (30, 20, 40, False)     # censored at the last record


def test_cell_and_compare_words():
    fast = pd.concat([_rows(216, [0, 30, 10, 2, 1], r) for r in range(4)])
    slow = pd.concat([_rows(216, [0, 30, 20, 10, 2], r) for r in range(4)])
    never = pd.concat([_rows(216, [0, 30, 20, 10, 9], r) for r in range(4)])
    f, s, nv = an.cell_stats(fast), an.cell_stats(slow), an.cell_stats(never)
    assert f["healed_share"] == 1.0 and f["median_time"] == 20
    assert nv["median_time"] == an.BEYOND and nv["healed_share"] == 0.0
    assert an.compare(s, f) == "FASTER IN 3D"
    assert an.compare(f, s) == "SLOWER IN 3D"
    assert an.compare(nv, f) == "FASTER IN 3D"
    assert an.compare(f, nv) == "SLOWER IN 3D"
    assert an.compare(f, f) == "MIXED"
    assert an.size_verdict(["FASTER IN 3D", "FASTER IN 3D"]) == "FASTER IN 3D"
    assert an.size_verdict(["FASTER IN 3D", "MIXED"]) == "MIXED"


def test_configs_match_registration():
    for name, dims in mk.cells() + mk.cells(held_out=True):
        p = ROOT / "configs" / (name + ".json")
        if not p.exists():
            assert name.endswith("n1000"), "stage-1 config missing: %s" % name
            continue
        cfg = json.loads(p.read_text())
        assert cfg["dims"] == dims and cfg["lambda"] == 1.25 and cfg["rho_per_point"] == [1.0, 2.0]
        assert cfg["capacity"] == "4*N" and cfg["replicas"] == 16 and cfg["n_sweeps"] == 8000 and cfg["record_every"] == 10
        assert cfg["section"] == "T66" and "T66" in cfg["_purpose"]
        assert all(x % 2 == 0 and x >= 6 for x in dims)
    assert {c[0] for c in mk.cells()} == {"t66_d2_n216", "t66_d3_n216", "t66_d2_n512", "t66_d3_n512"}
    assert {c[0] for c in mk.cells(held_out=True)} == {"t66_d2_n1000", "t66_d3_n1000"}


def test_runner_writes_csv_and_meta(tmp_path):
    cfg = dict(name="t66_smoke", section="T66", dims=[6, 6, 6], **{"lambda": 1.25}, rho_per_point=[1.0],
               capacity="4*N", replicas=1, n_sweeps=20, record_every=10, seed=1, _purpose="EXPLORATORY smoke")
    path = tmp_path / "t66_smoke.json"
    path.write_text(json.dumps(cfg))
    subprocess.run([sys.executable, str(ROOT / "scripts" / "run_heal_time.py"), str(path), str(tmp_path)], check=True)
    df = pd.read_csv(tmp_path / "t66_smoke.csv")
    assert list(df["sweep"]) == [0, 10, 20] and df["drift"].max() < 1e-6 and df["C"].iloc[0] == 864
    assert (tmp_path / "t66_smoke.meta.json").exists()
