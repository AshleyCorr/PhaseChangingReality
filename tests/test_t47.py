"""T47's runner and analyzer, before any run (PREREGISTRATION T47)."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

from graphity import cqg_d
from graphity.sealed_tie_d import energy_table_d, run_sealed_bath_table_d

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _front(bath, L, rep, times):
    """Rows reaching f = 0.1, 0.3, 0.5 at the given sweeps."""
    rows = [dict(bath=bath, L=L, replica=rep, sweep=0, f=0.0)]
    for t, f in zip(times, (0.1, 0.3, 0.5)):
        rows.append(dict(bath=bath, L=L, replica=rep, sweep=t, f=f))
    return rows


def test_part_a_verdicts():
    a = _load("analyse_t47")
    # fixed speed: equal time per equal distance, speed the same at every L (time scales with L)
    rows = []
    for L in (96, 192):
        for rep in range(3):
            rows += _front("local", L, rep, [L, 2 * L, 3 * L])
    table, verdict = a.summarize_a(rows)
    assert verdict["local"] == "FIXED SPEED" and abs(table[("local", 96)]["R"] - 1.0) < 1e-12
    rows = []
    for L in (96, 192):
        for rep in range(3):
            rows += _front("bath", L, rep, [100, 300, 700])              # R = 2
    assert a.summarize_a(rows)[1]["bath"] == "DIFFUSIVE"
    rows = []
    for L in (96, 192):
        for rep in range(3):
            rows += _front("bath", L, rep, [100, 300, 350])              # R = 0.25
    assert a.summarize_a(rows)[1]["bath"] == "ACCELERATING"
    rows = [dict(bath="bath", L=96, replica=r, sweep=10, f=0.05) for r in range(3)]
    assert a.summarize_a(rows)[1]["bath"] == "NO FRONT"


def test_part_b_outcomes():
    a = _load("analyse_t47")
    n = 100
    def last(d):
        return dict(N=n, **{"d%d" % k: (d[k] if k < len(d) else 0) for k in range(8)})
    assert a.read_recurl(last([0, 0, 0, 0, 100]), n) == "HEALS"
    assert a.read_recurl(last([0, 0, 0, 30, 70]), n) == "FRONT"
    assert a.read_recurl(last([0, 0, 0, 0, 70, 30]), n) == "MELTED"
    assert a.read_recurl(last([0, 0, 0, 10, 90]), n) == "STALLED"


def test_front_runner_conserves_energy(tmp_path):
    m = _load("run_front_speed")
    for bath in ("bath", "local"):
        cfg = dict(name="smoke_t47_" + bath, section="T47", side=[24, 4], bath=bath, replicas=1, n_sweeps=40,
                   record_every=10, seed=3)
        cfg["lambda"] = 1.25
        p = tmp_path / (cfg["name"] + ".json")
        p.write_text(json.dumps(cfg))
        m.main(str(p), str(tmp_path))
        rows = list(csv.DictReader(open(tmp_path / (cfg["name"] + ".csv"), newline="")))
        assert len(rows) == 5 and float(rows[0]["planted_cost"]) == 12.0
        assert all(float(r["drift"]) < 1e-8 for r in rows)


def test_table_tie_with_per_vertex_stores_conserves():
    lam = 1.30
    ftab = np.array([0.0, 1.0, -1.3711, -1.2, 0.0]) * 4.0 * (lam - 1.0)
    adj, part = cqg_d.torus([4, 4, 4, 6])
    side_u = np.flatnonzero(part == 0)
    stores = np.zeros(adj.shape[0]); stores[int(side_u[0])] = 300.0
    e0 = energy_table_d(adj, lam, ftab) + stores.sum()
    run_sealed_bath_table_d(adj, side_u, stores, 20, 4, lam, ftab, by_vertex=True)
    assert abs(energy_table_d(adj, lam, ftab) + stores.sum() - e0) < 1e-8
    assert cqg_d.is_valid(adj) and (stores >= -1e-12).all()
