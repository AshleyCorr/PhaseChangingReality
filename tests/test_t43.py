"""T43 (PREREGISTRATION.md): the reading rules of scripts/analyse_t43.py on synthetic rows, and the runner
scripts/run_planted_allotrope.py on a tiny run."""
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


a = load("analyse_t43")
run = load("run_planted_allotrope")

NS = 20000
SWEEPS = run.snapshot_sweeps(NS)


# ------------------------------------------------------------------------------------------------ the schedule
def test_snapshot_schedule():
    assert SWEEPS[:3] == [0, 10, 20]
    assert SWEEPS[200] == 2000 and SWEEPS[201] == 2100 and SWEEPS[-1] == 20000
    assert len(SWEEPS) == 1 + 200 + 180
    assert run.snapshot_sweeps(30) == [0, 10, 20, 30]


# ------------------------------------------------------------------------------------------------ the lifetime rule
def test_lifetime_needs_two_snapshots_in_a_row():
    sw = [0, 10, 20, 30, 40, 50]
    assert a.lifetime(sw, [1, 0.4, 0.9, 0.3, 0.2, 0.1], NS) == (30, False)   # the single dip at 10 does not count
    assert a.lifetime(sw, [1, 0.4, 0.4, 0.9, 0.9, 0.9], NS) == (10, False)   # it is the first of the two


def test_lifetime_is_censored_when_it_never_drops():
    sw = [0, 10, 20, 30]
    assert a.lifetime(sw, [1, 1, 0.9, 0.8], NS) == (NS, True)
    assert a.lifetime(sw, [1, 1, 0.9, 0.1], NS) == (NS, True)                # one low snapshot at the very end


def test_half_is_alive_and_a_start_below_is_zero():
    sw = [0, 10, 20]
    assert a.lifetime(sw, [0.5, 0.5, 0.5], NS) == (NS, True)                 # e >= 1/2 is alive
    assert a.lifetime(sw, [0.49, 0.49, 1.0], NS) == (0, False)


# ------------------------------------------------------------------------------------------------ synthetic rows
def rows_for(obj, g, rep, e_of_sweep, q_of_sweep, n_r=18, n=234):
    """Rows as csv.DictReader gives them: f_B = 0, so e = f_R."""
    out = []
    for s in SWEEPS:
        f = e_of_sweep(s)
        out.append(dict(object=obj, N=str(n), g=str(g), replica=str(rep), sweep=str(s), H="1008",
                        f_R=str(f) if n_r else "", f_B="0.0", q_R=str(f) if n_r else "", q_B=str(q_of_sweep(s)),
                        n_R=str(n_r), n_low=str(int(round(f * n_r))), low_R=""))
    return out


def step(at, high=1.0, low=0.0):
    return lambda s: high if s < at else low


def times(obj, g, taus_r, taus_b, n_r=18):
    rows = []
    for rep, (tr, tb) in enumerate(zip(taus_r, taus_b)):
        rows += rows_for(obj, g, rep, step(tr), step(tb), n_r)
    reps = a.replicas(rows)
    return [a.replica_times(b, NS) for b in reps.values()]


def test_replica_times_from_rows_and_the_far_subtraction():
    t = times("hyperbolic_fold", 2.0, [500], [3000])[0]
    assert (t["tau_R"], t["cens_R"], t["tau_B"], t["cens_B"]) == (500, False, 3000, False)
    rows = rows_for("hyperbolic_fold", 2.0, 0, lambda s: 0.9, lambda s: 1.0)
    for r in rows:                                      # f_R stays high but the far points catch up: e = 0.3
        r["f_B"] = "0.6" if int(r["sweep"]) >= 700 else "0.0"
    t = a.replica_times(rows, NS)
    assert t["tau_R"] == 700 and not t["cens_R"] and t["cens_B"]


def test_lasts_at_a_median_of_1000_and_dissolves_below():
    r = a.cell_reading(times("hyperbolic_fold", 2.0, [1000] * 3, [5000] * 3))
    assert r["tau_R"] == 1000 and r["reading"] == "LASTS"
    r = a.cell_reading(times("hyperbolic_fold", 2.0, [990] * 3, [5000] * 3))
    assert r["reading"] == "DISSOLVES"


def test_medians_count_censored_replicas_at_n_sweeps():
    # three of five never drop: the median is the censored value
    r = a.cell_reading(times("hyperbolic_fold", 2.0, [100, 200, NS + 1, NS + 1, NS + 1], [NS + 1] * 5))
    assert r["tau_R"] == NS and r["cens_R"] == 3 and r["reading"] == "LASTS"
    assert r["tau_B"] == NS and r["cens_B"] == 5 and r["tag"] == "WITH ITS BACKGROUND"
    r = a.cell_reading(times("hyperbolic_fold", 2.0, [300] * 3, [NS + 1] * 3))     # a background that outlives the run
    assert r["tau_B"] == NS and r["tag"] == "FIRST" and r["reading"] == "DISSOLVES"
    # an even number of replicas: the mean of the middle two, as numpy's median
    r = a.cell_reading(times("hyperbolic_fold", 2.0, [100, 300, 900, NS + 1], [NS + 1] * 4))
    assert r["tau_R"] == 600 and r["reading"] == "DISSOLVES"


def test_first_only_when_well_before_the_background():
    r = a.cell_reading(times("hyperbolic_fold", 3.0, [400] * 3, [900] * 3))
    assert r["tag"] == "FIRST"                           # 400 < 450
    r = a.cell_reading(times("hyperbolic_fold", 3.0, [500] * 3, [1000] * 3))
    assert r["tag"] == "WITH ITS BACKGROUND"             # 500 is not < 500
    r = a.cell_reading(times("hyperbolic_fold", 3.0, [800] * 3, [700] * 3))
    assert r["tag"] == "WITH ITS BACKGROUND"


def test_controls_are_read_for_the_background_only():
    r = a.cell_reading(times("flat_torus", 3.0, [0, 0], [300, 500], n_r=0))
    assert r["reading"] == "CONTROL" and r["tau_R"] is None and r["tau_B"] == 400


def test_verdict_is_read_on_the_fold_alone():
    lasts, gone = dict(reading="LASTS"), dict(reading="DISSOLVES")
    assert a.verdict({("hyperbolic_fold", 1.0): gone, ("hyperbolic_fold", 3.0): lasts}) == "ALLOTROPE LASTS"
    assert a.verdict({("hyperbolic_fold", 1.0): gone, ("hyperbolic_handle", 1.0): lasts,
                      ("flat_handle", 1.0): lasts}) == "DISSOLVES"
    assert a.verdict({("hyperbolic_handle", 1.0): lasts}) == "NOT READ"


# ------------------------------------------------------------------------------------------------ persistence
def pers_rows(sets, n_low, sweeps):
    return [dict(sweep=str(s), N="100", n_low=str(k), low_R=" ".join(map(str, sorted(x))))
            for s, x, k in zip(sweeps, sets, n_low)]


def test_excess_for_a_stuck_set_and_for_a_set_that_alternates():
    sw = [0, 10, 20, 30, 40]
    stuck = pers_rows([{1, 2, 3}] * 5, [10] * 5, sw)
    assert a.excess(stuck, 10) == pytest.approx(1 - 0.1)
    flip = pers_rows([{1, 2}, {3, 4}, {1, 2}, {3, 4}, {1, 2}], [10] * 5, sw)
    assert a.excess(flip, 10) == pytest.approx(-0.1)     # never there a snapshot later
    assert a.excess(flip, 20) == pytest.approx(0.9)      # always there two snapshots later


def test_excess_uses_only_pairs_the_schedule_has_and_skips_empty_sets():
    sw = [0, 100, 200]
    rows = pers_rows([set(), {5}, {5}], [0, 20, 20], sw)
    assert a.excess(rows, 100) == pytest.approx(1 - 40 / 300)   # only t = 100 counts; rho over all snapshots
    assert math.isnan(a.excess(rows, 50))                       # no pair 50 sweeps apart


# ------------------------------------------------------------------------------------------------ the runner
def tiny(tmp_path, name):
    cfg = dict(_purpose="test", name=name, objects=["hyperbolic_fold"], couplings=[2.0], replicas=2, n_sweeps=30,
               seed=12345)
    p = tmp_path / (name + ".json")
    p.write_text(json.dumps(cfg))
    run.main(str(p), str(tmp_path))
    return list(csv.DictReader(open(tmp_path / (name + ".csv"), newline="")))


def test_runner_writes_rows_in_range_and_repeats_with_the_same_seed(tmp_path):
    rows = tiny(tmp_path, "t43_tiny_a")
    assert (tmp_path / "t43_tiny_a.meta.json").exists()
    assert len(rows) == 2 * 4 and [int(r["sweep"]) for r in rows[:4]] == [0, 10, 20, 30]
    for r in rows:
        for k in ("f_R", "f_B", "q_R", "q_B"):
            assert 0.0 <= float(r[k]) <= 1.0
        assert int(r["N"]) == 234 and int(r["n_R"]) == 18
        assert len(r["low_R"].split()) == round(float(r["f_R"]) * 18)
    first = rows[0]                                      # the start is the planted fold exactly (design note, sec. 4)
    assert (float(first["f_R"]), float(first["f_B"]), float(first["q_R"]), float(first["q_B"])) == (1, 0, 1, 1)
    assert int(first["H"]) == 1008 and int(first["n_low"]) == 18
    assert any(r["H"] != first["H"] for r in rows)       # the chain moved
    again = tiny(tmp_path, "t43_tiny_b")
    assert again == rows
    with pytest.raises(FileExistsError):                 # results are append-only
        tiny(tmp_path, "t43_tiny_a")


def test_runner_energy_matches_the_graph(tmp_path):
    objs = run.bpa.build_all(240, 14)
    from graphity.cqg import hamiltonian
    for name, (adj, part, bg) in objs.items():
        h = run.energy(adj.shape[0], int(run.total_squares(adj)), int(run.surplus(adj)))
        assert h == hamiltonian(adj, 1.0)
        R, far = run.planted_and_far(adj, bg)
        e = run.link_squares(adj)
        assert (e.sum(axis=1) // 2 == run.bpa.square_counts(adj)).all()
        assert all(tuple(e_v) == run.bpa.link_pattern(adj, v) for v, e_v in enumerate(np.sort(e, axis=1)))
        m = run.measure(adj, R, far, bg)
        assert m["f_B"] == 0.0 and m["q_B"] == 1.0
        assert (m["f_R"], m["q_R"]) == ((1.0, 1.0) if len(R) else ("", ""))
