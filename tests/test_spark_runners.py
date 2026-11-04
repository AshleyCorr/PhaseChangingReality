"""Smoke tests of the T25 and T26/T27 runners on tiny configs: they run, conserve energy, write the columns
the analyzers read, and reproduce from their seeds."""
import csv
import importlib.util
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


spark = load("run_local_spark")
race = load("run_scrap_race")


def write_cfg(tmp_path, cfg):
    p = tmp_path / (cfg["name"] + ".json")
    p.write_text(json.dumps(cfg))
    return p


def rows_of(tmp_path, name):
    return list(csv.DictReader(open(tmp_path / (name + ".csv"), newline="")))


def test_local_spark_stores_protocol_conserves_energy_and_leaks_when_asked(tmp_path):
    cfg = dict(name="smoke_stores", side=[12, 12], **{"lambda": 1.25}, protocol="stores", points="named",
               energies=[32.0], radius=2, leak=0.0, block=20, n_blocks=3, replicas=2, seed=1, save_adjacency=True)
    spark.main(write_cfg(tmp_path, cfg), str(tmp_path))
    rows = rows_of(tmp_path, "smoke_stores")
    assert len(rows) == 2 * 4 and all(float(r["drift"]) < 1e-9 for r in rows)
    assert {r["final"] for r in rows if int(r["block"]) == 3} == {"True"}
    assert (tmp_path / "smoke_stores_adj" / "E32_rep0.npz").exists()
    leaky = dict(cfg, name="smoke_leak", leak=0.5, radius=None, section="T27")
    spark.main(write_cfg(tmp_path, leaky), str(tmp_path))
    rows = rows_of(tmp_path, "smoke_leak")
    last = [r for r in rows if r["final"] == "True"][0]
    assert float(last["lost"]) > 0 and float(last["drift"]) < 1e-9 and last["radius"] == ""


def test_local_spark_patch_protocol_starts_from_a_hot_patch_in_a_cold_box(tmp_path):
    cfg = dict(name="smoke_patch", side=[12, 12], **{"lambda": 1.25}, protocol="patch", points="named",
               energies=[48.0], radius=3, block=20, n_blocks=2, replicas=1, seed=2)
    spark.main(write_cfg(tmp_path, cfg), str(tmp_path))
    rows = rows_of(tmp_path, "smoke_patch")
    first = rows[0]
    assert float(first["achieved"]) == 48.0 and abs(float(first["h"]) - 48.0) < 1e-9
    assert float(first["stores_total"]) == 0.0 and int(first["flat"]) < 144
    assert all(float(r["drift"]) < 1e-9 for r in rows)


def test_local_spark_runs_are_reproducible_from_their_seed(tmp_path):
    cfg = dict(name="smoke_a", side=[12, 12], **{"lambda": 1.25}, protocol="stores", points="named",
               energies=[24.0], radius=1, block=15, n_blocks=2, replicas=1, seed=7)
    spark.main(write_cfg(tmp_path, cfg), str(tmp_path))
    spark.main(write_cfg(tmp_path, dict(cfg, name="smoke_b")), str(tmp_path))
    a, b = rows_of(tmp_path, "smoke_a"), rows_of(tmp_path, "smoke_b")
    assert [dict(r, replica="") for r in a] == [dict(r, replica="") for r in b]


def test_scrap_race_writes_followed_blocks_over_the_schedule(tmp_path):
    cfg = dict(name="smoke_race", side=[16, 4], **{"lambda": 1.25}, replicas=2, seed=3, make_sweeps=300,
               g_hot=1.25, g_cold=0.25, t_cool=[100], hold=100, block=50)
    race.main(write_cfg(tmp_path, cfg), str(tmp_path))
    rows = rows_of(tmp_path, "smoke_race")
    followed = [r for r in rows if r["followed"] == "True"]
    if followed:
        reps = {r["replica"] for r in followed}
        for rep in reps:
            blocks = [r for r in followed if r["replica"] == rep]
            assert len(blocks) == 1 + 2 + 2 and blocks[-1]["final"] == "True"
            assert float(blocks[0]["g"]) == 1.25 and float(blocks[-1]["g"]) == 0.25
    assert all(r["t_cool"] == "100" for r in rows)


def test_bath_protocol_is_t21s_box_and_patch_can_keep_its_heat_local(tmp_path):
    cfg = dict(name="smoke_bath", side=[12, 12], **{"lambda": 1.25}, protocol="bath", points="named",
               energies=[64.0], radius=None, leak=0.0, block=20, n_blocks=2, replicas=1, seed=4)
    spark.main(write_cfg(tmp_path, cfg), str(tmp_path))
    rows = rows_of(tmp_path, "smoke_bath")
    assert float(rows[0]["stores_total"]) == 64.0 and float(rows[0]["stores_max"]) == 64.0 and rows[0]["local_heat"] == "False"
    assert all(float(r["drift"]) < 1e-9 for r in rows)
    local = dict(name="smoke_patch_local", side=[12, 12], **{"lambda": 1.25}, protocol="patch", points="named",
                 energies=[32.0], radius=3, local_heat=True, block=20, n_blocks=2, replicas=1, seed=5)
    spark.main(write_cfg(tmp_path, local), str(tmp_path))
    rows = rows_of(tmp_path, "smoke_patch_local")
    assert rows[0]["local_heat"] == "True" and float(rows[0]["stores_total"]) == 0.0
    assert all(float(r["drift"]) < 1e-9 for r in rows)
