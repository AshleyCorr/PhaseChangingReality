"""Checks on result writing (the append-only rule) and on the CQG sweep runner."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from graphity.cqg import run_chain, torus
from graphity.results import ResultWriter

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    """scripts/ is not a package, so load a script by file path."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_runner():
    return load_script("run_cqg_sweep")


def test_comparison_with_published_points(tmp_path, capsys):
    """Matches each published point to the nearest coupling on the same leg, averaging replicas."""
    ours = tmp_path / "ours.csv"
    ours.write_text("leg,g,phi\ncool,10,0.30\ncool,10,0.34\ncool,5,0.60\nheat,10,0.90\n")
    published = tmp_path / "published.csv"
    published.write_text("series,log10_g,g,S_over_N\ncool_from_random_blue,0.996,9.9,0.31\n"
                         "heat_from_torus_red,0.996,9.9,0.50\n")
    load_script("compare_with_published").main(ours, published)
    lines = capsys.readouterr().out.splitlines()
    cool, heat = (next(l for l in lines if l.startswith(leg)).split() for leg in ("cool", "heat"))
    assert (float(cool[2]), float(cool[4]), float(cool[6])) == (10.0, 0.32, 0.01)     # mean of 0.30 and 0.34
    assert (float(heat[2]), float(heat[4]), float(heat[6])) == (10.0, 0.90, 0.40)     # own leg, not the cool one
    assert lines[-2] == "largest |difference| = 0.400"

    # interpolation, for published grids that differ from ours: linear in ln g, points outside skipped
    published.write_text("series,log10_g,g,S_over_N\ncool_N160,0,7.0711,0.40\ncool_N160,0,50,0.10\n"
                         "cool_N160,0,5.5,0.55\n")
    load_script("compare_with_published").main(ours, published, interpolate=True, g_min=6.0)
    rows = [l.split() for l in capsys.readouterr().out.splitlines() if l.startswith("cool")]
    assert len(rows) == 1                                   # g = 50 is outside our couplings, g = 5.5 below g_min
    assert float(rows[0][4]) == 0.46                        # halfway in ln g between 0.60 at g = 5 and 0.32 at g = 10


def tiny_config(tmp_path, **extra):
    cfg = dict(name="tiny", sides=[6], couplings=[50, 5, 1], n_melt=50, n_equil=5,
               n_meas=40, replicas=1, seed=7, **extra)
    path = tmp_path / "tiny.json"
    path.write_text(json.dumps(cfg))
    return cfg, path


def read_rows(path):
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def test_writer_streams_rows_then_renames(tmp_path):
    with ResultWriter("run", {"note": "meta"}, tmp_path) as out:
        out.write(dict(a=1, b=2.5))
        assert not out.csv_path.exists()                      # not final until the run completes
        assert len(read_rows(out.partial_path)) == 1          # but already on disk
        out.write(dict(a=3, b=4.5))
    assert read_rows(tmp_path / "run.csv") == [dict(a="1", b="2.5"), dict(a="3", b="4.5")]
    assert json.loads((tmp_path / "run.meta.json").read_text()) == {"note": "meta"}
    assert not out.partial_path.exists()


def test_writer_refuses_to_overwrite(tmp_path):
    with ResultWriter("run", {}, tmp_path) as out:
        out.write(dict(a=1))
    before = (tmp_path / "run.csv").read_bytes()
    with pytest.raises(FileExistsError):
        ResultWriter("run", {}, tmp_path)
    assert (tmp_path / "run.csv").read_bytes() == before


def test_failed_run_leaves_only_a_partial_file(tmp_path):
    with pytest.raises(RuntimeError):
        with ResultWriter("run", {}, tmp_path) as out:
            out.write(dict(a=1))
            raise RuntimeError("simulated crash")
    assert not (tmp_path / "run.csv").exists()
    assert not (tmp_path / "run.meta.json").exists()
    assert len(read_rows(out.partial_path)) == 1              # finished rows survive the crash


def test_runner_reproduces_direct_chain_calls(tmp_path):
    """The runner adds bookkeeping only: same seeds, same numbers, to the last bit."""
    cfg, path = tiny_config(tmp_path)
    load_runner().main(path, tmp_path)
    rows = read_rows(tmp_path / "tiny.csv")
    assert [(r["leg"], float(r["g"])) for r in rows] == [
        ("cool", 50), ("cool", 5), ("cool", 1), ("heat", 5), ("heat", 50)]

    side, seed = 6, cfg["seed"] + 6                           # seed scheme of the runner, replica 0
    adj, part = torus(side)
    side_u = np.flatnonzero(part == 0)
    run_chain(adj, side_u, 0.0, cfg["n_melt"], 1, seed)
    for k, row in enumerate(rows, start=1):
        s, _, acc = run_chain(adj, side_u, 1.0 / float(row["g"]), cfg["n_equil"], cfg["n_meas"], seed + k)
        phi = s / side**2
        assert float(row["phi"]) == phi.mean()
        assert float(row["chi"]) == side**2 * phi.var()
        assert float(row["acceptance"]) == acc
        assert float(row["phi_err"]) >= 0 and float(row["chi_err"]) >= 0
        assert (row["lam"], row["cap"], float(row["surplus"])) == ("1.0", "2", 0.0)   # the defaults
        assert float(row["pieces"]) >= 1 and 0 < float(row["largest_frac"]) <= 1
        assert float(row["baby_frac"]) == float(row["cube_frac"]) == 0   # three squares on an edge: not under the cap


def test_runner_model_keys(tmp_path):
    """cap: null lifts the cap, lambda and acceptance reach the kernel, and bad values are refused."""
    cfg, path = tiny_config(tmp_path, cap=None, acceptance="glauber", seed_scheme="independent")
    cfg["lambda"] = 0.0
    cfg["couplings"] = [50, 3]
    path.write_text(json.dumps(cfg))
    load_runner().main(path, tmp_path)
    rows = read_rows(tmp_path / "tiny.csv")
    assert {(r["lam"], r["cap"]) for r in rows} == {("0.0", "none")}
    cold = [r for r in rows if r["leg"] == "cool" and float(r["g"]) == 3][0]
    assert float(cold["surplus"]) > 0            # with no cap and no penalty, over-full edges appear
    assert {"pieces", "largest_frac", "baby_frac", "cube_frac"} <= set(cold)

    for key, bad in [("cap", 3), ("acceptance", "heat-bath")]:
        cfg2, path2 = tiny_config(tmp_path, **{key: bad})
        with pytest.raises(ValueError):
            load_runner().main(path2, tmp_path)


def test_runner_refuses_a_second_run(tmp_path):
    _, path = tiny_config(tmp_path)
    runner = load_runner()
    runner.main(path, tmp_path)
    with pytest.raises(FileExistsError):
        runner.main(path, tmp_path)


def test_heating_leg_can_start_from_the_torus(tmp_path):
    _, path = tiny_config(tmp_path, heat_start="torus")
    load_runner().main(path, tmp_path)
    heat = [r for r in read_rows(tmp_path / "tiny.csv") if r["leg"] == "heat"]
    assert [float(r["g"]) for r in heat] == [1, 5, 50]        # coldest point included: it is a fresh start
    assert float(heat[0]["phi"]) > 0.9                        # still the torus at g = 1
    assert float(heat[-1]["phi"]) < 0.6                       # and it melts on the way up


def test_unknown_heat_start_is_rejected(tmp_path):
    _, path = tiny_config(tmp_path, heat_start="hexagons")
    with pytest.raises(ValueError):
        load_runner().main(path, tmp_path)


def test_runner_accepts_a_rectangle(tmp_path):
    cfg, path = tiny_config(tmp_path, seed_scheme="independent")
    cfg["sides"] = [[8, 6]]
    path.write_text(json.dumps(cfg))
    load_runner().main(path, tmp_path)
    rows = read_rows(tmp_path / "tiny.csv")
    assert {(r["N"], r["lx"], r["ly"]) for r in rows} == {("48", "8", "6")}


def test_rectangle_is_refused_under_the_legacy_seed_scheme(tmp_path):
    cfg, path = tiny_config(tmp_path)
    cfg["sides"] = [[8, 6]]
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        load_runner().main(path, tmp_path)
    assert list(tmp_path.glob("*.csv*")) == []                # refused before any file was made


def test_seed_schemes():
    runner = load_runner()
    legacy = dict(seed=100)
    assert runner.seeder(legacy, 10, 0)(5) == runner.seeder(legacy, 14, 0)(1)     # the known flaw (Q7)

    cfg = dict(seed=100, seed_scheme="independent")
    seeds = {runner.seeder(cfg, entry, rep)(step)
             for entry in (10, 14, [16, 10], [10, 16]) for rep in (0, 1) for step in range(40)}
    assert len(seeds) == 4 * 2 * 40                            # no two alike
    assert all(0 <= s < 2**32 for s in seeds)                  # what numba's generator accepts
    assert runner.seeder(cfg, [16, 10], 1)(7) == runner.seeder(cfg, [16, 10], 1)(7)


def quench_config(tmp_path, **extra):
    cfg = dict(name="tinyq", sides=[[8, 4]], g=0, n_melt=30, n_sweeps=60, replicas=2, seed=9,
               seed_scheme="independent", cap=None, **extra)
    cfg.setdefault("lambda", 0.0)
    path = tmp_path / "tinyq.json"
    path.write_text(json.dumps(cfg))
    return cfg, path


@pytest.mark.parametrize("lam", [0.0, 1.0])
def test_zero_temperature_quench_never_raises_the_energy(tmp_path, lam):
    """g = 0: a switch is accepted only if H does not go up, so H per vertex falls or stays, sweep after sweep."""
    cfg, path = quench_config(tmp_path, **{"lambda": lam})
    load_script("run_cqg_quench").main(path, tmp_path)
    rows = read_rows(tmp_path / "tinyq.csv")
    assert len(rows) == 2 * 60 and {r["N"] for r in rows} == {"32"}
    for rep in ("0", "1"):
        mine = [r for r in rows if r["replica"] == rep]
        assert [int(r["sweep"]) for r in mine] == list(range(1, 61))
        h = [16 * (1 - float(r["phi"])) + 4 * lam * float(r["surplus"]) for r in mine]
        assert all(b <= a + 1e-12 for a, b in zip(h, h[1:])) and h[-1] < h[0]
        assert all(1 <= int(r["pieces"]) and 0 <= float(r["cube_frac"]) <= float(r["baby_frac"]) <= 1 for r in mine)


def test_quench_options(tmp_path):
    cfg, path = quench_config(tmp_path, record_every=20)
    load_script("run_cqg_quench").main(path, tmp_path)
    assert [int(r["sweep"]) for r in read_rows(tmp_path / "tinyq.csv") if r["replica"] == "0"] == [20, 40, 60]
    with pytest.raises(FileExistsError):                          # append-only, like the sweep runner
        load_script("run_cqg_quench").main(path, tmp_path)
    cfg2, path2 = quench_config(tmp_path, acceptance="glauber")
    (tmp_path / "tinyq.csv").unlink(), (tmp_path / "tinyq.meta.json").unlink()
    with pytest.raises(ValueError):                               # zero temperature is defined for Metropolis only
        load_script("run_cqg_quench").main(path2, tmp_path)


def test_ergodicity_script(tmp_path):
    """One row per class, a marker row for a size with no states, and the append-only rule as everywhere."""
    path = tmp_path / "erg.json"
    path.write_text(json.dumps(dict(name="erg", sizes=[6, 7], caps=[None])))
    load_script("check_ergodicity").main(path, tmp_path)
    rows = read_rows(tmp_path / "erg.csv")
    assert [(r["N"], r["class_id"], r["labelled_total"], r["all_joined"]) for r in rows] == [
        ("12", "-1", "0", ""), ("14", "0", "151200", "True")]
    with pytest.raises(FileExistsError):
        load_script("check_ergodicity").main(path, tmp_path)
    path.write_text(json.dumps(dict(name="erg2", sizes=[7], caps=[3])))
    with pytest.raises(ValueError):
        load_script("check_ergodicity").main(path, tmp_path)


def test_dip_census_script(tmp_path):
    """At N = 16 and lambda = 0.5 the 4-cube is the only dip and the cheapest way out of it costs 32 (1 - lambda)."""
    path = tmp_path / "dips.json"
    path.write_text(json.dumps(dict(name="dips", sizes=[6, 8], lambdas=[0.5, 1.5])))
    load_script("dip_census").main(path, tmp_path)
    rows = read_rows(tmp_path / "dips.csv")
    assert {r["N"] for r in rows} == {"16"} and len(rows) == 2 * 5        # n = 6 has no states and is skipped
    dips = [r for r in rows if r["is_dip"] == "True"]
    assert [(r["lam"], r["squares"], float(r["way_out"]), float(r["above_ground"])) for r in dips] == [
        ("0.5", "24", 16.0, 0.0)]


def test_exact_small_averages_script(capsys):
    """At N = 16 and lambda = 1 every state has H = 0, so phi is plain counting, 20.8 / 16, whatever g is; and no
    state that small is valid under the cap, so no 'cap' line is printed."""
    table = ROOT / "results" / "ergodicity_small.csv"
    load_script("exact_small_averages").main(table, 16, [5.0, 50.0], [1.0])
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "N = 16: 5 classes, 635,040,000 labelled states"
    values = [l.split("|")[1].split() for l in lines[2:]]
    assert [v[0] for v in values] == ["1.3000", "1.3000"] and not any("cap" in l for l in lines)


def test_tempering_runner(tmp_path):
    """One row per coupling, hottest first, with a swap rate for every coupling but the coldest."""
    cfg = dict(name="pt", sides=[[8, 6]], couplings=[9.0, 7.0, 5.5], n_melt=30, sweeps_per_round=2,
               rounds_equil=10, rounds_meas=40, replicas=1, seed=11)
    path = tmp_path / "pt.json"
    path.write_text(json.dumps(cfg))
    load_script("run_cqg_tempering").main(path, tmp_path)
    rows = read_rows(tmp_path / "pt.csv")
    assert [(r["N"], r["leg"], float(r["g"])) for r in rows] == [("48", "temper", 9.0), ("48", "temper", 7.0),
                                                                  ("48", "temper", 5.5)]
    assert [r["swap_rate"] == "" for r in rows] == [False, False, True]
    assert all(0 < float(r["phi"]) < 1.2 and float(r["acceptance"]) > 0 for r in rows)
    assert float(rows[0]["phi"]) < float(rows[2]["phi"])          # colder is more ordered
    with pytest.raises(FileExistsError):
        load_script("run_cqg_tempering").main(path, tmp_path)
    cfg.update(name="pt2", couplings=[5.0, 7.0])
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):                               # coldest first
        load_script("run_cqg_tempering").main(path, tmp_path)


def test_quench_from_the_flat_torus_at_several_couplings(tmp_path):
    """start = torus: every coupling gets a fresh flat sheet, whose energy is exactly 0 until something gives.
    At lambda = 0.5 and g = 1 nothing does (the cheapest way out costs 16 lambda = 8, against g = 1)."""
    cfg, path = quench_config(tmp_path, start="torus", **{"lambda": 0.5})
    cfg.update(g=[1.0, 40.0], sides=[[8, 6]], n_sweeps=40)
    path.write_text(json.dumps(cfg))
    load_script("run_cqg_quench").main(path, tmp_path)
    rows = read_rows(tmp_path / "tinyq.csv")
    cold = [r for r in rows if float(r["g"]) == 1.0]
    hot = [r for r in rows if float(r["g"]) == 40.0]
    assert len(cold) == len(hot) == 2 * 40
    assert all(float(r["energy"]) == 0 and float(r["phi"]) == 1 for r in cold)
    assert float(hot[-1]["energy"]) > 5                           # melted: most squares gone
    assert all(float(r["energy"]) == pytest.approx(16 * (1 - float(r["phi"])) + 4 * 0.5 * float(r["surplus"]))
               for r in rows)
    cfg2, path2 = quench_config(tmp_path, start="cubes")
    (tmp_path / "tinyq.csv").unlink(), (tmp_path / "tinyq.meta.json").unlink()
    with pytest.raises(ValueError):
        load_script("run_cqg_quench").main(path2, tmp_path)


def test_spark_threshold_script(tmp_path):
    """Sealed runs from a perfect tube: below the wall nothing moves, above it every run leaves."""
    cfg = dict(name="spark", sides=[[8, 4]], sparks=[4, 16], n_sweeps=600, replicas=2, seed=13,
               seed_scheme="independent", cap=None, acceptance="metropolis")
    cfg["lambda"] = 1.25
    path = tmp_path / "spark.json"
    path.write_text(json.dumps(cfg))
    load_script("run_spark_threshold").main(path, tmp_path)
    rows = read_rows(tmp_path / "spark.csv")
    assert len(rows) == 4 and {r["N"] for r in rows} == {"32"}
    small = [r for r in rows if float(r["spark"]) == 4]
    big = [r for r in rows if float(r["spark"]) == 16]
    assert all(r["left"] == "False" and r["first_change"] == "" for r in small)
    assert all(float(r["energy_start"]) == 1.0 and float(r["phi_start"]) == 1.25 for r in rows)
    assert all(r["left"] == "True" and int(r["first_change"]) > 0 for r in big)
    cfg["acceptance"] = "glauber"
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):                               # a sealed run has no acceptance rule to choose
        load_script("run_spark_threshold").main(path, tmp_path)


PROVENANCE_KEYS = {"git_commit", "git_dirty", "git_untracked_count", "git_untracked"}


def test_provenance_names_a_commit_or_says_it_cannot(tmp_path):
    from graphity.results import provenance
    here = provenance()
    assert set(here) == PROVENANCE_KEYS
    assert here["git_commit"] == "unknown" or (len(here["git_commit"]) == 40 and here["git_dirty"] in (True, False))
    assert set(provenance(tmp_path)) == PROVENANCE_KEYS          # a folder outside any checkout still answers


def test_provenance_counts_untracked_files_as_dirty(tmp_path):
    """An untracked file (not ignored) makes the tree dirty and is listed; an ignored one does not count. Until
    2026-10-12 untracked files were invisible to this helper, which let results name a commit holding none of
    their code (docs/theory/orh_findings.md, the provenance note)."""
    import subprocess
    from graphity.results import provenance

    def git(*args):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args], cwd=tmp_path,
                       check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / ".gitignore").write_text("*.log\n")
    (tmp_path / "tracked.txt").write_text("a\n")
    git("add", ".")
    git("commit", "-q", "-m", "start")
    clean = provenance(tmp_path)
    assert len(clean["git_commit"]) == 40
    assert (clean["git_dirty"], clean["git_untracked_count"], clean["git_untracked"]) == (False, 0, [])

    (tmp_path / "scratch.log").write_text("ignored\n")                   # matches .gitignore: not counted
    assert provenance(tmp_path)["git_dirty"] is False

    (tmp_path / "new_code.py").write_text("x = 1\n")                     # untracked, not ignored
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "inner.py").write_text("y = 2\n")
    dirty = provenance(tmp_path)
    assert dirty["git_commit"] == clean["git_commit"]
    assert dirty["git_dirty"] is True
    assert dirty["git_untracked_count"] == 2
    assert dirty["git_untracked"] == ["new_code.py", "pkg/inner.py"]     # files, not a collapsed folder
    assert provenance(tmp_path, max_listed=1)["git_untracked"] == ["new_code.py"]

    (tmp_path / "tracked.txt").write_text("b\n")                          # a modified tracked file still counts
    git("add", "new_code.py", "pkg")
    git("commit", "-q", "-m", "add code")
    assert provenance(tmp_path) == dict(git_commit=provenance(tmp_path)["git_commit"], git_dirty=True,
                                        git_untracked_count=0, git_untracked=[])
