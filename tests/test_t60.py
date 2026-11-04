"""T60's reading rules (scripts/analyse_t60.py): the wait is the first look at which a decay stopped being the
tube, the detector's own clock no longer decides anything, and the held-out size is scored against numbers fixed
from the smaller sizes."""
import csv
import json
import math
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import analyse_t23  # noqa: E402
import analyse_t60 as a  # noqa: E402
import make_t60_configs  # noqa: E402


def memoryless(k, mean):
    """k waits spread evenly over a memoryless distribution of the given mean, rounded up to a look."""
    return [5 * math.ceil(-mean * math.log(1 - (i + 0.5) / k) / 5) for i in range(k)]


def row(rep, n=64, lam=1.25, first_left=800, waiting=800, two=0.99, largest=0.95):
    d1 = int(round(two * n / 2))
    d2 = int(round(two * n)) - d1
    return dict(N=str(n), replica=str(rep), lam=str(lam), f_200="0.0", waiting=str(waiting), reached="0.98",
                released=str(4 * (lam - 1)), d1_50=str(d1), d2_50=str(d2), largest_50=str(largest), pieces_25="1",
                first_left=str(first_left))


def flat_states(k):
    return {str(i): dict(valid=True, h=0.0, kind="FLAT", release_exact=1.0, census=(0, 0, 64, 0, 0, 0, 0))
            for i in range(k)}


def test_a_memoryless_first_exit_is_sharp_and_a_slow_tail_is_not():
    lam, tau = 1.25, a.analyse_t8.tau(1.25)
    rows = [row(i, first_left=w) for i, w in enumerate(memoryless(120, tau))]
    c = a.cell(rows, 64, lam, flat_states(120))
    assert c["a"] and c["sharp"] and c["n_read"] == 120 and c["never_left"] == 0
    assert abs(c["ratio"] - 1.0) < 0.02 and c["law"]
    slow = memoryless(120, tau)
    slow[-3:] = [40 * tau, 50 * tau, 60 * tau]                      # three tubes that sat for tens of mean waits
    c = a.cell([row(i, first_left=w) for i, w in enumerate(slow)], 64, lam, flat_states(120))
    assert not c["a"] and not c["sharp"] and c["cv"] > c["band"][1]


def test_the_detectors_clock_no_longer_decides():
    """T24 failed two cells on one enormous detected wait each, which were not waits. Here `waiting` can say
    anything: the check reads the first look at which the tube stopped being the tube."""
    lam, tau = 1.30, a.analyse_t8.tau(1.30)
    waits = memoryless(120, tau)
    rows = [row(i, lam=lam, first_left=w, waiting=(31955 if i == 0 else w + 200)) for i, w in enumerate(waits)]
    c = a.cell(rows, 64, lam, flat_states(120))
    assert c["a"] and c["sharp"]
    old = analyse_t23.cell(rows, 64, lam, read=None)
    assert not old["a"]                                              # the same rows fail T23's check on `waiting`


def test_a_decay_that_never_left_enters_at_the_cap():
    rows = [row(i, first_left=w) for i, w in enumerate(memoryless(119, 845))]
    rows.append(dict(row(119), first_left=""))
    waits, never = a.first_exits(rows, 100000)
    assert never == 1 and waits[-1] == 100000 and len(waits) == 120
    c = a.cell(rows, 64, 1.25, flat_states(120))
    assert c["never_left"] == 1 and not c["a"]                       # one wait of 118 means breaks the band


def test_the_line_and_the_recipe():
    assert abs(a.line_at([64, 96, 144, 192], [0.9, 0.8, 0.65, 0.5], 288) - 0.2) < 1e-9
    cells = [dict(N=n, ratio=r, two=t, largest=f)
             for n, r, t, f in ((64, 1.10, 0.980, 0.98), (96, 1.06, 0.985, 0.95), (144, 1.04, 0.990, 0.90),
                                (192, 1.00, 0.995, 0.86))]
    p = a.predict(cells)
    assert abs(p["ratio"][0] - 1.05) < 1e-9 and abs(p["ratio"][1] - 1.05 * 0.68) < 1e-9
    assert p["two"][0] == 1.0 and p["two"][2] == 1.0 and abs(p["two"][1] - 0.97) < 1e-9     # clipped at 1
    front = a.line_at([64, 96, 144, 192], [0.98, 0.95, 0.90, 0.86], 288)
    assert abs(p["largest"][0] - front) < 1e-9 and abs(p["largest"][1] - (front - 0.10)) < 1e-9 and p["sharp"]
    falling = [dict(c, largest=v) for c, v in zip(cells, (0.95, 0.88, 0.78, 0.70))]
    assert not a.predict(falling)["sharp"]                           # the line reaches 0.52 at 288
    with pytest.raises(AssertionError):
        a.predict(cells[:3])


def held(two=0.99, largest=0.80, ratio=1.02, **more):
    c = dict(metastable=True, gate2=True, gate3=True, a=True, b=two >= 0.8, c=largest >= 0.7, cv=1.0, band=(0.756, 1.379),
             two=two, largest=largest, ratio=ratio)
    c.update(more)
    c["sharp"] = c["a"] and c["b"] and c["c"]
    return c


def test_the_held_out_size_is_scored_against_the_fixed_numbers():
    p = dict(ratio=[1.05, 0.714, 1.386], two=[0.99, 0.96, 1.0], largest=[0.78, 0.68, 0.88], sharp=True)
    predictions = {lam: p for lam in analyse_t23.WINDOW}
    cells = {lam: held() for lam in analyse_t23.WINDOW}
    first, second = a.held_out_verdict(cells, predictions)
    assert first == "AS PREDICTED AT THE HELD-OUT SIZE" and "1.05, 1.10, 1.15, 1.20, 1.25, 1.30" in second
    cells[1.30] = held(largest=0.55)                                  # the front broke up more than the line said
    first, second = a.held_out_verdict(cells, predictions)
    assert first.startswith("NOT AS PREDICTED AT 1.30 (one front 0.550 outside 0.680 to 0.880)")
    assert "not sharp or not read at 1.30" in second
    cells[1.30] = held(largest=0.69)                                  # inside the range, and not sharp: both are said
    first, second = a.held_out_verdict(cells, predictions)
    assert first == "AS PREDICTED AT THE HELD-OUT SIZE" and "not sharp or not read at 1.30" in second
    assert a.misses(held(a=False), p)[0].startswith("memoryless")
    assert a.misses(dict(metastable=False), p) == ["not metastable"]
    del cells[1.05]
    assert a.held_out_verdict(cells, predictions)[0].startswith("NOT READ")


def test_the_configs_are_t24s_with_fresh_seeds_and_the_first_look_recorded(tmp_path, monkeypatch):
    stage1 = make_t60_configs.configs()
    assert len(stage1) == 28 and len({c["name"] for c in stage1}) == 28
    old = json.loads((SCRIPTS.parent / "configs" / "t24_lam130_n64.json").read_text())
    new = next(c for c in stage1 if c["name"] == "t60_lam130_n64")
    for key in ("sides", "g", "replicas", "n_sweeps", "block", "stop_at", "settle", "settle_max", "lambda",
                "save_adjacency", "record_f_200"):
        assert new[key] == old[key], key
    assert new["record_detector"] is True and new["seed"] == 20266130 != old["seed"]
    assert {c["seed"] for c in stage1} == {20266000 + v for v in (105, 110, 115, 120, 125, 130, 135)}
    held_out = make_t60_configs.configs(held_out=True)
    assert len(held_out) == 7 and all(c["sides"] == [[72, 4]] and c["n_sweeps"] == 200000 for c in held_out)
    assert all(a.CAP[c["sides"][0][0] * 4] == c["n_sweeps"] for c in stage1 + held_out)
    monkeypatch.setattr(make_t60_configs, "PREDICTION", tmp_path / "none.json")
    with pytest.raises(SystemExit):
        make_t60_configs.main("--held-out")                           # no numbers written yet: no stage 2


def write_cell(folder, lam, n, largest):
    tag = "t60_lam%03d_n%d" % (round(lam * 100), n)
    rows = [row(i, n=n, lam=lam, first_left=w, largest=largest)
            for i, w in enumerate(memoryless(120, a.analyse_t8.tau(lam)))]
    with open(folder / (tag + ".csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_the_whole_report_prints_from_made_up_files(tmp_path, capsys, monkeypatch):
    """Both stages end to end on files shaped like the run's, so that the reading needs no repair afterwards.
    The saved graphs are stood in for (every decay ends flat); reading them is T24's and is tested there."""
    monkeypatch.setattr(a, "PREDICTION", tmp_path / "t60_heldout_prediction.json")
    monkeypatch.setattr(a.analyse_t24, "read_final_states", lambda rows, adj_dir, lam: flat_states(len(rows)))
    lams = (1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35)
    front = {64: 0.98, 96: 0.95, 144: 0.90, 192: 0.86}
    for lam in lams[:3]:
        for n in front:
            write_cell(tmp_path, lam, n, front[n])
    a.main(str(tmp_path))
    assert "STAGE 1 is not complete (12 of 28 cells)" in capsys.readouterr().out
    for lam in lams[3:]:
        for n in front:
            write_cell(tmp_path, lam, n, front[n])
    a.main(str(tmp_path))
    out = capsys.readouterr().out
    assert "WINDOW (1.05 to 1.30, N = 64 to 192): SHARP ACROSS THE WINDOW" in out
    assert "THE RECIPE'S NUMBERS FOR N = 288" in out and "no prediction file yet" in out
    a.main(str(tmp_path), "--write-prediction")
    capsys.readouterr()
    fixed = json.loads((tmp_path / "t60_heldout_prediction.json").read_text())
    assert sorted(fixed) == ["1.05", "1.10", "1.15", "1.20", "1.25", "1.30"]
    assert abs(fixed["1.30"]["largest"][0] - a.line_at(list(front), list(front.values()), 288)) < 1e-9
    with pytest.raises(SystemExit):
        a.main(str(tmp_path), "--write-prediction")                   # the numbers are fixed once
    a.main(str(tmp_path))
    assert "0 of 7 cells. Not complete; not read." in capsys.readouterr().out
    for lam in lams:
        write_cell(tmp_path, lam, 288, 0.75)
    a.main(str(tmp_path))
    out = capsys.readouterr().out
    assert "HELD-OUT SIZE: AS PREDICTED AT THE HELD-OUT SIZE" in out
    assert "sharp at N = 288 at 1.05, 1.10, 1.15, 1.20, 1.25, 1.30" in out
