"""T52's window builder (scripts/exact_hidden_relics.py): which points a window holds, which relics and which flat
squares it is built round. The count itself is `graphity.hidden`, tested in tests/test_hidden.py."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import exact_hidden_relics as t52                        # noqa: E402
from analyse_t37 import leftovers                        # noqa: E402
from graphity.cqg import NO_CAP, torus                   # noqa: E402
from graphity.hidden import hidden_count                 # noqa: E402

SAVED = ROOT / "results" / "t37_lam125_g125_L1024_a_adj" / "N4096_rep0.npz"
CFG = json.loads((ROOT / "configs" / "t52_hidden_relics.json").read_text())


def cut_of(adj, region):
    inside = set(region)
    return sum(1 for v in region for w in adj[v] if int(w) not in inside)


def test_ball_in_flat_space_grows_like_a_sheet():
    adj, _ = torus(16, 16)
    sizes = [(len(b), cut_of(adj, b)) for b in (t52.ball(adj, (0, 1, 16, 17), r) for r in range(4))]
    assert sizes == [(4, 8), (12, 16), (24, 24), (40, 32)]
    assert t52.ball(adj, (0, 1, 16, 17), 0) == [0, 1, 16, 17]


def test_no_relic_in_a_flat_torus_or_in_the_tube():
    assert t52.relics(torus(16, 16)[0]) == []
    assert t52.relics(torus(16, 4, NO_CAP)[0]) == []          # every point at d = 1, but one piece of 64, not four


def test_flat_squares_come_in_order_of_their_lowest_vertex():
    adj, _ = torus(16, 16)
    assert t52.flat_squares(adj, 3, 6) == [(0, 1, 16, 17), (0, 1, 240, 241), (0, 15, 16, 31)]
    assert t52.flat_squares(torus(16, 4, NO_CAP)[0], 3, 6) == []   # the tube has no flat point at all


def test_windows_of_a_flat_torus_are_three_squares_and_the_first_at_radius_two():
    adj, _ = torus(16, 16)
    w = t52.windows(adj, CFG)
    assert [(kind, radius) for kind, _, radius in w] == [("flat", 1), ("flat", 2), ("flat", 1), ("flat", 1)]
    assert len({anchors for _, anchors, _ in w}) == 3


def test_a_flat_window_hides_one_shape_under_four_names():
    # Amendment 1: the square's own four points have no link out of the window, so the two on each side can swap
    # names: four labeled wirings of one and the same graph.
    adj, part = torus(16, 16)
    r = hidden_count(adj, part, t52.ball(adj, (0, 1, 16, 17), 1), 1.25)
    assert (r["points"], r["cut"], r["count"], r["shapes"]) == (12, 16, 4, 1)


def test_the_two_ways_of_counting_agree_on_a_small_window(tmp_path):
    adj, part = torus(8, 8)
    state = tmp_path / "flat.npz"
    np.savez(state, adj=adj, part=part)
    full = t52.count_one(state, 1, (0, 1, 8, 9), 1.25, True)
    ties = t52.count_one(state, 1, (0, 1, 8, 9), 1.25, False)
    assert (full["count"], full["shapes"]) == (ties["count"], ties["shapes"]) == (4, 1)
    assert full["levels"] is not None and ties["levels"] is None and ties["valid"] is None


def test_the_config_is_the_preregistered_one():
    assert CFG["lambda"] == 1.25 and CFG["radii_all"] == [1] and CFG["radii_first"] == [2]
    assert (CFG["first_per_state"], CFG["flat_squares_per_state"], CFG["flat_distance"]) == (1, 3, 6)
    assert CFG["flat_first_per_state"] == 1
    assert CFG["time_limit_s"] == 1200 and CFG["states"] == "results/t37_lam125_g125_L1024_*_adj/*.npz"


def test_a_window_past_its_time_limit_is_abandoned(tmp_path):
    adj, part = torus(8, 8)
    state = tmp_path / "flat.npz"
    np.savez(state, adj=adj, part=part)
    assert t52.count_with_limit(state, 1, (0, 1, 8, 9), 1.25, 0.001) is None


@pytest.mark.skipif(not SAVED.exists(), reason="the saved T37 end state is not in this checkout")
def test_relics_are_the_columns_t37_reads_and_sit_in_narrow_windows():
    adj = np.load(SAVED)["adj"]
    found = t52.relics(adj)
    assert len(found) == leftovers(adj)[0] > 0
    assert found == sorted(found) and all(len(r) == 4 for r in found)
    dist = t52.distance_from_damage(adj)
    assert all(dist[list(r)].max() == 0 for r in found)
    for relic in found[:3]:                                   # the sizes disclosed in the pre-registration
        assert [len(t52.ball(adj, relic, r)) for r in (1, 2)] == [12, 20]
    for square in t52.flat_squares(adj, 3, 6):
        assert dist[list(square)].min() >= 6


def test_a_resumed_run_reuses_the_dead_runs_rows_and_recounts_the_rest(tmp_path):
    """--resume (9 October): the rows a dead run left in its .partial are written again unchanged, the other windows
    are counted, the result is the same file a fresh run gives apart from `seconds`, and the meta records the reuse."""
    adj, part = torus(8, 8)
    np.savez(tmp_path / "flat.npz", adj=adj, part=part)
    cfg = dict(name="t52_resume_test", **{"lambda": 1.25}, states=str(tmp_path / "*.npz"), radii_all=[1],
               radii_first=[], first_per_state=0, flat_squares_per_state=3, flat_distance=6, time_limit_s=120,
               flat_first_per_state=0)
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    fresh = tmp_path / "fresh"
    t52.main(str(cfg_path), str(fresh))
    rows_fresh = list(csv.DictReader(open(fresh / "t52_resume_test.csv", newline="")))
    assert len(rows_fresh) == 3 and all(r["counted"] == "True" for r in rows_fresh)

    dead = tmp_path / "dead"
    dead.mkdir()
    with open(fresh / "t52_resume_test.csv", newline="") as fh, \
            open(dead / "t52_resume_test.csv.partial", "w", newline="") as out:
        for i, line in enumerate(fh):
            if i < 3:                                   # the header and the first two rows: the run died after them
                out.write(line)
    t52.main(str(cfg_path), str(dead), resume=True)
    rows_dead = list(csv.DictReader(open(dead / "t52_resume_test.csv", newline="")))
    strip = lambda r: {k: v for k, v in r.items() if k != "seconds"}          # noqa: E731
    assert [strip(r) for r in rows_dead] == [strip(r) for r in rows_fresh]
    assert [r["seconds"] for r in rows_dead[:2]] == [r["seconds"] for r in rows_fresh[:2]]
    meta = json.loads((dead / "t52_resume_test.meta.json").read_text())
    assert meta["resumed"]["rows_reused"] == 2
    assert (dead / "t52_resume_test.csv.partial.resumed").is_file()
    assert not (dead / "t52_resume_test.csv.partial").exists()
