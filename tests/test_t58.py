"""T58's reading rules on made-up traces: each label is reached by the history it names, and the reproduction
gate sees a changed count but not a last-digit difference in an average."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyse_t58 as t58  # noqa: E402
import make_t58_configs  # noqa: E402

TUBE = dict(S=80, X=64, d0=0, d1=64, d2=0, baby=0, pieces=1)


def trace(events, blocks=400):
    """A tube that sits still except at the blocks named in `events` (block index -> changed fields)."""
    out = []
    for i in range(1, blocks + 1):
        t = dict(TUBE, sweep=5 * i)
        t.update(events.get(i, {}))
        out.append(t)
    return out


def test_a_tube_that_gains_squares_is_a_second_curl():
    knot = dict(S=84, X=80, d0=16, d1=48, baby=16, pieces=2)
    c = t58.classify_target(1500, trace({i: knot for i in range(100, 250)}))
    assert c["label"] == "SECOND CURL" and c["highest_s"] == 84 and c["most_d0"] == 16 and c["most_pieces"] == 2
    assert c["most_energy"] == 16 * (64 - 84) + 4 * 1.3 * 80 - (16 * (64 - 80) + 4 * 1.3 * 64)   # +19.2, a knot's cost


def test_eight_points_at_d0_count_but_two_do_not():
    assert t58.classify_target(1500, trace({120: dict(d0=8, d1=56)}))["label"] == "SECOND CURL"
    assert t58.classify_target(1500, trace({120: dict(d0=2, d1=62)}))["label"] == "TRUE WAIT"   # one block in 259
    c = t58.classify_target(1500, trace({120: dict(S=78, X=60, d0=2, d1=58, d2=4)}))
    assert c["label"] == "LOW THRESHOLD"


def test_squares_lost_unseen_is_a_low_threshold():
    exit_a = dict(S=78, X=60, d1=60, d2=4)
    c = t58.classify_target(1500, trace({i: exit_a for i in (60, 61, 62, 150)}))
    assert c["label"] == "LOW THRESHOLD"
    assert c["unseen_exits"] == 2 and c["off_blocks"] == 4 and c["lowest_s_unseen"] == 78
    assert c["zero_noise_wait"] == 300


def test_a_tube_that_sat_still_is_a_true_wait():
    c = t58.classify_target(1500, trace({}))
    assert c["label"] == "TRUE WAIT" and c["tube_share"] == 1.0 and c["zero_noise_wait"] is None
    # an excursion inside the watch (the first 40 blocks) is not part of the stretch
    c = t58.classify_target(1500, trace({i: dict(S=76, X=56, d1=56, d2=8) for i in range(10, 30)}))
    assert c["label"] == "TRUE WAIT"


def test_same_squares_other_wiring_is_another_state():
    other = dict(S=80, X=66, d1=60, d2=4)
    c = t58.classify_target(1500, trace({i: other for i in range(50, 200)}))
    assert c["label"] == "OTHER STATE"


def test_verdict_on_three():
    assert t58.verdict(["SECOND CURL", "SECOND CURL", "LOW THRESHOLD"]) == "SECOND CURL"
    assert t58.verdict(["LOW THRESHOLD"] * 3) == "LOW THRESHOLD"
    assert t58.verdict(["TRUE WAIT"] * 3) == "TRUE WAIT"
    assert t58.verdict(["SECOND CURL", "LOW THRESHOLD", "TRUE WAIT"]) == "MIXED"


def test_reproduction_gate():
    old = {1: dict(replica="1", waiting="505", released="1.2000000000000002", _file="a")}
    assert t58.reproduction(old, {1: dict(replica="1", waiting="505", released="1.2", extra="7")}) == []
    assert t58.reproduction(old, {1: dict(replica="1", waiting="510", released="1.2000000000000002")}) \
        == [(1, "waiting", "505", "510")]
    assert t58.reproduction(old, {}) == [(1, "missing", "", "")]


def test_population_times_the_first_exit_with_no_detector():
    rows = [dict(tube_at_200="1", first_left=str(200 + w), first_left_after_200=str(200 + w), thresh="1.2499",
                 waiting=str(200 + w), phi_max="1.25", d0_max="0") for w in (100, 300, 500)]
    rows.append(dict(tube_at_200="0", first_left="60", first_left_after_200="205", thresh="1.1", waiting="",
                     phi_max="1.25", d0_max="2"))
    rows.append(dict(tube_at_200="1", first_left="40", first_left_after_200="400", thresh="1.20", waiting="5200",
                     phi_max="1.3125", d0_max="16"))
    rows.append(dict(tube_at_200="1", first_left="", first_left_after_200="", thresh="1.2499", waiting="",
                     phi_max="1.25", d0_max="0"))
    p = t58.population(rows, 300.0)
    assert p["n"] == 6 and p["never"] == 1 and p["mean_first"] == (300 + 500 + 700 + 60 + 40) / 5
    assert p["beyond_10"] == 1                       # the decay that never left counts as beyond
    assert p["perfect"] == 5 and p["never_left"] == 1 and p["mean_left"] == (100 + 300 + 500 + 200) / 4
    assert p["beyond_10_left"] == 1
    assert p["low"] == 1 and p["wait_low"] == (5000.0, 1) and p["plain"] == 4
    assert p["above_tube"] == 1 and p["d0_curl"] == 1 and p["d0_any"] == 2 and p["d0_most"] == 16


def test_configs_replay_the_original_seeds():
    import json

    configs = Path(make_t58_configs.CONFIGS)
    keys = ("sides", "g", "n_sweeps", "block", "stop_at", "settle", "settle_max", "seed", "lambda")
    for i in range(make_t58_configs.FILES):
        old = json.loads((configs / ("t38_lam130_n64_%02d.json" % i)).read_text())
        new = make_t58_configs.config(i)
        for key in keys + ("replica_ids",):
            assert new[key] == old[key]
        assert new["record_detector"] is True and "save_adjacency" not in new and "trace_replicas" not in new
    old = json.loads((configs / "t38_lam130_n64_02.json").read_text())
    new = make_t58_configs.targets_config()
    for key in keys:
        assert new[key] == old[key]
    assert new["replica_ids"] == new["trace_replicas"] == sorted(make_t58_configs.TARGETS + make_t58_configs.CONTROLS)
    assert tuple(make_t58_configs.TARGETS) == t58.TARGETS and tuple(make_t58_configs.CONTROLS) == t58.CONTROLS
