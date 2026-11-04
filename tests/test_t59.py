"""T59's reading rules on made-up rows: each label is reached by the history it names, and the configs replay the
seeds they say they replay."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyse_t59 as t59  # noqa: E402
import make_t59_configs  # noqa: E402

TUBE96 = dict(S=120, X=96, d0=0, d1=96, d2=0, baby=0, pieces=1)


def trace(events, blocks=400):
    out = []
    for i in range(1, blocks + 1):
        t = dict(TUBE96, sweep=5 * i)
        t.update(events.get(i, {}))
        out.append(t)
    return out


def test_a_tube_of_96_that_sits_until_detected_is_a_true_wait():
    c = t59.classify_wait(1500, trace({300: dict(S=118, X=92, d1=92, d2=4)}), 96)
    assert c["label"] == "TRUE WAIT" and c["first_left"] == 1500 and c["tube_share"] == 1.0


def test_squares_lost_before_the_detector_fired_is_a_low_threshold():
    c = t59.classify_wait(1500, trace({i: dict(S=118, X=92, d1=92, d2=4) for i in (100, 101)}), 96)
    assert c["label"] == "LOW THRESHOLD" and c["first_left"] == 500 and c["off_blocks"] == 2


def test_more_squares_than_the_tube_is_a_second_curl_at_this_size_too():
    assert t59.classify_wait(1500, trace({120: dict(S=124, X=112, d0=16, d1=80)}), 96)["label"] == "SECOND CURL"


def stretch(a, b, sweeps=5000, whole=1):
    return dict(sweeps_in_stretch=str(sweeps), whole_stretch=str(whole), offered_a=str(a), offered_b=str(b))


def test_offers_within_five_per_cent_are_as_counted_and_fewer_are_starved():
    assert t59.offers_label([stretch(15000, 10000), stretch(14400, 10400)]) == "OFFERED AS COUNTED"
    assert t59.offers_label([stretch(15000, 10000), stretch(14000, 10000)]) == "STARVED"      # A at 2.8 a sweep
    assert t59.offers_label([stretch(15000, 9000)]) == "STARVED"                              # B at 1.8 a sweep
    assert t59.offers_label([stretch(900, 600, sweeps=313.4, whole=0)]) == "NO WHOLE STRETCH"
    # a short last stretch is not scored, whatever it holds
    assert t59.offers_label([stretch(15000, 10000), stretch(10, 0, sweeps=40.0, whole=0)]) == "OFFERED AS COUNTED"


def test_expected_exits_from_the_offers_made():
    lam = 1.05
    want = 240000 * math.exp(-15.2 / 1.5) + 160000 * math.exp(-22.0 / 1.5)
    assert abs(t59.expected_taken([stretch(240000, 160000)], lam) - want) < 1e-12
    assert abs(t59.cost_a(1.25) - 12.0) < 1e-12 and abs(t59.cost_b(1.25) - 14.0) < 1e-12


def tubes(first, seen=None, exits=None, tau=100.0):
    """Rows whose first exits, in mean waits, are `first`; the look clock reads `seen` (default: the next multiple
    of five sweeps), and `exits` counts the exits before that look (default 1)."""
    rows = []
    for i, f in enumerate(first):
        sweeps = f * tau
        look = (seen[i] * tau) if seen else 5 * math.ceil(sweeps / 5)
        rows.append(dict(first_exit_sweeps=str(sweeps), seen_every_look=str(look), seen_every_sweep=str(math.ceil(sweeps)),
                         exits_before_look=str(exits[i] if exits else 1)))
    return rows


def test_a_cell_on_the_count_and_one_with_a_slow_tail():
    even = [(i + 0.5) / 1000.0 for i in range(1000)]
    on = [-math.log(1.0 - u) for u in even]                         # one memoryless population, mean 1
    m = t59.clock_numbers(tubes(on), 100.0, 200000)
    assert abs(m["move"][0] - 1.0) < 0.01 and m["tail"][8][0] == 0
    assert t59.verdict_b(m, 0.10, 3) == (True, True, "ON THE COUNT")
    slow = on[:-10] + [9.0] * 10                                    # ten tubes that sat nine mean waits
    m = t59.clock_numbers(tubes(slow), 100.0, 200000)
    assert m["tail"][8][0] == 10
    assert t59.verdict_b(m, 0.10, 3)[2] == "SLOW TAIL"
    late = [1.2 * v for v in on]                                    # every wait a fifth longer, no tail to speak of
    m = t59.clock_numbers(tubes(late), 100.0, 200000)
    assert t59.verdict_b(m, 0.10, 3) == (False, True, "OFF THE COUNT")


def test_a_tube_that_never_left_counts_as_beyond_and_enters_at_the_cap():
    rows = tubes([1.0, 1.0, 1.0])
    rows.append(dict(first_exit_sweeps="", seen_every_look="", seen_every_sweep="", exits_before_look="0"))
    m = t59.clock_numbers(rows, 100.0, 5000)
    assert m["never"] == 1 and m["tail"][10][0] == 1 and abs(m["move"][0] - (3 * 1.0 + 50.0) / 4) < 1e-9


def test_the_four_verdicts_on_the_two_clocks():
    even = [(i + 0.5) / 2000.0 for i in range(2000)]
    on = [-math.log(1.0 - u) for u in even]
    # seven tubes in a hundred have a first exit that came back, and are seen one mean wait later
    seen = [v + (1.0 if i % 100 < 7 else 0.0) for i, v in enumerate(on)]
    exits = [2 if i % 100 < 7 else 1 for i in range(len(on))]
    m = t59.clock_numbers(tubes(on, seen, exits), 100.0, 50000)
    assert abs(m["gap"][0] - 0.07) < 1e-6 and m["hidden_share"] > 0.99 and m["tubes_hidden"] == 140
    assert t59.verdict_c(m) == (True, True, True, "HIDDEN EXITS")
    # the same gap, but every tube is merely seen late and none came back
    m = t59.clock_numbers(tubes(on, [v + 0.07 for v in on]), 100.0, 50000)
    assert t59.verdict_c(m) == (True, True, False, "THE LOOK'S ROUNDING")
    # the looks see the first exit at once: no gap to explain T58's seven per cent
    m = t59.clock_numbers(tubes(on, on), 100.0, 50000)
    assert t59.verdict_c(m)[3] == "NOT THE LOOKS"
    # the first exit itself is late
    late = [1.07 * v for v in on]
    m = t59.clock_numbers(tubes(late, [v + 0.07 for v in late], exits), 100.0, 50000)
    assert t59.verdict_c(m)[3] == "OFF THE COUNT"


def test_the_configs_replay_the_seeds_they_name_and_split_the_tubes_evenly():
    a1 = make_t59_configs.t8_wait_config()
    assert a1["seed"] == 20261201 and a1["lambda"] == 1.05 and a1["sides"] == [[24, 4]]
    assert a1["replica_ids"] == [9, 10, 11] == a1["trace_replicas"] and a1["block"] == 5 and a1["record_detector"]
    a2 = make_t59_configs.offers_config()
    by_size = {64: [], 96: []}
    for t in a2["targets"]:
        n = t["side"][0] * t["side"][1]
        assert t["seed"] == {64: 20261801, 96: 20261802}[n]
        by_size[n].append((t["role"], t["replica"]))
    assert sorted(r for role, r in by_size[64] if role == "target") == list(t59.T22_TARGETS[64]) == [6, 7, 11, 17]
    assert sorted(r for role, r in by_size[96] if role == "target") == list(t59.T22_TARGETS[96]) == [6, 12, 29]
    assert not set(r for _, r in by_size[64] if _ == "control") & set(t59.T22_TARGETS[64])
    cells = {}
    for cfg in make_t59_configs.clock_configs():
        assert cfg["mode"] == "clocks" and cfg["look"] == 5 and cfg["seed"] == 20265900
        key = (cfg["lambda"], cfg["side"][0] * cfg["side"][1])
        cells.setdefault(key, []).extend(range(cfg["replica_start"], cfg["replica_start"] + cfg["replicas"]))
    assert {k: len(v) for k, v in cells.items()} == {(1.05, 64): 2000, (1.05, 96): 1000, (1.30, 64): 4000}
    assert all(v == list(range(len(v))) for v in cells.values())
    assert {n: c["tubes"] for n, c in t59.B_CELLS.items()} == {64: 2000, 96: 1000} and t59.C_TUBES == 4000


def write_csv(path, rows):
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def test_the_whole_report_prints_from_made_up_files(tmp_path, capsys):
    """Every part of the report, on files shaped like the run's, so that nothing about the reading has to be
    repaired after the results exist."""
    import numpy as np
    # A1: three decays whose replay matches, each a tube until its detector fires
    old = [dict(N="96", replica=str(r), waiting=str(w), f_200="0.0") for r, w in ((9, 300), (10, 400), (11, 1000))]
    old.append(dict(N="64", replica="11", waiting="250", f_200="0.0"))            # another size: not read
    write_csv(tmp_path / "t8_lam105.csv", old)
    write_csv(tmp_path / "t59_t8wait_lam105_n96.csv", [dict(r, thresh="1.25") for r in old[:3]])
    for r in (9, 10, 11):
        for folder, name in (("t8_lam105_adj", "N96_rep%d.npz"), ("t59_t8wait_lam105_n96_trace", "N96_rep%d_final.npz")):
            (tmp_path / folder).mkdir(exist_ok=True)
            np.savez(tmp_path / folder / (name % r), adj=np.arange(8).reshape(2, 4))
        blocks = [dict(sweep=5 * i, S=120, X=96, d0=0, d1=96, d2=0, d3=0, d4=0, d5=0, d6=0, pieces=1, largest=96,
                       baby=0, cubes=0) for i in range(1, 260)]
        write_csv(tmp_path / "t59_t8wait_lam105_n96_trace" / ("N96_rep%d.csv" % r), blocks)
    # A2: fourteen replays that match T22, each with one whole stretch offered as counted
    offers = []
    for n in (64, 96):
        wanted = t59.T22_TARGETS[n] + t59.T22_CONTROLS[n]
        write_csv(tmp_path / ("t22_exits_n%d_lam105.csv" % n),
                  [dict(N=str(n), replica=str(rep), first_exit_sweeps="6000.5") for rep in wanted])
        for rep in wanted:
            base = dict(N=str(n), replica=str(rep), first_exit_sweeps="6000.5", smallest_draw_a="0.0001")
            offers.append(dict(base, stretch="0", sweeps_in_stretch="5000", whole_stretch="1", offered_a="15010",
                               offered_b="9990"))
            offers.append(dict(base, stretch="1", sweeps_in_stretch="1000.5", whole_stretch="0", offered_a="3001",
                               offered_b="2002"))
    write_csv(tmp_path / "t59_offers_lam105.csv", offers)
    # B and C: one memoryless population on the count; in C seven tubes in a hundred are seen one mean wait late
    for name, tubes_wanted, tau, files in (("t59_lam105_n64", 2000, 8329.7, 16), ("t59_lam105_n96", 1000, 8329.7, 20),
                                           ("t59_lam130_n64", 4000, 419.0, 16)):
        per = tubes_wanted // files
        for f in range(files):
            rows = []
            for i in range(f * per, (f + 1) * per):
                first = -math.log(1.0 - (i + 0.5) / tubes_wanted) * tau
                late = tau if (name == "t59_lam130_n64" and i % 100 < 7) else 0.0
                rows.append(dict(first_exit_sweeps=repr(first), seen_every_sweep=str(math.ceil(first)),
                                 seen_every_look=str(5 * math.ceil((first + late) / 5)),
                                 exits_before_look="2" if late else "1"))
            write_csv(tmp_path / ("%s_%02d.csv" % (name, f)), rows)
    t59.main(str(tmp_path))
    out = capsys.readouterr().out
    assert "A1. T8's long wait" in out and out.count("-> TRUE WAIT") == 3
    assert "VERDICT ON THE SEVEN: OFFERED AS COUNTED" in out and out.count("-> OFFERED AS COUNTED") == 14
    assert "N = 64, 2000 tubes -> ON THE COUNT" in out and "N = 96, 1000 tubes -> ON THE COUNT" in out
    assert "4000 tubes -> HIDDEN EXITS" in out and "Not read" not in out and "NOT REPRODUCED" not in out
