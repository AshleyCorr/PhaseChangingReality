"""The sweep-by-sweep look at T58's long waits (exploratory, 10 October 2026): the reader's arithmetic, the exhaustive
census of a tube's single switches, and what the saved traces say."""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import read_t58_sweeps as reader  # noqa: E402
from analyse_curled_revision import neutral_closure  # noqa: E402


def test_neighbouring_blocks_join_into_stretches():
    assert reader.stretches([5, 10, 15, 40]) == [(1, 15), (36, 40)]
    assert reader.stretches([]) == []


def test_summary_of_a_made_up_trace():
    # twenty sweeps of an 8-point "tube" (S = 10, X = 8): free moves in block 2, an exit and return in block 3
    rows = []
    for sweep in range(1, 21):
        block = (sweep - 1) // 5
        s = 8 if sweep in (12, 13) else 10
        rows.append(dict(sweep=str(sweep), S=str(s), X="8" if s == 10 else "4", pieces="1", largest="8", baby="0",
                         cubes="0", accepted_in_block=str([0, 3, 2, 0][block])))
    d = reader.summarize(rows, 20, 10, 8)
    assert d["blocks"] == 4 and d["quiet"] == 2 and d["moves"] == 5 and d["busy"] == [(6, 15)]
    assert d["off"] == 2 and d["first_off"] == 12 and d["s_max"] == 10 and d["cubes"] == 0 and d["pieces"] == 1
    assert reader.summarize(rows, 10, 10, 8)["off"] == 0          # only up to the sweep asked for


def test_no_switch_of_any_re_glued_tube_adds_a_square():
    """The moves a waiting tube makes for nothing are paper 1's neutral switches. The exhaustive census of every
    arrangement they reach (three classes at N = 64) is one list of switches, and nothing in it adds a square."""
    records = neutral_closure(64)
    assert len(records) == 3 and len({r["census"] for r in records}) == 1
    kinds = dict(records[0]["census"])
    assert kinds[(0, 0)] == 32                                     # half a neutral switch a sweep
    assert kinds[(-2, -4)] == 192 and kinds[(-4, -10)] == 128      # paper 1's two exits, three and two a sweep
    assert max(ds for ds, _dx in kinds) == 0                       # a second curl is more squares; no move adds one


def test_the_saved_traces_say_what_the_record_says():
    """Pins the reading of results/t58_exploratory_sweeps_lam130_n64 recorded in ASSUMPTIONS O110."""
    assert reader.same_history()
    rows = {int(r["replica"]): r for r in reader.rows_of(reader.RESULTS / (reader.NEW + ".csv"))}
    assert sorted(rows) == [551, 552, 553, 2746, 2747, 2748, 3070, 3071, 3072]

    def off_stretches(rep):
        wait = int(rows[rep]["waiting"])
        path = reader.RESULTS / (reader.NEW + "_trace") / ("N64_rep%d_sweeps.csv" % rep)
        trace = [t for t in csv.DictReader(open(path, newline="")) if int(t["sweep"]) <= wait]
        assert len(trace) == wait
        d = reader.summarize(trace, wait, 80, 64)
        assert d["s_max"] == 80 and d["cubes"] == 0 and d["baby"] == 0 and d["pieces"] == 1
        runs = []
        for t in trace:
            if (int(t["S"]), int(t["X"])) != (80, 64):
                s = int(t["sweep"])
                if runs and s == runs[-1][1] + 1:
                    runs[-1][1] = s
                else:
                    runs.append([s, s])
        return [tuple(r) for r in runs], d["moves"]

    runs, moves = off_stretches(2748)
    assert runs == [(4999, 5000)] and moves == 2580               # the tube at every sweep until its one exit
    runs, moves = off_stretches(3072)
    assert runs == [(23, 32), (3166, 3168), (4701, 4705)] and moves == 2363   # one exit between two looks
    runs, _moves = off_stretches(553)
    assert runs[0] == (42, 112) and (2868, 2870) in runs and runs[-1] == (5030, 5080)
    for rep in (551, 552, 2746, 2747, 3070, 3071):
        off_stretches(rep)                                        # no control ever gains a square or a knot
