"""T37: counting seeds reads the graph and draws no random numbers, so a run with it is the run without it."""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_tube_decay  # noqa: E402


def _run(tmp_path, name, extra):
    cfg = {"name": name, "sides": [[24, 4]], "g": 1.5, "replicas": 2, "n_sweeps": 30000, "block": 5,
           "stop_at": 0.98, "settle": 100, "settle_max": 400, "seed": 7, "lambda": 1.25}
    cfg.update(extra)
    path = tmp_path / (name + ".json")
    path.write_text(json.dumps(cfg))
    run_tube_decay.main(str(path), str(tmp_path))
    return list(csv.DictReader(open(tmp_path / (name + ".csv"), newline="")))


def test_seed_count_does_not_change_the_chain(tmp_path):
    a = _run(tmp_path, "plain", {})
    b = _run(tmp_path, "counted", {"count_patches_every": 50, "patch_min": 8})
    for ra, rb in zip(a, b):
        for key in ("waiting", "phi_final", "released", "sweeps"):
            assert ra[key] == rb[key]
        if float(rb["reached"]) >= 0.5:
            assert int(rb["seeds"]) >= 1


def test_saving_waiting_tubes_does_not_change_the_chain(tmp_path):
    """T38: a tube still waiting at a named sweep is saved; the chain is the chain without it."""
    a = _run(tmp_path, "plain2", {})
    b = _run(tmp_path, "saved", {"save_waiting_at": [300, 600]})
    for ra, rb in zip(a, b):
        for key in ("waiting", "phi_final", "released", "sweeps"):
            assert ra[key] == rb[key]
    saved = sorted(p.name for p in (tmp_path / "saved_waiting").glob("*.npz")) if (tmp_path / "saved_waiting").exists() else []
    for rb in b:
        w = int(rb["waiting"]) if rb["waiting"] else 10 ** 9
        for s in (300, 600):
            assert (("N96_rep%s_sweep%d.npz" % (rb["replica"], s)) in saved) == (w > s)


def test_recording_the_detector_does_not_change_the_chain(tmp_path):
    """T58: the detector's threshold, what it cannot see and a block-by-block trace are read off the chain. The
    chain is the chain without them, and the recorded numbers are the ones the detector used."""
    import numpy as np

    a = _run(tmp_path, "plain3", {"record_f_200": True})
    b = _run(tmp_path, "watched", {"record_f_200": True, "record_detector": True, "trace_replicas": [0, 1]})
    n, s_tube = 96, 120
    for ra, rb in zip(a, b):
        for key in ra:                                    # every column the plain run writes
            assert ra[key] == rb[key]
        trace = list(csv.DictReader(open(tmp_path / "watched_trace" / ("N96_rep%s.csv" % rb["replica"]), newline="")))
        assert [int(t["sweep"]) for t in trace] == list(range(5, 5 * len(trace) + 1, 5))
        phis = [int(t["S"]) / n for t in trace]
        rest = np.array(phis[:40])
        thresh = 1.25 - 3.0 * max(rest.std(), 1e-6) - 1e-9
        assert float(rb["thresh"]) == thresh and float(rb["rest_sd"]) == float(rest.std())
        fired = [int(t["sweep"]) for t, phi in zip(trace, phis) if int(t["sweep"]) > 200 and phi < thresh]
        assert rb["waiting"] == (str(fired[0]) if fired else "")
        assert float(rb["phi_max"]) == max([1.25] + phis)
        assert int(rb["d0_max"]) == max(int(t["d0"]) for t in trace)
        tube = [int(t["S"]) == s_tube and int(t["X"]) == n and int(t["d1"]) == n for t in trace]
        assert all(sum(int(t["d%d" % k]) for k in range(7)) == n for t in trace)
        assert rb["tube_at_200"] == str(int(tube[39]))
        left = [int(t["sweep"]) for t, ok in zip(trace, tube) if not ok]
        assert rb["first_left"] == (str(left[0]) if left else "")
        after = [s for s in left if s > 200]
        assert rb["first_left_after_200"] == (str(after[0]) if after else "")
        stop = fired[0] if fired else 10 ** 9
        unseen = [i for i, t in enumerate(trace) if 200 < int(t["sweep"]) < stop and not tube[i]]
        assert int(rb["off_blocks"]) == len(unseen)
        assert int(rb["exits_unseen"]) == sum(1 for i in unseen if tube[i - 1])
        final = np.load(tmp_path / "watched_trace" / ("N96_rep%s_final.npz" % rb["replica"]))["adj"]
        assert final.shape == (n, 4)


def test_writing_every_sweep_does_not_change_the_chain(tmp_path):
    """10 October 2026: with `trace_sweeps` every sweep of a traced replica is written, with the pieces, baby
    universes and 4-cubes after it and the number of moves accepted in its block. The chain is the chain without it,
    every fifth row is the block-by-block trace, and a block with no move accepted is a block with no change."""
    base = {"record_f_200": True, "record_detector": True, "trace_replicas": [0, 1]}
    a = _run(tmp_path, "blocks", base)
    b = _run(tmp_path, "sweeps", dict(base, trace_sweeps=True))
    assert a == b                                          # every column of every row
    seen_change = False
    for row in b:
        name = "N96_rep%s" % row["replica"]
        blocks_a = (tmp_path / "blocks_trace" / (name + ".csv")).read_text()
        blocks_b = (tmp_path / "sweeps_trace" / (name + ".csv")).read_text()
        assert blocks_a == blocks_b and not (tmp_path / "blocks_trace" / (name + "_sweeps.csv")).exists()
        blocks = list(csv.DictReader(blocks_b.splitlines()))
        sweeps = list(csv.DictReader(open(tmp_path / "sweeps_trace" / (name + "_sweeps.csv"), newline="")))
        assert [int(t["sweep"]) for t in sweeps] == list(range(1, 5 * len(blocks) + 1))
        before = (120, 96)                                 # the tube it starts as: S, X
        for k, block in enumerate(blocks):
            rows = sweeps[5 * k:5 * k + 5]
            last = rows[-1]
            for key in ("S", "X", "pieces", "largest", "baby", "cubes"):
                assert last[key] == block[key]             # the fifth sweep is the block's own look
            moves = {int(t["accepted_in_block"]) for t in rows}
            assert len(moves) == 1 and min(moves) >= 0
            states = [before] + [(int(t["S"]), int(t["X"])) for t in rows]
            if len(set(states)) > 1:
                assert min(moves) >= 1                     # a change needs a move
                seen_change = True
            if min(moves) == 0:
                assert len(set(states)) == 1               # no move, no change
            before = states[-1]
    assert seen_change
