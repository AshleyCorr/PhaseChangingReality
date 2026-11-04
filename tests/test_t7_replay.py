"""T7 amendment 3: replayed decays replace their originals only when they are the same decay."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyse_t7 as a7  # noqa: E402


def _row(n, rep, waiting, first, released, settle):
    return dict(N=str(n), replica=str(rep), waiting=str(waiting), released_first_window=str(first),
                released=str(released), settle_sweeps=str(settle))


def test_replays_replace_the_same_decay_and_others_are_left_alone():
    base = {64: [_row(64, 0, 500, 1.0, 1.0, 600), _row(64, 1, 700, 0.34, 0.28, 30000), _row(64, 2, 650, 1.0, 1.0, 600)]}
    replay = {64: [_row(64, 1, 700, 0.34, 1.0, 45000)]}
    merged, replaced, rejected = a7.merge_replays(base, replay)
    assert replaced == {64: [1]} and rejected == {}
    assert [r["released"] for r in merged[64]] == ["1.0", "1.0", "1.0"]
    assert merged[64][1]["settle_sweeps"] == "45000"


def test_a_replay_that_is_not_the_same_decay_is_rejected_and_the_original_kept():
    base = {64: [_row(64, 1, 700, 0.34, 0.28, 30000)]}
    replay = {64: [_row(64, 1, 705, 0.34, 1.0, 45000)]}       # different wait: not the same stream
    merged, replaced, rejected = a7.merge_replays(base, replay)
    assert replaced == {} and rejected == {64: [1]}
    assert merged[64][0]["released"] == "0.28"


def test_replays_for_a_size_not_in_the_base_are_ignored():
    base = {64: [_row(64, 0, 500, 1.0, 1.0, 600)]}
    replay = {96: [_row(96, 3, 900, 0.83, 1.0, 50000)]}
    merged, replaced, rejected = a7.merge_replays(base, replay)
    assert set(merged) == {64} and replaced == {} and rejected == {}
