"""Known-answer tests for the seam check: the circular arithmetic and the verdict branches."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyse_t11 as a11          # noqa: E402
import run_seam_check as seam      # noqa: E402


def test_circular_mean_handles_the_wraparound():
    assert abs(seam.circular_mean([15, 0], 16) - 15.5) < 1e-9
    assert abs(seam.circular_mean([3, 4], 16) - 3.5) < 1e-9


def test_circular_distance_never_exceeds_half_the_loop():
    assert seam.circular_distance(0, 8, 16) == 8
    assert seam.circular_distance(1, 15, 16) == 2
    assert seam.circular_distance(15.5, 7.5, 16) == 8


def _rows(dists_by_n, half=8.0, rings=1):
    return [dict(N=str(n), rings=str(rings), dist=str(d), half_length=str(half if n == 64 else 12.0))
            for n, ds in dists_by_n.items() for d in ds]


def test_rings_at_the_far_side_give_seam():
    rng = np.random.default_rng(0)
    rows = _rows({64: list(np.clip(rng.normal(8, 1.0, 40), 0, 8)), 96: list(np.clip(rng.normal(12, 1.5, 30), 0, 12))})
    v, lines = a11.verdict(rows)
    assert v == "SEAM", "\n".join(lines)


def test_rings_at_the_start_give_at_the_seed():
    rng = np.random.default_rng(1)
    rows = _rows({64: list(np.abs(rng.normal(0, 0.6, 40))), 96: list(np.abs(rng.normal(0, 0.6, 30)))})
    v, lines = a11.verdict(rows)
    assert v == "AT THE SEED", "\n".join(lines)


def test_uniform_positions_give_neither():
    rng = np.random.default_rng(2)
    rows = _rows({64: list(rng.uniform(0, 8, 40)), 96: list(rng.uniform(0, 12, 30))})
    v, lines = a11.verdict(rows)
    assert v == "NEITHER", "\n".join(lines)


def test_too_few_single_ring_replicas_is_inconclusive():
    rows = _rows({64: [8.0] * 10, 96: [12.0] * 30})
    v, lines = a11.verdict(rows)
    assert v == "INCONCLUSIVE"


def test_distance_is_derived_from_the_recorded_columns_when_the_runner_left_it_blank():
    # leftover across columns 7 and 8 (mean 7.5), change started at 15.5: 8 columns apart on a 16-loop
    r = dict(N="64", rings="1", dist="nan", half_length="8.0", ring_cols="7 8", start="15.5")
    d, half = a11.distance_of(r)
    assert abs(d - 8.0) < 1e-9 and half == 8.0
    # wraparound: columns 0 and 15 (mean 15.5), start at 0.5: one column apart
    r = dict(N="64", rings="1", dist="nan", half_length="8.0", ring_cols="0 15", start="0.5")
    assert abs(a11.distance_of(r)[0] - 1.0) < 1e-9
    # a numeric dist from the runner is used as is
    r = dict(N="64", rings="1", dist="3.0", half_length="8.0", ring_cols="4", start="1.0")
    assert a11.distance_of(r)[0] == 3.0


def test_multi_ring_replicas_are_left_out():
    rows = _rows({64: [8.0] * 40, 96: [12.0] * 30}) + _rows({64: [0.0] * 40}, rings=2)
    v, lines = a11.verdict(rows)
    assert v == "SEAM", "\n".join(lines)
