"""scripts/exact_t54_between_patches.py: T54's completed-patch energies are not a barrier (ASSUMPTIONS O118,
correction of 11 October 2026). The states between two completed patches go higher than the fitted peak, as an
outside review found with its own code, and a line of columns goes lower than any square."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import exact_t54_between_patches as b                             # noqa: E402

TIED = "all_at_the_last"


def test_the_columns_that_take_one_square_patch_to_the_next():
    assert b.new_columns(1) == [(0, 1), (1, 0), (1, 1)]
    assert b.new_columns(2) == [(0, 2), (1, 2), (2, 0), (2, 1), (2, 2)]


def test_between_the_first_two_patches_the_energy_goes_above_the_fitted_peak():
    start, end, peak, path = b.climb(8, 1, 1.25, TIED)
    assert path == pytest.approx([24.0, 20.0, 45.0, 24.0]) and (start, end, peak) == pytest.approx((24.0, 24.0, 45.0))
    a_fit, b_fit = 176 - 112 * 1.25, 48 * (1.25 - 1)                     # T54's exact quadratic under the tie
    assert [a_fit * k - b_fit * k * k for k in (1, 2, 3)] == pytest.approx([24.0, 24.0, 0.0])
    assert a_fit ** 2 / (4 * b_fit) == pytest.approx(27.0) and peak > 27.0    # the fitted peak bounds nothing
    assert b.climb(8, 2, 1.25, TIED)[:3] == pytest.approx((24.0, 0.0, 33.0))
    assert b.climb(8, 1, 1.25, "none")[:3] == pytest.approx((56.0, 104.0, 113.0))


def test_a_line_of_columns_goes_lower_than_any_square_under_the_tie():
    for lam in (1.10, 1.15, 1.20, 1.25, 1.40):
        first, step = 176 - 112 * lam - 48 * (lam - 1), 96 - 80 * lam    # one column; each further one along y
        assert b.row(8, lam, TIED) == pytest.approx([first + i * step for i in range(6)], abs=1e-9)
    assert b.row(8, 1.25, TIED) == pytest.approx([24.0, 20.0, 16.0, 12.0, 8.0, 4.0])
    assert b.row(8, 1.25, TIED, along_y=False) == pytest.approx([24.0, 40.0, 56.0, 72.0, 88.0, 104.0])
    for lam in (1.10, 1.25, 1.40):                                       # untied: every further column costs
        line = b.row(8, lam, "none")
        assert [y - x for x, y in zip(line, line[1:])] == pytest.approx([32 - 16 * lam] * 5)
