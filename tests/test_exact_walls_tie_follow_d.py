"""The follow form of the direction tie (VISION Update 30, amended): exact checks."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import exact_walls_tie_follow_d as fol  # noqa: E402
import exact_walls_tie_d as tie  # noqa: E402


def test_the_form():
    assert [fol.f(d, 3) for d in range(6)] == [0, 2, 1, 0, 0, 0]          # curled 0; one open 2; two open 1; flat 0; broken 0
    assert [fol.f(d, 4) for d in range(6)] == [0, 3, 2, 1, 0, 0]


def test_kappa_zero_is_the_model_and_flat_space_is_untouched():
    (adj, part), _ = tie.parse("6,6,8")
    ks = tie.kinds(adj, part)
    best0 = min((k for k in ks if not tie.is_null(k)), key=lambda k: (fol.cost(k, 1.25, 0.0, 3), k))
    assert fol.cost(best0, 1.25, 0.0, 3) == 64.0
    for kappa in (1.0, 4.0):
        best = min((k for k in ks if not tie.is_null(k)), key=lambda k: (fol.cost(k, 1.25, kappa, 3), k))
        assert fol.cost(best, 1.25, kappa, 3) == 64.0


def test_two_curled_wall_at_kappa_zero_matches_O49():
    (adj, part), _ = tie.parse("4,4,18")
    ks = tie.kinds(adj, part)
    best = min((k for k in ks if not tie.is_null(k)), key=lambda k: (fol.cost(k, 1.25, 0.0, 3), k))
    assert fol.cost(best, 1.25, 0.0, 3) == 16.0
