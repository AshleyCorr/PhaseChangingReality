"""The counting drive to curl against the counts of O55 (N = 512, six links) and against direct counts at small k."""
import importlib.util
import math
from pathlib import Path

spec = importlib.util.spec_from_file_location("ecd", Path(__file__).resolve().parents[1] / "scripts" / "exact_counting_drive.py")
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)


def test_matches_o55_at_512_points():
    assert abs(e.ln_a_cube(3) - math.log(23040)) < 1e-12              # the 6-cube, side-preserving
    assert abs(e.ln_a_flat(512, 3) - math.log(12288)) < 1e-12        # 8 x 8 x 8
    ln_gas = 8 * math.log(23040) + math.log(math.factorial(8))
    assert abs(ln_gas - 90.96) < 0.01                                 # O55: ln A = 90.96
    assert abs(e.crossing_lambda(512, 3, 1.5) - 1.020) < 0.001
    assert abs(e.crossing_lambda(512, 3, 1.0) - 1.013) < 0.001


def test_stirling_form_agrees_with_the_exact_one_and_the_crossing_rises_with_size():
    for k in (8, 1000, 10 ** 6):
        assert abs(e.crossing_lambda(k * 64, 3, 1.5) - e.crossing_lambda_ln(math.log(k), 3, 1.5)) < 1e-5
    lams = [e.crossing_lambda_ln(x * math.log(10), 3, 1.5) for x in (1, 3, 6, 12, 24)]
    assert all(a < b for a, b in zip(lams, lams[1:]))
