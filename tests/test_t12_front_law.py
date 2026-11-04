"""The T12 analyser, checked against data whose answer is known in advance.

The project's rule for a verdict script: build runs where the right answer is put in by hand, and
check the script returns it — both when the criteria should pass and when each should fail. Without
that, a verdict is only the script's opinion of itself.
"""
import numpy as np
import pytest

from scripts.analyse_t12_front_law import per_run, verdict

LAM = 1.25
DELTA = 4 * (LAM - 1)          # 1.0 per point, exact
SIGMA = 7.0                    # the one fitted number, put in by hand here


def make_rows(rate_per_sweep=lambda n: 0.4, sigma=SIGMA, delta=DELTA, rules=("metropolis", "glauber"),
              sizes=(64, 96, 144, 192), noise=0.0, seed=1):
    """A perfect coarse-law run: H falls by `delta` a point, the front costs `sigma`, the front
    advances at `rate_per_sweep(n)` points a sweep."""
    rng = np.random.default_rng(seed)
    rows = []
    for rule in rules:
        for n in sizes:
            for rep in range(3):
                rate = rate_per_sweep(n)
                h_tube = delta * n
                for step in range(1, 400):
                    sweep = 25 * step
                    converted = min(rate * sweep, n)
                    if converted >= 0.99 * n:
                        break
                    h = h_tube - delta * converted + 2 * sigma + (rng.normal(0, noise) if noise else 0)
                    rows.append(dict(rule=rule, n=str(n), g="1.5", replica=str(rep), sweep=str(sweep),
                                     h=str(h), h_per_point=str(h / n), converted=str(converted),
                                     fraction=str(converted / n), patches="1", largest_patch=str(int(converted))))
    return rows


def test_a_perfect_coarse_law_passes_and_the_numbers_come_back():
    runs = per_run(make_rows(), LAM)
    good = [r for r in runs if r["nucleated"]]
    assert len(good) == len(runs) and good
    assert all(abs(r["slope"] + DELTA) < 1e-6 for r in good)      # slope is -Delta
    assert all(abs(r["sigma"] - SIGMA) < 1e-6 for r in good)      # sigma recovered exactly
    lines, passed = verdict(good, LAM)
    assert passed, lines


def test_a_wrong_gap_fails_criterion_one():
    """A run whose energy falls at the wrong rate per point must fail linearity, however tidy it is."""
    runs = per_run(make_rows(delta=DELTA * 1.2), LAM)
    lines, passed = verdict([r for r in runs if r["nucleated"]], LAM)
    assert not passed and "1 linearity" in lines[0] and "FAIL" in lines[0]


def test_a_front_cost_that_grows_with_size_fails_criterion_two():
    """If sigma is not one number but grows with N, the coarse model has a size in it and must fail."""
    rows = []
    for n in (64, 96, 144, 192):
        rows += make_rows(sizes=(n,), sigma=SIGMA + 0.1 * n, rules=("metropolis",))
    runs = per_run(rows, LAM)
    lines, passed = verdict([r for r in runs if r["nucleated"]], LAM)
    assert not passed
    assert any("2 front cost" in line and "FAIL" in line for line in lines)


def test_a_front_that_speeds_up_with_size_fails_criterion_three():
    """The pre-registered prediction is a fixed number of points a sweep whatever the tube's length."""
    runs = per_run(make_rows(rate_per_sweep=lambda n: 0.004 * n, rules=("metropolis",)), LAM)
    lines, passed = verdict([r for r in runs if r["nucleated"]], LAM)
    assert not passed
    assert any("3 local advance" in line and "FAIL" in line for line in lines)


def test_one_acceptance_rule_is_not_enough():
    runs = per_run(make_rows(rules=("metropolis",)), LAM)
    lines, passed = verdict([r for r in runs if r["nucleated"]], LAM)
    assert not passed
    assert any("NOT TESTED" in line for line in lines)


def test_runs_that_never_convert_are_excluded_not_counted_as_passing():
    rows = [r for r in make_rows() if float(r["fraction"]) < 0.5]      # every run stalls half way
    runs = per_run(rows, LAM)
    assert all(not r["nucleated"] for r in runs)
    lines, passed = verdict([r for r in runs if r.get("nucleated")], LAM)
    assert not passed and "NO VERDICT" in lines[0]


@pytest.mark.parametrize("noise", [0.05, 0.2])
def test_the_verdict_survives_a_little_measurement_noise(noise):
    runs = per_run(make_rows(noise=noise, seed=4), LAM)
    lines, passed = verdict([r for r in runs if r["nucleated"]], LAM)
    assert passed, lines
