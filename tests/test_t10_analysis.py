"""Known-answer tests for the T10 verdict: synthetic end states with a ring count we choose."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyse_t10 as a10  # noqa: E402

SIZES = (64, 96, 192, 288)


def _sheet_with_rings(n, rings, largest=4, bath=0.5, f_final=1.0):
    """A perfect sheet plus `rings` separate rings: S = N + rings, X = 6 * rings, so the excess is
    exactly 14 per ring at lambda = 1.25."""
    s = n + rings
    x = 6 * rings
    phi = s / n
    return dict(N=str(n), phi=str(phi), surplus=str(x / n), bath_T=str(bath), f_final=str(f_final),
                pieces_d1=str(rings), largest_d1=str(largest if rings else 0), final="1")


def _make(ring_counts, bath=0.5, drift=0.0, f_final=1.0):
    finals, traces = {}, {}
    for n in SIZES:
        for rep, k in enumerate(ring_counts[n]):
            finals[(n, rep)] = _sheet_with_rings(n, k, bath=bath, f_final=f_final)
            f_end = (1.25 - (n + k) / n) / 0.25
            traces[(n, rep)] = [(s, f_end + (drift if s == 30000 else 0.0)) for s in range(100, 30001, 100)]
    return finals, traces


def test_rings_growing_with_n_gives_the_grows_verdict():
    rng = np.random.default_rng(1)
    counts = {n: list(rng.poisson(n / 64, 20)) for n in SIZES}
    v, lines = a10.verdict(*_make(counts))
    assert v == "THE LEFTOVER GROWS WITH THE SPACE", "\n".join(lines)


def test_one_ring_at_every_size_gives_the_one_ring_verdict():
    rng = np.random.default_rng(2)
    counts = {n: [1 if rng.random() < 0.95 else 0 for _ in range(20)] for n in SIZES}
    v, lines = a10.verdict(*_make(counts))
    assert v == "ONE RING, HOWEVER LARGE", "\n".join(lines)


def test_a_ring_leaving_late_makes_it_inconclusive():
    counts = {n: [max(1, n // 64)] * 20 for n in SIZES}
    v, lines = a10.verdict(*_make(counts, drift=14 / 64 + 0.01))
    assert v == "INCONCLUSIVE", "\n".join(lines)
    assert any("FAILS" in l and "(c)" in l for l in lines)


def test_a_warm_bath_fails_c():
    counts = {n: [max(1, n // 64)] * 20 for n in SIZES}
    v, lines = a10.verdict(*_make(counts, bath=1.5))
    assert v == "INCONCLUSIVE"


def test_gate_3_excludes_a_size_and_says_so():
    counts = {n: [max(1, n // 64)] * 20 for n in SIZES}
    finals, traces = _make(counts)
    finals[(288, 3)]["f_final"] = "0.2"
    v, lines = a10.verdict(finals, traces)
    assert any("GATE 3 FAILS" in l and "N=288" in l for l in lines)
    # three sizes remain and still grow
    assert v == "THE LEFTOVER GROWS WITH THE SPACE", "\n".join(lines)


def test_additivity_is_exact_on_the_synthetic_rows():
    r = _sheet_with_rings(192, 3)
    assert abs(a10.excess_energy(r) - 42.0) < 1e-9


def test_slope_error_matches_a_hand_calculation():
    slope, err = a10.slope_with_error([0, 1, 2], [0, 1, 2], [1, 1, 1])
    assert abs(slope - 1.0) < 1e-12
    assert abs(err - np.sqrt(1 / 2)) < 1e-12
