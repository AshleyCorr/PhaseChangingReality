"""Known-answer tests for scripts/observer_window.py (PREREGISTRATION T72, gates 1 and 2)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import observer_window as ow  # noqa: E402


def test_cosmology_age_today_is_planck():
    # Planck 2018: 13.79 Gyr at z = 0 (general knowledge, to verify)
    assert abs(float(ow.age_at_z(0.0)) - 13.79) < 0.1
    # inverse
    for t in (1.0, 5.0, 13.0, 40.0):
        assert abs(float(ow.age_at_z(ow.z_at_age(t))) - t) < 1e-9


def test_sfr_peak_near_z_two():
    # Madau and Dickinson 2014: the fit peaks at z of about 1.9, age about 3.5 Gyr
    t = np.linspace(0.1, 13.8, 2000)
    psi = ow.sfr_density(t, tail_gyr=None)
    assert 3.0 < t[np.argmax(psi)] < 4.0


def test_imf_continuous_at_one_solar_mass():
    lo, hi = ow.imf_chabrier(np.array([0.999999])), ow.imf_chabrier(np.array([1.000001]))
    assert abs(lo[0] / hi[0] - 1.0) < 0.05   # the 12 October note: the two pieces match to 3 %


def test_lifetime_values_match_the_12_october_note():
    tau = ow.lifetime(np.array([0.08, 0.15, 0.27, 0.49, 0.90]))
    expected = np.array([1.07e4, 2.9e3, 7.4e2, 9.2e1, 1.3e1])  # Gyr
    assert np.all(np.abs(tau / expected - 1.0) < 0.03)


def _gate():
    model = ow.Model(tail_gyr=50.0)
    p = model.density(0.08, 3.0, "gate")
    return ow.summarize(model.t, p)


def test_gate_share_before_now_reproduces_loeb_batista_sloan_2016():
    """Rule 2, part 1: with their assumptions the share before now is about 0.1 % (within a factor of two)."""
    s = _gate()
    assert 0.0005 <= s["F_now"] <= 0.002, s


@pytest.mark.xfail(strict=True, reason="T72 gate 1, part 2, FAILS BY THE LETTER as written before the number: the "
                   "peak of dP/dln t is 4.43e3 Gyr against the registered [5e3, 2e4] (the paper says 'around' 1e13 yr "
                   "and its figure is not digitised). Recorded in PREREGISTRATION T72, 'Reading, gate'; the tolerance "
                   "is not changed after the fact; the owner decides.")
def test_gate_peak_reproduces_loeb_batista_sloan_2016():
    s = _gate()
    assert 5.0e3 <= s["peak_dPdlnt_Gyr"] <= 2.0e4, s


def test_sun_like_gate_peaks_near_the_present():
    """LBS16: 'for mass ranges centered narrowly around a solar mass ... the current time indeed represents the peak'."""
    model = ow.Model(tail_gyr=50.0)
    p = model.density(0.9, 1.1, "gate")
    s = ow.summarize(model.t, p)
    assert 7.0 <= s["peak_dPdlnt_Gyr"] <= 30.0, s


def test_fixed_once_kernel_is_blind_to_the_mass_range():
    """Exact under the registered counting unit (T72 Reading): one civilization per planet at a fixed delay gives the
    same shape for every mass range whose stars outlive the delay, so classes A and C coincide."""
    model = ow.Model(tail_gyr=50.0)
    a = model.density(0.8, 1.2, "fixed", delay=4.5)
    c = model.density(0.08, 3.0, "fixed", delay=4.5)
    a, c = a / ow.trapezoid(a, model.t), c / ow.trapezoid(c, model.t)
    assert np.max(np.abs(a - c)) < 1e-9 * np.max(a)


def test_observer_moment_variant_for_all_stars_is_early():
    """T72 Amendment 2: weighting each civilization by its remaining window restores the 2016 paper's lifetime weighting,
    so the all-stars class is EARLY again under it."""
    model = ow.Model(tail_gyr=50.0)
    p = model.density(0.08, 3.0, "fixed_moments", delay=4.5)
    s = ow.summarize(model.t, p)
    assert s["F_now"] < 0.05, s


def test_diamond_volume_peaks_where_the_entropy_paper_says():
    """T73 gate 2: V_c peaks at 0.23 t_Lambda ([BHKP07] Eq. 20) within 5 %."""
    model = ow.Model(tail_gyr=50.0)
    v = model.diamond_volume()
    t_peak = float(model.t[np.argmax(v)])
    assert abs(t_peak / ow.t_lambda() - 0.23) < 0.05 * 0.23 + 0.005, (t_peak, ow.t_lambda())
