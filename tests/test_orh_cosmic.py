"""Stage E benchmarks against published numbers (orh.cosmic): the Planck 2018 age, the Madau-Dickinson peak, the
Loeb et al. lifetimes and mass function, and the shape of Loeb et al.'s own result, which the model must reproduce
before any knob of ours is turned (CLAUDE.md rule 2)."""
import numpy as np
import pytest

from orh import cosmic


def test_planck_2018_age_and_scale_factor_today():
    c = cosmic.Cosmology()
    assert abs(float(c.age(0.0)) - 13.797) < 0.01                    # Planck 2018 Table 2: 13.797 +- 0.023
    assert abs(float(c.scale_factor(c.age(0.0))) - 1.0) < 1e-6
    assert abs(float(c.redshift(c.age(1.85))) - 1.85) < 1e-6
    assert abs(float(c.age(1.85)) - 3.52) < 0.02                      # the note: z = 1.85 at 3.52 Gyr


def test_madau_dickinson_peak_and_present_value():
    z = np.linspace(0, 6, 60001)
    psi = cosmic.madau_dickinson(z)
    assert abs(cosmic.madau_dickinson(0.0) / 0.015 - 1) < 0.003              # 0.015 / (1 + 2.9^-5.6)
    assert abs(z[np.argmax(psi)] - 1.86) < 0.01                       # the note: peaks at z = 1.86
    assert abs(psi.max() / psi[0] - 8.9) < 0.1                        # "approximately nine times higher than today"


def test_loeb_lifetimes_are_continuous_and_give_the_quoted_values():
    tau = cosmic.stellar_lifetime_gyr
    assert abs(tau(0.08) * 1e9 - 1.07e13) / 1.07e13 < 0.01
    assert abs(tau(1.0) * 1e9 - 1.0e10) < 1e-6
    assert abs(tau(0.75 + 1e-9) / tau(0.75 - 1e-9) - 1) < 0.02        # continuous to 1.5 %
    assert abs(tau(0.25 + 1e-9) / tau(0.25 - 1e-9) - 1) < 0.005       # continuous to 0.1 %


def test_chabrier_pieces_meet_at_one_solar_mass_to_three_percent():
    lo = float(cosmic.chabrier_imf(1.0 - 1e-9))
    hi = float(cosmic.chabrier_imf(1.0 + 1e-9))
    assert abs(lo / hi - 1) < 0.035


def test_future_star_formation_falls_and_the_cutoff_bites():
    c = cosmic.Cosmology()
    t = np.array([13.8, 50.0, 100.0])
    bare = cosmic.star_formation_rate(t, c, cutoff_gyr=None)
    cut = cosmic.star_formation_rate(t, c, cutoff_gyr=50.0)
    assert bare[0] > bare[1] > bare[2] > 0
    assert cut[1] / bare[1] == pytest.approx(np.exp(-1.0))
    assert cosmic.star_formation_rate(0.01, c) == 0.0                 # before the first stars


@pytest.fixture(scope="module")
def model():
    return cosmic.PopulationModel(n_t=3000, n_m=200)


def test_loeb_baseline_shape_is_reproduced(model):
    """Loeb et al. Sec. 3: for 0.08 to 3 M_sun the peak of dP/d ln t is near 10^13 yr and P(< today) is about 0.1 %;
    for masses near a solar mass the current time is near the peak. Written before the model was run."""
    w = model.weight(0.08)
    s = model.summarize(w)
    assert s["converged"]
    assert 3e3 < s["t_peak_gyr"] < 3e4                                # 3e12 to 3e13 yr
    assert 2e-4 < s["F_today"] < 5e-3                                 # "0.1 %" in the paper
    w9 = model.weight(0.9)
    s9 = model.summarize(w9)
    assert 5 < s9["t_peak_gyr"] < 25                                  # the present is near the peak


def test_normalization_monotone_cumulative_and_convergence_flags(model):
    w = model.weight(0.5, tau_evo=4.5, L_obs=1.0)
    s = model.summarize(w)
    assert abs(float(np.trapezoid(s["p"], model.t)) - 1) < 1e-9
    assert np.all(np.diff(s["F"]) >= -1e-12)
    assert 0 < s["F_today"] < 1
    assert np.isfinite(s["t9999_gyr"])


def test_origins_class_is_the_delayed_formation_rate(model):
    w = model.weight(0.9, tau_evo=4.0, ref_class="origins")
    direct = cosmic.star_formation_rate(model.t - 4.0, model.cosmo, model.t_first, model.cutoff)
    ratio = w[w > 0] / direct[w > 0]
    assert np.allclose(ratio, ratio[0])                               # the same shape, up to the IMF weight


def test_exotic_late_observers_make_the_integral_diverge(model):
    short = cosmic.PopulationModel(n_t=3000, n_m=200, t_max_gyr=300.0)
    longer = cosmic.PopulationModel(n_t=3000, n_m=200, t_max_gyr=1000.0)
    early = short.summarize(short.weight(0.08) + short.exotic_weight(1e-12))
    later = longer.summarize(longer.weight(0.08) + longer.exotic_weight(1e-12))
    assert later["F_today"] < early["F_today"]                        # F(today) keeps falling as the grid extends
    assert later["tail_weight"] > 0.5                                 # the weight sits at the end of any grid
    on_default = model.summarize(model.weight(0.08) + model.exotic_weight(1e-12))
    assert on_default["converged"] is False                           # on the 10^5 Gyr grid a^3 is not even finite


def test_pooled_and_averaged_normalizations_differ_when_totals_differ():
    t = np.linspace(0, 100, 1001)
    w1 = np.where(t < 20, 1.0, 0.0)                                   # a history whose observers all come early
    w2 = np.where(t < 100, 1.0, 0.0)                                  # five times as many, mostly late
    out = cosmic.pooled_and_averaged_F([dict(t=t, w=w1, p=0.5), dict(t=t, w=w2, p=0.5)], t_today=13.8)
    # the grid is 0.1 wide and the step functions lose half a cell at their edges: tolerance 5e-3
    assert abs(out["averaged"] - 0.5 * (13.8 / 20 + 13.8 / 100)) < 5e-3
    assert abs(out["pooled"] - (0.5 * 13.8 + 0.5 * 13.8) / (0.5 * 20 + 0.5 * 100)) < 5e-3
    assert abs(out["pooled"] - out["averaged"]) > 0.1
