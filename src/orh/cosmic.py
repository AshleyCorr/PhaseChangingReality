"""Stage E: the observer-moment rate over cosmic time in one fixed FLRW background (handoff section 5).

This is a classical population model, separate from the quantum toy, and it imports published equations wherever
there are any. Every published input is transcribed in docs/reading/notes/2026-10-12_orh_cosmic_time_observer_
measure.md with its equation number; the synthetic knobs are named as such here and in docs/theory/orh_assumptions.md.

Published inputs
    Planck 2018 results VI, Table 2 (TT,TE,EE+lowE+lensing): H_0 = 67.36 km/s/Mpc, Omega_m = 0.3153, Omega_Lambda =
        0.6847, age 13.797 Gyr. Matter + Lambda only; radiation changes the age by 0.006 Gyr (the note checks this).
    Cosmic time from the scale factor, matter + Lambda, closed form:
        a(t) = (Omega_m/Omega_Lambda)^(1/3) sinh^(2/3)(1.5 sqrt(Omega_Lambda) H_0 t);  1 + z = 1/a.
    Madau and Dickinson 2014, Eq. (15): psi(z) = 0.015 (1+z)^2.7 / (1 + ((1+z)/2.9)^5.6) M_sun / yr / Mpc^3, used for
        its shape in time only (its normalization cancels), continued into the future through a(t) > 1 as Loeb,
        Batista and Sloan 2016 do (their Eq. 2.4), with their exponential cutoff exp(-t/t_1), t_1 = 50 Gyr.
    Loeb, Batista and Sloan 2016, Eq. (2.2): the Chabrier initial mass function per unit linear mass,
        xi(m) = m^-2.3 for m > 1 M_sun, and 790 exp(-ln^2(m/0.08) / (2 x 0.69^2)) / m for m <= 1 M_sun.
    Loeb, Batista and Sloan 2016, Eq. (2.3): main-sequence lifetimes, piecewise in m (yr):
        1.0e10 m^-2.5 (0.75 < m < 3), 7.6e9 m^-3.5 (0.25 < m <= 0.75), 5.3e10 m^-2.1 (0.08 <= m <= 0.25).
    Loeb, Batista and Sloan 2016, Eq. (2.1): dP/dt proportional to the integral over birth time and mass of
        xi(m) psi(t') g(t - t', m) with the top-hat window g = 1 for 0 < t - t' < tau(m): the number of habitable
        planets alive at t per comoving volume, with the constant factors eta_Earth and p(life|HZ) cancelling.

Synthetic knobs (ours; each is a declared assumption, not data)
    tau_evo      a delay from a planet's birth to the first observers on it (Gyr); Loeb et al. have none (their delays
                 to biology enter only through m_max).
    L_obs        how long observers last on one planet once they appear (Gyr), capped by the star's lifetime.
    m_min        the lowest stellar mass whose planets count as habitable: 0.08 (red-dwarf-friendly), 0.5
                 (red-dwarf-suppressed), 0.9 (Sun-like only). Loeb et al. scan the same knob.
    reference class
        "moments"  weight = inhabited planets alive at t (observer-moments at a constant rate per inhabited planet);
        "origins"  weight = planets whose observers first appear at t (each inhabited planet counted once);
        "late"     observer-moments weighted by the age of the planet's observer era, Loeb et al.'s remark (Sec. 4)
                   that a given observer is more likely to exist later in a planet's existence, in its simplest
                   (linear) form.
    gamma        an interaction factor: the weight is multiplied by (n(t)/n_max)^gamma with n(t) the density of
                 inhabited planets; gamma = 0 is none; positive means company helps, negative that it hurts. Purely
                 speculative, as the handoff says of its "interactions and civilizations" scenario.
    cutoff_gyr   the e-folding time of the future star-formation cutoff (Loeb et al.: 50; alternative 20); None
                 keeps the bare Madau-Dickinson extrapolation.
    exotic_eps   a rate of exotic late observers per unit physical volume, constant in time, so growing as a(t)^3 per
                 comoving volume: the late-time stress test. Any eps > 0 makes the integral of W diverge.

All results are per unit comoving volume and relative: W is normalized to unit integral where it converges, so a
finite comoving region only makes the integrals finite and drops out.
"""
from __future__ import annotations

import numpy as np

GYR_TIMES_H0_UNIT = 977.79          # 1/H_0 in Gyr for H_0 = 1 km/s/Mpc
PLANCK18 = dict(H0=67.36, Omega_m=0.3153, Omega_L=0.6847, age_Gyr=13.797)
MD14 = dict(a=0.015, b=2.7, c=2.9, d=5.6)
CHABRIER = dict(a=790.0, sigma=0.69, m_c=0.08, slope=-2.3)


class Cosmology:
    """Flat matter + Lambda background with the Planck 2018 baseline parameters; times in Gyr."""

    def __init__(self, H0=PLANCK18["H0"], Omega_m=PLANCK18["Omega_m"], Omega_L=PLANCK18["Omega_L"]):
        self.H0, self.Om, self.OL = float(H0), float(Omega_m), float(Omega_L)
        self.t_H = GYR_TIMES_H0_UNIT / self.H0

    def age(self, z=0.0):
        """Cosmic time at redshift z (Gyr), matter + Lambda closed form."""
        z = np.asarray(z, dtype=float)
        return (2 * self.t_H / (3 * np.sqrt(self.OL))) * np.arcsinh(np.sqrt(self.OL / self.Om) * (1 + z) ** -1.5)

    def scale_factor(self, t):
        """a(t); infinite (with no warning) once sinh overflows, which happens beyond about 5 x 10^3 Gyr."""
        t = np.asarray(t, dtype=float)
        with np.errstate(over="ignore"):
            return (self.Om / self.OL) ** (1 / 3) * np.sinh(1.5 * np.sqrt(self.OL) * t / self.t_H) ** (2 / 3)

    def redshift(self, t):
        return 1 / self.scale_factor(t) - 1


def madau_dickinson(z):
    """Eq. (15) of Madau and Dickinson 2014: the cosmic star-formation-rate density, M_sun / yr / Mpc^3."""
    z = np.asarray(z, dtype=float)
    x = 1 + z
    return MD14["a"] * x ** MD14["b"] / (1 + (x / MD14["c"]) ** MD14["d"])


def star_formation_rate(t, cosmo, t_first_gyr=0.03, cutoff_gyr=50.0):
    """psi at cosmic time t (Gyr), zero before the first stars, continued into the future through a(t), times the
    exponential cutoff of Loeb et al. (None for no cutoff)."""
    t = np.asarray(t, dtype=float)
    out = madau_dickinson(cosmo.redshift(np.maximum(t, 1e-9)))
    if cutoff_gyr is not None:
        out = out * np.exp(-t / cutoff_gyr)
    return np.where(t >= t_first_gyr, out, 0.0)


def chabrier_imf(m):
    """Loeb et al. Eq. (2.2): the initial mass function per unit linear mass (shape only)."""
    m = np.asarray(m, dtype=float)
    low = CHABRIER["a"] * np.exp(-np.log(m / CHABRIER["m_c"]) ** 2 / (2 * CHABRIER["sigma"] ** 2)) / m
    high = m ** CHABRIER["slope"]
    return np.where(m > 1.0, high, low)


def stellar_lifetime_gyr(m):
    """Loeb et al. Eq. (2.3), in Gyr."""
    m = np.asarray(m, dtype=float)
    yr = np.where(m > 0.75, 1.0e10 * m ** -2.5,
                  np.where(m > 0.25, 7.6e9 * m ** -3.5, 5.3e10 * m ** -2.1))
    return yr / 1e9


def _cumulative(x, y):
    """Running trapezoid integral of y over x, starting at zero."""
    out = np.zeros_like(y)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out


class PopulationModel:
    """W(t) per comoving volume for one history, on a time grid, under one set of knobs."""

    def __init__(self, cosmo=None, t_min_gyr=1e-3, t_max_gyr=1e5, n_t=6000, n_m=400, m_max=3.0,
                 t_first_gyr=0.03, cutoff_gyr=50.0):
        self.cosmo = cosmo or Cosmology()
        self.t = np.logspace(np.log10(t_min_gyr), np.log10(t_max_gyr), n_t)
        self.m_max = float(m_max)
        self.n_m = int(n_m)
        self.t_first = t_first_gyr
        self.cutoff = cutoff_gyr
        self.psi = star_formation_rate(self.t, self.cosmo, t_first_gyr, cutoff_gyr)
        self.S = _cumulative(self.t, self.psi)                    # stellar mass formed per comoving volume by t
        self.M = _cumulative(self.t, self.t * self.psi)           # its first moment in time

    def _interp(self, y, x):
        """y(x) on the grid, zero below the grid's start (nothing formed before the first stars)."""
        return np.interp(x, self.t, y, left=0.0, right=float(y[-1]))

    def mass_grid(self, m_min):
        m = np.logspace(np.log10(m_min), np.log10(self.m_max), self.n_m)
        w = chabrier_imf(m) * np.gradient(m)                      # xi(m) dm
        return m, w

    def weight(self, m_min, tau_evo=0.0, L_obs=np.inf, ref_class="moments", gamma=0.0):
        """The unnormalized W(t) on self.t. See the module docstring for the knobs."""
        m, wm = self.mass_grid(m_min)
        tau = stellar_lifetime_gyr(m)
        t = self.t
        total = np.zeros_like(t)
        density = np.zeros_like(t)                                # inhabited planets alive at t, for gamma
        for mi, wi, taui in zip(m, wm, tau):
            if taui <= tau_evo:
                continue                                          # the star dies before observers appear
            end = min(taui, tau_evo + L_obs)
            alive = self._interp(self.S, t - tau_evo) - self._interp(self.S, t - end)
            density += wi * alive
            if ref_class == "moments":
                total += wi * alive
            elif ref_class == "origins":
                total += wi * star_formation_rate(t - tau_evo, self.cosmo, self.t_first, self.cutoff)
            elif ref_class == "late":
                first = (t - tau_evo) * alive
                second = self._interp(self.M, t - tau_evo) - self._interp(self.M, t - end)
                total += wi * np.maximum(first - second, 0.0)
            else:
                raise ValueError(f"unknown reference class {ref_class!r}")
        if gamma != 0.0 and density.max() > 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                factor = np.where(density > 0, (density / density.max()) ** gamma, 0.0)
            total = total * factor
        return total

    def summarize(self, w, t_today=PLANCK18["age_Gyr"]):
        """Normalize W, and read off F(t_today), the median, the 99.99 % time and the peak of t W(t)."""
        t = self.t
        if not np.all(np.isfinite(w)):
            return dict(converged=False)
        norm = float(np.trapezoid(w, t))
        if not np.isfinite(norm) or norm <= 0:
            return dict(converged=False)
        p = w / norm
        F = _cumulative(t, p)
        F_today = float(np.interp(t_today, t, F))
        t_peak = float(t[np.argmax(t * p)])                       # peak of dP/d ln t
        t50 = float(np.interp(0.5, F, t))
        t9999 = float(np.interp(0.9999, F, t)) if F[-1] >= 0.9999 else float("nan")
        # last time with any observer support: where W falls below 1e-4 of its peak for good
        above = np.flatnonzero(w > 1e-4 * w.max())
        T_support = float(t[above[-1]]) if above.size else float("nan")
        tail = float(1 - np.interp(0.9 * t[-1], t, F))             # weight in the last tenth of the grid
        return dict(converged=True, norm=norm, F_today=F_today, t_peak_gyr=t_peak, t50_gyr=t50,
                    t9999_gyr=t9999, T_support_gyr=T_support, tail_weight=tail, p=p, F=F)

    def exotic_weight(self, eps):
        """A constant rate of exotic late observers per physical volume: eps a(t)^3 per comoving volume. Beyond a few
        thousand Gyr a(t)^3 overflows double precision; the weight is then infinite and `summarize` reports that the
        integral did not converge, which is the physical statement too."""
        with np.errstate(over="ignore"):
            return eps * self.cosmo.scale_factor(self.t) ** 3


def pooled_and_averaged_F(histories, t_today=PLANCK18["age_Gyr"]):
    """Two ways to combine histories, each a dict with p (Born weight), t grid, w (unnormalized W):
    pooled   F = sum_a p_a N_a(<= t0) / sum_a p_a N_a(total)   (the handoff's F(t_0); Sebens-Carroll's form),
    averaged F = sum_a p_a [N_a(<= t0) / N_a(total)]           (Srednicki-Hartle's xerographic recipe: normalize within
                                                               each history, then weight by third-person probability).
    They agree only when every history holds the same total."""
    num = den = avg = 0.0
    for h in histories:
        t, w, p = h["t"], h["w"], h["p"]
        tot = float(np.trapezoid(w, t))
        upto = float(np.trapezoid(np.where(t <= t_today, w, 0.0), t))
        num += p * upto
        den += p * tot
        avg += p * (upto / tot if tot > 0 else 0.0)
    return dict(pooled=num / den if den > 0 else float("nan"), averaged=avg)
