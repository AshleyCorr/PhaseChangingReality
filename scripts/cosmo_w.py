"""Flat cosmologies with a constant dark-energy equation of state w, for PREREGISTRATION T77 (the stand-ins of a fading
dark energy). Standard cosmology, not the toy.

Model: H(a)^2 = H0^2 [Omega_m a^-3 + (1 - Omega_m) a^(-3 (1 + w))], flat, a = 1 today, Omega_m = 0.315 (as T72/T73).
Declared choice (T77): H0 is refitted for each w so that the age of the universe today is T_TODAY = 13.8 Gyr, so that
"our age" is the same observed number in every row; the fitted H0 is recorded beside each row. The w = -1 control is
NOT computed here: T77's control reuses scripts/observer_window.py's own closed forms, which are T73's.

Quantities: t(a) = integral of d ln a / H (Gyr); a(t) and z(t) by interpolation; the conformal-time-like coordinate
eta(t) = integral of (a0 / a) dt, in units of today's comoving scale, as observer_window.diamond_volume (BHKP07 Eq. 18
and 19); eta_inf = eta at the far future, infinite where the integral diverges (w = -1/3, a logarithm; for w < -1/3 it
converges). Diamond weight V_c(t) = (4 pi / 3) r^3, r = min(eta(t), eta_inf - eta(t)), with r = eta where eta_inf is
infinite, as T73 places the bottom tip at t_i = 0.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq

OMEGA_M = 0.315
KM_S_MPC_TO_PER_GYR = 1.0220e-3
H0_PLANCK = 67.4
T_TODAY = 13.8
T_MAX = 3.0e4                       # the observer_window grid's end (Gyr)


class Cosmo:
    """A flat cosmology with constant w and H0 refitted so that t(a = 1) = T_TODAY."""

    def __init__(self, w: float, n: int = 60_000):
        self.w = float(w)
        self.lna = np.linspace(np.log(1e-9), 60.0, n)          # ln a from the early universe to far future
        self.a = np.exp(self.lna)
        self.h0 = brentq(lambda h: self._age_today(h) - T_TODAY, 0.02, 0.2)   # per Gyr
        self.t = self._t_of_lna(self.h0)
        self.H0_km_s_Mpc = self.h0 / KM_S_MPC_TO_PER_GYR
        self.eta, self.eta_inf = self._eta(self.h0)

    def _hub(self, a, h0):
        return h0 * np.sqrt(OMEGA_M * a ** -3 + (1 - OMEGA_M) * a ** (-3 * (1 + self.w)))

    def _t_of_lna(self, h0):
        # t(a) = integral of d ln a / H; the piece below the grid is matter-like, (2/3) a^(3/2) / (H0 sqrt(Omega_m))
        integrand = 1.0 / self._hub(self.a, h0)
        t = cumulative_trapezoid(integrand, self.lna, initial=0.0)
        return t + (2.0 / 3.0) * self.a[0] ** 1.5 / (h0 * np.sqrt(OMEGA_M))

    def _age_today(self, h0):
        t = self._t_of_lna(h0)
        return float(np.interp(0.0, self.lna, t))

    def _eta(self, h0):
        a0 = 1.0
        # eta = integral of (a0 / a) d ln a / H. Early piece (matter era, H = H0 sqrt(Omega_m) a^(-3/2)): 2 a^(1/2) / (H0 sqrt(Omega_m)).
        integrand = (a0 / self.a) / self._hub(self.a, h0)
        e = cumulative_trapezoid(integrand, self.lna, initial=0.0) + 2.0 * np.sqrt(self.a[0]) / (h0 * np.sqrt(OMEGA_M))
        # Far future: integrand ~ exp(s ln a) with s the late log-slope. Converges iff s < 0, i.e. w < -1/3; then the tail
        # beyond the grid is f(L) / (-s). At w = -1/3 the slope is zero and the integral diverges logarithmically.
        slope = (np.log(integrand[-1]) - np.log(integrand[-2])) / (self.lna[-1] - self.lna[-2])
        eta_inf = float(e[-1] + integrand[-1] / (-slope)) if slope < -1e-9 else np.inf
        return e, eta_inf

    def age_of_a(self, a):
        return np.interp(np.log(a), self.lna, self.t)

    def z_at_age(self, t):
        """Redshift at cosmic time t (Gyr); -1 beyond the grid's far future."""
        a = np.interp(np.asarray(t, dtype=float), self.t, self.a)
        return 1.0 / a - 1.0

    def diamond_weight(self, t_grid):
        """V_c on t_grid (Gyr) for the bottom tip at t_i = 0 (the reheating point, as observer_window)."""
        e = np.interp(t_grid, self.t, self.eta)
        r = np.clip(e if not np.isfinite(self.eta_inf) else np.minimum(e, self.eta_inf - e), 0.0, None)
        return (4.0 * np.pi / 3.0) * r ** 3
