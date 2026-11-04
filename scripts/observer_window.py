"""PREREGISTRATION T72: when is an observer like us most likely to find itself, by its universe's own age?

A calculation in standard cosmology, not a run of the toy. Everything here is a published input (the reading notes of
12 and 19 October 2026: ``docs/reading/notes/2026-10-12_orh_cosmic_time_observer_measure.md``,
``2026-10-19_t72_inputs.md``, ``2026-10-19_observer_window_endpoints.md``) or a declared choice of T72.

The model, in the notation of Loeb, Batista and Sloan 2016 [LBS16] (their Eq. 2.1, as recorded on 12 October):

    dP/dt (t) proportional to  integral dm  xi(m)  integral dt'  psi(t')  u(t - t'; m)

with xi the Chabrier initial mass function (LBS16 Eq. 2.2), psi the cosmic star-formation rate density (Madau and
Dickinson 2014 Eq. 15 against redshift, converted to cosmic time with a flat Lambda-CDM cosmology, truncated before the
first stars and multiplied by LBS16's exponential tail exp(-t/t1) into the future), and u the per-planet kernel that says
when, after a planet of a star of mass m forms at t', the counted thing exists:

* gate (LBS16 as published): u = 1 while the star lives (planet-years, no delay); the known-answer test.
* fixed delay d (classes A, B, C): one civilization per planet, once, at t' + d, if d is less than the planet's window.
* hard steps as a selected fast case (class D): one civilization per planet at an age drawn uniformly from the last
  1/(n+1) of the planet's window.

The window is the star's main-sequence lifetime tau(m) (LBS16 Eq. 2.3) or a fixed length per planet (class B).
The occurrence of habitable planets and the probability of life are constant within a class and cancel in the
normalization, as in LBS16; the metallicity dependence of planet formation is not applied (T72, Amendment 1).

Usage:  python scripts/observer_window.py configs/t72_observer_window.json
Writes  results/<name>.csv and results/<name>.meta.json; refuses to overwrite.
"""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np

# np.trapz was renamed np.trapezoid in NumPy 2.0 and later removed; the same trapezoid rule either way.
trapezoid = getattr(np, "trapezoid", None) or np.trapz

GYR = 1.0
T_TODAY = 13.8          # Gyr, Planck 2018 (13.79; general knowledge, to verify)
H0_KM_S_MPC = 67.4      # Planck 2018
OMEGA_M = 0.315
OMEGA_L = 1.0 - OMEGA_M
KM_S_MPC_TO_PER_GYR = 1.0220e-3   # 1 km/s/Mpc = 1.0220e-3 / Gyr


# ---------------------------------------------------------------- cosmology (flat matter + Lambda)
def _h0_per_gyr() -> float:
    return H0_KM_S_MPC * KM_S_MPC_TO_PER_GYR


def age_at_z(z: np.ndarray | float) -> np.ndarray | float:
    """Cosmic time in Gyr at redshift z for flat matter + Lambda (analytic)."""
    h0 = _h0_per_gyr()
    a = np.sqrt(OMEGA_L / OMEGA_M) * (1.0 + np.asarray(z, dtype=float)) ** (-1.5)
    return (2.0 / (3.0 * h0 * np.sqrt(OMEGA_L))) * np.arcsinh(a)


def z_at_age(t: np.ndarray | float) -> np.ndarray | float:
    """Inverse of age_at_z; z -> -1 as t -> infinity."""
    h0 = _h0_per_gyr()
    s = np.sinh(1.5 * h0 * np.sqrt(OMEGA_L) * np.asarray(t, dtype=float))
    one_plus_z = (np.sqrt(OMEGA_L / OMEGA_M) / s) ** (2.0 / 3.0)
    return one_plus_z - 1.0


# ---------------------------------------------------------------- inputs
def sfr_density(t: np.ndarray, t_first: float = 0.03, tail_gyr: float | None = 50.0) -> np.ndarray:
    """Madau and Dickinson 2014 Eq. 15 (Msun/yr/Mpc^3) at cosmic time t (Gyr); zero before the first stars;
    multiplied by exp(-t/tail) into the future as LBS16 do (tail None = no cutoff)."""
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    ok = t > t_first
    z = z_at_age(t[ok])
    opz = 1.0 + z
    psi = 0.015 * opz ** 2.7 / (1.0 + (opz / 2.9) ** 5.6)
    if tail_gyr is not None:
        psi = psi * np.exp(-t[ok] / tail_gyr)
    out[ok] = psi
    return out


def imf_chabrier(m: np.ndarray) -> np.ndarray:
    """LBS16 Eq. 2.2 (Chabrier), per unit linear mass, unit coefficient at 1 Msun."""
    m = np.asarray(m, dtype=float)
    out = np.empty_like(m)
    hi = m > 1.0
    out[hi] = m[hi] ** -2.3
    lo = ~hi
    out[lo] = 790.0 * np.exp(-np.log(m[lo] / 0.08) ** 2 / (2 * 0.69 ** 2)) / m[lo]
    return out


def lifetime(m: np.ndarray) -> np.ndarray:
    """Main-sequence lifetime in Gyr, LBS16 Eq. 2.3 (piecewise)."""
    m = np.asarray(m, dtype=float)
    out = np.empty_like(m)
    a = m > 0.75
    b = (m > 0.25) & ~a
    c = ~a & ~b
    out[a] = 10.0 * m[a] ** -2.5
    out[b] = 7.6 * m[b] ** -3.5
    out[c] = 53.0 * m[c] ** -2.1
    return out


# ---------------------------------------------------------------- the calculation
class Model:
    def __init__(self, t_max: float = 3.0e4, n_t: int = 6000, tail_gyr: float | None = 50.0, t_first: float = 0.03):
        # log-spaced time grid for reading out p(t); a fine uniform grid for the cumulative SFR
        self.t = np.geomspace(0.01, t_max, n_t)
        self.tf = np.linspace(0.0, t_max, 400_001)
        psi = sfr_density(self.tf, t_first=t_first, tail_gyr=tail_gyr)
        self.Psi = np.concatenate([[0.0], np.cumsum(0.5 * (psi[1:] + psi[:-1]) * np.diff(self.tf))])

    def cum_sfr(self, t: np.ndarray) -> np.ndarray:
        """Integral of psi from 0 to t (clipped at 0 below)."""
        return np.interp(np.clip(t, 0.0, self.tf[-1]), self.tf, self.Psi)

    def density(self, m_min: float, m_max: float, kernel: str, delay: float = 0.0, window_fixed: float | None = None,
                n_steps: int = 5, n_m: int = 400) -> np.ndarray:
        """dP/dt on self.t (unnormalized), for stars m_min..m_max."""
        m = np.geomspace(m_min, m_max, n_m)
        w_m = imf_chabrier(m)
        win = np.full_like(m, window_fixed) if window_fixed is not None else lifetime(m)
        t = self.t[:, None]
        if kernel == "gate":            # LBS16: planet-years while the star lives
            k = self.cum_sfr(t) - self.cum_sfr(t - win[None, :])
        elif kernel == "fixed":         # one civilization per planet at formation + delay, if the window allows
            alive = (win > delay).astype(float)[None, :]
            k = sfr_density_cached(self, t - delay) * alive
        elif kernel == "fixed_moments":  # T72's reported-not-scored variant (observer-moments): a civilization that
            alive = (win > delay).astype(float)[None, :]          # arises at t' + delay exists at every age until its
            k = (self.cum_sfr(t - delay) - self.cum_sfr(t - win[None, :])) * alive   # planet's window ends at t' + win
        elif kernel == "hardsteps":     # uniform over the last 1/(n+1) of the window
            lo = win * n_steps / (n_steps + 1.0)
            width = win - lo
            k = (self.cum_sfr(t - lo[None, :]) - self.cum_sfr(t - win[None, :])) / width[None, :]
        else:
            raise ValueError(kernel)
        return trapezoid(k * w_m[None, :], m, axis=1)

    # ------------------------------------------------------------ T73: the causal diamond's comoving volume
    def diamond_volume(self, t_i: float = 0.0) -> np.ndarray:
        """V_c(t) = (4 pi / 3) r^3 with r = min(eta(t) - eta(t_i), eta_inf - eta(t)) ([BHKP07] Eqs. 18, 19), eta the
        conformal time of the flat Planck 2018 cosmology, on self.t; the bottom tip at reheating, t_i = 0, as the paper
        places it (not at the first stars: that offset moves the peak to 5.4 Gyr). Known answer: peaks at 0.23 t_Lambda (Eq. 20)."""
        h0 = _h0_per_gyr()
        tt = np.geomspace(1e-9, self.t[-1], 2_000_001)
        with np.errstate(over="ignore"):
            a = (OMEGA_M / OMEGA_L) ** (1.0 / 3.0) * np.sinh(1.5 * h0 * np.sqrt(OMEGA_L) * tt) ** (2.0 / 3.0)
            a0 = (OMEGA_M / OMEGA_L) ** (1.0 / 3.0) * np.sinh(1.5 * h0 * np.sqrt(OMEGA_L) * T_TODAY) ** (2.0 / 3.0)
            inv = a0 / a
        inv = np.where(np.isfinite(inv), inv, 0.0)
        eta = np.concatenate([[0.0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(tt))])
        eta += 3.0 * tt[0] * inv[0]          # the piece below the first grid point, a ~ t^(2/3)
        eta_inf = eta[-1]
        e = np.interp(self.t, tt, eta)
        e_i = np.interp(t_i, tt, eta)
        r = np.clip(np.minimum(e - e_i, eta_inf - e), 0.0, None)
        return (4.0 * np.pi / 3.0) * r ** 3


def t_lambda() -> float:
    """The dark-energy timescale 1 / H_infinity = 1 / (H0 sqrt(Omega_Lambda)), as [BHKP07] define t_Lambda (to verify)."""
    return 1.0 / (_h0_per_gyr() * np.sqrt(OMEGA_L))


def sfr_density_cached(model: Model, t: np.ndarray) -> np.ndarray:
    """psi(t) read off the model's fine grid (so the tail choice is shared)."""
    tf, Psi = model.tf, model.Psi
    psi_f = np.gradient(Psi, tf)
    return np.interp(np.clip(t, 0.0, tf[-1]), tf, psi_f) * (t > 0)


def summarize(t: np.ndarray, p: np.ndarray, cap_gyr: float | None = None) -> dict:
    p = np.where(np.isfinite(p), p, 0.0)
    if cap_gyr is not None:
        p = p * (t <= cap_gyr)
    norm = trapezoid(p, t)
    p = p / norm
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(t))])
    f_now = float(np.interp(T_TODAY, t, cum))
    peak_dpdt = float(t[np.argmax(p)])
    peak_dplnt = float(t[np.argmax(p * t)])
    median = float(np.interp(0.5, cum, t))
    return dict(F_now=f_now, peak_dPdt_Gyr=peak_dpdt, peak_dPdlnt_Gyr=peak_dplnt, median_Gyr=median)


def verdict(f: float) -> str:
    if f < 0.05:
        return "EARLY"
    if f > 0.95:
        return "LATE"
    return "TYPICAL"


def run_class(spec: dict, tail_gyr: float | None) -> list[dict]:
    model = Model(tail_gyr=tail_gyr)
    rows = []
    kern = spec["kernel"]
    delays = [spec.get("delay_Gyr", 0.0)]
    if kern in ("fixed", "fixed_moments"):
        delays = [spec["delay_Gyr"] - 1.0, spec["delay_Gyr"], spec["delay_Gyr"] + 1.0]
    steps = spec.get("n_steps", [5]) if kern == "hardsteps" else [None]
    caps = [None, 100.0] if spec.get("cap_Gyr") is None else [spec["cap_Gyr"], None]
    weight = model.diamond_volume() if spec.get("diamond") else 1.0
    for d in delays:
        for n in steps:
            p = model.density(spec["m_min"], spec["m_max"], kern, delay=d, window_fixed=spec.get("window_fixed_Gyr"),
                              n_steps=n or 5) * weight
            for cap in caps:
                s = summarize(model.t, p, cap_gyr=cap)
                rows.append(dict(cls=spec["name"], kernel=kern, diamond=bool(spec.get("diamond")), delay_Gyr=d, n_steps=n, tail_Gyr=tail_gyr, cap_Gyr=cap,
                                 scored=(d == spec.get("delay_Gyr", 0.0) and (n in (None, 5)) and tail_gyr == 50.0
                                         and cap == spec.get("cap_Gyr")),
                                 verdict=verdict(s["F_now"]), **s))
    return rows


def main(path: str) -> None:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    out = Path("results") / f"{cfg['name']}.csv"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    rows: list[dict] = []
    for tail in cfg["tails_Gyr"]:
        for spec in cfg["classes"]:
            rows.extend(run_class(spec, None if tail == "none" else float(tail)))
    cols = list(rows[0].keys())
    with out.open("w", encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join("" if r[c] is None else str(r[c]) for c in cols) + "\n")
    meta = dict(config=cfg, config_path=str(path), python=platform.python_version(), numpy=np.__version__,
                cosmology=dict(H0=H0_KM_S_MPC, Omega_m=OMEGA_M, t_today_Gyr=T_TODAY))
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for r in rows:
        if r["scored"]:
            print(f"{r['cls']:<4} F_now={r['F_now']:.4f} peak(dP/dt)={r['peak_dPdt_Gyr']:.2f} Gyr "
                  f"peak(dP/dlnt)={r['peak_dPdlnt_Gyr']:.1f} Gyr median={r['median_Gyr']:.1f} Gyr  {r['verdict']}")


if __name__ == "__main__":
    main(sys.argv[1])
