"""PREREGISTRATION T77: does a dark energy that fades open the causal diamond, and does our age become early?

Usage:  python scripts/run_t77.py configs/t77_diamond_w.json
Writes  results/<name>.csv and results/<name>.meta.json; refuses to overwrite.

Each w row is T73's computation (the four classes of T72 weighted by the causal diamond, the three tails, the class's own
delay and window, and the sensitivity variants) under a flat cosmology with constant w and H0 refitted to a 13.8 Gyr age
(scripts/cosmo_w.py). The w = -1 row is T73 itself: it uses observer_window's own closed forms, so it reproduces T73's
numbers exactly (the check is in the output). Scored rows follow T73's rule: the registered delay, n = 5, tail 50 Gyr and
the class's own cap setting.
"""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import observer_window as O                      # noqa: E402
from cosmo_w import Cosmo, T_TODAY               # noqa: E402

TRAPZ = O.trapezoid


def patched_model(cosmo: Cosmo | None, tail):
    """A Model whose star history is computed in the given cosmology (None: T73's own)."""
    if cosmo is None:
        return O.Model(tail_gyr=tail)
    saved = O.z_at_age
    O.z_at_age = cosmo.z_at_age
    try:
        return O.Model(tail_gyr=tail)
    finally:
        O.z_at_age = saved


def weight_for(cosmo: Cosmo | None, model):
    if cosmo is None:
        return model.diamond_volume()
    return cosmo.diamond_weight(model.t)


def run_w(w: float, classes: list, tails: list) -> list[dict]:
    cosmo = None if w == -1.0 else Cosmo(w)
    rows = []
    for tail in tails:
        tail_g = None if tail == "none" else float(tail)
        model = patched_model(cosmo, tail_g)
        weight = weight_for(cosmo, model)
        for spec in classes:
            kern = spec["kernel"]
            delays = [spec.get("delay_Gyr", 0.0)]
            if kern in ("fixed", "fixed_moments"):
                delays = [spec["delay_Gyr"] - 1.0, spec["delay_Gyr"], spec["delay_Gyr"] + 1.0]
            steps = spec.get("n_steps", [5]) if kern == "hardsteps" else [None]
            caps = [None, 100.0] if spec.get("cap_Gyr") is None else [spec["cap_Gyr"], None]
            for d in delays:
                for n in steps:
                    p = model.density(spec["m_min"], spec["m_max"], kern, delay=d,
                                      window_fixed=spec.get("window_fixed_Gyr"), n_steps=n or 5)
                    if spec.get("diamond"):
                        p = p * weight
                    for cap in caps:
                        s = O.summarize(model.t, p, cap_gyr=cap)
                        rows.append(dict(
                            w=w, H0_km_s_Mpc=(O.H0_KM_S_MPC if cosmo is None else round(cosmo.H0_km_s_Mpc, 4)),
                            cls=spec["name"], kernel=kern, diamond=bool(spec.get("diamond")), delay_Gyr=d,
                            n_steps=n, tail_Gyr=tail, cap_Gyr=cap,
                            scored=(d == spec.get("delay_Gyr", 0.0) and (n in (None, 5)) and tail == 50
                                    and cap == spec.get("cap_Gyr")),
                            verdict=O.verdict(s["F_now"]), **s))
    return rows


def main(path: str) -> None:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    out = Path("results") / f"{cfg['name']}.csv"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    classes = json.loads(Path(cfg["classes_from"]).read_text(encoding="utf-8"))["classes"]
    rows: list[dict] = []
    for w in cfg["w_values"]:
        rows.extend(run_w(float(w), classes, cfg["tails_Gyr"]))
    cols = list(rows[0].keys())
    with out.open("w", encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join("" if r[c] is None else str(r[c]) for c in cols) + "\n")
    meta = dict(config=cfg, config_path=str(path), python=platform.python_version(), numpy=np.__version__,
                cosmology=dict(Omega_m=0.315, age_today_Gyr=T_TODAY,
                               H0_km_s_Mpc={str(w): (O.H0_KM_S_MPC if w == -1.0 else round(Cosmo(w).H0_km_s_Mpc, 4))
                                            for w in cfg["w_values"]},
                               eta_inf={str(w): (None if w == -1.0 else str(Cosmo(w).eta_inf))
                                        for w in cfg["w_values"]}))
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for w in cfg["w_values"]:
        for r in rows:
            if r["w"] == float(w) and r["scored"]:
                print(f"w={r['w']:+.3f} {r['cls']:<4} F_now={r['F_now']:.4f} peak(dP/dt)={r['peak_dPdt_Gyr']:.2f} Gyr  {r['verdict']}")


if __name__ == "__main__":
    main(sys.argv[1])
