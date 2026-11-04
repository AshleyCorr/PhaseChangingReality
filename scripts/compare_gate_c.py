"""Gate C: our six-link sweeps against [T22] Fig. 3, under the criterion accepted before any comparison. Usage:

    python scripts/compare_gate_c.py results/gatec_t22_fig3_p1a.csv [more result files ...]

A config's optional "coupling_scale" maps our coupling onto the published axis as g / scale (the reading in
which [T22] plots the coupling divided by N^(1 - 2/D)). TASKS, six links per point, Gate C: where our two legs agree (within 0.1 squares per vertex, with tau_int below
n_meas / 20), the largest difference from the digitised points, interpolated in ln g, is below 0.3 squares per vertex;
and the coupling where the curve crosses 6 squares per vertex agrees within 0.05 in ln g. The cold plateau is
reported, not scored, unless both legs agree there.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

TARGET = Path(__file__).resolve().parents[1] / "docs" / "published" / "T22_fig3_digitised.csv"
AGREE, TOL, CROSS_TOL, CROSS_AT = 0.1, 0.3, 0.05, 6.0


def crossing(ln_g, y, level=CROSS_AT):
    """ln g where y first falls through level, going from cold (small g) to hot, by linear interpolation."""
    order = np.argsort(ln_g)
    x, y = np.asarray(ln_g)[order], np.asarray(y)[order]
    for i in range(len(x) - 1):
        if (y[i] - level) * (y[i + 1] - level) <= 0 and y[i] != y[i + 1]:
            return float(x[i] + (level - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))
    return float("nan")


def compare(path):
    rows = list(csv.DictReader(open(path, newline="")))
    cfg = json.loads(Path(path).with_suffix(".meta.json").read_text())["config"]
    n_meas = cfg["n_meas"]
    scale = float(cfg.get("coupling_scale", 1.0))     # the published axis is g / scale (reading (ii): N^(1/3))
    tgt = list(csv.DictReader(open(TARGET, newline="")))
    tx = np.array([float(r["ln_hbar_g"]) for r in tgt]); ty = np.array([float(r["squares_per_vertex"]) for r in tgt])
    cool = {float(r["g"]): r for r in rows if r["leg"] == "cool"}
    heat = {float(r["g"]): r for r in rows if r["leg"] == "heat"}
    both = sorted(set(cool) & set(heat))
    agree, diffs = [], []
    for g in both:
        c, h = float(cool[g]["squares_per_vertex"]), float(heat[g]["squares_per_vertex"])
        taus = [float(cool[g]["tau_int"]), float(heat[g]["tau_int"])]
        ok = abs(c - h) <= AGREE and all(np.isfinite(t) and t < n_meas / 20 for t in taus)
        if ok and tx.min() <= np.log(g / scale) <= tx.max():
            agree.append(g)
            diffs.append((g, (c + h) / 2 - float(np.interp(np.log(g / scale), tx, ty))))
    worst = max(diffs, key=lambda d: abs(d[1])) if diffs else (float("nan"), float("nan"))
    gs = sorted(cool)
    ours = crossing(np.log(np.array(gs) / scale), [float(cool[g]["squares_per_vertex"]) for g in gs])
    theirs = crossing(tx, ty)
    cold = min(gs)
    print(f"{Path(path).name}: legs agree at {len(agree)} of {len(both)} couplings "
          f"(published axis ln g {np.log(min(agree) / scale):.2f} to {np.log(max(agree) / scale):.2f}; scale {scale:.3f})" if agree else f"{Path(path).name}: legs agree nowhere")
    print(f"   largest difference from [T22] Fig. 3 where they agree: {worst[1]:+.3f} squares per vertex at g = {worst[0]}"
          f" -> {'PASS' if abs(worst[1]) < TOL else 'FAIL'} (tolerance {TOL})")
    print(f"   crossing of 6: ours at ln g = {ours:.3f}, theirs {theirs:.3f}, difference {ours - theirs:+.3f}"
          f" -> {'PASS' if abs(ours - theirs) <= CROSS_TOL else 'FAIL'} (tolerance {CROSS_TOL})")
    print(f"   cold end g = {cold}: cooling {float(cool[cold]['squares_per_vertex']):.2f}, published "
          f"{float(np.interp(np.log(cold / scale), tx, ty)):.2f} (reported, not scored)")
    for g0 in (4.0, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5):
        g = min(cool, key=lambda v: abs(v / scale - g0))
        if abs(g / scale - g0) < 1e-3:
            c = float(cool[g]["squares_per_vertex"]); h = float(heat[g]["squares_per_vertex"]) if g in heat else float("nan")
            print(f"      published g = {g0:4.2f} (ours {g:6.2f}): cool {c:6.2f}  heat {h:6.2f}  "
                  f"published {float(np.interp(np.log(g0), tx, ty)):6.2f}"
                  f"  tau {float(cool[g]['tau_int']):7.1f}  acc {float(cool[g]['acceptance']):.4f}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        compare(p)
