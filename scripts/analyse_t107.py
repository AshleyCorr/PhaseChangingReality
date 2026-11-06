"""T107's reading by its registered words (PREREGISTRATION.md T107). Usage:

    python scripts/analyse_t107.py [results/t107_boundary_feet] [results/t107_boundary_feet_heldout]

Prints, per length: (1) the share of replicas with a foot on each side at 200 sweeps after the escape; (2) the share
with the outside foot larger than the inside at 500, and whether the inside foot ever exceeds the band's points;
(3) the share joined at 500 (read, not scored); (4) under her kind rule, among replicas that made a straddling pair,
the share whose outside foot carries the outward kind's sign and inside foot the reverse, and the same under the
coin (control); (5) her size ratio, |net twist outside| over (curl left on squares + held in stores), median; the
median escape sweep. Verdicts by the registered thresholds. Pure reading; writes nothing.
"""
import sys

import numpy as np
import pandas as pd


def read(stem):
    feet = pd.read_csv(f"{stem}_feet.csv")
    kind = pd.read_csv(f"{stem}_kind.csv")
    return feet, kind


def report(feet, kind):
    out = {}
    for L, f in feet.groupby("L"):
        esc = f[f["escape"].notna()]
        n = f["replica"].nunique()
        f200 = esc[esc["dt"] == 200]
        f500 = esc[esc["dt"] == 500]
        both = float(((f200["inside"] > 0) & (f200["outside"] > 0)).sum()) / n
        grow = float((f500["outside"] > f500["inside"]).sum()) / n
        bounded = bool((esc["inside"] <= esc["band_points"]).all())
        joined = float(f500["joined"].sum()) / n
        k = kind[kind["L"] == L]
        kb = k[(k["rule"] == "boundary") & (k["outward_kind"] != 0)]
        mirror_b = float(((np.sign(kb["twist_out"]) == kb["outward_kind"]) & (np.sign(kb["twist_in"]) == -kb["outward_kind"])).mean()) if len(kb) else float("nan")
        kc = k[k["rule"] == "coin"]
        # the coin has no outward kind: score opposite signs on the two feet, any orientation
        mirror_c = float(((np.sign(kc["twist_out"]) != 0) & (np.sign(kc["twist_out"]) == -np.sign(kc["twist_in"]))).mean())
        denom = kb["stored_on_squares"] + kb["held_in_stores"]
        ratio = float(np.median(np.abs(kb["twist_out"]) / denom)) if len(kb) else float("nan")
        out[L] = dict(n=n, escaped=int(esc["replica"].nunique()), escape_median=float(esc.groupby("replica")["escape"].first().median()),
                      foot_each_side_200=both, outside_larger_500=grow, inside_bounded=bounded, joined_500=joined,
                      straddling=int(len(kb)), mirror_rule=mirror_b, mirror_coin=mirror_c, size_ratio=ratio,
                      books_ok=bool(k["books_ok"].all()))
    return out


def verdicts(out):
    v = {}
    v["1 foot each side"] = "HOLDS" if all(r["foot_each_side_200"] > 0.5 for r in out.values()) else "FAILS"
    v["2a outside larger"] = "HOLDS" if all(r["outside_larger_500"] > 0.5 for r in out.values()) else "FAILS"
    v["2b inside bounded"] = "HOLDS" if all(r["inside_bounded"] for r in out.values()) else "FAILS"
    v["4 mirror images"] = "HOLDS" if all(r["mirror_rule"] > 0.5 for r in out.values()) else "FAILS"
    return v


if __name__ == "__main__":
    stems = sys.argv[1:] or ["results/t107_boundary_feet"]
    for stem in stems:
        feet, kind = read(stem)
        out = report(feet, kind)
        print(f"== {stem}")
        for L, r in sorted(out.items()):
            print(f"L {L:4d}: escaped {r['escaped']}/{r['n']}, median escape sweep {r['escape_median']:.0f}; "
                  f"foot each side at +200: {r['foot_each_side_200']:.2f}; outside larger at +500: {r['outside_larger_500']:.2f}; "
                  f"inside bounded: {r['inside_bounded']}; joined at +500: {r['joined_500']:.2f}; "
                  f"mirror under the rule: {r['mirror_rule']:.2f} of {r['straddling']} straddling; opposite signs under the coin: {r['mirror_coin']:.2f}; "
                  f"size ratio (median): {r['size_ratio']:.4f}; books {r['books_ok']}")
        for k, val in verdicts(out).items():
            print(f"  {k}: {val}")
