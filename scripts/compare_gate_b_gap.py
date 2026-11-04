"""Gate B check, 2026-09-23: is [T25] Fig. 3 the author's protocol trapping the lattice branch? Usage:

    python scripts/compare_gate_b_gap.py

Prints only; writes nothing. Every input is a committed file.

THE QUESTION. Gate B has never passed because two published curves at N = 160 disagree with each
other in the middle of the transition: [T25] Fig. 3 sits up to 0.41 above [KTB19] Fig. 8a, and our
equilibrium runs match Fig. 8a to rms 0.005. On 23 September the author wrote that his procedure
of starting each coupling from the previous coupling's final graph may accentuate the jump by
trapping configurations in the wrong phase. If Fig. 3 was made that way, its excess over Fig. 8a
should have the shape of the excess our copy of his protocol (T13 protocol P) shows over our
equilibrium (T13 protocol E) at the neighbouring size N = 196: one-sided (on the leg that carries
the lattice), and confined to the couplings below the transition where the lattice can survive.

WHAT IS COMPARED. Six differences, each interpolated linearly in ln g onto a common grid:
Fig. 3 heat minus Fig. 8a and Fig. 3 cool minus Fig. 8a (published against published, N = 160);
Fig. 3 heat and cool minus our N = 160 tempering (published against ours); protocol P heat and
cool minus protocol E from the melt at N = 196 (ours against ours). For each: the peak, where it
is, and the range of g over which it exceeds 0.10. Also the coupling at which each curve crosses
phi = 0.6, the middle of the transition.

Sources: docs/published/T25_fig3_digitised.csv, docs/published/KTB19_fig8a_N160_digitised.csv,
results/cqg_n160_lam1_nocap_tempering.csv, results/t13_seq_n196.csv,
results/t13_temper_n196_melt.csv. Ours, unverified; a comparison, not a run.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GRID = [3.0, 3.5, 4.0, 4.5, 5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 7.0, 8.0]
ABOVE = 0.10


def load(rel):
    with open(ROOT / rel, newline="") as fh:
        return list(csv.DictReader(fh))


def curve(rows, gkey="g", pkey="phi", where=lambda r: True):
    """(g ascending, mean phi over whatever rows share each g)."""
    d = defaultdict(list)
    for r in rows:
        if where(r):
            d[float(r[gkey])].append(float(r[pkey]))
    gs = np.array(sorted(d))
    return gs, np.array([np.mean(d[g]) for g in gs])


def at(c, g):
    """Linear interpolation in ln g; nan outside the curve's range."""
    gs, ph = c
    lg, x = np.log(gs), np.log(g)
    if x < lg.min() or x > lg.max():
        return np.nan
    return float(np.interp(x, lg, ph))


def crossing(c, level=0.6):
    gs, ph = c
    if not ph.min() < level < ph.max():
        return np.nan
    order = np.argsort(ph)
    return float(np.exp(np.interp(level, ph[order], np.log(gs)[order])))


def gap_summary(a, b, lo=2.2, hi=9.0):
    fine = np.exp(np.linspace(np.log(lo), np.log(hi), 400))
    gap = np.array([at(a, g) - at(b, g) for g in fine])
    ok = ~np.isnan(gap)
    if not ok.any():
        return None
    i = int(np.nanargmax(gap))
    above = fine[ok][gap[ok] > ABOVE]
    return dict(peak=float(gap[i]), at=float(fine[i]),
                above=(float(above.min()), float(above.max())) if len(above) else None)


def main():
    fig3 = load("docs/published/T25_fig3_digitised.csv")
    fig8 = load("docs/published/KTB19_fig8a_N160_digitised.csv")
    curves = {
        "Fig3 heat": curve(fig3, "g", "S_over_N", lambda r: r["series"].startswith("heat")),
        "Fig3 cool": curve(fig3, "g", "S_over_N", lambda r: r["series"].startswith("cool")),
        "Fig8a": curve(fig8, "g", "S_over_N"),
        "ours N160": curve(load("results/cqg_n160_lam1_nocap_tempering.csv")),
        "P196 heat": curve(load("results/t13_seq_n196.csv"), where=lambda r: r["leg"] == "heat"),
        "P196 cool": curve(load("results/t13_seq_n196.csv"), where=lambda r: r["leg"] == "cool"),
        "E196 melt": curve(load("results/t13_temper_n196_melt.csv")),
    }
    names = list(curves)
    print("phi on a common grid, interpolated in ln g (nan = outside that curve's range)\n")
    print("%6s |" % "g" + "".join(" %10s" % n for n in names))
    for g in GRID:
        print("%6.2f |" % g + "".join(" %10.3f" % at(curves[n], g) for n in names))

    pairs = [("Fig3 heat", "Fig8a", "published vs published, N = 160"),
             ("Fig3 cool", "Fig8a", "published vs published, N = 160"),
             ("Fig3 heat", "ours N160", "published vs our tempering, N = 160"),
             ("Fig3 cool", "ours N160", "published vs our tempering, N = 160"),
             ("P196 heat", "E196 melt", "his protocol vs equilibrium, ours, N = 196"),
             ("P196 cool", "E196 melt", "his protocol vs equilibrium, ours, N = 196")]
    print("\nThe gaps: peak, where, and the range of g over which the gap exceeds %.2f\n" % ABOVE)
    for a, b, what in pairs:
        s = gap_summary(curves[a], curves[b])
        label = "%s - %s" % (a, b)
        if s is None:
            print("%-24s %-44s no overlap" % (label, what))
            continue
        rng = "g in [%.2f, %.2f]" % s["above"] if s["above"] else "nowhere"
        print("%-24s %-44s peak %+.3f at g = %.2f; above %.2f for %s"
              % (label, what, s["peak"], s["at"], ABOVE, rng))

    print("\nWhere each curve crosses phi = 0.6 (the middle of the transition)\n")
    for n in names:
        print("   %-12s g = %.2f" % (n, crossing(curves[n])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
