"""Apply the T6 pre-registered criteria to a density-of-states run. Usage:

    python scripts/analyse_t6.py results/t6_pilot_n36.csv [more.csv ...]

Prints to the screen and writes nothing. It implements PREREGISTRATION.md section T6 and makes
no choices of its own; every threshold below is quoted from it.

    FIRST ORDER    latent heat extrapolates non-zero at 3 sigma, AND the barrier slope is
                   positive at 3 sigma, AND the Binder minimum sits below 2/3 and is not
                   rising towards it. All three.
    NO EVIDENCE    latent heat extrapolates to zero within 2 sigma, AND the barrier does not
                   grow, AND the Binder minimum approaches 2/3. Deliberately NOT called
                   "continuous": at a weak first-order transition these analyses can return
                   continuous-looking answers.
    INCONCLUSIVE   anything else, including the three disagreeing. A real outcome.

Gate 3 of the pre-registration is applied first: seeds must agree on the barrier to 20 % and on
the latent heat to 10 %, or the size is excluded and the exclusion reported. Gate 4 asks for at
least 20 round trips.
"""
import csv
import sys
from collections import defaultdict

import numpy as np

GATE_TRIPS = 20
GATE_BARRIER = 0.20
GATE_LATENT = 0.10


def load(paths):
    rows = []
    for path in paths:
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                rows.append({k: (float(v) if v not in ("", "nan") else float("nan"))
                             if k not in ("N", "side", "seed", "bins", "moves", "discovery_round",
                                          "round_trips", "end_pieces")
                             else (int(float(v)) if v else -1) for k, v in r.items()})
    return rows


def line_fit(x, y, sig):
    """Weighted straight-line fit. Returns (slope, slope error, intercept, intercept error)."""
    x, y, sig = np.asarray(x, float), np.asarray(y, float), np.asarray(sig, float)
    sig = np.where((sig > 0) & np.isfinite(sig), sig, np.nanmax(sig[sig > 0]) if (sig > 0).any() else 1.0)
    w = 1.0 / sig ** 2
    s, sx, sy = w.sum(), (w * x).sum(), (w * y).sum()
    sxx, sxy = (w * x * x).sum(), (w * x * y).sum()
    d = s * sxx - sx * sx
    if abs(d) < 1e-300:
        return (float("nan"),) * 4
    slope = (s * sxy - sx * sy) / d
    inter = (sxx * sy - sx * sxy) / d
    return slope, np.sqrt(s / d), inter, np.sqrt(sxx / d)


def summarise(rows, lam):
    """Per size: mean and spread over seeds, and whether gate 3 and gate 4 passed."""
    by_size = defaultdict(list)
    for r in rows:
        if abs(r["lam"] - lam) < 1e-12:
            by_size[r["N"]].append(r)
    out = []
    for n in sorted(by_size):
        rs = by_size[n]
        take = lambda k: np.array([r[k] for r in rs], float)
        lat, bar, bnd = take("latent_heat"), take("barrier"), take("binder_min")
        trips = min(r["round_trips"] for r in rs)
        m_lat, m_bar = np.nanmean(lat), np.nanmean(bar)
        e_lat = np.nanstd(lat, ddof=1) / np.sqrt(len(rs)) if len(rs) > 1 else float("nan")
        e_bar = np.nanstd(bar, ddof=1) / np.sqrt(len(rs)) if len(rs) > 1 else float("nan")
        rel_lat = abs(np.nanmax(lat) - np.nanmin(lat)) / abs(m_lat) if m_lat else float("inf")
        rel_bar = abs(np.nanmax(bar) - np.nanmin(bar)) / abs(m_bar) if m_bar else float("inf")
        why = []
        if len(rs) < 2:
            why.append("only one seed")
        if not np.isfinite(m_lat):
            why.append("no two humps found at any coupling")
        else:
            if rel_bar > GATE_BARRIER:
                why.append("seeds differ on the barrier by %.0f%%" % (100 * rel_bar))
            if rel_lat > GATE_LATENT:
                why.append("seeds differ on the latent heat by %.0f%%" % (100 * rel_lat))
        if trips < GATE_TRIPS:
            why.append("only %d round trips" % trips)
        out.append(dict(N=n, L=np.sqrt(n), seeds=len(rs), trips=trips,
                        latent=m_lat, latent_err=e_lat, barrier=m_bar, barrier_err=e_bar,
                        binder=np.nanmean(bnd), g_c=np.nanmean(take("g_c")),
                        largest=np.nanmean(take("end_largest")), passes=not why, why="; ".join(why)))
    return out


def verdict(sizes):
    """The three pre-registered criteria, on the sizes that passed gate 3."""
    good = [s for s in sizes if s["passes"]]
    if len(good) < 3:
        return "INCONCLUSIVE", ["fewer than three sizes passed the gates (%d)" % len(good)], {}

    L = [s["L"] for s in good]
    inv = [1.0 / s["L"] for s in good]
    lat, lat_e = [s["latent"] for s in good], [s["latent_err"] for s in good]
    bar, bar_e = [s["barrier"] for s in good], [s["barrier_err"] for s in good]

    _, _, lat0, lat0_e = line_fit(inv, lat, lat_e)          # latent heat extrapolated to 1/L -> 0
    slope, slope_e, _, _ = line_fit(L, bar, bar_e)          # barrier against L
    binder_trend = good[-1]["binder"] - good[0]["binder"]
    binder_last = good[-1]["binder"]

    facts = dict(latent_limit=lat0, latent_limit_err=lat0_e, barrier_slope=slope,
                 barrier_slope_err=slope_e, binder_last=binder_last, binder_trend=binder_trend)

    lat_nonzero = np.isfinite(lat0_e) and lat0 > 3 * lat0_e
    lat_zero = np.isfinite(lat0_e) and abs(lat0) < 2 * lat0_e
    bar_grows = np.isfinite(slope_e) and slope > 3 * slope_e
    bar_flat = np.isfinite(slope_e) and abs(slope) < 2 * slope_e
    bnd_low = binder_last < 2.0 / 3.0 - 1e-6 and binder_trend <= 0
    bnd_two_thirds = abs(binder_last - 2.0 / 3.0) < 1e-3

    notes = ["latent heat at infinite size: %+.5f +/- %.5f  -> %s"
             % (lat0, lat0_e, "non-zero at 3 sigma" if lat_nonzero
                else "zero within 2 sigma" if lat_zero else "neither"),
             "barrier slope against L:    %+.5f +/- %.5f  -> %s"
             % (slope, slope_e, "grows at 3 sigma" if bar_grows
                else "no growth" if bar_flat else "neither"),
             "Binder minimum at largest size: %.6f (2/3 = %.6f), trend %+.6f  -> %s"
             % (binder_last, 2 / 3, binder_trend,
                "below 2/3 and not rising" if bnd_low
                else "at 2/3" if bnd_two_thirds else "neither")]

    if lat_nonzero and bar_grows and bnd_low:
        return "FIRST ORDER", notes, facts
    if lat_zero and bar_flat and not bnd_low:
        return "NO EVIDENCE OF FIRST ORDER AT THESE SIZES", notes, facts
    return "INCONCLUSIVE", notes, facts


def main(paths):
    rows = load(paths)
    lams = sorted({r["lam"] for r in rows})
    print("T6 analysis, against PREREGISTRATION.md section T6 (revised 2026-09-21)")
    print("Files: %s\n" % ", ".join(paths))
    for lam in lams:
        sizes = summarise(rows, lam)
        print("=" * 100)
        print("PENALTY lambda = %.2f" % lam)
        print("%-6s %-6s %-7s %-12s %-12s %-10s %-9s %-8s %s"
              % ("N", "seeds", "trips", "latent heat", "barrier", "Binder", "g_c", "largest", "gate"))
        for s in sizes:
            print("%-6d %-6d %-7d %-12.5f %-12.4f %-10.6f %-9.3f %-8.3f %s"
                  % (s["N"], s["seeds"], s["trips"], s["latent"], s["barrier"],
                     s["binder"], s["g_c"], s["largest"],
                     "pass" if s["passes"] else "EXCLUDED: " + s["why"]))
        call, notes, _ = verdict(sizes)
        print("\n  " + "\n  ".join(notes))
        print("\n  VERDICT: %s\n" % call)
    print("=" * 100)
    print("""Reminders from the pre-registration, so a reader does not have to go and find them.
  * "No evidence of first order" is NOT "continuous". At a weak first-order transition these
    analyses can return continuous-looking answers with plausible exponents.
  * If the latent heat is consistent with zero, the result is the measured UPPER BOUND on it,
    and the argument that closes the escape: a transition too weak to detect here is also too
    weak to do claim 4's job, which needs a lump big enough to become a universe.
  * A size that failed a gate is excluded and the exclusion is printed above. Gates are not
    relaxed afterwards to admit a size that would change the verdict.""")


if __name__ == "__main__":
    main(sys.argv[1:] or ["results/t6_pilot_n36.csv"])
