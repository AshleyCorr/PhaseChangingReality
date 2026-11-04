"""The pre-registered T16 reading: the correlation length of [KTB19] Fig. 9a. Usage:

    python scripts/analyse_t16.py

Implements PREREGISTRATION.md section T16, "Rules for reading, fixed now", on the committed files
results/t16_{temper,seq}_n<N>_rep<k>.csv. Pure functions of parsed rows, so that
tests/test_t16.py can check each rule on rows whose answer is known.

One detail the section does not spell out, stated here rather than decided silently: where
protocol P is read (couplings at which the cooling and heating legs agree within 0.03 in phi),
both legs are equilibrium samples by that same rule, so both legs' replicas are pooled there.

Standard errors: each row carries the mean and standard error of xi / diameter over its own
snapshots; a coupling's value is the mean over the rows pooled there, and its error the root sum
of the rows' squared errors divided by their number.
"""
import csv
import glob
import math
import sys
from collections import defaultdict

TRIPS = 5                 # T16 E gate: every replica at least 5 round trips
SWAP = (0.15, 0.6)        # T16 E gate: every swap rate inside this range
LEGS_AGREE = 0.03         # T16: P is read where the legs' replica-mean phi differ by less than this


def num(x):
    return float(x) if x not in ("", None) else math.nan


def e_gate(rows):
    """(passes, round trips per replica, lowest swap rate, highest swap rate) for one size of E."""
    trips = {}
    swaps = []
    for r in rows:
        trips[r["replica"]] = int(r["round_trips"])
        if r["swap_rate"] != "":
            swaps.append(float(r["swap_rate"]))
    ok = min(trips.values()) >= TRIPS and SWAP[0] <= min(swaps) and max(swaps) <= SWAP[1]
    return ok, [trips[k] for k in sorted(trips)], min(swaps), max(swaps)


def pooled(rows):
    """Per coupling: phi, chi, xi / diameter with its error, and the share of snapshots skipped."""
    by_g = defaultdict(list)
    for r in rows:
        by_g[float(r["g"])].append(r)
    out = {}
    for g, rs in by_g.items():
        xi = [(num(r["xi_over_diam"]), num(r["xi_over_diam_err"])) for r in rs]
        xi = [(m, e) for m, e in xi if not math.isnan(m)]
        snaps = sum(int(r["snapshots"]) for r in rs)
        skipped = sum(int(r["snap_no_fluct"]) + int(r["snap_no_xi"]) for r in rs)
        out[g] = dict(
            phi=sum(num(r["phi"]) for r in rs) / len(rs),
            chi=sum(num(r["chi"]) for r in rs) / len(rs),
            xi=(sum(m for m, _ in xi) / len(xi)) if xi else math.nan,
            xi_err=(math.sqrt(sum((0.0 if math.isnan(e) else e) ** 2 for _, e in xi)) / len(xi)) if xi else math.nan,
            skipped=(skipped / snaps) if snaps else 1.0)
    return out


def read_couplings_p(rows):
    """The couplings at which P is read: cooling and heating replica-mean phi within LEGS_AGREE."""
    cool = pooled([r for r in rows if r["leg"] == "cool"])
    heat = pooled([r for r in rows if r["leg"] == "heat"])
    return sorted((g for g in cool if g in heat and abs(cool[g]["phi"] - heat[g]["phi"]) < LEGS_AGREE),
                  reverse=True)


def peak(curve, gs, key):
    """(g, value, error, at_edge) of the largest `key` among the couplings `gs` (hottest first)."""
    usable = [g for g in gs if not math.isnan(curve[g][key])]
    if not usable:
        return None
    best = max(usable, key=lambda g: curve[g][key])
    at_edge = best in (usable[0], usable[-1])
    return best, curve[best][key], curve[best].get(key + "_err", math.nan), at_edge


def tendency(peaks):
    """The pre-registered reading of xi / diameter across the read sizes of one protocol.

    peaks: {N: (g, value, error, at_edge, skipped share at that g)}.
    """
    if len(peaks) < 2:
        return "NOT READABLE (fewer than two read sizes)"
    for n, (g, v, e, edge, skipped) in peaks.items():
        if edge:
            return "NOT READABLE (the peak at N = %d is at the edge of the read couplings, g = %.2f)" % (n, g)
        if skipped > 0.5:
            return "NOT READABLE (more than half the snapshots skipped at the peak, N = %d)" % n
    sizes = sorted(peaks)
    heights = [peaks[n][1] for n in sizes]
    rising = all(a < b for a, b in zip(heights, heights[1:]))
    lo, hi = peaks[sizes[0]], peaks[sizes[-1]]
    gap = hi[1] - lo[1]
    sigma = math.sqrt(lo[2] ** 2 + hi[2] ** 2)
    if rising and gap > 2 * sigma:
        return "DIVERGENT TENDENCY"
    return "NONE (peak heights %s; largest minus smallest %.3f against 2 sigma = %.3f)" % (
        ", ".join("%.3f" % h for h in heights), gap, 2 * sigma)


def load(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        with open(f, newline="") as fh:
            rows += list(csv.DictReader(fh))
    return rows


def main():
    out = []
    for proto, sizes in (("temper", (36, 100, 196)), ("seq", (196, 484, 676))):
        xi_peaks = {}
        for n in sizes:
            rows = load("results/t16_%s_n%d_rep*.csv" % (proto, n))
            if proto == "temper":
                ok, trips, lo, hi = e_gate(rows)
                out.append("E N=%d gate: round trips %s, swap rates %.3f to %.3f -> %s"
                           % (n, trips, lo, hi, "passes" if ok else "FAILS, not read"))
                if not ok:
                    continue
                curve = pooled(rows)
                gs = sorted(curve, reverse=True)
            else:
                gs = read_couplings_p(rows)
                curve = pooled([r for r in rows if float(r["g"]) in gs])
                out.append("P N=%d read at %d of 40 couplings (g = %.2f to %.2f where the legs agree; "
                           "not read at %s)" % (n, len(gs), gs[0], gs[-1],
                                                ", ".join("%.2f" % g for g in sorted(
                                                    {float(r["g"]) for r in rows} - set(gs), reverse=True))))
            pc = peak(curve, gs, "chi")
            px = peak(curve, gs, "xi")
            out.append("   chi peak %.3f at g = %.2f%s" % (pc[1], pc[0], " (edge)" if pc[3] else ""))
            out.append("   xi/diam peak %.3f +- %.3f at g = %.2f%s, snapshots skipped there %.0f %%"
                       % (px[1], px[2], px[0], " (edge)" if px[3] else "", 100 * curve[px[0]]["skipped"]))
            xi_peaks[n] = (px[0], px[1], px[2], px[3], curve[px[0]]["skipped"])
        out.append("%s: xi / diameter across read sizes -> %s" % ("E" if proto == "temper" else "P", tendency(xi_peaks)))
    print("\n".join(out))


if __name__ == "__main__":
    sys.exit(main())
