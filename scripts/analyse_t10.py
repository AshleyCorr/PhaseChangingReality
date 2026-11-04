"""The T10 verdict table. Usage:

    python scripts/analyse_t10.py

Reads every results/t10_n*.csv and applies PREREGISTRATION.md section T10 as written:
  (a) the mean number of leftover rings rises with N: positive slope at more than three standard
      errors, and mean(largest N) - mean(smallest N) exceeds the replica scatter at either size;
  (b) rings are additive: excess energy over the perfect sheet = 14 units per ring, checked on the
      replicas whose d = 1 pieces are all rings of four;
  (c) nothing leaves: the conversion fraction changes by less than one ring's worth (14/N) over the
      last 3,000 sweeps in every replica, and the bath ends below g = 1 at every N.
The historical implementation of (c) compares mean store energy, bath_T, with 1;
it does NOT measure g<1. This legacy scoring is retained to reproduce the verdict.
Even the conditional delta=1 canonical proxies range above 1 in the saved data.
"Rings" below are connected d=1 pieces; their geometry requires a wiring check.
Gate 3: every replica converts (f_final >= 0.9); a size at which any replica fails is reported and
excluded, and the verdict is over the sizes that converted.
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

LAM = 1.25                 # T10 is at lambda = 1.25 by pre-registration; the csv does not carry it
RING_UNITS = 14.0          # one ring of the tube: one extra square (-16) and six surplus units (+30)
LAST_SWEEPS = 3000
ADDITIVE_TOL = 0.5


def load(out_dir="results"):
    finals, traces = {}, defaultdict(list)          # (N, rep) -> final row ; (N, rep) -> [(sweep, f)]
    for f in sorted(Path(out_dir).glob("t10_n*.csv")):
        for r in csv.DictReader(open(f, newline="")):
            key = (int(r["N"]), int(r["replica"]))
            traces[key].append((int(r["sweep"]), float(r["f"])))
            if r["final"] == "1":
                finals[key] = r
    return finals, traces


def excess_energy(r):
    n = int(r["N"])
    return 16.0 * n * (1.0 - float(r["phi"])) + 4.0 * LAM * n * float(r["surplus"])


def slope_with_error(xs, means, ses):
    """Weighted least-squares slope of means against xs, and its standard error."""
    x, y, w = np.asarray(xs, float), np.asarray(means, float), 1.0 / np.maximum(np.asarray(ses, float), 1e-9) ** 2
    xm = np.sum(w * x) / np.sum(w)
    sxx = np.sum(w * (x - xm) ** 2)
    slope = np.sum(w * (x - xm) * y) / sxx
    return slope, np.sqrt(1.0 / sxx)


def verdict(finals, traces):
    """Returns (verdict, report lines). Pure function of the parsed rows so that it can be tested."""
    lines = []
    by_n = defaultdict(list)
    for (n, rep), r in finals.items():
        by_n[n].append((rep, r))
    sizes = sorted(by_n)

    # gate 3
    kept = []
    for n in sizes:
        fails = [rep for rep, r in by_n[n] if float(r["f_final"]) < 0.9]
        if fails:
            lines.append("   N=%-4d GATE 3 FAILS: %d of %d replicas did not convert -> size excluded" % (n, len(fails), len(by_n[n])))
        else:
            kept.append(n)

    lines.append("%-5s %-5s %-12s %-8s %-12s %-14s %-9s %-8s" % ("N", "reps", "rings mean", "sd", "max piece", "additive", "mean E", "f drift"))
    means, ses, sds, add_ok, c_ok = [], [], [], True, True
    for n in kept:
        rows = by_n[n]
        rings = np.array([int(r["pieces_d1"]) for _, r in rows], float)
        largest = max(int(r["largest_d1"]) for _, r in rows)
        sep = [(rep, r) for rep, r in rows if int(r["largest_d1"]) <= 4]
        add = [abs(excess_energy(r) - RING_UNITS * int(r["pieces_d1"])) <= ADDITIVE_TOL for _, r in sep]
        add_frac = np.mean(add) if add else float("nan")
        if add and add_frac < 1.0:
            add_ok = False
        bath = np.array([float(r["bath_T"]) for _, r in rows])
        drift = []
        for rep, r in rows:
            tail = [f for s, f in traces[(n, rep)] if s > 30000 - LAST_SWEEPS]
            drift.append((max(tail) - min(tail)) if tail else float("nan"))
        drift = np.array(drift)
        this_c = bool(np.all(drift < RING_UNITS / n)) and bool(np.all(bath < 1.0))
        c_ok = c_ok and this_c
        means.append(rings.mean()); sds.append(rings.std(ddof=1) if len(rings) > 1 else 0.0)
        ses.append(sds[-1] / np.sqrt(len(rings)))
        lines.append("%-5d %-5d %-12.2f %-8.2f %-12d %-14s %-9.2f %-8s"
                     % (n, len(rows), means[-1], sds[-1], largest,
                        ("%d/%d" % (sum(add), len(add))) if add else "-", bath.mean(),
                        "ok" if this_c else "MOVES"))

    a = False
    if len(kept) >= 2:
        slope, err = slope_with_error(kept, means, ses)
        gap = means[-1] - means[0]
        a = slope / err > 3.0 and gap > max(sds[0], sds[-1])
        lines.append("\n  (a) slope of mean rings against N: %.4f +- %.4f per point (%.1f sigma); mean(N=%d) - mean(N=%d) = %.2f against scatter %.2f -> %s"
                     % (slope, err, slope / err, kept[-1], kept[0], gap, max(sds[0], sds[-1]), "holds" if a else "does not hold"))
        one_ring = all(abs(m - 1.0) <= max(s, 1e-9) for m, s in zip(means, sds)) and not (slope / err > 3.0)
    else:
        lines.append("\n  (a) needs at least two sizes that passed gate 3")
        one_ring = False
    lines.append("  (b) additive, 14 units per ring on replicas with separate rings: %s" % ("holds" if add_ok else "FAILS"))
    lines.append("  (c) nothing leaves in the last %d sweeps and mean store energy < 1 (legacy proxy, NOT g < 1): %s" % (LAST_SWEEPS, "holds" if c_ok else "FAILS"))

    if a and c_ok:
        v = "THE LEFTOVER GROWS WITH THE SPACE"
    elif one_ring and c_ok:
        v = "ONE RING, HOWEVER LARGE"
    else:
        v = "INCONCLUSIVE"
    return v, lines


def main(out_dir="results"):
    finals, traces = load(out_dir)
    if not finals:
        print("no T10 results yet"); return
    print("T10, leftover rings in a cold sealed box, lambda = %.2f, C = 2N. PREREGISTRATION.md section T10.\n" % LAM)
    v, lines = verdict(finals, traces)
    print("\n".join(lines))
    print("\n  PRE-REGISTERED VERDICT, T10:  %s" % v)


if __name__ == "__main__":
    main()
