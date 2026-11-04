"""The four pre-registered criteria of PREREGISTRATION T12. Usage:

    python scripts/analyse_t12_front_law.py front_law_lam125

Reads `results/<name>.csv`, written by `scripts/run_front_law.py`, and issues the verdict by the
rules written before the runs. One number is fitted in the whole model, the cost of a front; the gap
per point is exact and is never fitted.

    H = H(tube) - Delta * converted + 2 sigma ,   Delta = 4(lambda - 1)

Criterion 1  linearity: the fitted slope is within 2 % of -Delta, at every size and coupling.
Criterion 2  the front cost does not change with size: sigma shows no trend with N beyond 2 s.e.
Criterion 3  the front advances locally: points converted per sweep shows no trend with N beyond 2 s.e.
Criterion 4  both acceptance rules give the same Delta and sigma.

Only the stretch between 10 % and 90 % converted is used, as pre-registered, so that nucleation at
one end and the last few defects at the other cannot flatter the fit. Runs that never nucleate are
reported and excluded; runs whose sheet arrives in more than one patch are reported separately,
because the coarse model assumes one converted region.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

LOW, HIGH = 0.10, 0.90


def line_fit(x, y):
    """(slope, intercept, standard error of the slope) by least squares."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 3 or np.ptp(x) == 0:
        return np.nan, np.nan, np.nan
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    dof = max(n - 2, 1)
    se = np.sqrt((resid @ resid) / dof / ((x - x.mean()) @ (x - x.mean())))
    return slope, intercept, se


def trend(xs, ys):
    """(slope, standard error) of ys against xs: a trend is 'absent' when |slope| < 2 se."""
    slope, _, se = line_fit(xs, ys)
    return slope, se


def no_trend(xs, ys):
    """(absent, slope, se). A trend counts as present only if it is BOTH big enough to matter and
    distinguishable from noise: the value must move by more than a tenth of a per cent of itself
    across the sizes run, AND the slope must exceed two standard errors. Either test alone is
    wrong. Size alone would call any tidy systematic a trend; significance alone fails on data with
    almost no scatter, where the standard error shrinks faster than the slope and float-level
    residue tests as significant -- which is exactly what the known-answer test on perfect data
    threw up here.

    **Recorded because this project is strict about it:** the rule was written as significance
    alone, and was repaired on synthetic data BEFORE any real run was analysed. Nothing in
    `results/front_law_lam125.csv` had been looked at. Twice before, a verdict rule failed on a case
    it should have passed and had to be repaired after the fact (O11, O17); this one was caught the
    way it should be, by a test whose answer was known in advance."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    slope, se = trend(xs, ys)
    if not np.isfinite(slope):
        return False, slope, se
    change = abs(slope) * (xs.max() - xs.min())           # how much it moves across the sizes run
    scale = max(abs(ys.mean()), 1e-12)
    if change < 1e-3 * scale:                             # a tenth of a per cent is not a trend
        return True, slope, se if np.isfinite(se) else 0.0
    if not np.isfinite(se) or se == 0:
        return False, slope, 0.0
    return abs(slope) < 2 * se, slope, se


def per_run(rows, lam):
    """One row per replica: fitted slope, front cost, points converted per sweep."""
    out = []
    by = defaultdict(list)
    for r in rows:
        by[(r["rule"], int(r["n"]), float(r["g"]), int(r["replica"]))].append(r)
    for key, series in sorted(by.items()):
        rule, n, g, rep = key
        series.sort(key=lambda r: float(r["sweep"]))
        frac = np.array([float(r["fraction"]) for r in series])
        if frac.max() < HIGH:
            out.append(dict(rule=rule, n=n, g=g, replica=rep, nucleated=False))
            continue
        take = (frac >= LOW) & (frac <= HIGH)
        if take.sum() < 3:
            out.append(dict(rule=rule, n=n, g=g, replica=rep, nucleated=False))
            continue
        converted = np.array([float(r["converted"]) for r in series])[take]
        h = np.array([float(r["h"]) for r in series])[take]
        sweeps = np.array([float(r["sweep"]) for r in series])[take]
        patches = np.array([float(r["patches"]) for r in series])[take]
        slope, intercept, _ = line_fit(converted, h)
        h_tube = 4 * (lam - 1) * n
        rate, _, _ = line_fit(sweeps, converted)          # points converted per sweep
        out.append(dict(rule=rule, n=n, g=g, replica=rep, nucleated=True,
                        slope=slope, exact=-4 * (lam - 1),
                        sigma=(intercept - h_tube) / 2, rate=rate,
                        one_patch=bool((patches <= 1).mean() > 0.5)))
    return out


def verdict(runs, lam):
    good = [r for r in runs if r.get("nucleated")]
    lines = []
    exact = -4 * (lam - 1)
    if not good:
        return ["NO VERDICT: nothing nucleated"], False

    slopes = np.array([r["slope"] for r in good])
    worst = np.max(np.abs(slopes - exact) / abs(exact))
    c1 = worst <= 0.02
    lines.append("1 linearity      : slope %.4f to %.4f against the exact %.3f; worst %.2f %% -> %s"
                 % (slopes.min(), slopes.max(), exact, 100 * worst, "PASS" if c1 else "FAIL"))

    sig = np.array([r["sigma"] for r in good])
    ns = np.array([r["n"] for r in good], float)
    c2, s_slope, s_se = no_trend(ns, sig)
    lines.append("2 front cost     : sigma %.1f +- %.1f; trend with N %.4f +- %.4f -> %s"
                 % (sig.mean(), sig.std(ddof=1) if len(sig) > 1 else 0, s_slope, s_se,
                    "PASS (no trend)" if c2 else "FAIL (moves with size)"))

    rates = np.array([r["rate"] for r in good])
    c3, r_slope, r_se = no_trend(ns, rates)
    lines.append("3 local advance  : %.3f to %.3f points a sweep; trend with N %.5f +- %.5f -> %s"
                 % (rates.min(), rates.max(), r_slope, r_se,
                    "PASS (flat in N)" if c3 else "FAIL (depends on size)"))

    by_rule = defaultdict(list)
    for r in good:
        by_rule[r["rule"]].append(r)
    c4 = True
    if len(by_rule) < 2:
        c4 = False
        lines.append("4 both rules     : only one acceptance rule in the file -> NOT TESTED")
    else:
        parts = []
        for rule, rs in sorted(by_rule.items()):
            sl = np.mean([r["slope"] for r in rs])
            sg = np.mean([r["sigma"] for r in rs])
            parts.append("%s slope %.3f sigma %.1f" % (rule, sl, sg))
            if abs(sl - exact) / abs(exact) > 0.02:
                c4 = False
        sigs = [np.mean([r["sigma"] for r in rs]) for rs in by_rule.values()]
        spread = max(sigs) - min(sigs)
        pooled = np.std(sig, ddof=1) if len(sig) > 1 else 0
        if pooled > 0 and spread > 2 * pooled:
            c4 = False
        lines.append("4 both rules     : %s; sigma differs by %.1f (scatter %.1f) -> %s"
                     % ("; ".join(parts), spread, pooled, "PASS" if c4 else "FAIL"))

    passed = c1 and c2 and c3 and c4
    return lines, passed


def main(name, results="results"):
    rows = list(csv.DictReader(open(Path(results) / f"{name}.csv")))
    lam = 1.25
    runs = per_run(rows, lam)
    dead = [r for r in runs if not r.get("nucleated")]
    many = [r for r in runs if r.get("nucleated") and not r.get("one_patch")]
    print("%d replicas, %d nucleated, %d never did, %d converted in more than one patch"
          % (len(runs), len(runs) - len(dead), len(dead), len(many)))
    lines, passed = verdict([r for r in runs if r.get("nucleated")], lam)
    for line in lines:
        print("  " + line)
    print("\nVERDICT: %s" % ("THE COARSE LAW GOVERNS" if passed else "NOT ESTABLISHED"))
    return passed


if __name__ == "__main__":
    main(*sys.argv[1:])
