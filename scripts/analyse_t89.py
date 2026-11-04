"""T89, read under the registered rules (PREREGISTRATION.md section T89). Usage:

    python scripts/analyse_t89.py [results_dir]

The seven sizes already on the record (replica-mean phi(g) at lambda = 1, no cap) and the held-out N = 1156 run
(`t89_drift_n1156_lam1`, both legs). For each level phi* in (0.5, 0.65): the crossing g(phi*) of the replica-mean
curve, interpolated linearly in ln g between the two neighbouring couplings; the fit 1/g = a + b ln N over the seven
smaller sizes (equal weights); the prediction for N = 1156 and its registered range; the held-out crossing from the
pooled legs, with the gate (legs within 0.03 in phi at the bracketing couplings; tau_int below n_meas / 20 on both).
Verdict words: LOGARITHMIC DRIFT (both scorable crossings inside their ranges), FASTER (both below), SLOWER (both
above), MIXED, UNSCORED (a gate fails or a level is not bracketed).
"""
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

LEVELS = (0.5, 0.65)
RANGES = {0.5: (3.43, 3.30, 3.56), 0.65: (2.79, 2.68, 2.90)}          # registered: prediction, low, high
SMALL = {36: ["t16_temper_n36_rep0", "t16_temper_n36_rep1", "t16_temper_n36_rep2", "t16_temper_n36_rep3"],
         64: ["t6c_lam1_n64"], 100: ["t6c_lam1_n100"], 160: ["cqg_n160_lam1_nocap_tempering"],
         196: ["t13_temper_n196_melt"], 484: ["t13_temper_n484_melt"], 676: ["t13_temper_n676_melt"]}
HELD_OUT = "t89_drift_n1156_lam1"
N_MEAS = 20000
LEG_TOL = 0.03


def load(results_dir, names):
    rows = []
    for name in names:
        with open(Path(results_dir) / (name + ".csv"), newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def mean_curve(rows, leg=None):
    """Replica-mean phi at each coupling, as a sorted list of (g, phi)."""
    acc = defaultdict(list)
    for r in rows:
        if leg is not None and r.get("leg") != leg:
            continue
        acc[float(r["g"])].append(float(r["phi"]))
    return sorted((g, sum(v) / len(v)) for g, v in acc.items())


def crossing(curve, level):
    """The coupling where the curve crosses `level`, linear in ln g between neighbours; None if not bracketed."""
    for (g1, p1), (g2, p2) in zip(curve, curve[1:]):
        if (p1 - level) * (p2 - level) < 0:
            t = (level - p1) / (p2 - p1)
            return math.exp(math.log(g1) + t * (math.log(g2) - math.log(g1))), (g1, g2)
    return None, None


def fit(points):
    """Least squares 1/g = a + b ln N over [(N, g)]. Returns a, b, rms."""
    xs = [math.log(n) for n, _ in points]
    ys = [1.0 / g for _, g in points]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    a = my - b * mx
    rms = math.sqrt(sum((y - a - b * x) ** 2 for x, y in zip(xs, ys)) / n)
    return a, b, rms


def gate(rows, bracket):
    """Legs agree within LEG_TOL at the bracketing couplings and tau_int is below N_MEAS / 20 on both."""
    if bracket is None:
        return False, "not bracketed"
    for g in bracket:
        by_leg = defaultdict(list)
        taus = []
        for r in rows:
            if abs(float(r["g"]) - g) < 1e-9:
                by_leg[r["leg"]].append(float(r["phi"]))
                taus.append(float(r["tau_int"]))
        if len(by_leg) < 2:
            return False, "one leg only at g = %.3f" % g
        means = {leg: sum(v) / len(v) for leg, v in by_leg.items()}
        if abs(means["cool"] - means["heat"]) > LEG_TOL:
            return False, "legs differ by %.3f at g = %.3f" % (abs(means["cool"] - means["heat"]), g)
        if any(math.isnan(t) or t >= N_MEAS / 20 for t in taus):
            return False, "tau_int %.0f at g = %.3f" % (max(taus), g)
    return True, "ok"


def word(scores):
    """scores: list of 'inside' / 'below' / 'above' / 'unscored' per level."""
    if any(s == "unscored" for s in scores):
        return "UNSCORED"
    if all(s == "inside" for s in scores):
        return "LOGARITHMIC DRIFT"
    if all(s == "below" for s in scores):
        return "FASTER"
    if all(s == "above" for s in scores):
        return "SLOWER"
    return "MIXED"


def analyse(results_dir="results"):
    out = {}
    scores = []
    for level in LEVELS:
        pts = []
        for n, names in SMALL.items():
            g, _ = crossing(mean_curve(load(results_dir, names)), level)
            if g is not None:
                pts.append((n, g))
        a, b, rms = fit(pts)
        pred = 1.0 / (a + b * math.log(1156))
        held = load(results_dir, [HELD_OUT])
        g_held, bracket = crossing(mean_curve(held), level)
        ok, why = gate(held, bracket)
        reg_pred, lo, hi = RANGES[level]
        if g_held is None or not ok:
            score = "unscored"
        elif lo <= g_held <= hi:
            score = "inside"
        else:
            score = "below" if g_held < lo else "above"
        scores.append(score)
        a2, b2, _ = fit(pts + ([(1156, g_held)] if g_held else []))
        out[level] = dict(points=pts, a=a, b=b, rms=rms, pred=pred, registered=(reg_pred, lo, hi), held=g_held,
                          bracket=bracket, gate=(ok, why), score=score, slope16_with_held=16 * b2)
    out["verdict"] = word(scores)
    return out


def main(results_dir="results"):
    res = analyse(results_dir)
    for level in LEVELS:
        r = res[level]
        print("phi* = %.2f" % level)
        for n, g in r["points"]:
            print("   N = %5d  g = %.3f  1/g = %.4f" % (n, g, 1 / g))
        print("   fit: a = %.4f  b = %.4f  rms(1/g) = %.4f  16 b = %.3f  -> predicted g(1156) = %.3f (registered %.2f, range %.2f to %.2f)"
              % (r["a"], r["b"], r["rms"], 16 * r["b"], r["pred"], *r["registered"]))
        print("   held out N = 1156: g = %s  bracket = %s  gate = %s (%s)  score = %s  16 b with 1156 = %.3f"
              % ("%.3f" % r["held"] if r["held"] else "none", r["bracket"], r["gate"][0], r["gate"][1], r["score"], r["slope16_with_held"]))
    print("VERDICT:", res["verdict"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
