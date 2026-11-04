"""Gate B''' (PREREGISTRATION, 23 October 2026): is [T25] Fig. 3 the Hamiltonian as printed (our lambda = 1/2)? Usage:

    python scripts/analyse_gateb3.py [results_dir]

Reads results/gateb3_n160_lam{025,050,075}_vs_t25fig3.csv and the lambda = 1 curve results/cqg_n160_lam1_nocap_vs_t25fig3.csv,
and docs/published/T25_fig3_digitised.csv. For each lambda: the read couplings (legs' replica means within 0.03, tau_int
below n_meas / 20 = 200), the largest and rms difference from Fig. 3 (each series interpolated linearly in ln g; the
difference taken against the mean of Fig. 3's two series), the coupling at which the curve crosses phi = 0.6 on each leg,
and the verdict words of the registration.
"""
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

LEGS_AGREE = 0.03
TAU_MAX = 4000 / 20
RUNS = {0.25: "gateb3_n160_lam025_vs_t25fig3", 0.5: "gateb3_n160_lam050_vs_t25fig3",
        0.75: "gateb3_n160_lam075_vs_t25fig3", 1.0: "cqg_n160_lam1_nocap_vs_t25fig3"}


def load_fig3(root):
    series = defaultdict(list)
    with open(Path(root) / "docs" / "published" / "T25_fig3_digitised.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            series[r["series"]].append((math.log(float(r["g"])), float(r["S_over_N"])))
    return {k: sorted(v) for k, v in series.items()}


def interp(points, x):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    if x < min(xs) or x > max(xs):
        return float("nan")
    return float(np.interp(x, xs, ys))


def crossing(curve, level=0.6):
    """g at which a curve (sorted by g ascending; (g, phi) with phi falling as g rises) crosses `level`, in ln g."""
    pts = sorted(curve)
    for (g1, p1), (g2, p2) in zip(pts, pts[1:]):
        if (p1 - level) * (p2 - level) <= 0 and p1 != p2:
            t = (level - p1) / (p2 - p1)
            return math.exp(math.log(g1) + t * (math.log(g2) - math.log(g1)))
    return float("nan")


def read_run(path):
    by = defaultdict(lambda: defaultdict(list))        # leg -> g -> [(phi, tau)]
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            by[r["leg"]][float(r["g"])].append((float(r["phi"]), float(r["tau_int"])))
    gs = sorted(set(by["cool"]) & set(by["heat"]))
    out = {}
    for g in gs:
        cool = np.mean([p for p, _ in by["cool"][g]]); heat = np.mean([p for p, _ in by["heat"][g]])
        tau = max(max(t for _, t in by["cool"][g]), max(t for _, t in by["heat"][g]))
        out[g] = dict(cool=cool, heat=heat, tau=tau, read=(abs(cool - heat) < LEGS_AGREE and tau < TAU_MAX))
    return out


def main(results_dir="results"):
    root = Path(__file__).resolve().parents[1]
    fig3 = load_fig3(root)
    fig3_mean = sorted({x for s in fig3.values() for x, _ in s})
    fig3_cross = {k: crossing([(math.exp(x), y) for x, y in v]) for k, v in fig3.items()}
    print("Fig. 3 crosses phi = 0.6 at g = %s" % ", ".join("%.2f (%s)" % (v, k) for k, v in fig3_cross.items()))
    verdict_inputs = {}
    for lam, name in sorted(RUNS.items()):
        path = Path(results_dir) / (name + ".csv")
        if not path.exists():
            print("lambda = %.2f: %s missing" % (lam, path)); continue
        run = read_run(path)
        diffs = []
        for g, v in run.items():
            if not v["read"]:
                continue
            x = math.log(g)
            f = np.nanmean([interp(s, x) for s in fig3.values()])
            if not math.isnan(f):
                diffs.append((g, (v["cool"] + v["heat"]) / 2 - f))
        n_read_mid = sum(1 for g, v in run.items() if v["read"] and 2 <= g <= 10)
        largest = max(abs(d) for _, d in diffs) if diffs else float("nan")
        rms = math.sqrt(np.mean([d * d for _, d in diffs])) if diffs else float("nan")
        cross_cool = crossing([(g, v["cool"]) for g, v in run.items()])
        cross_heat = crossing([(g, v["heat"]) for g, v in run.items()])
        print("lambda = %.2f: read %d of %d couplings (%d in 2..10); largest |diff| from Fig. 3 %.3f, rms %.3f; crosses 0.6 at g = %.2f (cool), %.2f (heat); cold end cool %.3f heat %.3f"
              % (lam, sum(v["read"] for v in run.values()), len(run), n_read_mid, largest, rms, cross_cool, cross_heat,
                 run[min(run)]["cool"], run[min(run)]["heat"]))
        verdict_inputs[lam] = dict(largest=largest, n_mid=n_read_mid, cross_heat=cross_heat)
    ref = fig3_cross.get("heat_from_torus_red", list(fig3_cross.values())[0])
    words = []
    for lam, v in verdict_inputs.items():
        if v["n_mid"] < 8:
            words.append((lam, "NOT READ")); continue
        if v["largest"] < 0.05 and abs(math.log(v["cross_heat"]) - math.log(ref)) < 0.1:
            words.append((lam, "REPRODUCED"))
    if any(w == "REPRODUCED" for _, w in words):
        verdict = "REPRODUCED at lambda = " + ", ".join("%.2f" % l for l, w in words if w == "REPRODUCED")
    elif 0.5 in verdict_inputs and verdict_inputs[0.5]["n_mid"] >= 8 and verdict_inputs[0.5]["largest"] < 0.41 / 2:
        verdict = "CLOSER (lambda = 1/2 largest difference %.3f against 0.41 at lambda = 1)" % verdict_inputs[0.5]["largest"]
    elif 0.5 in verdict_inputs and verdict_inputs[0.5]["n_mid"] < 8:
        verdict = "NOT READ at lambda = 1/2"
    else:
        verdict = "NOT REPRODUCED"
    print("Verdict: %s" % verdict)
    print("Ours: CLOSER at least. The owner's: owed before this is read.")


if __name__ == "__main__":
    main(*sys.argv[1:2])
