"""Compare a sweep result with digitised published points. Usage:

    python scripts/compare_with_published.py results/<name>.csv docs/published/<points>.csv [interpolate [g_min]]

For every published point it prints our phi on the matching leg: the mean over
replicas, the standard error between replicas, and the difference. Published
series names start with "cool" or "heat".

Without "interpolate" our value is taken at the nearest simulated coupling, which
is right when the run used the published couplings. With it, our curve is
interpolated linearly in ln g between the two couplings either side, and
published points outside our range are skipped; use this when the grids differ.
g_min drops published points at couplings below it. A comparison means something
only where our chain is in equilibrium, so first find, in the result file, the
coupling below which the two legs stop agreeing or tau_int stops being small,
and pass that.
Prints only; writes nothing.
"""
import csv
import math
import statistics
import sys
from collections import defaultdict


def main(result_path, published_path, interpolate=False, g_min=0.0):
    ours = defaultdict(list)                                   # (leg, g) -> phi of each replica
    with open(result_path, newline="") as fh:
        for r in csv.DictReader(fh):
            ours[(r["leg"], float(r["g"]))].append(float(r["phi"]))
    with open(published_path, newline="") as fh:
        published = list(csv.DictReader(fh))

    print(f"{'leg':4s} {'g publ.':>8s} {'g ours':>8s} {'publ.':>6s} {'ours':>6s} {'+/-':>6s} {'diff':>7s}")
    diffs = []
    for p in sorted(published, key=lambda p: (p["series"][:4], -float(p["g"]))):
        leg, g_pub, phi_pub = p["series"][:4], float(p["g"]), float(p["S_over_N"])
        if g_pub < g_min:
            continue
        gs = sorted(g for (l, g) in ours if l == leg)
        flag = ""
        if interpolate:
            if not gs[0] <= g_pub <= gs[-1]:
                continue
            hi = next(g for g in gs if g >= g_pub)
            lo = max(g for g in gs if g <= g_pub)
            t = 0.0 if hi == lo else math.log(g_pub / lo) / math.log(hi / lo)
            mean = (1 - t) * statistics.mean(ours[(leg, lo)]) + t * statistics.mean(ours[(leg, hi)])
            g, err = g_pub, float("nan")
        else:
            g = min(gs, key=lambda g: abs(math.log(g / g_pub)))
            phis = ours[(leg, g)]
            mean = statistics.mean(phis)
            err = statistics.stdev(phis) / math.sqrt(len(phis)) if len(phis) > 1 else float("nan")
            if abs(math.log10(g / g_pub)) >= 0.01:
                flag = "   (nearest coupling is not close)"
        diffs.append(mean - phi_pub)
        print(f"{leg:4s} {g_pub:8.3f} {g:8.3f} {phi_pub:6.3f} {mean:6.3f} {err:6.3f} {mean - phi_pub:+7.3f}{flag}")
    print(f"largest |difference| = {max(abs(d) for d in diffs):.3f}")
    print(f"rms difference = {math.sqrt(statistics.mean(d * d for d in diffs)):.3f} over {len(diffs)} points")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], len(sys.argv) > 3 and sys.argv[3] == "interpolate",
         float(sys.argv[4]) if len(sys.argv) > 4 else 0.0)
