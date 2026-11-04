"""T78, read under the registered rules (PREREGISTRATION.md section T78). Usage:

    python scripts/analyse_t78.py [results_dir]

Per coupling g, pooling the snapshots of all replicas with equal weight per replica:
    rho   = <D>/N                              the deficit per point
    F_X   = -g ln(<n_X>/N)                     free energy per object, X in {ring, patch}
    E_X   = <H n_X>/<n_X> - <H>                energy per object (fluctuation estimator)
    S_X   = (E_X - F_X)/g                      entropy lost per object (negative: the object costs entropy)
    Delta = S_patch - S_ring                   the entropy the curl loses beyond the flat patch
    lambda_eff - 1 = g Delta / 16              four surplus squares at 4 (lambda - 1) each; compared with nothing
Errors: bootstrap over replicas (each replica one draw), 1,000 resamples, seed 78; 2 sigma is the bar.
A cell with fewer than 50 ring sightings or fewer than 50 patch sightings is NOT READ.
Verdict over the read cells: CURL PENALISED, GROWING / CURL PENALISED, FLAT / NO GAP / CURL FAVOURED / MIXED; fewer
than two read cells: NOT READ.
"""
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

MIN_SIGHTINGS = 50
BOOT = 1000


NAMES = ["t78_curl_vs_flat_n196_g3300", "t78_curl_vs_flat_n196_g3433",
         "t78_curl_vs_flat_n196_g3550", "t78_curl_vs_flat_n196_g3697"]     # the registered set, one per coupling


def load(paths):
    cells = defaultdict(lambda: defaultdict(list))         # g -> replica -> rows
    for path in paths:
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                cells[float(r["g"])][int(r["replica"])].append(
                    (int(r["N"]), float(r["H"]), int(r["D"]), int(r["n_ring"]), int(r["n_patch"])))
    return cells


def estimate(reps, g):
    """Pooled estimates from a list of replicas, each a list of (N, H, D, n_ring, n_patch) rows."""
    rows = np.array([row for rep in reps for row in rep], dtype=float)
    n = rows[0, 0]
    h, d, nr, npch = rows[:, 1], rows[:, 2], rows[:, 3], rows[:, 4]
    out = dict(rho=d.mean() / n, H=h.mean(), rings=int(nr.sum()), patches=int(npch.sum()))
    for name, cnt in (("ring", nr), ("patch", npch)):
        m = cnt.mean()
        if m <= 0:
            out[name] = None
            continue
        f = -g * math.log(m / n)
        e = (h * cnt).mean() / m - h.mean()
        out[name] = dict(F=f, E=e, S=(e - f) / g, mean=m)
    if out["ring"] and out["patch"]:
        out["delta"] = out["patch"]["S"] - out["ring"]["S"]
        out["lam_eff_minus_1"] = g * out["delta"] / 16.0
    else:
        out["delta"] = None
    return out


def bootstrap(reps, g, rng):
    vals = []
    k = len(reps)
    for _ in range(BOOT):
        pick = [reps[i] for i in rng.integers(0, k, size=k)]
        est = estimate(pick, g)
        vals.append(est["delta"] if est["delta"] is not None else float("nan"))
    vals = np.array(vals)
    vals = vals[~np.isnan(vals)]
    return (float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")), (vals if len(vals) else None)


def read(cells):
    rng = np.random.default_rng(78)
    table = []
    for g in sorted(cells):
        reps = [cells[g][r] for r in sorted(cells[g])]
        est = estimate(reps, g)
        est["g"] = g
        est["n_rep"] = len(reps)
        est["read"] = est["rings"] >= MIN_SIGHTINGS and est["patches"] >= MIN_SIGHTINGS and est["delta"] is not None
        est["sigma"], est["boot"] = (bootstrap(reps, g, rng) if est["read"] else (float("nan"), None))
        table.append(est)
    return table, verdict(table)


def verdict(table):
    read = [t for t in table if t["read"]]
    if len(read) < 2:
        return "NOT READ"
    pos = all(t["delta"] > 2 * t["sigma"] for t in read)
    neg = all(t["delta"] < -2 * t["sigma"] for t in read)
    zero = all(abs(t["delta"]) <= 2 * t["sigma"] for t in read)
    lo, hi = min(read, key=lambda t: t["rho"]), max(read, key=lambda t: t["rho"])
    diff = hi["delta"] - lo["delta"]
    sig = math.sqrt(hi["sigma"] ** 2 + lo["sigma"] ** 2)
    if pos:
        return "CURL PENALISED, GROWING" if diff > 2 * sig else ("CURL PENALISED, FLAT" if abs(diff) <= 2 * sig else "MIXED")
    if zero:
        return "NO GAP"
    if neg:
        return "CURL FAVOURED"
    return "MIXED"


def main(results_dir="results", *names):
    names = list(names) or NAMES
    paths = [Path(results_dir) / (name + ".csv") for name in names]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit("not all of the registered set is present; missing: %s" % ", ".join(map(str, missing)))
    cells = load(paths)
    table, word = read(cells)
    print("T78, read under PREREGISTRATION T78 (%s)" % ", ".join(map(str, paths)))
    print("%6s %5s %7s %8s %8s | %9s %9s %9s | %9s %9s %9s | %8s %8s %10s %s"
          % ("g", "reps", "rho", "rings", "patches", "F_ring", "E_ring", "S_ring", "F_patch", "E_patch", "S_patch",
             "Delta", "sigma", "lam_eff-1", "read"))
    for t in table:
        r, p = t["ring"], t["patch"]
        fmt = lambda o, k: ("%9.3f" % o[k]) if o else "%9s" % "-"
        print("%6.3f %5d %7.3f %8d %8d | %s %s %s | %s %s %s | %8s %8s %10s %s"
              % (t["g"], t["n_rep"], t["rho"], t["rings"], t["patches"],
                 fmt(r, "F"), fmt(r, "E"), fmt(r, "S"), fmt(p, "F"), fmt(p, "E"), fmt(p, "S"),
                 ("%+8.3f" % t["delta"]) if t["delta"] is not None else "-",
                 ("%8.3f" % t["sigma"]) if t["read"] else "-",
                 ("%+10.4f" % t["lam_eff_minus_1"]) if t["delta"] is not None else "-",
                 "yes" if t["read"] else "NOT READ"))
    print("Verdict: %s" % word)
    print("Her prediction: CURL PENALISED, GROWING. Ours: NO GAP (or CURL PENALISED, FLAT, read as the point-count limit).")


if __name__ == "__main__":
    main(*sys.argv[1:])
