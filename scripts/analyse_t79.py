"""T79, read under the registered rules (PREREGISTRATION.md section T79). Usage:

    python scripts/analyse_t79.py [results_dir]

Identity checks first: the replayed ring3 count equals T78's n_ring and the replayed 4 x 4 count equals T78's n_patch
on every snapshot of every cell; the reading refuses otherwise. Then per coupling: S_X = (E_X - F_X)/g for every
object (T78's estimators), the flat line (least squares of S on links over the flat grids with >= 50 sightings, at
least four of them), the residual r = S_ring3 - S_fit(20) and the twin gap S_ring2 - S_ring3, with bootstrap errors
over replicas (1,000 resamples, seed 79, the line refitted in each). Verdict words: PENALTY ON CURL / DENSITY, NOT
CURL / CURL FAVOURED / MIXED / NOT READ.
"""
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

MIN_SIGHTINGS = 50
BOOT = 1000
GRIDS = [(2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (3, 5), (4, 4), (4, 5)]
T79 = ["t79_curl_family_n196_g3300", "t79_curl_family_n196_g3433", "t79_curl_family_n196_g3550", "t79_curl_family_n196_g3697"]
T78 = ["t78_curl_vs_flat_n196_g3300", "t78_curl_vs_flat_n196_g3433", "t78_curl_vs_flat_n196_g3550", "t78_curl_vs_flat_n196_g3697"]


def links_of(a, b):
    return a * (b - 1) + b * (a - 1)


OBJECTS = {"ring3": dict(points=12, links=20, squares=9, flat=False),
           "ring2": dict(points=12, links=16, squares=5, flat=False)}
for a, b in GRIDS:
    OBJECTS["flat_%dx%d" % (a, b)] = dict(points=a * b, links=links_of(a, b), squares=(a - 1) * (b - 1), flat=True)


def load(paths):
    cells = defaultdict(lambda: defaultdict(list))
    for path in paths:
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                cells[float(r["g"])][int(r["replica"])].append(r)
    return cells


def identity_check(t79, t78):
    for g in t79:
        for rep in t79[g]:
            a, b = t79[g][rep], t78[g][rep]
            if len(a) != len(b):
                raise SystemExit("replay length differs at g=%s rep %d" % (g, rep))
            for x, y in zip(a, b):
                if x["snap"] != y["snap"] or x["n_ring3"] != y["n_ring"] or x["n_patch"] != y["n_patch"] or x["H"] != y["H"]:
                    raise SystemExit("replay differs from T78 at g=%s rep %d snap %s" % (g, rep, x["snap"]))


def estimates(reps, g):
    """reps: list of replicas, each a list of row dicts. Returns per-object dict(mean, F, E, S, sightings)."""
    rows = [r for rep in reps for r in rep]
    n = float(rows[0]["N"])
    h = np.array([float(r["H"]) for r in rows])
    out = {}
    for name in OBJECTS:
        cnt = np.array([float(r["n_" + name]) for r in rows])
        tot = int(cnt.sum())
        m = cnt.mean()
        if tot < MIN_SIGHTINGS or m <= 0:
            out[name] = dict(sightings=tot, S=None)
            continue
        f = -g * math.log(m / n)
        e = (h * cnt).mean() / m - h.mean()
        out[name] = dict(sightings=tot, mean=m, F=f, E=e, S=(e - f) / g)
    return out


def fit_line(est, key="links"):
    xs = [OBJECTS[k][key] for k in OBJECTS if OBJECTS[k]["flat"] and est[k]["S"] is not None]
    ys = [est[k]["S"] for k in OBJECTS if OBJECTS[k]["flat"] and est[k]["S"] is not None]
    if len(xs) < 4:
        return None
    slope, intercept = np.polyfit(np.array(xs, float), np.array(ys, float), 1)
    return float(slope), float(intercept), len(xs)


def read_cell(reps, g, rng):
    est = estimates(reps, g)
    line = fit_line(est)
    readable = est["ring3"]["S"] is not None and est["ring2"]["S"] is not None and line is not None
    out = dict(g=g, est=est, line=line, read=readable, lines_other={k: fit_line(est, k) for k in ("points", "squares")})
    if not readable:
        return out
    r = est["ring3"]["S"] - (line[0] * 20 + line[1])
    twin = est["ring2"]["S"] - est["ring3"]["S"]
    rs, twins = [], []
    k = len(reps)
    for _ in range(BOOT):
        pick = [reps[i] for i in rng.integers(0, k, size=k)]
        e2 = estimates(pick, g)
        l2 = fit_line(e2)
        if l2 is None or e2["ring3"]["S"] is None or e2["ring2"]["S"] is None:
            continue
        rs.append(e2["ring3"]["S"] - (l2[0] * 20 + l2[1]))
        twins.append(e2["ring2"]["S"] - e2["ring3"]["S"])
    out.update(residual=r, sigma_r=float(np.std(rs, ddof=1)) if len(rs) > 1 else float("nan"),
               twin=twin, sigma_twin=float(np.std(twins, ddof=1)) if len(twins) > 1 else float("nan"))
    return out


def verdict(cells):
    read = [c for c in cells if c["read"]]
    if len(read) < 2:
        return "NOT READ"
    below = all(c["residual"] < -2 * c["sigma_r"] and c["twin"] > 2 * c["sigma_twin"] for c in read)
    on = all(abs(c["residual"]) <= 2 * c["sigma_r"] for c in read)
    above = all(c["residual"] > 2 * c["sigma_r"] for c in read)
    if below:
        return "PENALTY ON CURL"
    if on:
        return "DENSITY, NOT CURL"
    if above:
        return "CURL FAVOURED"
    return "MIXED"


def main(results_dir="results"):
    p79 = [Path(results_dir) / (n + ".csv") for n in T79]
    p78 = [Path(results_dir) / (n + ".csv") for n in T78]
    missing = [p for p in p79 + p78 if not p.exists()]
    if missing:
        raise SystemExit("missing: %s" % ", ".join(map(str, missing)))
    t79, t78 = load(p79), load(p78)
    identity_check(t79, t78)
    print("identity check passed: ring3 = T78 n_ring, 4 x 4 = T78 n_patch, H equal, on every snapshot")
    rng = np.random.default_rng(79)
    cells = []
    for g in sorted(t79):
        reps = [t79[g][r] for r in sorted(t79[g])]
        cells.append(read_cell(reps, g, rng))
    print("%-10s %6s %6s %7s | " % ("object", "points", "links", "squares") + " ".join("%12s" % ("g=%.3f" % c["g"]) for c in cells))
    for name, o in OBJECTS.items():
        vals = []
        for c in cells:
            e = c["est"][name]
            vals.append(("%7.3f (%d)" % (e["S"], e["sightings"])) if e["S"] is not None else ("     - (%d)" % e["sightings"]))
        print("%-10s %6d %6d %7d | " % (name, o["points"], o["links"], o["squares"]) + " ".join("%12s" % v for v in vals))
    print("\nflat line S = slope x links + intercept (flat grids with >= 50 sightings); fits on points and squares beside")
    for c in cells:
        if c["line"]:
            lp, ls = c["lines_other"]["points"], c["lines_other"]["squares"]
            print("  g=%.3f: links slope %+.4f intercept %+.3f (%d grids); points slope %+.4f; squares slope %+.4f"
                  % (c["g"], c["line"][0], c["line"][1], c["line"][2], lp[0] if lp else float("nan"), ls[0] if ls else float("nan")))
    print("\n%6s %10s %8s %10s %8s %s" % ("g", "residual", "sigma", "twin gap", "sigma", "read"))
    for c in cells:
        if c["read"]:
            print("%6.3f %+10.3f %8.3f %+10.3f %8.3f yes" % (c["g"], c["residual"], c["sigma_r"], c["twin"], c["sigma_twin"]))
        else:
            print("%6.3f %10s %8s %10s %8s NOT READ" % (c["g"], "-", "-", "-", "-"))
    print("Verdict: %s" % verdict(cells))
    print("Her prediction: PENALTY ON CURL. Ours: DENSITY, NOT CURL.")


if __name__ == "__main__":
    main(*sys.argv[1:2])
