"""T82: read the twin-gate runs by the rules of PREREGISTRATION.md section T82. Usage:

    python scripts/analyse_t82.py

Reads the named stage-1 files (results/t82_{model,layer}_lam{110,125,140}.csv) and, for the comparisons P3 and P4, T71's
stage-1 files of the same arm and curling cost (results/t71_{model,layer}_lam{110,125,140}.csv).

Per replica: the births (sweeps in which the daughter count rose; under release_to "twins" each birth is two twins), the
first birth, the refolds after it, the median period between births, the energy per twin, and whether a clean move
fired. Word: CYCLES (two or more births and at least one refold), ONE BIRTH, NEVER SPLITS. Per cell (lam, L, rate,
arm): the count of each word, the median first birth against 4a / r, the median energy per twin against a N_slab, the
median period and its ratio to T71's median period in the same cell, and the replicas in which a clean move fired; then
P1 to P4 scored against their thresholds (P4 with T71's clean-move count and energy share printed beside it). The
held-out size is not scored here; that waits for the dated amendment.
"""
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

CELLS = [(arm, lam) for arm in ("model", "layer") for lam in ("110", "125", "140")]


def read_replica(rows):
    rows = sorted(rows, key=lambda r: int(r["sweep"]))
    lam, n_slab, rate = float(rows[0]["lam"]), int(rows[0]["N_slab"]), float(rows[0]["rate"])
    a = 4.0 * (lam - 1.0)
    births = [int(r["sweep"]) for prev, r in zip(rows[:-1], rows[1:]) if int(r["daughters"]) > int(prev["daughters"])]
    if int(rows[0]["daughters"]) > 0:
        births = [int(rows[0]["sweep"])] + births
    n_dau = int(rows[-1]["daughters"])
    first_split = rows[0]["first_split"]
    first_split = int(first_split) if str(first_split) not in ("", "None") else None
    refolds = None
    if first_split is not None:
        at = [r for r in rows if int(r["sweep"]) >= first_split]
        refolds = int(rows[-1]["n_contract"]) - int(at[0]["n_contract"])
    out = dict(a_n=a * n_slab, t_gate2=4 * a / rate, births=len(births), first_birth=births[0] if births else None,
               refolds=refolds, period=float(np.median(np.diff(births))) if len(births) >= 2 else None,
               per_twin=float(rows[-1]["daughter_energy"]) / n_dau if n_dau else None,
               clean=int(rows[-1]["n_clean"]) > 0)
    if first_split is None:
        out["word"] = "NEVER SPLITS"
    elif len(births) >= 2 and refolds and refolds >= 1:
        out["word"] = "CYCLES"
    else:
        out["word"] = "ONE BIRTH"
    return out


def read_cells(path):
    groups = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(path, newline="")):
        groups[(float(r["lam"]), int(r["L"]), float(r["rate"]), r["move_set"])][int(r["replica"])].append(r)
    return {k: [read_replica(rs) for _, rs in sorted(reps.items())] for k, reps in sorted(groups.items())}


def med(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


def score(key, reads, t71_reads):
    """P1 to P4 of PREREGISTRATION T82 for one cell; None where a prediction does not apply to the cell."""
    lam, L, rate, arm = key
    fast = abs(lam - 1.10) < 1e-9 and abs(rate - 1.0 / 16) < 1e-9
    clean_any = any(x["clean"] for x in reads)
    first = med([x["first_birth"] for x in reads])
    per_twin = med([x["per_twin"] for x in reads])
    period = med([x["period"] for x in reads])
    t71_period = med([x["period"] for x in t71_reads]) if t71_reads else None
    p = {}
    p["P1"] = sum(x["word"] == "CYCLES" for x in reads) >= 15
    p["P2"] = (abs(per_twin / reads[0]["a_n"] - 1) <= 1e-6) if (per_twin and (arm == "layer" or not clean_any)) else None
    p["P3 first"] = (first is not None and reads[0]["t_gate2"] <= first <= reads[0]["t_gate2"] + 10)
    p["P3 period"] = (None if fast or not (period and t71_period) else 1.7 <= period / t71_period <= 2.3)
    if fast and arm == "model" and t71_reads:
        t71_clean = sum(x["clean"] for x in t71_reads)
        t71_share = med([x["per_twin"] for x in t71_reads]) / t71_reads[0]["a_n"]
        p["P4"] = sum(x["clean"] for x in reads) >= t71_clean and per_twin / reads[0]["a_n"] <= t71_share
        p["P4 comparators (T71 clean, T71 share)"] = (t71_clean, round(t71_share, 4))
    return p


def main():
    for arm, lam in CELLS:
        path = Path("results") / f"t82_{arm}_lam{lam}.csv"
        if not path.exists():
            print("missing", path)
            continue
        t71_path = Path("results") / f"t71_{arm}_lam{lam}.csv"
        t71 = read_cells(t71_path) if t71_path.exists() else {}
        for key, reads in read_cells(path).items():
            words = Counter(x["word"] for x in reads)
            period = med([x["period"] for x in reads])
            t71_period = med([x["period"] for x in t71.get(key, [])])
            ratio = period / t71_period if period and t71_period else None
            print("lam=%.2f L=%2d r=%-8g %-6s | %s | first birth median %s vs 4a/r %.1f | per twin %s vs a N_slab %.1f | "
                  "period %s, ratio to T71 %s | clean fired %d/%d"
                  % (key[0], key[1], key[2], key[3], dict(words), med([x["first_birth"] for x in reads]),
                     reads[0]["t_gate2"], med([x["per_twin"] for x in reads]), reads[0]["a_n"], period,
                     None if ratio is None else round(ratio, 3), sum(x["clean"] for x in reads), len(reads)))
            print("    scored:", score(key, reads, t71.get(key)))


if __name__ == "__main__":
    main()
