"""The pre-registered T24 reading: the lambda map a third time, with the energy check read from the saved
wiring. Usage:

    python scripts/analyse_t24.py

Implements PREREGISTRATION.md section T24 (written 2026-09-24, before the runs) on results/t24_lam<tag>_n<N>.csv
and their _adj directories. Everything is T23's (scripts/analyse_t23.py: (a') memoryless sized to the sample,
(b) two orders side by side, (c) one front, gate 2, the window and edge verdicts) except gate 3, which
becomes gate 3': for every decay that reached 75 % conversion the saved final graph must exist and be
valid; each such decay is then classified by the **exact** energy of its final wiring above the flat
torus, h = 16(N - S) + 4 lambda X computed from the saved graph: FLAT (h = 0), LEDGE (h = 24 lambda - 16,
one curled column), or OTHER (any other resting state; its local-dimension census is reported). Nothing
about the classification fails a cell: the old catalog of allowed states was an assumption the data
outgrew (VISION Update 14; O44). Reported beside it, not scored: the exact release per point, the window-
mean release the runner recorded, and their difference, which is the thermal excitation of the final state
at g = 1.5 that tripped the old gate. Pure functions of parsed rows, tested in tests/test_t24.py.
"""
import csv
import glob
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import analyse_t8                                                 # noqa: E402
import analyse_t23                                                # noqa: E402
from graphity.cqg import NO_CAP, is_valid, surplus, total_squares  # noqa: E402
from graphity.dimension import local_dimension                    # noqa: E402

EXACT_TOL = 1e-6          # exact arithmetic on integers times lambda; this is rounding, not a tolerance
D_BINS = 7


def exact_energy(adj, lam):
    n = adj.shape[0]
    return 16.0 * (n - int(total_squares(adj))) + 4.0 * lam * int(surplus(adj))


def classify_exact(h, n, lam):
    """FLAT, LEDGE or OTHER from the exact energy above the flat torus."""
    if abs(h) <= EXACT_TOL:
        return "FLAT"
    if abs(h - (24.0 * lam - 16.0)) <= EXACT_TOL:
        return "LEDGE"
    return "OTHER"


def read_final_states(rows, adj_dir, lam):
    """{replica: dict(valid, h, kind, release_exact, census)} for every row with a saved graph."""
    out = {}
    for r in rows:
        f = Path(adj_dir) / ("N%s_rep%s.npz" % (r["N"], r["replica"]))
        if not f.exists():
            continue
        z = np.load(f)
        adj = z["adj"]
        n = adj.shape[0]
        h = exact_energy(adj, lam)
        d = local_dimension(adj)
        out[r["replica"]] = dict(valid=bool(is_valid(adj, NO_CAP)), h=h, kind=classify_exact(h, n, lam),
                                 release_exact=(float(z["h0"]) - h) / n,
                                 census=tuple(int(c) for c in np.bincount(d, minlength=D_BINS)[:D_BINS]))
    return out


def gate3_prime(rows, states):
    """Every decay that reached 75 % has a saved, valid final graph."""
    reached = [r for r in rows if r["reached"] and float(r["reached"]) >= 0.75]
    return all(r["replica"] in states and states[r["replica"]]["valid"] for r in reached)


def cell(rows, n, lam, states):
    """T23's cell with gate 3 replaced by gate 3', plus the exact resting-state census."""
    out = analyse_t23.cell(rows, n, lam, read=None)
    reached = [r for r in rows if r["reached"] and float(r["reached"]) >= 0.75]
    kinds = [states[r["replica"]]["kind"] for r in reached if r["replica"] in states]
    out.update({k.lower(): kinds.count(k) for k in ("FLAT", "LEDGE", "OTHER")})
    out["gate3"] = gate3_prime(rows, states) if out["metastable"] else False
    ex = [states[r["replica"]]["release_exact"] for r in reached if r["replica"] in states]
    rec = [float(r["released"]) for r in reached if r["replica"] in states]
    out["release_exact"] = float(np.mean(ex)) if ex else math.nan
    out["release_recorded"] = float(np.mean(rec)) if rec else math.nan
    out["thermal_excess"] = float(np.mean(np.array(ex) - np.array(rec))) if ex else math.nan
    out["other_census"] = sorted({states[r["replica"]]["census"] for r in reached
                                  if r["replica"] in states and states[r["replica"]]["kind"] == "OTHER"})
    return out


def load(out_dir="results"):
    data = defaultdict(dict)
    for f in sorted(glob.glob(str(Path(out_dir) / "t24_lam*_n*.csv"))):
        m = re.search(r"t24_lam(\d+)_n(\d+)\.csv$", f)
        lam, n = int(m.group(1)) / 100.0, int(m.group(2))
        rows = list(csv.DictReader(open(f, newline="")))
        data[lam][n] = (rows, Path(f).with_suffix("").as_posix() + "_adj")
    return data


def main(out_dir="results"):
    data = load(out_dir)
    status, pooled = {}, {}
    for lam in sorted(data):
        cells, pooled[lam] = [], []
        for n in sorted(data[lam]):
            rows, adj_dir = data[lam][n]
            states = read_final_states(rows, adj_dir, lam)
            c = cell(rows, n, lam, states)
            cells.append(c)
            pooled[lam] += [(r, r["replica"] in states and states[r["replica"]]["kind"] == "FLAT") for r in rows]
            if not c["metastable"]:
                print("lambda=%.2f N=%-4d NOT METASTABLE (median f_200 %.2f)" % (lam, n, c["f200_median"]))
                continue
            print("lambda=%.2f N=%-4d gate2 %s gate3' %s | read %d, CV of wait-200 %.3f in [%.3f, %.3f] %s | "
                  "d in {1,2} %.3f | largest %.2f | sharp %s | wait %.0f vs tau %.0f | ends flat %d ledge %d other %d "
                  "| release exact %.4f recorded %.4f (thermal %.4f)"
                  % (lam, n, "pass" if c["gate2"] else "FAIL", "pass" if c["gate3"] else "FAIL", c.get("n_read", 0),
                     c["cv"], *c.get("band", (math.nan, math.nan)), "Y" if c["a"] else "n", c["two"], c["largest"],
                     "Y" if c["sharp"] else "n", c["wait"], analyse_t8.tau(lam), c["flat"], c["ledge"], c["other"],
                     c["release_exact"], c["release_recorded"], c["thermal_excess"]))
            for cen in c["other_census"]:
                print("      other resting state, census by d: %s" % (cen,))
        status[lam] = analyse_t8.lam_status(cells)
    window = {l: s for l, s in status.items() if l in analyse_t23.WINDOW}
    print("\nWINDOW (1.05 to 1.30):", analyse_t23.window_verdict(window))
    if analyse_t23.EDGE in pooled and analyse_t23.BELOW_EDGE in pooled:
        e = analyse_t23.edge_report(pooled[analyse_t23.BELOW_EDGE], pooled[analyse_t23.EDGE])
        print("EDGE (1.35 against 1.30): ending flat %.2f -> %.2f (E1 %s); converted region in more than one piece "
              "at 25%% %.2f -> %.2f (E2 %s): %s"
              % (*e["flat"], "yes" if e["e1"] else "no", *e["multi"], "yes" if e["e2"] else "no", e["verdict"]))
    else:
        print("EDGE: no data at 1.30 and 1.35")


if __name__ == "__main__":
    main(*sys.argv[1:])
